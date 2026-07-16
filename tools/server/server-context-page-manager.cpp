// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration using kv_ssd_cache

#include "server-context-page-manager.h"
#include "server-context-ssd-cache.h"
#include "server-context.h"
#include "server-task.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <system_error>
namespace fs = std::filesystem;

namespace llama {

// FNV-1a 64-bit hash of a byte string. Mirrors kv_ssd_hash_tokens in shape
// but operates on raw bytes so it works for any string, not just tokens.
static uint64_t fnv1a_string(const std::string & s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ULL;
    }
    return h;
}

server_context_page_manager::server_context_page_manager(
    const char* ssd_path,
    const kv_eviction_config* cfg,
    size_t /* n_tokens_total */,
    size_t max_cross_slot_checkpoints
) : max_cross_slot_checkpoints_(max_cross_slot_checkpoints)
{
    ssd_base_path_ = ssd_path;
    std::error_code ec_fs;
    fs::create_directories(ssd_path, ec_fs);

    kv_ssd_config ssd_cfg;
    if (cfg) {
        ssd_cfg.hot_ram_bytes = cfg->max_hot_bytes > 0 ? cfg->max_hot_bytes : 2ULL * 1024 * 1024 * 1024;
        ssd_cfg.warm_ram_bytes = cfg->max_warm_bytes > 0 ? cfg->max_warm_bytes : 1ULL * 1024 * 1024 * 1024;
        ssd_cfg.hot_window_tokens = cfg->hot_window_tokens;
        ssd_cfg.hot_turns = cfg->turn_inactivity_threshold > 0 ? cfg->turn_inactivity_threshold : 2;
        ssd_cfg.warm_turns = cfg->turn_inactivity_threshold > 0 ? cfg->turn_inactivity_threshold * 2 : 4;
        ssd_cfg.auto_size = cfg->auto_size;
        ssd_cfg.unified_memory = cfg->unified_memory;
        ssd_cfg.max_cold_checkpoints = cfg->max_cold_checkpoints;
        ssd_cfg.memory_reserve = cfg->memory_reserve;
    }
    if (ssd_cfg.hot_ram_bytes == 0) ssd_cfg.hot_ram_bytes = 2ULL * 1024 * 1024 * 1024;
    if (ssd_cfg.warm_ram_bytes == 0) ssd_cfg.warm_ram_bytes = 1ULL * 1024 * 1024 * 1024;
    if (ssd_cfg.hot_turns == 0) ssd_cfg.hot_turns = 2;
    if (ssd_cfg.warm_turns == 0) ssd_cfg.warm_turns = 4;

    // Store config for creating per-conversation caches later
    // (save a copy of the config)
    config_ = ssd_cfg;
}

void server_context_page_manager::set_no_fsync(bool no_fsync) {
    config_.no_fsync = no_fsync;
}

server_context_page_manager::~server_context_page_manager() {
    // Each unique_ptr in conv_caches_ handles its own kv_ssd_free
}

void server_context_page_manager::set_model_info(const struct llama_model* model,
                                                   int cache_type_k, int cache_type_v) {
    if (!model) return;

    char desc_buf[2048];
    int desc_len = llama_model_desc(model, desc_buf, sizeof(desc_buf));
    if (desc_len < 0) {
        LOG_WRN("SSD cache: llama_model_desc() failed, skipping compat_hash\n");
        return;
    }

    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < desc_len; i++) {
        h ^= (uint64_t)(unsigned char)desc_buf[i];
        h *= 1099511628211ULL;
    }
    // Include build commit in compat_hash so checkpoints from different
    uint32_t tk = (uint32_t)cache_type_k;
    h ^= (uint64_t)(tk & 0xFF);         h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 8) & 0xFF);  h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 16) & 0xFF); h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 24) & 0xFF); h *= 1099511628211ULL;
    uint32_t tv = (uint32_t)cache_type_v;
    h ^= (uint64_t)(tv & 0xFF);         h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 8) & 0xFF);  h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 16) & 0xFF); h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 24) & 0xFF); h *= 1099511628211ULL;

    model_compat_hash_ = h;

    // Set compat_hash on any already-created cache instances
    for (auto& [conv, wrapper] : conv_wrappers_) {
        wrapper->set_compat_hash(h);
    }

    LOG_INF("SSD cache: model compat_hash %016lx (arch dims + type_k=%d type_v=%d)\n",
            (unsigned long)h, cache_type_k, cache_type_v);
}

server_ssd_cache* server_context_page_manager::get_or_create_cache(uint64_t conv_hash) {
    if (conv_hash == 0) return nullptr;

    auto it = conv_wrappers_.find(conv_hash);
    if (it != conv_wrappers_.end()) {
        return it->second.get();
    }

    // Evict oldest conversation if at max
    if ((int)conv_caches_.size() >= max_conversations) {
        uint64_t oldest_conv = 0;
        time_t oldest_mtime = 0;

        for (const auto& [cv, cache] : conv_caches_) {
            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)cv);
            fs::path dir = fs::path(ssd_base_path_) / hex;

            std::error_code ec;
            auto ftime = fs::last_write_time(dir, ec);
            if (!ec) {
                auto mtime = std::chrono::system_clock::to_time_t(
                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()));
                if (oldest_conv == 0 || mtime < oldest_mtime) {
                    oldest_mtime = mtime;
                    oldest_conv = cv;
                }
            }
        }

        if (oldest_conv != 0) {
            LOG_WRN("SSD cache: evicting conversation %016lx (max=%d reached)\n",
                     (unsigned long)oldest_conv, max_conversations);

            // Delete conversation directory and all its files
            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)oldest_conv);
            fs::path dir = fs::path(ssd_base_path_) / hex;

            for (const auto& entry : fs::directory_iterator(dir)) {
                fs::remove(entry.path());
            }
            fs::remove(dir);

            conv_wrappers_.erase(oldest_conv);
            conv_caches_.erase(oldest_conv);
        }
    }

    // Create new cache for this conversation
    auto raw = kv_ssd_init(ssd_base_path_.c_str(), &config_, conv_hash);
    if (!raw) return nullptr;

    auto cache_ptr = std::unique_ptr<kv_ssd_cache>(raw);
    auto wrapper = std::make_unique<server_ssd_cache>(raw);

    // Apply model compat_hash if already set
    if (model_compat_hash_ != 0) {
        wrapper->set_compat_hash(model_compat_hash_);
    }

    server_ssd_cache* result = wrapper.get();
    conv_caches_[conv_hash] = std::move(cache_ptr);
    conv_wrappers_[conv_hash] = std::move(wrapper);

    LOG_INF("SSD cache: created new conversation cache conv=%016lx (total=%zu)\n",
             (unsigned long)conv_hash, conv_caches_.size());

    return result;
}

uint64_t server_context_page_manager::get_timestamp_ms() const {
    auto now = std::chrono::system_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

void server_context_page_manager::evict_slot_internal(uint32_t slot_id) {
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return;
    checkpoints_.erase(it);
}

bool server_context_page_manager::store_checkpoint(
    uint32_t slot_id,
    struct llama_context* ctx,
    const common_prompt_checkpoint& ckpt,
    uint32_t turn_id
) {
    return store_checkpoint_with_tokens(slot_id, ctx, nullptr, ckpt, nullptr, 0, turn_id);
}

bool server_context_page_manager::store_checkpoint_with_tokens(
    uint32_t slot_id,
    struct llama_context* ctx,
    struct llama_context* ctx_dft,
    const common_prompt_checkpoint& ckpt,
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t turn_id,
    uint64_t conv_hash,
    const std::string& user_id
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!ckpt.data_tgt.data()) return false;

    // Get or create the appropriate cache. user_id routes to a user-scoped
    // cache in the "u/" namespace; conv_hash routes to the anonymous bucket.
    server_ssd_cache* sc = user_id.empty()
        ? get_or_create_cache(conv_hash)
        : get_or_create_user_cache(user_id);
    if (!sc) return false;

    // Evict if needed
    if (checkpoints_.size() >= max_cross_slot_checkpoints_) {
        auto it = std::min_element(checkpoints_.begin(), checkpoints_.end(),
            [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });
        if (it != checkpoints_.end()) evict_slot_internal(it->first);
    }

    uint64_t ckpt_id = sc->store(slot_id, ctx, ctx_dft, ckpt, tokens, tokens_size, turn_id);
    if (ckpt_id == 0) return false;

    stored_checkpoint sc2;
    sc2.checkpoint_id = ckpt_id;
    sc2.slot_id = slot_id;
    sc2.turn_id = turn_id;
    sc2.size_bytes = ckpt.data_tgt.size() + ckpt.data_dft.size();
    sc2.n_tokens = ckpt.n_tokens;
    sc2.pos_min = ckpt.pos_min;
    sc2.pos_max = ckpt.pos_max;
    sc2.last_access = get_timestamp_ms();
    sc2.access_count = 0;
    if (tokens && tokens_size > 0) {
        sc2.tokens.assign(tokens, tokens + std::min(tokens_size, (size_t)256));
    }

    checkpoints_.emplace(slot_id, std::move(sc2));
    return true;
}

bool server_context_page_manager::load_checkpoint(
    uint32_t slot_id,
    uint32_t /* turn_id */,
    struct llama_context* ctx,
    struct llama_context* ctx_dft,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    std::vector<uint8_t>* out_spec_data
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return false;

    // Find which cache has this checkpoint
    server_ssd_cache* sc = nullptr;
    // We need to know which conversation cache stores this checkpoint.
    // Since we removed conv_hash from checkpoint metadata, we iterate all caches.
    for (auto& [conv, wrapper] : conv_wrappers_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(
            conv_caches_[conv].get(), it->second.checkpoint_id);
        if (meta) {
            sc = wrapper.get();
            break;
        }
    }
    if (!sc) return false;

    // Load from SSD cache, which will promote to hot tier
    bool ok = sc->load(it->second.checkpoint_id, ctx, ctx_dft, out_pos_min, out_pos_max, out_n_tokens, out_spec_data);

    if (ok) {
        it->second.last_access = get_timestamp_ms();
        it->second.access_count++;
        cache_hits_++;
    } else {
        cache_misses_++;
    }

    return ok;
}

bool server_context_page_manager::load_checkpoint_by_id(
    uint64_t checkpoint_id,
    struct llama_context* ctx,
    struct llama_context* ctx_dft,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    std::vector<uint8_t>* out_spec_data
) {
    if (checkpoint_id == 0) return false;

    // Find which cache has this checkpoint
    server_ssd_cache* sc = nullptr;
    for (auto& [conv, wrapper] : conv_wrappers_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(
            conv_caches_[conv].get(), checkpoint_id);
        if (meta) {
            sc = wrapper.get();
            break;
        }
    }
    if (!sc) return false;

    bool ok = sc->load(checkpoint_id, ctx, ctx_dft, out_pos_min, out_pos_max, out_n_tokens, out_spec_data);

    if (ok) {
        cache_hits_++;
    } else {
        cache_misses_++;
    }

    return ok;
}

void server_context_page_manager::prefetch_for_slot(uint32_t slot_id, uint32_t /* turn_id */) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return;

    // Prefetch all cold checkpoints for this slot across all conversation caches.
    // This triggers kernel page cache readahead so the SSD I/O overlaps with
    // subsequent CPU work (token matching, state restoration, etc.).
    for (auto& [conv, cache] : conv_caches_) {
        kv_ssd_prefetch_slot(cache.get(), slot_id);
    }
    for (auto& [key, cache] : user_caches_) {
        kv_ssd_prefetch_slot(cache.get(), slot_id);
    }
}

void server_context_page_manager::on_turn_complete(uint32_t turn_id) {
    // Notify all cache instances
    for (auto& [conv, wrapper] : conv_wrappers_) {
        wrapper->on_turn_complete(turn_id);
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto& [slot_id, sc] : checkpoints_) {
        sc.turn_id = turn_id;
    }
}

bool server_context_page_manager::find_matching_checkpoint(
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t current_turn,
    uint32_t& out_slot_id,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    uint64_t conv_hash,
    int32_t n_past,
    uint64_t max_n_tokens,
    const std::string& user_id
) {
    if (!user_id.empty()) {
        // user-scoped lookups never escape the user's own cache. cross-user
        // continuation matching is a privacy violation, so we skip it.
        const uint64_t key = fnv1a_string(user_id);
        server_ssd_cache* sc = get_or_create_user_cache(user_id);
        if (!sc) return false;

        uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past);
        if (ckpt_id == 0) { cache_misses_++; return false; }

        for (const auto& [slot_id, cp] : checkpoints_) {
            if (cp.checkpoint_id == ckpt_id) {
                out_slot_id = slot_id;
                out_pos_min = cp.pos_min;
                out_pos_max = cp.pos_max;
                out_n_tokens = cp.n_tokens;
                cache_hits_++;
                return true;
            }
        }

        kv_ssd_cache* raw = user_caches_[key].get();
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(raw, ckpt_id);
        if (meta) {
            out_slot_id = meta->slot_id;
            out_pos_min = meta->pos_min;
            out_pos_max = meta->pos_max;
            out_n_tokens = meta->n_tokens;
            cache_hits_++;
            return true;
        }
        cache_misses_++;
        return false;
    }

    // Try exact conversation match first
    uint64_t effective_conv = conv_hash;

    // If this conv_hash doesn't have a cache yet, try continuation matching
    if (effective_conv != 0 && conv_wrappers_.find(effective_conv) == conv_wrappers_.end()) {
        uint64_t continuation = kv_ssd_find_continuation(
            ssd_base_path_.c_str(),
            (const uint32_t*)tokens, tokens_size,
            0.90f, model_compat_hash_);
        if (continuation != 0) {
            effective_conv = continuation;
            LOG_INF("SSD cache: reusing conversation %016lx (90%%+ prefix match)\n",
                     (unsigned long)continuation);
        }
    }

    server_ssd_cache* sc = get_or_create_cache(effective_conv);
    if (!sc) return false;

    uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past);
    if (ckpt_id == 0) {
        cache_misses_++;
        return false;
    }

    // Look up in checkpoints_ map first
    for (const auto& [slot_id, cp] : checkpoints_) {
        if (cp.checkpoint_id == ckpt_id) {
            out_slot_id = slot_id;
            out_pos_min = cp.pos_min;
            out_pos_max = cp.pos_max;
            out_n_tokens = cp.n_tokens;
            cache_hits_++;
            return true;
        }
    }

    // Look up in the cache's own metadata
    kv_ssd_cache* raw = conv_caches_[effective_conv].get();
    const kv_ssd_checkpoint* meta = kv_ssd_get_meta(raw, ckpt_id);
    if (meta) {
        out_slot_id = meta->slot_id;
        out_pos_min = meta->pos_min;
        out_pos_max = meta->pos_max;
        out_n_tokens = meta->n_tokens;
        cache_hits_++;
        return true;
    }

    cache_misses_++;
    return false;
}

bool server_context_page_manager::find_and_load_checkpoint(
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t current_turn,
    struct llama_context* ctx,
    struct llama_context* ctx_dft,
    uint32_t dest_seq_id,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    std::vector<uint8_t>* out_spec_data,
    uint64_t conv_hash,
    int32_t n_past,
    uint64_t max_n_tokens,
    int32_t* out_lcp,
    float* out_overlap,
    bool* out_is_continuation,
    const std::string& user_id
) {
    if (!user_id.empty()) {
        // user-scoped cold-start lookups never escape the user's own cache.
        // cross-user continuation matching is a privacy violation.
        server_ssd_cache* sc = get_or_create_user_cache(user_id);
        if (!sc) return false;

        int32_t match_lcp = 0;
        uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past, &match_lcp);
        if (ckpt_id == 0) { cache_misses_++; return false; }

        // Prefetch the checkpoint file from SSD while we prepare to load it.
        sc->prefetch(ckpt_id);

        bool ok = sc->load(ckpt_id, ctx, ctx_dft, out_pos_min, out_pos_max, out_n_tokens, out_spec_data, dest_seq_id);
        if (ok) {
            cache_hits_++;
            if (out_lcp) *out_lcp = match_lcp;
            if (out_is_continuation) *out_is_continuation = false;
            // Same-user match: out_overlap must be set so the Case 2 cold-start
            // validation in server-context.cpp can recognize a full-prefix
            // match. The conversation hash was already verified to be in
            // user_wrappers_, so by construction this is the same conversation
            // and the LCP reflects how much of the stored prefix matched.
            // 1.0 signals "same conversation, full coverage" to the caller.
            // Case 2 still gates on ssd_lcp >= PREFIX_MAX, so this is a no-op
            // when the LCP is too small to trust beyond the stored prefix.
            if (out_overlap) *out_overlap = 1.0f;
        } else {
            cache_misses_++;
        }
        return ok;
    }

    uint64_t effective_conv = conv_hash;
    bool is_continuation = false;

    // Try continuation matching if no cache exists for this conv_hash
    if (effective_conv != 0 && conv_wrappers_.find(effective_conv) == conv_wrappers_.end()) {
        float overlap = 0.0f;
        uint64_t continuation = kv_ssd_find_continuation(
            ssd_base_path_.c_str(),
            (const uint32_t*)tokens, tokens_size,
            0.90f, model_compat_hash_, &overlap);
        if (continuation != 0) {
            effective_conv = continuation;
            is_continuation = true;
            LOG_INF("SSD cache: reusing conversation %016lx for cold restart\n",
                    (unsigned long)continuation);
            if (out_overlap) *out_overlap = overlap;
        }
    }

    server_ssd_cache* sc = get_or_create_cache(effective_conv);
    if (!sc) return false;

    int32_t match_lcp = 0;
    uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past, &match_lcp);
    if (ckpt_id == 0) {
        cache_misses_++;
        return false;
    }

    // Prefetch the checkpoint file from SSD while we prepare to load it.
    // This triggers kernel page cache readahead so the SSD I/O overlaps
    // with the state restoration setup in load().
    sc->prefetch(ckpt_id);

    bool ok = sc->load(ckpt_id, ctx, ctx_dft, out_pos_min, out_pos_max, out_n_tokens, out_spec_data);
    if (ok) {
        cache_hits_++;
        if (out_lcp) *out_lcp = match_lcp;
        if (out_is_continuation) *out_is_continuation = is_continuation;
        // Same-conversation match (effective_conv matched a loaded cache and
        // find_match returned a hit on the stored prefix). The continuation
        // path above already set out_overlap from kv_ssd_find_continuation,
        // so only set it here when this is NOT a continuation. Same-conv
        // overlap is 1.0 by construction: we matched the cache for THIS
        // conv_hash, and the LCP shows how much of the stored prefix aligned.
        // Case 2 in server-context.cpp still requires ssd_lcp >= PREFIX_MAX,
        // so a short LCP safely falls through to the partial-coverage branch.
        if (out_overlap && !is_continuation) *out_overlap = 1.0f;
    } else {
        cache_misses_++;
    }
    return ok;
}

void server_context_page_manager::evict_slot(uint32_t slot_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    evict_slot_internal(slot_id);
}

bool server_context_page_manager::get_checkpoint_data(uint32_t slot_id, std::vector<uint8_t>& out_data) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return false;

    // Find which cache has this checkpoint
    for (auto& [conv, cache] : conv_caches_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache.get(), it->second.checkpoint_id);
        if (meta) {
            return kv_ssd_load(cache.get(), it->second.checkpoint_id, out_data);
        }
    }
    return false;
}

void server_context_page_manager::get_stats(
    size_t* hot_bytes, size_t* warm_bytes, size_t* cold_bytes,
    size_t* total_checkpoints, size_t* max_checkpoints,
    uint64_t* hits, uint64_t* misses, float* hit_rate
) const {
    size_t hot_sum = 0, warm_sum = 0, cold_sum = 0, total_sum = 0;
    for (const auto& [conv, cache] : conv_caches_) {
        size_t h, w, c, t;
        kv_ssd_get_stats(cache.get(), &h, &w, &c, &t, nullptr, nullptr);
        hot_sum += h;
        warm_sum += w;
        cold_sum += c;
        total_sum += t;
    }
    if (hot_bytes) *hot_bytes = hot_sum;
    if (warm_bytes) *warm_bytes = warm_sum;
    if (cold_bytes) *cold_bytes = cold_sum;
    if (total_checkpoints) *total_checkpoints = total_sum;
    if (max_checkpoints) *max_checkpoints = max_cross_slot_checkpoints_;
    if (hits) *hits = cache_hits_;
    if (misses) *misses = cache_misses_;
    if (hit_rate) {
        uint64_t h = cache_hits_, m = cache_misses_;
        *hit_rate = (h + m) > 0 ? (float)h / (float)(h + m) : 0.0f;
    }
}

uint32_t server_context_page_manager::get_max_turn_id() const {
    return kv_ssd_get_max_turn_id_global(ssd_base_path_.c_str());
}

server_ssd_cache* server_context_page_manager::get_or_create_user_cache(const std::string& user_id) {
    if (user_id.empty()) return nullptr;

    const uint64_t key = fnv1a_string(user_id);

    auto it = user_wrappers_.find(key);
    if (it != user_wrappers_.end()) {
        return it->second.get();
    }

    // Evict oldest user cache if at max. share the max_conversations cap
    // with the anonymous bucket so the total SSD directory count stays
    // bounded.
    if ((int)user_caches_.size() >= max_conversations) {
        uint64_t oldest = 0;
        time_t oldest_mtime = 0;

        for (const auto& [uk, cache] : user_caches_) {
            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)uk);
            fs::path dir = fs::path(ssd_base_path_) / "u" / hex;

            std::error_code ec;
            auto ftime = fs::last_write_time(dir, ec);
            if (!ec) {
                auto mtime = std::chrono::system_clock::to_time_t(
                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()));
                if (oldest == 0 || mtime < oldest_mtime) {
                    oldest_mtime = mtime;
                    oldest = uk;
                }
            }
        }

        if (oldest != 0) {
            LOG_WRN("SSD cache: evicting user %016lx (max=%d reached)\n",
                     (unsigned long)oldest, max_conversations);

            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)oldest);
            fs::path dir = fs::path(ssd_base_path_) / "u" / hex;

            for (const auto& entry : fs::directory_iterator(dir)) {
                fs::remove(entry.path());
            }
            fs::remove(dir);

            user_wrappers_.erase(oldest);
            user_caches_.erase(oldest);
        }
    }

    auto raw = kv_ssd_init(ssd_base_path_.c_str(), &config_, key, "u/");
    if (!raw) return nullptr;

    auto cache_ptr = std::unique_ptr<kv_ssd_cache>(raw);
    auto wrapper = std::make_unique<server_ssd_cache>(raw);

    if (model_compat_hash_ != 0) {
        wrapper->set_compat_hash(model_compat_hash_);
    }

    server_ssd_cache* result = wrapper.get();
    user_caches_[key] = std::move(cache_ptr);
    user_wrappers_[key] = std::move(wrapper);

    LOG_INF("SSD cache: created new user cache user=%s key=%016lx (total=%zu)\n",
             user_id.c_str(), (unsigned long)key, user_caches_.size());

    return result;
}

} // namespace llama