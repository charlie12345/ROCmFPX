// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Global System Prompt KV Cache
//
// See kv-ssd-system-cache.h for design.

#include "kv-ssd-system-cache.h"
#include "kv-ssd-cache.h"  // for kv_ssd_hash_tokens
#include "kv-ssd-posix.h"
#include "log.h"
#include "llama.h"        // for llama_vocab / token attrs

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cinttypes>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fcntl.h>

namespace {

// I/O helpers (duplicated minimally from kv-ssd-cache.cpp to keep
// the system cache as a self-contained translation unit).
bool pwrite_all(int fd, const void* buf, size_t count, int64_t offset) {
    static const size_t chunk_max = 64 * 1024 * 1024; // 64 MiB
    const char* ptr = (const char*)buf;
    size_t remaining = count;
    int64_t off = offset;
    while (remaining > 0) {
        size_t chunk = remaining;
        if (chunk > chunk_max) {
            chunk = chunk_max;
        }
        ssize_t n = pwrite(fd, ptr, chunk, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) {
            errno = ENOSPC;
            return false;
        }
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

bool pread_all(int fd, void* buf, size_t count, int64_t offset) {
    static const size_t chunk_max = 64 * 1024 * 1024; // 64 MiB
    char* ptr = (char*)buf;
    size_t remaining = count;
    int64_t off = offset;
    while (remaining > 0) {
        size_t chunk = remaining;
        if (chunk > chunk_max) {
            chunk = chunk_max;
        }
        ssize_t n = pread(fd, ptr, chunk, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) { errno = EIO; return false; }
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

uint32_t aligned_size(uint32_t n) {
    // 4-byte alignment for the prefix array boundary
    return (n + 3u) & ~3u;
}

// Print a 16-char hex hash for log messages
std::string hex16(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016lx", (unsigned long)h);
    return std::string(buf);
}

} // namespace

// =============================================================================
// Static helpers
// =============================================================================

uint64_t kv_ssd_system_cache::hash_tokens(const uint32_t* tokens, size_t n) {
    return kv_ssd_hash_tokens(tokens, n);
}

std::string kv_ssd_system_cache::entry_path(const std::string& model_dir, uint64_t hash) {
    return model_dir + "/sys-" + hex16(hash) + ".bin";
}

// =============================================================================
// Init
// =============================================================================

bool kv_ssd_system_cache::init(const std::string& model_dir, uint64_t compat_hash) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized) {
        return true;
    }

    model_dir_ = model_dir;
    compat_hash_ = compat_hash;

    // Ensure directory exists
    namespace fs = std::filesystem;
    fs::path model_path(model_dir_);
    if (!fs::exists(model_path)) {
        std::error_code ec;
        if (!fs::create_directory(model_path, ec) && ec.value() != 0) {
            LOG_ERR("system cache: failed to create %s: %s\n",
                    model_dir_.c_str(), ec.message().c_str());
            return false;
        }
    }

    if (!fs::is_directory(model_path)) {
        LOG_ERR("system cache: %s is not a directory\n", model_dir_.c_str());
        return false;
    }

    size_t loaded = 0;
    size_t rejected = 0;
    for (const auto& entry : fs::directory_iterator(model_path)) {
        std::string fname = entry.path().filename().string();
        if (fname.size() < 9) continue;
        if (fname.compare(0, 4, "sys-") != 0) continue;
        if (fname.compare(fname.size() - 4, 4, ".bin") != 0) continue;

        std::string filepath = entry.path().string();
        kv_ssd_system_entry sys_entry;
        if (load_entry_from_disk(filepath, sys_entry)) {
            if (compat_hash != 0 && sys_entry.compat_hash != 0 && sys_entry.compat_hash != compat_hash) {
                LOG_WRN("system cache: rejecting %s (compat_hash mismatch: stored=%016lx current=%016lx)\n",
                        fname.c_str(),
                        (unsigned long)sys_entry.compat_hash,
                        (unsigned long)compat_hash);
                rejected++;
                continue;
            }
            bytes_ += sys_entry.data.size();
            entries_[sys_entry.hash] = std::move(sys_entry);
            loaded++;
        } else {
            LOG_WRN("system cache: failed to load %s\n", filepath.c_str());
        }
    }

    initialized = true;

    LOG_INF("system cache: initialized at %s (loaded=%zu, entries=%zu, bytes=%.1f MiB, rejected=%zu, max_entries=%zu, max_unused_days=%d)\n",
            model_dir_.c_str(), loaded, entries_.size(),
            (double)bytes_ / (1024.0 * 1024.0),
            rejected, max_entries, max_unused_days);

    return true;
}

// =============================================================================
// Store
// =============================================================================

bool kv_ssd_system_cache::store(const uint32_t* tokens, uint32_t n_tokens,
                                 const uint8_t* data, size_t data_size) {
    if (!initialized) {
        LOG_WRN("system cache: store called before init\n");
        return false;
    }
    if (!tokens || n_tokens == 0 || !data || data_size == 0) {
        LOG_WRN("system cache: store called with invalid args\n");
        return false;
    }

    uint64_t hash = hash_tokens(tokens, n_tokens);

    std::lock_guard<std::mutex> lock(mutex_);

    // Build entry
    kv_ssd_system_entry entry;
    entry.hash = hash;
    entry.n_tokens = n_tokens;
    entry.tokens.assign(tokens, tokens + n_tokens);
    entry.data.assign(data, data + data_size);
    entry.created_at = now_unix();
    entry.last_used = entry.created_at;
    entry.access_count = 0;
    entry.compat_hash = compat_hash_;
    entry.filepath = entry_path(model_dir_, hash);

    // If we already have this entry, just update it in place. The byte
    // count for the old entry is the same as for the new one (same data
    // size since n_tokens is the same), so no LRU pressure changes.
    auto existing = entries_.find(hash);
    if (existing != entries_.end()) {
        // Replace data and metadata; preserve file path
        size_t old_size = existing->second.data.size();
        size_t new_size = entry.data.size();

        existing->second = std::move(entry);
        bytes_ -= old_size;
        bytes_ += new_size;
        // Update last_used since this is a fresh store
        existing->second.last_used = now_unix();
        existing->second.access_count++;  // count refreshes as access

        // Persist (re-write file)
        if (!write_entry_to_disk(existing->second)) {
            LOG_WRN("system cache: failed to update %s on disk\n",
                    existing->second.filepath.c_str());
            // In-memory update succeeded, keep going
        }

        stats_stores++;
        LOG_INF("system cache: refreshed entry hash=%s tokens=%u size=%.1f MiB\n",
                hex16(hash).c_str(), n_tokens, (double)new_size / (1024.0 * 1024.0));
        return true;
    }

    // New entry - persist to disk first, then add to in-memory
    if (!write_entry_to_disk(entry)) {
        int se = errno;
        LOG_ERR("system cache: failed to write %s: %s (errno=%d, win32_err=%lu)\n",
                entry.filepath.c_str(), std::strerror(se), se, (unsigned long)portable_get_last_win32_error());
        return false;
    }

    entries_[hash] = std::move(entry);
    bytes_ += entries_[hash].data.size();
    stats_stores++;

    LOG_INF("system cache: stored entry hash=%s tokens=%u size=%.1f MiB (total=%zu, %.1f MiB)\n",
            hex16(hash).c_str(), n_tokens,
            (double)entries_[hash].data.size() / (1024.0 * 1024.0),
            entries_.size(), (double)bytes_ / (1024.0 * 1024.0));

    // Apply retention policy
    apply_retention_policy();

    return true;
}

// =============================================================================
// Find
// =============================================================================

const kv_ssd_system_entry* kv_ssd_system_cache::find(const uint32_t* tokens, uint32_t n_tokens) {
    if (!initialized || !tokens || n_tokens == 0) return nullptr;

    uint64_t hash = hash_tokens(tokens, n_tokens);

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(hash);
    if (it == entries_.end()) {
        stats_misses++;
        return nullptr;
    }

    // Verify n_tokens matches and the first n_tokens are identical
    // (defense in depth against hash collisions or file corruption).
    if (it->second.n_tokens != n_tokens) {
        LOG_WRN("system cache: hash collision? hash=%s stored_tokens=%u requested=%u\n",
                hex16(hash).c_str(), it->second.n_tokens, n_tokens);
        stats_misses++;
        return nullptr;
    }
    if (n_tokens > KV_SSD_SYS_TOKEN_MAX) {
        // We can't fully verify beyond the stored prefix. Trust the hash.
    } else {
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (it->second.tokens[i] != tokens[i]) {
                LOG_WRN("system cache: token mismatch at index %u hash=%s\n",
                        i, hex16(hash).c_str());
                stats_misses++;
                return nullptr;
            }
        }
    }

    it->second.last_used = now_unix();
    it->second.access_count++;
    stats_hits++;

    LOG_DBG("system cache: hit hash=%s tokens=%u access_count=%u\n",
            hex16(hash).c_str(), n_tokens, it->second.access_count);

    return &it->second;
}

bool kv_ssd_system_cache::load(const uint32_t* tokens, uint32_t n_tokens,
                                std::vector<uint8_t>& out_data) {
    const kv_ssd_system_entry* entry = find(tokens, n_tokens);
    if (!entry) return false;
    out_data = entry->data;
    return true;
}

// =============================================================================
// Eviction
// =============================================================================

void kv_ssd_system_cache::evict_entry(uint64_t hash) {
    auto it = entries_.find(hash);
    if (it == entries_.end()) return;

    LOG_INF("system cache: evicting entry hash=%s tokens=%u size=%.1f MiB (reason=%s)\n",
            hex16(hash).c_str(), it->second.n_tokens,
            (double)it->second.data.size() / (1024.0 * 1024.0),
            // Reason reported below by caller
            "retention");

    bytes_ -= it->second.data.size();

    if (!it->second.filepath.empty()) {
        if (unlink(it->second.filepath.c_str()) != 0 && errno != ENOENT) {
            int se = errno;
            LOG_WRN("system cache: failed to delete %s: %s (errno=%d)\n",
                    it->second.filepath.c_str(), std::strerror(se), se);
        }
    }
    entries_.erase(it);
    stats_evicts++;
}

uint64_t kv_ssd_system_cache::find_lru_hash() const {
    if (entries_.empty()) return 0;

    uint64_t lru_hash = 0;
    uint64_t lru_time = UINT64_MAX;

    for (const auto& [hash, entry] : entries_) {
        if (entry.last_used < lru_time) {
            lru_time = entry.last_used;
            lru_hash = hash;
        }
    }
    return lru_hash;
}

size_t kv_ssd_system_cache::evict_lru_to_limit() {
    if (!initialized) return 0;
    std::lock_guard<std::mutex> lock(mutex_);

    size_t evicted = 0;
    while (entries_.size() > max_entries) {
        uint64_t lru = find_lru_hash();
        if (lru == 0) break;
        evict_entry(lru);
        evicted++;
    }
    return evicted;
}

size_t kv_ssd_system_cache::expire_old_entries(int unused_days) {
    if (!initialized) return 0;
    if (unused_days < 0) unused_days = max_unused_days;
    if (unused_days <= 0) return 0;  // 0 = never expire

    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t threshold = now_unix() - (uint64_t)unused_days * 24 * 3600;

    size_t evicted = 0;
    // Collect hashes first to avoid iterator invalidation
    std::vector<uint64_t> to_evict;
    for (const auto& [hash, entry] : entries_) {
        if (entry.last_used < threshold) {
            to_evict.push_back(hash);
        }
    }
    for (uint64_t hash : to_evict) {
        evict_entry(hash);
        evicted++;
    }
    if (evicted > 0) {
        LOG_INF("system cache: expired %zu entries (unused > %d days)\n",
                evicted, unused_days);
    }
    return evicted;
}

void kv_ssd_system_cache::apply_retention_policy() {
    // LRU cap
    while (entries_.size() > max_entries) {
        uint64_t lru = find_lru_hash();
        if (lru == 0) break;
        evict_entry(lru);
    }
}

// =============================================================================
// Disk I/O
// =============================================================================

bool kv_ssd_system_cache::load_entry_from_disk(const std::string& filepath, kv_ssd_system_entry& out) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    kv_ssd_system_record rec;
    bool ok = pread_all(fd, &rec, sizeof(rec), 0);
    if (!ok || rec.magic != KV_SSD_SYS_MAGIC_REC || rec.version != KV_SSD_SYS_VERSION) {
        close(fd);
        return false;
    }

    // Read data payload
    std::vector<uint8_t> data(rec.data_size);
    if (rec.data_size > 0) {
        if (!pread_all(fd, data.data(), rec.data_size, (int64_t)sizeof(kv_ssd_system_record))) {
            close(fd);
            return false;
        }
    }
    close(fd);

    out.hash = rec.hash;
    out.n_tokens = rec.n_tokens;
    out.tokens.assign(rec.token_prefix, rec.token_prefix + rec.token_count);
    out.data = std::move(data);
    out.created_at = rec.created_at;
    out.last_used = rec.last_used;
    out.access_count = rec.access_count;
    out.compat_hash = rec.compat_hash;
    out.filepath = filepath;

    return true;
}

bool kv_ssd_system_cache::write_entry_to_disk(const kv_ssd_system_entry& entry) {
    kv_ssd_system_record rec;
    std::memset(&rec, 0, sizeof(rec));
    rec.magic = KV_SSD_SYS_MAGIC_REC;
    rec.version = KV_SSD_SYS_VERSION;
    rec.hash = entry.hash;
    rec.n_tokens = entry.n_tokens;
    rec.data_size = (uint32_t)entry.data.size();
    rec.compat_hash = entry.compat_hash;
    rec.created_at = entry.created_at;
    rec.last_used = entry.last_used;
    rec.access_count = entry.access_count;
    rec.token_count = (uint32_t)std::min((size_t)entry.n_tokens, (size_t)KV_SSD_SYS_TOKEN_MAX);
    if (rec.token_count > 0) {
        std::memcpy(rec.token_prefix, entry.tokens.data(),
                    rec.token_count * sizeof(uint32_t));
    }

    int fd = open(entry.filepath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    bool ok = pwrite_all(fd, &rec, sizeof(rec), 0);
    if (ok && entry.data.size() > 0) {
        ok = pwrite_all(fd, entry.data.data(), entry.data.size(),
                        (int64_t)sizeof(kv_ssd_system_record));
    }
    fsync(fd);
    close(fd);
    return ok;
}

// =============================================================================
// Stats
// =============================================================================

size_t kv_ssd_system_cache::size() const {
    if (!initialized) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

size_t kv_ssd_system_cache::bytes() const {
    if (!initialized) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
}

// =============================================================================
// System prompt boundary detection
// =============================================================================

namespace {
// Check if a token's decoded text indicates a role/structure marker
// (as opposed to a true end-of-generation token). Some vocabularies
// (GLM-4, Gemma) classify role tokens like <|system|>, <|user|>,
// <|assistant|> as EOG. We need to distinguish these from actual
// section-closing tokens like <|im_end|>, <|end_of_turn|>, <|eot_id|>.
//
// Substring match is sufficient here because:
//   - Role markers are short special tokens (<|user|>, <|system|>, etc.)
//   - The role name substring only appears in role markers, not in
//     typical system prompt content (and even if it does, the worst
//     case is a slightly shorter cached system prompt - still works)
static bool is_role_marker_text(const std::string& text) {
    if (text.empty()) return false;
    if (text.find("system")    != std::string::npos) return true;
    if (text.find("user")      != std::string::npos) return true;
    if (text.find("assistant") != std::string::npos) return true;
    if (text.find("human")     != std::string::npos) return true;
    if (text.find("model")     != std::string::npos) return true;
    if (text.find("tool")      != std::string::npos) return true;
    if (text.find("function")  != std::string::npos) return true;
    if (text.find("[INST]")    != std::string::npos) return true;
    return false;
}

// Decode a token to text using the vocab. Special tokens are rendered
// (e.g. <|im_start|>) so we can match role markers by their text.
// Returns empty string on error.
static std::string decode_token_for_boundary(
    const llama_vocab* vocab,
    llama_token token)
{
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n <= 0) {
        n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    }
    if (n <= 0) return "";
    return std::string(buf, n);
}

// Try to detect the system prompt end in tokens[0..n_tokens] using a
// template-aware scan. We look for the first occurrence of one of these
// template-specific end markers:
//
// Qwen3 / ChatML:  <|im_start|>system\n...\n<|im_end|>
//                  -> end position is right after <|im_end|>\n
//
// Llama-3:         <|start_header_id|>system<|end_header_id|>...<|eot_id|>
//                  -> end position is right after <|eot_id|>
//
// Mistral:         [INST] ... [/INST]
//                  -> system prompt typically doesn't use this format
//
// Gemma:           <start_of_turn>user...<end_of_turn>
//                  -> similar
//
// Returns -1 if no system section is detected at all.

struct template_marker {
    const char* name;
    // Token IDs that mark the end of the system section
    // (typically a single special token like <|im_end|>).
    // We just look for the first such token after the system role marker.
    std::vector<llama_token> end_tokens;
};

bool tokens_match(const llama_token* tokens, size_t n, size_t pos,
                  const std::vector<llama_token>& pattern) {
    if (pos + pattern.size() > n) return false;
    for (size_t i = 0; i < pattern.size(); i++) {
        if (tokens[pos + i] != pattern[i]) return false;
    }
    return true;
}

} // namespace

int32_t kv_detect_system_prompt_boundary(
    const llama_vocab* vocab,
    const llama_token* tokens,
    int32_t n_tokens,
    const char* chat_template_hint)
{
    (void)chat_template_hint; // reserved for future template-aware detection
    if (!vocab || !tokens || n_tokens <= 0) return 0;

    // Two-phase boundary detection. We compute a candidate boundary from
    // each phase and return the minimum. The minimum is the most
    // conservative (smallest) system prompt, which is the original
    // design intent: cache just the system section, nothing more.
    //
    // Phase 1: find the first user-role marker token. The user role
    //   marker is a special token whose decoded text contains "user" or
    //   "human" (case-insensitive). This is the most reliable method
    //   because the user role name is always present in any chat
    //   template that has a user/assistant turn structure, regardless
    //   of whether the template uses EOG to close the system section.
    //
    //   Works for: GLM (<|user|>), Gemma (text "user"), Llama-3
    //   (text "user"), ChatML (text "user"), Command-R, etc.
    //
    //   We use llama_vocab_is_control() to filter out regular content
    //   tokens that happen to contain "user" in their text (e.g. a
    //   system prompt that says "user-facing assistant"). Control
    //   tokens are always special/structural, never content words.
    //   We also accept the text "user" or "User" etc. as exact matches
    //   for templates like Llama-3 where "user" is a regular text
    //   token (not a control token) but is still a role marker by
    //   position.
    //
    // Phase 2: EOG-based detection, skipping role-marker EOGs. This
    //   preserves the original behavior for templates where the
    //   system section is closed by a true EOG token (<|im_end|>,
    //   <|end_of_turn|>, <|eot_id|>, etc.) and the role tokens
    //   themselves are NOT classified as EOG (ChatML, Llama-3, standard
    //   Gemma).
    int32_t boundary_user = n_tokens;
    int32_t boundary_eog  = n_tokens;

    // Phase 1: find first user-role marker token
    // Start from i=1 (skip BOS). The preceding-control-token check
    // filters out content words that happen to be "user" or "human".
    // Starting from i=2 would miss templates where the user role
    // marker is at position 1 (Qwen3.6 with no system section:
    // <|im_start|>user\n...).
    for (int i = 1; i < n_tokens; i++) {
        std::string text = decode_token_for_boundary(vocab, tokens[i]);
        if (text.empty()) continue;

        // Case-insensitive substring match for "user" or "human"
        std::string lower;
        lower.reserve(text.size());
        for (char c : text) {
            lower.push_back((char)std::tolower((unsigned char)c));
        }
        bool is_user_text = (lower.find("user")   != std::string::npos ||
                             lower.find("human")  != std::string::npos);
        if (!is_user_text) continue;

        // Filter false positives: if the token is NOT a control/special
        // token AND not an exact match for a known user-role text, it's
        // probably content (e.g. "user_name" in the system prompt).
        // For control tokens, the role-marker interpretation is correct.
        // For exact text matches like "user" or "User", the role-marker
        // interpretation is correct ONLY if the preceding token is a
        // role-start control token (ChatML: <|im_start|>user, Llama-3:
        // <|start_header_id|>user). Without this check, content that
        // contains the word "user" (e.g. "user-facing" or a 19th-century
        // novel that uses the word) gets falsely identified as a role
        // marker, and the boundary lands inside the user content.
        bool is_control = llama_vocab_is_control(vocab, tokens[i]);
        bool is_exact_role = (text == "user" || text == "User" ||
                              text == "USER" || text == "Human" ||
                              text == "human" || text == "H" ||
                              text == "<|user|>" || text == "<|USER|>");

        if (is_control) {
            // Control token with "user" in text - always a role marker
            // (e.g. GLM's <|user|>, Gemma's <|user|>)
        } else if (is_exact_role) {
            // Exact text match (e.g. plain "user" token in ChatML/Llama-3)
            // - verify preceded by a role-start control token
            if (i < 1 || !llama_vocab_is_control(vocab, tokens[i-1])) {
                continue;  // Content word "user", not a role marker
            }
        } else {
            // Substring match but not control and not exact - content
            // (e.g. "users", "user-facing"). Skip.
            continue;
        }

        boundary_user = i;
        break;
    }

    // Phase 2: EOG-based detection, skipping role-marker EOGs
    const int ROLE_HEADER_MAX = 6;
    int scan_start = 0;
    for (int i = 0; i < ROLE_HEADER_MAX && i < n_tokens; i++) {
        if (llama_vocab_is_eog(vocab, tokens[i])) {
            continue;
        }
        if (i < 1) continue;  // always skip token 0
        scan_start = i + 1;
        break;
    }

    const int MAX_SYSTEM_SCAN = 64;
    int scan_end = std::min(scan_start + MAX_SYSTEM_SCAN, n_tokens);
    for (int j = scan_start; j < scan_end; j++) {
        if (!llama_vocab_is_eog(vocab, tokens[j])) continue;
        // Skip role-marker EOGs (GLM/Gemma classify role tokens as EOG)
        std::string text = decode_token_for_boundary(vocab, tokens[j]);
        if (is_role_marker_text(text)) continue;
        boundary_eog = j + 1;
        break;
    }

    // Return the minimum - most conservative system prompt boundary.
    // This preserves original behavior for templates that work
    // (ChatML, Llama-3, standard Gemma) and fixes GLM/Gemma where the
    // original returned n_tokens (no cache) or a tiny boundary (3-7
    // tokens of chat template header).
    return std::min(boundary_user, boundary_eog);
}
