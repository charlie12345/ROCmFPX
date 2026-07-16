// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration

#ifndef SERVER_CONTEXT_PAGE_MANAGER_H
#define SERVER_CONTEXT_PAGE_MANAGER_H

#include "common/kv-ssd-cache.h"
#include "common/kv_page_manager.h"
#include "llama.h"
#include "server-task.h"
#include "server-common.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace llama {

class server_ssd_cache;

/**
 * Manages server-level checkpoint storage with SSD backing.
 * Uses kv_ssd_cache for hot/warm/cold tiering with disk persistence.
 */
class server_context_page_manager {
public:
    // Checkpoint stored in the manager
    struct stored_checkpoint {
        uint64_t checkpoint_id = 0;       // kv_ssd cache checkpoint ID
        uint32_t slot_id = 0;             // Original slot
        uint32_t turn_id = 0;             // Last turn accessed
        uint64_t size_bytes = 0;           // Checkpoint size
        uint64_t n_tokens = 0;            // Token coverage
        int32_t pos_min = 0;              // Position range
        int32_t pos_max = 0;
        uint64_t last_access = 0;          // Timestamp
        uint32_t access_count = 0;         // Access count
        std::vector<llama_token> tokens;  // Token sequence for matching
    };

    server_context_page_manager(
        const char* ssd_path,
        const kv_eviction_config* cfg,
        size_t n_tokens_total,
        size_t max_cross_slot_checkpoints
    );
    ~server_context_page_manager();

    // Store a checkpoint (written to SSD and kept in hot tier)
    // ctx is required to compute full state (recurrent + KV cache) for SSD storage
    bool store_checkpoint(
        uint32_t slot_id,
        struct llama_context* ctx,
        const common_prompt_checkpoint& ckpt,
        uint32_t turn_id
    );

    // Store a checkpoint with token sequence for cross-slot matching
    // conv_hash routes to the correct per-conversation cache.
    // Creates the conversation directory automatically on first use.
    // ctx is required to compute full state (recurrent + KV cache) for SSD storage.
    // ctx_dft is the MTP/draft context (nullptr if none or mem-shared).
    bool store_checkpoint_with_tokens(
        uint32_t slot_id,
        struct llama_context* ctx,
        struct llama_context* ctx_dft,
        const common_prompt_checkpoint& ckpt,
        const llama_token* tokens,
        size_t tokens_size,
        uint32_t turn_id,
        uint64_t conv_hash = 0,
        const std::string& user_id = std::string()
    );

    // Load a checkpoint back to slot memory
    bool load_checkpoint(
        uint32_t slot_id,
        uint32_t turn_id,
        struct llama_context* ctx,
        struct llama_context* ctx_dft,
        int32_t& out_pos_min,
        int32_t& out_pos_max,
        uint64_t& out_n_tokens,
        std::vector<uint8_t>* out_spec_data = nullptr
    );

    // Load a checkpoint by its SSD cache ID (for cross-slot restore)
    bool load_checkpoint_by_id(
        uint64_t checkpoint_id,
        struct llama_context* ctx,
        struct llama_context* ctx_dft,
        int32_t& out_pos_min,
        int32_t& out_pos_max,
        uint64_t& out_n_tokens,
        std::vector<uint8_t>* out_spec_data = nullptr
    );

    // Prefetch checkpoints for a slot before processing
    void prefetch_for_slot(uint32_t slot_id, uint32_t turn_id);

    // Notify turn completion (triggers tier demotion)
    void on_turn_complete(uint32_t turn_id);

    // Find a checkpoint matching the given tokens (for cross-slot reuse)
    // Routes to per-conversation cache based on conv_hash.
    // On cold start where conv_hash doesn't match any existing cache,
    // falls back to continuation matching across all conversation directories.
    bool find_matching_checkpoint(
        const llama_token* tokens,
        size_t tokens_size,
        uint32_t current_turn,
        uint32_t& out_slot_id,
        int32_t& out_pos_min,
        int32_t& out_pos_max,
        uint64_t& out_n_tokens,
        uint64_t conv_hash = 0,
        int32_t n_past = -1,
        uint64_t max_n_tokens = UINT64_MAX,
        const std::string& user_id = std::string()
    );

    // Find matching checkpoint by token prefix and restore to VRAM (cross-session restart)
    // Routes to per-conversation cache. Falls back to continuation detection
    // across all conversation directories if conv_hash isn't known yet.
    // dest_seq_id: current slot's seq_id — KV cells are restored under this id so that
    // llama_memory_seq_pos_min() returns a valid value for the slot after restore.
    bool find_and_load_checkpoint(
        const llama_token* tokens,
        size_t tokens_size,
        uint32_t current_turn,
        struct llama_context* ctx,
        struct llama_context* ctx_dft,
        uint32_t dest_seq_id,
        int32_t& out_pos_min,
        int32_t& out_pos_max,
        uint64_t& out_n_tokens,
        std::vector<uint8_t>* out_spec_data = nullptr,
        uint64_t conv_hash = 0,
        int32_t n_past = -1,
        uint64_t max_n_tokens = UINT64_MAX,
        int32_t* out_lcp = nullptr,
        float* out_overlap = nullptr,
        bool* out_is_continuation = nullptr,
        const std::string& user_id = std::string()
    );

   // Evict all checkpoints for a specific slot
    void evict_slot(uint32_t slot_id);

    // Get checkpoint data for a specific slot
    bool get_checkpoint_data(uint32_t slot_id, std::vector<uint8_t>& out_data);

    // Get stats for monitoring
    void get_stats(
        size_t* hot_bytes, size_t* warm_bytes, size_t* cold_bytes,
        size_t* total_checkpoints, size_t* max_checkpoints,
        uint64_t* hits, uint64_t* misses, float* hit_rate
    ) const;

    // Get the maximum turn_id across all checkpoints (for seeding turn counter on restart)
    uint32_t get_max_turn_id() const;

    // Set model compatibility info after model init.
    // Computes compat_hash from model description + cache types.
    // Called once after llama_model is loaded.
    void set_model_info(const struct llama_model* model,
                        int cache_type_k, int cache_type_v);
    // Configure SSD checkpoint writes: when true, skip fsync for lower write latency.
    void set_no_fsync(bool no_fsync);


    // Max conversations: LRU eviction of entire conversation directories when exceeded.
    // Set before any store/find calls. Default: 16.
    int max_conversations = 16;

    std::unordered_map<uint32_t, stored_checkpoint> checkpoints_; // slot_id -> checkpoint
    size_t max_cross_slot_checkpoints_;
    mutable std::shared_mutex mutex_;

    // Statistics
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;

private:
    void evict_slot_internal(uint32_t slot_id);
    uint64_t get_timestamp_ms() const;

    // Get or create cache for a conversation hash
    server_ssd_cache* get_or_create_cache(uint64_t conv_hash);

    // Get or create a user-scoped cache. user_id is hashed and the cache
    // is created under the "u/" namespace, isolated from anonymous
    // conv_hash caches on disk. Continuation matching across user_id
    // caches is disabled (privacy).
    server_ssd_cache* get_or_create_user_cache(const std::string& user_id);

    // Per-conversation caches (conv_hash -> cache instance)
    std::string ssd_base_path_;
    kv_ssd_config config_;
    uint64_t model_compat_hash_ = 0;
    std::unordered_map<uint64_t, std::unique_ptr<kv_ssd_cache>> conv_caches_;
    std::unordered_map<uint64_t, std::unique_ptr<server_ssd_cache>> conv_wrappers_;

    // User-scoped caches. Parallel to conv_caches_ but isolated on disk
    // under the "u/" namespace. Keyed by fnv1a(user_id) so the on-disk
    // layout is the same hash format as anonymous caches.
    std::unordered_map<uint64_t, std::unique_ptr<kv_ssd_cache>> user_caches_;
    std::unordered_map<uint64_t, std::unique_ptr<server_ssd_cache>> user_wrappers_;
};

} // namespace llama

#endif // SERVER_CONTEXT_PAGE_MANAGER_H