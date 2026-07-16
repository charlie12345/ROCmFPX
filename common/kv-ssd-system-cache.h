// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Global System Prompt KV Cache
//
// Stores the recurrent + attention state for system prompt sections in a
// hash-indexed global pool. Lookup is by FNV-1a of the system prompt token
// sequence, so multiple harnesses (CLIO, SAM, etc.) can share their system
// prompts across all conversations and across server restarts.
//
// LRU eviction keeps the most recently used N entries in RAM (configurable).
// Day-based expiry removes entries that have not been used in N days.
// SSD-backed files live under {model_dir}/sys-{HASH16}.bin.

#ifndef KV_SSD_SYSTEM_CACHE_H
#define KV_SSD_SYSTEM_CACHE_H

#include "llama.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <ctime>

// Magic numbers for the system prompt cache file format
#define KV_SSD_SYS_MAGIC_REC  0x4B565359  // "KVSM" - system record
#define KV_SSD_SYS_VERSION    1

// Maximum number of system prompt tokens stored per entry for verification
// and display. The actual KV state covers all tokens up to n_tokens.
#define KV_SSD_SYS_TOKEN_MAX  4096

// Per-system-prompt file header (fixed size, followed by KV state data).
// Mirrors the layout of kv_ssd_record but uses a separate magic so we never
// accidentally cross-load a regular checkpoint as a system prompt entry.
struct kv_ssd_system_record {
    uint32_t magic;                              // KV_SSD_SYS_MAGIC_REC
    uint32_t version;                             // KV_SSD_SYS_VERSION
    uint64_t hash;                                // FNV-1a of system tokens
    uint32_t n_tokens;                            // System prompt token count
    uint32_t data_size;                           // Bytes of state data following
    uint64_t compat_hash;                         // Model config hash
    uint64_t created_at;                          // Unix seconds
    uint64_t last_used;                           // Unix seconds (updated on load)
    uint32_t access_count;                        // Total loads
    uint32_t token_count;                         // Tokens stored in prefix
    uint32_t token_prefix[KV_SSD_SYS_TOKEN_MAX];  // Cached tokens for verification
};

// In-memory entry for a system prompt cache.
struct kv_ssd_system_entry {
    uint64_t hash = 0;                  // FNV-1a of system prompt tokens
    uint32_t n_tokens = 0;              // Total system prompt tokens
    std::vector<uint32_t> tokens;       // First n_tokens for verification
    std::vector<uint8_t> data;          // Serialized KV state (attention + recurrent)
    uint64_t created_at = 0;            // Unix seconds
    uint64_t last_used = 0;             // Unix seconds
    uint32_t access_count = 0;          // Total loads
    uint64_t compat_hash = 0;           // Model config hash

    // Disk file path
    std::string filepath;
};

// Manager for a single model's global system prompt pool.
// One instance per (model, server) pair. Thread-safe.
class kv_ssd_system_cache {
public:
    // Configuration
    size_t max_entries = 8;             // Max entries kept (LRU)
    int    max_unused_days = 30;        // Expire entries unused for N days (0 = never)
    size_t max_hot_bytes = 0;           // 0 = auto-size from max_entries (~150 MiB/entry)

    kv_ssd_system_cache() = default;
    ~kv_ssd_system_cache() = default;

    kv_ssd_system_cache(const kv_ssd_system_cache&) = delete;
    kv_ssd_system_cache& operator=(const kv_ssd_system_cache&) = delete;

    // Initialize: scan {model_dir}/ for existing sys-*.bin files, load index.
    // model_dir is the per-model directory (e.g. kv-cache/{MODEL_DIR}/).
    // compat_hash is the model config hash; entries with mismatched hash are
    // rejected on load.
    bool init(const std::string& model_dir, uint64_t compat_hash);

    // Check if the manager has been initialized.
    bool is_initialized() const { return initialized; }

    // Store a system prompt's KV state.
    // tokens/n_tokens identifies the system prompt content.
    // data/data_size is the serialized state.
    // Overwrites an existing entry with the same hash (updates last_used).
    // Returns true on success.
    bool store(const uint32_t* tokens, uint32_t n_tokens,
               const uint8_t* data, size_t data_size);

    // Find a matching system prompt entry by content hash.
    // Returns a pointer to the entry (valid until next store/evict) or nullptr.
    // Bumps last_used and access_count on hit.
    const kv_ssd_system_entry* find(const uint32_t* tokens, uint32_t n_tokens);

    // Load a system prompt's KV state into out_data.
    // Returns true on success. Caller is responsible for restoring the data
    // to the llama_context.
    bool load(const uint32_t* tokens, uint32_t n_tokens,
              std::vector<uint8_t>& out_data);

    // Force-evict entries that have not been used in N days.
    // Returns the number of entries evicted.
    size_t expire_old_entries(int unused_days = -1);

    // Evict the least-recently-used entry until size() <= max_entries.
    // Returns the number of entries evicted.
    size_t evict_lru_to_limit();

    // Stats
    size_t size() const;
    size_t bytes() const;
    uint64_t hits() const   { return stats_hits; }
    uint64_t misses() const { return stats_misses; }
    uint64_t stores() const { return stats_stores; }
    uint64_t evicts() const { return stats_evicts; }

    // Hash helper (also used by callers that want to test for existence
    // without locking).
    static uint64_t hash_tokens(const uint32_t* tokens, size_t n);

private:
    // File system layout helpers
    static std::string entry_path(const std::string& model_dir, uint64_t hash);

    // Apply size/time limits. Called after store and periodically.
    void apply_retention_policy();

    // Evict one specific entry by hash. Removes from in-memory and disk.
    void evict_entry(uint64_t hash);

    // Load one entry from disk. Does not modify in-memory state.
    bool load_entry_from_disk(const std::string& filepath, kv_ssd_system_entry& out);

    // Write one entry to disk.
    bool write_entry_to_disk(const kv_ssd_system_entry& entry);

    // Get the current entry to evict (LRU).
    uint64_t find_lru_hash() const;

    // State
    std::string model_dir_;
    uint64_t compat_hash_ = 0;
    bool initialized = false;

    // In-memory index, keyed by hash
    std::unordered_map<uint64_t, kv_ssd_system_entry> entries_;

    // In-memory byte usage
    size_t bytes_ = 0;

    // Stats
    uint64_t stats_hits = 0;
    uint64_t stats_misses = 0;
    uint64_t stats_stores = 0;
    uint64_t stats_evicts = 0;

    mutable std::mutex mutex_;
};

// =============================================================================
// System prompt boundary detection
// =============================================================================

// Detects the end position (exclusive) of the system prompt section in a
// tokenized prompt. Returns n_tokens if no system prompt section is found
// (treats the entire prompt as system content), or 0 if the very first
// tokens already diverge from a recognized pattern.
//
// Strategy:
//   1. Template-aware: if chat_template_hint is non-null, look for the
//      template's system-end marker tokens in order.
//   2. Fallback: scan for known end-marker tokens via vocab:
//        <|im_end|>, <|eot_id|>, </s>, [INST] close, etc.
//
// The function only inspects tokens; callers must ensure the tokens were
// produced by the same vocab as chat_template_hint.
//
// Returns the position one past the last system prompt token. If the
// entire prompt is system content, returns n_tokens. If the prompt does
// not look like a chat template, returns 0 (caller can decide what to do).

int32_t kv_detect_system_prompt_boundary(
    const struct llama_vocab* vocab,
    const llama_token* tokens,
    int32_t n_tokens,
    const char* chat_template_hint = nullptr
);

#endif // KV_SSD_SYSTEM_CACHE_H
