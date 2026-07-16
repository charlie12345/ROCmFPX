// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// SSD-Backed KV Cache Paging Manager

#ifndef KV_PAGE_MANAGER_H
#define KV_PAGE_MANAGER_H

/**
 * SSD-Backed KV Cache Paging Manager
 * 
 * Extends in-memory KV cache with SSD storage to enable 128K+ context
 * windows on memory-constrained hardware.
 * 
 * Features:
 * - Tiered eviction (hot/warm/cold)
 * - LRU with access count scoring
 * - Turn-based expiry
 * - Auto-sizing based on available memory
 * - Write-back with periodic flush
 * 
 * Default page size: 1024 tokens (configurable)
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <chrono>

namespace llama {

// Page states
enum kv_page_state {
    PAGE_COLD     = 0,  // On SSD, not in RAM
    PAGE_HOT      = 1,  // In RAM, actively used
    PAGE_WARM     = 2,  // In RAM, less frequently used
    PAGE_LOADING  = 3,  // Being loaded from SSD
    PAGE_EVICTING = 4,  // Being written to SSD
    PAGE_DIRTY    = 5,  // In RAM, modified, not yet on SSD
    PAGE_NEW      = 6,  // Never been written to SSD
};

// SSD file header
struct kv_page_header {
    uint32_t magic;              // "KVPG" magic
    uint32_t version;            // Format version
    uint32_t n_pages;            // Number of pages
    uint32_t page_size_tokens;   // Tokens per page
    uint64_t data_offset;        // Byte offset to page data
    uint64_t total_size_bytes;   // Total size of all pages
    uint64_t reserved[6];       // Reserved for future use
};

// Page table entry
struct kv_page_entry {
    uint64_t page_id;            // Sequential ID (0, 1, 2...)
    uint32_t token_start;        // First token in this page
    uint32_t token_end;          // Last token (exclusive)
    kv_page_state state;         // Current state
    uint64_t ssd_offset;         // Byte offset in SSD file (if cold)
    uint64_t size_bytes;         // Serialized size
    uint64_t last_access;        // Timestamp for LRU (ms since epoch)
    uint32_t access_count;       // Number of times accessed
    uint32_t turn_id;            // Last turn this page was part of
    uint64_t reserved;          // Reserved
};

// Eviction configuration
struct kv_eviction_config {
    // Window sizes (in tokens)
    size_t hot_window_tokens = 16384;     // Always-keep window
    size_t warm_window_tokens = 32768;    // Evict warm before hot
    
    // Thresholds
    int min_access_count = 3;            // System prompt pages accessed many times
    int turn_inactivity_threshold = 2;    // Evict after N turns without access
    
    // Memory limits
    size_t max_hot_bytes = 6ULL * 1024 * 1024 * 1024;  // Max RAM for hot cache (6GB default)
    size_t max_warm_bytes = 2ULL * 1024 * 1024 * 1024;  // Max RAM for warm cache (2GB)

    // Page size
    size_t page_size_tokens = 1024;       // Default 1024 tokens per page

    // Auto-sizing
    bool auto_size = true;               // Automatically adjust based on available memory
    float memory_reserve = 0.15f;        // Keep 15% RAM free for system
    bool unified_memory = false;         // iGPU/UMA system: RAM tiers share the pool with "VRAM", keep them small
    
    // Cold cache limits
    int max_cold_checkpoints = 0;        // Max cold tier entries (0=unlimited)
    
    // Flush behavior
    int flush_interval_ms = 30000;       // Flush dirty pages every 30s
    bool write_back = true;              // Use write-back policy
};

// I/O task types
enum io_task_type {
    IO_LOAD = 0,
    IO_SAVE = 1,
    IO_FLUSH = 2,
};

// I/O task for background thread
struct io_task {
    io_task_type type;
    uint64_t page_id;
    std::vector<uint8_t> data;  // For save tasks
};

// Main page manager class
class kv_page_manager {
public:
    // Default constructor
    kv_page_manager() = default;
    
    // Configuration
    std::string ssd_path;             // Directory for SSD storage
    std::string ssd_file_path;        // Full path to SSD file
    size_t page_size_tokens;          // Tokens per page
    size_t max_hot_bytes;             // Max RAM for hot cache
    size_t max_warm_bytes;            // Max RAM for warm cache
    size_t current_hot_bytes;         // Current hot cache usage
    size_t current_warm_bytes;        // Current warm cache usage
    size_t n_tokens_total;            // Total context size
    size_t n_pages;                   // Total pages
    uint64_t next_page_id;            // Next page ID to allocate
    
    // Eviction config
    kv_eviction_config config;
    
    // State
    std::vector<kv_page_entry> page_table;  // Page metadata
    std::unordered_map<uint64_t, std::vector<uint8_t>> hot_cache; // Hot page data (in RAM)
    std::unordered_map<uint64_t, std::vector<uint8_t>> warm_cache; // Warm page data
    std::fstream ssd_file;                 // SSD file handle
    
    // I/O thread
    std::thread io_thread;
    std::mutex io_mutex;
    std::condition_variable io_cv;
    std::queue<io_task> io_queue;
    bool running;
    
    // Statistics
    uint64_t page_hits = 0;
    uint64_t page_misses = 0;
    uint64_t pages_evicted = 0;
    uint64_t pages_loaded = 0;
    size_t total_bytes_written = 0;
    size_t total_bytes_read = 0;
    
    // Auto-sizing
    size_t detected_memory_bytes = 0;
    bool auto_sized = false;
    
    // Deleted copy/move
    kv_page_manager(const kv_page_manager&) = delete;
    kv_page_manager& operator=(const kv_page_manager&) = delete;
};

// =============================================================================
// Public API
// =============================================================================

/**
 * Initialize the page manager.
 * 
 * @param ssd_path Directory for SSD storage
 * @param cfg Eviction configuration (nullptr for defaults)
 * @param n_tokens_total Total context size (e.g., 131072 for 128K)
 * @return Page manager instance, or nullptr on failure
 */
kv_page_manager* kv_page_manager_init(
    const char* ssd_path,
    const kv_eviction_config* cfg,
    size_t n_tokens_total,
    const char* model_filename = nullptr
);

/**
 * Free the page manager and flush all dirty pages.
 */
void kv_page_manager_free(kv_page_manager* km);

/**
 * Allocate a new page ID.
 * For dynamically growing contexts.
 */
uint64_t kv_page_manager_alloc_page(kv_page_manager* km);

/**
 * Store a page in the hot/warm cache.
 * Evicts cold/warm pages via LRU if needed.
 * 
 * @return true on success
 */
bool kv_page_manager_put(kv_page_manager* km, uint64_t page_id,
                         const uint8_t* data, size_t size);

/**
 * Retrieve a page data.
 * If page is cold, loads from SSD synchronously.
 * Use kv_page_manager_prefetch() for async loading.
 * 
 * @return true if page exists and was copied
 */
bool kv_page_manager_get(kv_page_manager* km, uint64_t page_id,
                         uint8_t* out_data, size_t* out_size);

/**
 * Check if a page is hot (in RAM).
 * Fast path for attention to avoid I/O checks.
 * 
 * @return true if page is in hot cache
 */
bool kv_page_manager_is_hot(kv_page_manager* km, uint64_t page_id);

/**
 * Check if a page is in RAM (hot or warm).
 */
bool kv_page_manager_is_cached(kv_page_manager* km, uint64_t page_id);

/**
 * Prefetch a range of pages asynchronously.
 * Queues page-in operations on the background I/O thread.
 */
void kv_page_manager_prefetch(kv_page_manager* km, uint64_t page_start, uint64_t page_end);

/**
 * Mark a page as dirty (modified in RAM).
 * Will be written to SSD on next flush or eviction.
 */
void kv_page_manager_dirty(kv_page_manager* km, uint64_t page_id);

/**
 * Flush all dirty pages to SSD.
 * Also updates access counts and handles turn transitions.
 */
void kv_page_manager_flush_dirty(kv_page_manager* km);

/**
 * Notify page manager of turn completion.
 * Triggers turn-based eviction of inactive pages.
 * 
 * @param turn_id Current turn ID
 */
void kv_page_manager_on_turn_complete(kv_page_manager* km, uint32_t turn_id);

/**
 * Get memory statistics.
 */
void kv_page_manager_stats(kv_page_manager* km, 
                           size_t* hot_bytes, size_t* warm_bytes, 
                           size_t* cold_bytes, size_t* total_pages);

/**
 * Get performance statistics.
 */
void kv_page_manager_perf_stats(kv_page_manager* km,
                                 uint64_t* hits, uint64_t* misses,
                                 uint64_t* evicted, uint64_t* loaded,
                                 float* hit_rate);

/**
 * Auto-size memory limits based on available system RAM.
 * Call this on systems with variable memory availability.
 */
void kv_page_manager_auto_size(kv_page_manager* km);

/**
 * Get eviction configuration.
 */
const kv_eviction_config* kv_page_manager_get_config(const kv_page_manager* km);

/**
 * Update eviction configuration at runtime.
 */
void kv_page_manager_set_config(kv_page_manager* km, const kv_eviction_config* cfg);

} // namespace llama

#endif // KV_PAGE_MANAGER_H