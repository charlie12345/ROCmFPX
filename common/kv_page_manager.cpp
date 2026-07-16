// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// SSD-Backed KV Cache Paging Manager

#include "kv_page_manager.h"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>

#if defined(_WIN32)
#include <direct.h>
// Windows _mkdir has no mode argument
#define mkdir(path, mode) _mkdir(path)
#else
#include <unistd.h>
#endif

// Platform-specific memory detection
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace llama {

// Magic number to identify KV page files
static const uint32_t KV_PAGE_MAGIC = 0x4B565047; // "KVPG"
static const uint32_t KV_PAGE_VERSION = 2;
static const size_t HEADER_SIZE = 4096;
static const size_t PAGE_ALIGN = 4096;

// =============================================================================
// Forward declarations
// =============================================================================

static void write_header(kv_page_manager* km);
static bool load_page_from_ssd(kv_page_manager* km, uint64_t page_id);
static bool save_page_to_ssd(kv_page_manager* km, uint64_t page_id);
static void evict_lru_pages(kv_page_manager* km, size_t needed_bytes, bool evict_warm_too);
static void evict_turn_inactive(kv_page_manager* km, uint32_t turn_id);
static void queue_page_load(kv_page_manager* km, uint64_t page_id);
static void queue_page_save(kv_page_manager* km, uint64_t page_id);
static void wait_for_page_loaded(kv_page_manager* km, uint64_t page_id);
static void io_thread_func(kv_page_manager* km);
static uint64_t get_timestamp_ms();
static size_t get_available_memory_bytes();
static uint64_t calculate_eviction_score(kv_page_manager* km, const kv_page_entry& entry);

// =============================================================================
// Public API implementation
// =============================================================================

kv_page_manager* kv_page_manager_init(
    const char* ssd_path,
    const kv_eviction_config* cfg,
    size_t n_tokens_total,
    const char* model_filename
) {
    kv_page_manager* km = new kv_page_manager();
    
    // Set default config or use provided
    if (cfg) {
        km->config = *cfg;
    }
    
    // Override page size if configured
    km->page_size_tokens = km->config.page_size_tokens;
    km->n_tokens_total = n_tokens_total;
    km->running = true;
    km->next_page_id = 0;
    km->current_hot_bytes = 0;
    km->current_warm_bytes = 0;
    
    // Calculate total pages needed
    km->n_pages = (n_tokens_total + km->page_size_tokens - 1) / km->page_size_tokens;
    
    // Auto-size if enabled
    if (km->config.auto_size) {
        kv_page_manager_auto_size(km);
    }
    
    // Initialize page table
    km->page_table.resize(km->n_pages);
    for (size_t i = 0; i < km->n_pages; i++) {
        km->page_table[i].page_id = i;
        km->page_table[i].token_start = static_cast<uint32_t>(i * km->page_size_tokens);
        km->page_table[i].token_end = static_cast<uint32_t>(std::min((i + 1) * km->page_size_tokens, n_tokens_total));
        km->page_table[i].state = PAGE_COLD;
        km->page_table[i].ssd_offset = 0;
        km->page_table[i].size_bytes = 0;
        km->page_table[i].last_access = 0;
        km->page_table[i].access_count = 0;
        km->page_table[i].turn_id = 0;
    }
    
    // Create SSD directory if needed
    mkdir(ssd_path, 0755);
    
    // Open SSD file for this session
    km->ssd_path = ssd_path;
    if (model_filename && model_filename[0]) {
        km->ssd_file_path = std::string(ssd_path) + "/" + std::string(model_filename) + ".bin";
    } else {
        km->ssd_file_path = std::string(ssd_path) + "/kv_cache.bin";
    }
    km->ssd_file.open(km->ssd_file_path, 
                      std::ios::in | std::ios::out | std::ios::binary);
    if (!km->ssd_file.is_open()) {
        // File doesn't exist, create it
        km->ssd_file.open(km->ssd_file_path,
                          std::ios::out | std::ios::binary);
        km->ssd_file.close();
        km->ssd_file.open(km->ssd_file_path,
                          std::ios::in | std::ios::out | std::ios::binary);
    }
    
    // Write header
    write_header(km);
    
    // Start I/O thread
    km->io_thread = std::thread(io_thread_func, km);
    
    return km;
}

void kv_page_manager_free(kv_page_manager* km) {
    if (!km) return;
    
    // Signal I/O thread to stop
    km->running = false;
    km->io_cv.notify_all();
    
    // Wait for I/O thread
    if (km->io_thread.joinable()) {
        km->io_thread.join();
    }
    
    // Flush any dirty pages synchronously
    for (auto& entry : km->page_table) {
        if (entry.state == PAGE_DIRTY || entry.state == PAGE_NEW) {
            // Get data from cache and save
            auto hit = km->hot_cache.find(entry.page_id);
            if (hit != km->hot_cache.end()) {
                save_page_to_ssd(km, entry.page_id);
            }
        }
    }
    
    // Update header with final state
    write_header(km);
    
    // Close file
    if (km->ssd_file.is_open()) {
        km->ssd_file.close();
    }
    
    delete km;
}

uint64_t kv_page_manager_alloc_page(kv_page_manager* km) {
    uint64_t page_id = km->next_page_id++;
    if (page_id >= km->page_table.size()) {
        // Expand page table
        km->page_table.push_back(kv_page_entry{});
        auto& entry = km->page_table.back();
        entry.page_id = page_id;
        entry.token_start = static_cast<uint32_t>(page_id * km->page_size_tokens);
        entry.token_end = static_cast<uint32_t>(std::min((page_id + 1) * km->page_size_tokens, km->n_tokens_total));
        entry.state = PAGE_COLD;
        entry.ssd_offset = 0;
        entry.size_bytes = 0;
        entry.last_access = 0;
        entry.access_count = 0;
        entry.turn_id = 0;
    }
    return page_id;
}

bool kv_page_manager_put(kv_page_manager* km, uint64_t page_id, 
                         const uint8_t* data, size_t size) {
    if (page_id >= km->page_table.size()) return false;
    
    auto& entry = km->page_table[page_id];
    
    // Check if we need to evict to make room
    size_t total_used = km->current_hot_bytes + km->current_warm_bytes;
    if (total_used + size > km->max_hot_bytes + km->max_warm_bytes) {
        evict_lru_pages(km, size, true);
    }
    
    // Determine tier based on position
    uint32_t token_pos = entry.token_start;
    size_t hot_window_pages = km->config.hot_window_tokens / km->page_size_tokens;
    size_t warm_window_pages = km->config.warm_window_tokens / km->page_size_tokens;
    
    // Calculate the "tail" of the context (most recent tokens)
    // Pages closer to the end (higher token_pos) are hotter
    size_t context_end_page = (km->n_tokens_total + km->page_size_tokens - 1) / km->page_size_tokens;
    size_t tail_distance = (context_end_page > 0) ? (context_end_page - 1 - page_id) : 0;
    
    bool is_hot = (tail_distance < hot_window_pages) || 
                  (entry.access_count >= km->config.min_access_count);
    bool is_warm = (tail_distance < warm_window_pages) && !is_hot;
    
    // Store in appropriate cache
    if (is_hot) {
        km->hot_cache[page_id].assign(data, data + size);
        entry.state = PAGE_HOT;
    } else if (is_warm) {
        km->warm_cache[page_id].assign(data, data + size);
        entry.state = PAGE_WARM;
        km->current_warm_bytes += size;
    } else {
        // Not in a hot/warm window, but still store for now
        // Will be evicted on next LRU pass
        km->hot_cache[page_id].assign(data, data + size);
        entry.state = PAGE_HOT;
    }
    
    entry.size_bytes = size;
    entry.last_access = get_timestamp_ms();
    entry.access_count++;
    
    if (entry.state == PAGE_HOT) {
        km->current_hot_bytes += size;
    }
    
    return true;
}

bool kv_page_manager_get(kv_page_manager* km, uint64_t page_id,
                         uint8_t* out_data, size_t* out_size) {
    if (page_id >= km->page_table.size()) return false;
    
    auto& entry = km->page_table[page_id];
    
    // Check hot cache first
    auto hit = km->hot_cache.find(page_id);
    if (hit != km->hot_cache.end()) {
        if (out_data && !hit->second.empty()) {
            memcpy(out_data, hit->second.data(), hit->second.size());
        }
        if (out_size) *out_size = entry.size_bytes;
        entry.last_access = get_timestamp_ms();
        entry.access_count++;
        km->page_hits++;
        return true;
    }
    
    // Check warm cache
    auto warm_it = km->warm_cache.find(page_id);
    if (warm_it != km->warm_cache.end()) {
        if (out_data && !warm_it->second.empty()) {
            memcpy(out_data, warm_it->second.data(), warm_it->second.size());
        }
        if (out_size) *out_size = entry.size_bytes;
        entry.last_access = get_timestamp_ms();
        entry.access_count++;
        km->page_hits++;
        return true;
    }
    
    // If loading, wait for it
    if (entry.state == PAGE_LOADING) {
        wait_for_page_loaded(km, page_id);
        hit = km->hot_cache.find(page_id);
        if (hit != km->hot_cache.end()) {
            if (out_data && !hit->second.empty()) {
                memcpy(out_data, hit->second.data(), hit->second.size());
            }
            if (out_size) *out_size = entry.size_bytes;
            return true;
        }
    }
    
    // Cold page - load from SSD
    km->page_misses++;
    bool loaded = load_page_from_ssd(km, page_id);
    if (loaded) {
        hit = km->hot_cache.find(page_id);
        if (hit != km->hot_cache.end() && out_data) {
            memcpy(out_data, hit->second.data(), hit->second.size());
        }
        if (out_size) *out_size = entry.size_bytes;
    }
    return loaded;
}

bool kv_page_manager_is_hot(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return false;
    return km->page_table[page_id].state == PAGE_HOT &&
           km->hot_cache.find(page_id) != km->hot_cache.end();
}

bool kv_page_manager_is_cached(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return false;
    auto& entry = km->page_table[page_id];
    if (entry.state == PAGE_HOT || entry.state == PAGE_WARM || entry.state == PAGE_LOADING) {
        return true;
    }
    if (entry.state == PAGE_HOT) {
        return km->hot_cache.find(page_id) != km->hot_cache.end();
    }
    if (entry.state == PAGE_WARM) {
        return km->warm_cache.find(page_id) != km->warm_cache.end();
    }
    return false;
}

void kv_page_manager_prefetch(kv_page_manager* km, uint64_t page_start, uint64_t page_end) {
    for (uint64_t page_id = page_start; page_id <= page_end && page_id < km->page_table.size(); page_id++) {
        auto& entry = km->page_table[page_id];
        if (entry.state == PAGE_COLD) {
            queue_page_load(km, page_id);
        }
    }
}

void kv_page_manager_dirty(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return;
    
    auto& entry = km->page_table[page_id];
    // Mark page dirty so it gets persisted to SSD
    // Can be called on HOT, WARM, or COLD pages
    if (entry.state == PAGE_HOT || entry.state == PAGE_WARM || entry.state == PAGE_COLD) {
        entry.state = PAGE_DIRTY;
    }
}

void kv_page_manager_flush_dirty(kv_page_manager* km) {
    std::vector<uint64_t> to_save;
    
    // Collect dirty pages
    for (auto& entry : km->page_table) {
        if (entry.state == PAGE_DIRTY) {
            to_save.push_back(entry.page_id);
        }
    }
    
    // Queue saves
    for (uint64_t page_id : to_save) {
        queue_page_save(km, page_id);
    }
}

void kv_page_manager_on_turn_complete(kv_page_manager* km, uint32_t turn_id) {
    // Evict pages that haven't been accessed in too many turns
    evict_turn_inactive(km, turn_id);
    
    // Promote accessed pages based on new access counts
    for (auto& entry : km->page_table) {
        if (entry.turn_id < turn_id - 1 && entry.state == PAGE_WARM) {
            // This page wasn't accessed in the last turn
            // Consider demoting to cold
            if (entry.turn_id < turn_id - km->config.turn_inactivity_threshold) {
                auto it = km->warm_cache.find(entry.page_id);
                if (it != km->warm_cache.end()) {
                    km->current_warm_bytes -= it->second.size();
                    km->warm_cache.erase(it);
                }
                entry.state = PAGE_COLD;
            }
        }
        entry.turn_id = turn_id;
    }
}

void kv_page_manager_stats(kv_page_manager* km, 
                           size_t* hot_bytes, size_t* warm_bytes,
                           size_t* cold_bytes, size_t* total_pages) {
    if (hot_bytes) *hot_bytes = km->current_hot_bytes;
    if (warm_bytes) *warm_bytes = km->current_warm_bytes;
    if (cold_bytes) {
        *cold_bytes = 0;
        for (const auto& entry : km->page_table) {
            if (entry.state == PAGE_COLD) {
                *cold_bytes += entry.size_bytes;
            }
        }
    }
    if (total_pages) *total_pages = km->n_pages;
}

void kv_page_manager_perf_stats(kv_page_manager* km,
                                 uint64_t* hits, uint64_t* misses,
                                 uint64_t* evicted, uint64_t* loaded,
                                 float* hit_rate) {
    if (hits) *hits = km->page_hits;
    if (misses) *misses = km->page_misses;
    if (evicted) *evicted = km->pages_evicted;
    if (loaded) *loaded = km->pages_loaded;
    if (hit_rate) {
        uint64_t total = km->page_hits + km->page_misses;
        *hit_rate = total > 0 ? (float)km->page_hits / total : 0.0f;
    }
}

void kv_page_manager_auto_size(kv_page_manager* km) {
    size_t available = get_available_memory_bytes();
    km->detected_memory_bytes = available;
    
    // Reserve some memory for system
    size_t usable = static_cast<size_t>(available * (1.0f - km->config.memory_reserve));
    
    // Split between hot and warm
    // Default: 75% hot, 25% warm
    km->max_hot_bytes = (usable * 3) / 4;
    km->max_warm_bytes = usable / 4;
    
    km->auto_sized = true;
}

const kv_eviction_config* kv_page_manager_get_config(const kv_page_manager* km) {
    return &km->config;
}

void kv_page_manager_set_config(kv_page_manager* km, const kv_eviction_config* cfg) {
    if (cfg) {
        km->config = *cfg;
        // Update derived values
        km->page_size_tokens = cfg->page_size_tokens;
        if (!cfg->auto_size) {
            km->max_hot_bytes = cfg->max_hot_bytes;
            km->max_warm_bytes = cfg->max_warm_bytes;
        }
    }
}

// =============================================================================
// Internal implementation
// =============================================================================

static void write_header(kv_page_manager* km) {
    if (!km->ssd_file.is_open()) return;
    
    km->ssd_file.seekp(0, std::ios::beg);
    
    kv_page_header header;
    header.magic = KV_PAGE_MAGIC;
    header.version = KV_PAGE_VERSION;
    header.n_pages = static_cast<uint32_t>(km->n_pages);
    header.page_size_tokens = static_cast<uint32_t>(km->page_size_tokens);
    header.data_offset = HEADER_SIZE + km->n_pages * sizeof(kv_page_entry);
    header.data_offset = (header.data_offset + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
    header.total_size_bytes = 0;  // Will be updated as pages are written
    memset(header.reserved, 0, sizeof(header.reserved));
    
    km->ssd_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write page table
    km->ssd_file.write(reinterpret_cast<const char*>(km->page_table.data()),
                       km->n_pages * sizeof(kv_page_entry));
}

static bool load_page_from_ssd(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return false;
    
    auto& entry = km->page_table[page_id];
    if (entry.ssd_offset == 0 || entry.size_bytes == 0) return false;
    
    // Check if we need to evict to make room
    size_t total_used = km->current_hot_bytes + km->current_warm_bytes;
    if (total_used + entry.size_bytes > km->max_hot_bytes + km->max_warm_bytes) {
        evict_lru_pages(km, entry.size_bytes, true);
    }
    
    // Seek and read
    km->ssd_file.seekg(static_cast<std::streamoff>(entry.ssd_offset), std::ios::beg);
    km->hot_cache[page_id].resize(entry.size_bytes);
    km->ssd_file.read(reinterpret_cast<char*>(km->hot_cache[page_id].data()), entry.size_bytes);
    
    if (km->ssd_file.gcount() != static_cast<std::streamsize>(entry.size_bytes)) {
        km->hot_cache.erase(page_id);
        return false;
    }
    
    entry.state = PAGE_HOT;
    entry.last_access = get_timestamp_ms();
    km->current_hot_bytes += entry.size_bytes;
    km->pages_loaded++;
    km->total_bytes_read += entry.size_bytes;
    
    return true;
}

static bool save_page_to_ssd(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return false;
    
    auto& entry = km->page_table[page_id];
    
    // Get data from cache
    auto hot_it = km->hot_cache.find(page_id);
    auto warm_it = km->warm_cache.find(page_id);
    const std::vector<uint8_t>* data_ptr = nullptr;
    size_t size = 0;
    
    if (hot_it != km->hot_cache.end()) {
        data_ptr = &hot_it->second;
        size = hot_it->second.size();
    } else if (warm_it != km->warm_cache.end()) {
        data_ptr = &warm_it->second;
        size = warm_it->second.size();
    } else {
        return false;
    }
    
    const auto& data = *data_ptr;
    
    // Find end of file for new page
    km->ssd_file.seekp(0, std::ios::end);
    entry.ssd_offset = static_cast<uint64_t>(km->ssd_file.tellp());
    
    // Align to page boundary
    size_t padded_size = (data.size() + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
    
    // Write page data
    km->ssd_file.write(reinterpret_cast<const char*>(data.data()), data.size());
    
    // Pad to alignment
    if (padded_size > data.size()) {
        std::vector<uint8_t> padding(padded_size - data.size(), 0);
        km->ssd_file.write(reinterpret_cast<const char*>(padding.data()), padding.size());
    }
    
    entry.size_bytes = size;
    entry.state = PAGE_COLD;
    km->total_bytes_written += size;
    
    return true;
}

static void evict_lru_pages(kv_page_manager* km, size_t needed_bytes, bool evict_warm_too) {
    // Collect all evictable candidates with scores
    struct candidate {
        uint64_t page_id;
        uint64_t score;
        bool in_warm;
    };
    std::vector<candidate> candidates;
    
    uint64_t now = get_timestamp_ms();
    
    for (auto& entry : km->page_table) {
        // Can't evict pages that are already cold (not cached)
        if (entry.state == PAGE_COLD && !evict_warm_too) continue;
        
        if (entry.state == PAGE_HOT) {
            candidates.push_back({
                entry.page_id,
                calculate_eviction_score(km, entry),
                false
            });
        } else if (evict_warm_too && entry.state == PAGE_WARM) {
            candidates.push_back({
                entry.page_id,
                calculate_eviction_score(km, entry),
                true
            });
        }
    }
    
    // Sort by score (lowest = most evictable first)
    std::sort(candidates.begin(), candidates.end(), 
              [](const candidate& a, const candidate& b) {
                  return a.score < b.score;
              });
    
    // Evict until we have enough room
    for (const auto& c : candidates) {
        if (km->current_hot_bytes + km->current_warm_bytes <= 
            km->max_hot_bytes + km->max_warm_bytes - needed_bytes) break;
        
        auto& entry = km->page_table[c.page_id];
        
        if (c.in_warm) {
            auto it = km->warm_cache.find(c.page_id);
            if (it != km->warm_cache.end()) {
                // Save to SSD if dirty
                if (entry.state == PAGE_DIRTY) {
                    save_page_to_ssd(km, c.page_id);
                }
                km->current_warm_bytes -= it->second.size();
                km->warm_cache.erase(it);
                entry.state = PAGE_COLD;
            }
        } else {
            auto it = km->hot_cache.find(c.page_id);
            if (it != km->hot_cache.end()) {
                // Save to SSD if dirty
                if (entry.state == PAGE_DIRTY) {
                    save_page_to_ssd(km, c.page_id);
                }
                km->current_hot_bytes -= it->second.size();
                km->hot_cache.erase(it);
                entry.state = PAGE_COLD;
            }
        }
        
        km->pages_evicted++;
    }
}

static void evict_turn_inactive(kv_page_manager* km, uint32_t turn_id) {
    int threshold = km->config.turn_inactivity_threshold;
    
    for (auto& entry : km->page_table) {
        if (entry.turn_id == 0) continue;  // Never accessed
        if (entry.turn_id >= turn_id - threshold) continue;  // Still active
        // Check if page is cached (can't evict cold pages here)
        
        // Check hot cache
        auto hot_it = km->hot_cache.find(entry.page_id);
        if (hot_it != km->hot_cache.end()) {
            if (entry.state == PAGE_DIRTY) {
                save_page_to_ssd(km, entry.page_id);
            }
            km->current_hot_bytes -= hot_it->second.size();
            km->hot_cache.erase(hot_it);
            entry.state = PAGE_COLD;
            km->pages_evicted++;
            continue;
        }
        
        // Check warm cache
        auto warm_it = km->warm_cache.find(entry.page_id);
        if (warm_it != km->warm_cache.end()) {
            if (entry.state == PAGE_DIRTY) {
                save_page_to_ssd(km, entry.page_id);
            }
            km->current_warm_bytes -= warm_it->second.size();
            km->warm_cache.erase(warm_it);
            entry.state = PAGE_COLD;
            km->pages_evicted++;
        }
    }
}

static uint64_t calculate_eviction_score(kv_page_manager* km, const kv_page_entry& entry) {
    // Lower score = more evictable
    uint64_t now = get_timestamp_ms();
    
    // Base score from age (older = more evictable, max 1000 points)
    uint64_t age_ms = now - entry.last_access;
    uint64_t age_score = std::min(age_ms / 1000, static_cast<uint64_t>(1000));
    
    // Access count bonus (frequently accessed = less evictable, max 2000 points)
    uint64_t access_score = std::min(static_cast<uint64_t>(entry.access_count) * 200, static_cast<uint64_t>(2000));
    
    // Hot window penalty (pages in hot window = much less evictable)
    uint64_t hot_window_pages = km->config.hot_window_tokens / km->page_size_tokens;
    uint64_t context_end_page = (km->n_tokens_total + km->page_size_tokens - 1) / km->page_size_tokens;
    uint64_t tail_distance = (context_end_page > 0 && entry.page_id < context_end_page) 
                            ? (context_end_page - 1 - entry.page_id) : 0;
    uint64_t hot_penalty = (tail_distance < hot_window_pages) ? 10000ULL : 0ULL;
    
    // Page size bonus (larger pages = more evictable, max 500 points)
    uint64_t size_score = std::min(entry.size_bytes / (1024 * 1024), static_cast<uint64_t>(500));
    
    return age_score + size_score - access_score + hot_penalty;
}

static void queue_page_load(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return;
    
    auto& entry = km->page_table[page_id];
    if (entry.state != PAGE_COLD) return;
    
    std::lock_guard<std::mutex> lock(km->io_mutex);
    
    io_task task;
    task.type = IO_LOAD;
    task.page_id = page_id;
    km->io_queue.push(task);
    
    km->io_cv.notify_one();
}

static void queue_page_save(kv_page_manager* km, uint64_t page_id) {
    if (page_id >= km->page_table.size()) return;
    
    auto& entry = km->page_table[page_id];
    
    std::lock_guard<std::mutex> lock(km->io_mutex);
    
    io_task task;
    task.type = IO_SAVE;
    task.page_id = page_id;
    
    // Copy data for async save
    auto hot_it = km->hot_cache.find(page_id);
    if (hot_it != km->hot_cache.end()) {
        task.data = hot_it->second;
    }
    
    km->io_queue.push(task);
    km->io_cv.notify_one();
}

static void wait_for_page_loaded(kv_page_manager* km, uint64_t page_id) {
    auto& entry = km->page_table[page_id];
    
    // Simple spin-wait
    int max_spins = 1000;
    while (entry.state == PAGE_LOADING && max_spins-- > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void io_thread_func(kv_page_manager* km) {
    while (km->running) {
        io_task task;
        
        {
            std::unique_lock<std::mutex> lock(km->io_mutex);
            km->io_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !km->io_queue.empty() || !km->running;
            });
            
            if (!km->running) break;
            if (km->io_queue.empty()) continue;
            
            task = km->io_queue.front();
            km->io_queue.pop();
        }
        
        // Process I/O task
        switch (task.type) {
            case IO_LOAD: {
                auto& entry = km->page_table[task.page_id];
                entry.state = PAGE_LOADING;
                
                if (entry.ssd_offset > 0 && entry.size_bytes > 0) {
                    km->ssd_file.seekg(static_cast<std::streamoff>(entry.ssd_offset), std::ios::beg);
                    km->hot_cache[task.page_id].resize(entry.size_bytes);
                    km->ssd_file.read(reinterpret_cast<char*>(km->hot_cache[task.page_id].data()), entry.size_bytes);
                    
                    if (km->ssd_file.gcount() == static_cast<std::streamsize>(entry.size_bytes)) {
                        entry.state = PAGE_HOT;
                        entry.last_access = get_timestamp_ms();
                        km->current_hot_bytes += entry.size_bytes;
                        km->pages_loaded++;
                        km->total_bytes_read += entry.size_bytes;
                    }
                }
                break;
            }
            
            case IO_SAVE: {
                auto& entry = km->page_table[task.page_id];
                
                if (!task.data.empty()) {
                    km->ssd_file.seekp(0, std::ios::end);
                    entry.ssd_offset = static_cast<uint64_t>(km->ssd_file.tellp());
                    
                    size_t padded_size = (task.data.size() + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
                    km->ssd_file.write(reinterpret_cast<const char*>(task.data.data()), task.data.size());
                    
                    if (padded_size > task.data.size()) {
                        std::vector<uint8_t> padding(padded_size - task.data.size(), 0);
                        km->ssd_file.write(reinterpret_cast<const char*>(padding.data()), padding.size());
                    }
                    
                    entry.size_bytes = task.data.size();
                    entry.state = PAGE_COLD;
                    km->total_bytes_written += task.data.size();
                }
                break;
            }
            
            case IO_FLUSH: {
                // Periodic flush - save all dirty pages
                for (auto& entry : km->page_table) {
                    if (entry.state == PAGE_DIRTY) {
                        auto hot_it = km->hot_cache.find(entry.page_id);
                        if (hot_it != km->hot_cache.end() && !hot_it->second.empty()) {
                            km->ssd_file.seekp(0, std::ios::end);
                            entry.ssd_offset = static_cast<uint64_t>(km->ssd_file.tellp());
                            
                            size_t padded_size = (hot_it->second.size() + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
                            km->ssd_file.write(reinterpret_cast<const char*>(hot_it->second.data()), hot_it->second.size());
                            
                            if (padded_size > hot_it->second.size()) {
                                std::vector<uint8_t> padding(padded_size - hot_it->second.size(), 0);
                                km->ssd_file.write(reinterpret_cast<const char*>(padding.data()), padding.size());
                            }
                            
                            entry.size_bytes = hot_it->second.size();
                            entry.state = PAGE_COLD;
                            km->total_bytes_written += hot_it->second.size();
                        }
                    }
                }
                break;
            }
        }
    }
}

static uint64_t get_timestamp_ms() {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return static_cast<uint64_t>(ms);
}

static size_t get_available_memory_bytes() {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
#endif
    // Default fallback: assume 8GB
    return 8ULL * 1024 * 1024 * 1024;
}

} // namespace llama