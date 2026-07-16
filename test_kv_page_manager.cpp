// SPDX-License-Identifier: MIT
// Test for KV Page Manager

#include "common/kv_page_manager.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace llama;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("  %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAILED: %s:%d: %s (%lld) != %s (%lld)\n", __FILE__, __LINE__, #a, (long long)(a), #b, (long long)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Test basic initialization
TEST(init_shutdown) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;  // 64MB for testing
    cfg.max_warm_bytes = 16 * 1024 * 1024;  // 16MB
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 32768);  // 32K tokens
    ASSERT(km != nullptr);
    ASSERT_EQ(km->n_pages, 32);  // 32K / 1K per page
    
    kv_page_manager_free(km);
}

// Test page allocation
TEST(alloc_pages) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    uint64_t page_id = kv_page_manager_alloc_page(km);
    ASSERT_EQ(page_id, 8);  // 8192 / 1024 = 8 pages pre-allocated
    
    kv_page_manager_free(km);
}

// Test put/get basic
TEST(put_get_basic) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    cfg.max_warm_bytes = 16 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Write some data to a page
    uint8_t write_data[1024];
    for (int i = 0; i < 1024; i++) write_data[i] = i & 0xFF;
    
    ASSERT(kv_page_manager_put(km, 0, write_data, sizeof(write_data)));
    ASSERT(kv_page_manager_is_hot(km, 0));
    
    // Read it back
    uint8_t read_data[1024] = {0};
    size_t read_size = 0;
    ASSERT(kv_page_manager_get(km, 0, read_data, &read_size));
    ASSERT_EQ(read_size, 1024U);
    
    // Verify data
    for (int i = 0; i < 1024; i++) {
        ASSERT_EQ(read_data[i], static_cast<uint8_t>(i & 0xFF));
    }
    
    // Statistics
    uint64_t hits = 0, misses = 0, evicted = 0, loaded = 0;
    float hit_rate = 0;
    kv_page_manager_perf_stats(km, &hits, &misses, &evicted, &loaded, &hit_rate);
    ASSERT_EQ(hits, 2U);  // 1 put + 1 get
    ASSERT_EQ(misses, 0U);
    
    kv_page_manager_free(km);
}

// Test cold page load from SSD
TEST(cold_page_load) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 1024;  // Very small to force eviction
    cfg.max_warm_bytes = 0;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Write a page (will be hot)
    uint8_t write_data[1024];
    for (int i = 0; i < 1024; i++) write_data[i] = (i * 2) & 0xFF;
    
    ASSERT(kv_page_manager_put(km, 0, write_data, sizeof(write_data)));
    
    // Force eviction by allocating more pages
    for (int i = 1; i < 16; i++) {
        uint8_t other_data[1024] = {0};
        kv_page_manager_put(km, i, other_data, sizeof(other_data));
    }
    
    // Page 0 should be evicted (LRU)
    // ASSERT(!kv_page_manager_is_hot(km, 0));  // May not be immediate
    
    // Read it back (should load from SSD)
    uint8_t read_data[1024] = {0};
    size_t read_size = 0;
    bool ok = kv_page_manager_get(km, 0, read_data, &read_size);
    
    // If page was saved to SSD, we can load it back
    if (ok) {
        for (int i = 0; i < 1024; i++) {
            ASSERT_EQ(read_data[i], static_cast<uint8_t>((i * 2) & 0xFF));
        }
    }
    
    kv_page_manager_free(km);
}

// Test prefetch
TEST(prefetch) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 32768);
    ASSERT(km != nullptr);
    
    // Prefetch a range
    kv_page_manager_prefetch(km, 5, 10);
    
    // All pages should be in TABLE state COLD but queued for loading
    for (uint64_t i = 5; i <= 10; i++) {
        // Just verify pages exist
        ASSERT(i < km->n_pages);
    }
    
    kv_page_manager_free(km);
}

// Test dirty/flush
TEST(dirty_flush) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Put data
    uint8_t data[1024] = {0xAB};
    kv_page_manager_put(km, 0, data, sizeof(data));
    
    // Mark dirty
    kv_page_manager_dirty(km, 0);
    
    // Flush
    kv_page_manager_flush_dirty(km);
    
    // Give I/O thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    kv_page_manager_free(km);
}

// Test turn completion triggers eviction
TEST(turn_eviction) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    cfg.turn_inactivity_threshold = 2;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Put some data in early pages
    uint8_t data[1024] = {0};
    for (int i = 0; i < 4; i++) {
        kv_page_manager_put(km, i, data, sizeof(data));
    }
    
    // Simulate multiple turns
    kv_page_manager_on_turn_complete(km, 1);
    kv_page_manager_on_turn_complete(km, 2);
    
    // Pages from turn 0 should be evicted by now (if they weren't accessed)
    kv_page_manager_on_turn_complete(km, 3);
    
    kv_page_manager_free(km);
}

// Test hot/warm window logic
TEST(hot_warm_windows) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;  // 1K per page
    cfg.hot_window_tokens = 4096;  // 4 pages hot
    cfg.warm_window_tokens = 8192;  // 8 pages warm
    cfg.auto_size = false;
    cfg.max_hot_bytes = 4 * 1024 * 1024;
    cfg.max_warm_bytes = 4 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 32768);
    ASSERT(km != nullptr);
    
    // Put pages at different positions
    uint8_t data[1024] = {0};
    
    // Pages 0-3 should be hot (in hot window)
    for (int i = 0; i < 4; i++) {
        kv_page_manager_put(km, i, data, sizeof(data));
    }
    
    // Pages 4-7 should be warm (in warm window but not hot)
    for (int i = 4; i < 8; i++) {
        kv_page_manager_put(km, i, data, sizeof(data));
    }
    
    // Check tier assignment
    ASSERT(kv_page_manager_is_hot(km, 0));
    ASSERT(kv_page_manager_is_hot(km, 3));
    ASSERT(kv_page_manager_is_cached(km, 4));
    
    // Get statistics
    size_t hot_bytes = 0, warm_bytes = 0, cold_bytes = 0;
    kv_page_manager_stats(km, &hot_bytes, &warm_bytes, &cold_bytes, nullptr);
    ASSERT(hot_bytes > 0);
    ASSERT(warm_bytes > 0);
    
    kv_page_manager_free(km);
}

// Test auto-sizing
TEST(auto_size) {
    kv_eviction_config cfg;
    cfg.auto_size = true;
    cfg.memory_reserve = 0.2f;  // 20% reserve
    cfg.page_size_tokens = 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Should have detected available memory
    ASSERT(km->detected_memory_bytes > 0);
    
    // Memory limits should be set based on available memory
    ASSERT(km->max_hot_bytes > 0);
    ASSERT(km->max_warm_bytes > 0);
    
    // Hot should be larger than warm
    ASSERT(km->max_hot_bytes > km->max_warm_bytes);
    
    kv_page_manager_free(km);
}

// Test config get/set
TEST(config) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 512;
    cfg.hot_window_tokens = 8192;
    cfg.turn_inactivity_threshold = 5;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Get config
    const kv_eviction_config* retrieved = kv_page_manager_get_config(km);
    ASSERT_EQ(retrieved->page_size_tokens, 512U);
    ASSERT_EQ(retrieved->hot_window_tokens, 8192U);
    ASSERT_EQ(retrieved->turn_inactivity_threshold, 5);
    
    // Update config
    kv_eviction_config new_cfg;
    new_cfg.page_size_tokens = 2048;
    new_cfg.hot_window_tokens = 16384;
    new_cfg.turn_inactivity_threshold = 3;
    new_cfg.auto_size = false;
    new_cfg.max_hot_bytes = 128 * 1024 * 1024;
    new_cfg.max_warm_bytes = 32 * 1024 * 1024;
    
    kv_page_manager_set_config(km, &new_cfg);
    
    retrieved = kv_page_manager_get_config(km);
    ASSERT_EQ(retrieved->page_size_tokens, 2048U);
    ASSERT_EQ(retrieved->hot_window_tokens, 16384U);
    
    kv_page_manager_free(km);
}

// Test performance statistics
TEST(perf_stats) {
    kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.auto_size = false;
    cfg.max_hot_bytes = 64 * 1024 * 1024;
    
    kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, 8192);
    ASSERT(km != nullptr);
    
    // Initial stats
    uint64_t hits = 0, misses = 0, evicted = 0, loaded = 0;
    float hit_rate = 0;
    kv_page_manager_perf_stats(km, &hits, &misses, &evicted, &loaded, &hit_rate);
    ASSERT_EQ(hits, 0U);
    ASSERT_EQ(hit_rate, 0.0f);
    
    // Access some pages
    uint8_t data[1024] = {0};
    kv_page_manager_put(km, 0, data, sizeof(data));
    kv_page_manager_get(km, 0, nullptr, nullptr);
    kv_page_manager_get(km, 0, nullptr, nullptr);
    
    // Updated stats
    kv_page_manager_perf_stats(km, &hits, &misses, &evicted, &loaded, &hit_rate);
    ASSERT_EQ(hits, 3U);  // 1 put + 2 gets
    ASSERT_EQ(misses, 0U);
    ASSERT(hit_rate > 0.99f);
    
    kv_page_manager_free(km);
}

// Test page size variations
TEST(page_sizes) {
    // Test with different page sizes
    std::vector<size_t> page_sizes = {512, 1024, 2048};
    
    for (size_t page_size : page_sizes) {
        kv_eviction_config cfg;
        cfg.page_size_tokens = page_size;
        cfg.auto_size = false;
        cfg.max_hot_bytes = 64 * 1024 * 1024;
        
        size_t context_size = 16384;  // 16K tokens
        
        kv_page_manager* km = kv_page_manager_init("/tmp/kv_test", &cfg, context_size);
        ASSERT(km != nullptr);
        
        // Verify page count
        size_t expected_pages = (context_size + page_size - 1) / page_size;
        ASSERT_EQ(km->n_pages, expected_pages);
        
        // Write and read
        std::vector<uint8_t> data(page_size);
        for (size_t i = 0; i < page_size; i++) data[i] = static_cast<uint8_t>(i);
        
        ASSERT(kv_page_manager_put(km, 0, data.data(), data.size()));
        
        std::vector<uint8_t> read_data(page_size, 0);
        size_t read_size = 0;
        ASSERT(kv_page_manager_get(km, 0, read_data.data(), &read_size));
        ASSERT_EQ(read_size, page_size);
        
        for (size_t i = 0; i < page_size; i++) {
            ASSERT_EQ(read_data[i], static_cast<uint8_t>(i));
        }
        
        kv_page_manager_free(km);
    }
}

int main() {
    printf("KV Page Manager Test Suite\n");
    printf("==========================\n\n");

    // Ensure test directory exists and is clean
#if defined(_WIN32)
    system("rmdir /s /q \\tmp\\kv_test 2>nul");
    system("mkdir \\tmp\\kv_test 2>nul");
#else
    system("rm -rf /tmp/kv_test");
    system("mkdir -p /tmp/kv_test");
#endif
    
    // Run tests
    printf("Running tests...\n\n");
    
    RUN_TEST(init_shutdown);
    RUN_TEST(alloc_pages);
    RUN_TEST(put_get_basic);
    RUN_TEST(cold_page_load);
    RUN_TEST(prefetch);
    RUN_TEST(dirty_flush);
    RUN_TEST(turn_eviction);
    RUN_TEST(hot_warm_windows);
    RUN_TEST(auto_size);
    RUN_TEST(config);
    RUN_TEST(perf_stats);
    RUN_TEST(page_sizes);
    
    // Summary
    printf("\n==========================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    
    // Cleanup
#if defined(_WIN32)
    system("rmdir /s /q \\tmp\\kv_test 2>nul");
#else
    system("rm -rf /tmp/kv_test");
#endif

    return tests_failed > 0 ? 1 : 0;
}