#include "llama-kv-recency-plan.h"
#include "llama-kv-cache-recency.h"
#include "llama-kv-cells.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <cstdio>
#include <limits>
#include <stdexcept>

#undef NDEBUG
#include <cassert>

static std::vector<llama_kv_recency_row> make_rows(const llama_kv_recency_plan_params & p, uint32_t cell) {
    std::vector<llama_kv_recency_row> rows(p.layers.size());
    for (size_t layer = 0; layer < p.layers.size(); ++layer) {
        for (bool key : {true, false}) {
            const auto width = key ? p.layers[layer].k_width : p.layers[layer].v_width;
            if (!width) {
                continue;
            }
            std::vector<float> values(width);
            for (size_t d = 0; d < width; ++d) {
                values[d] = std::sin(float(d+cell)*0.17f)*3 + float((cell+layer)%7)*0.13f - (key ? 0.0f : 1.0f);
            }
            auto & packed = key ? rows[layer].k : rows[layer].v;
            packed.resize(ggml_row_size(GGML_TYPE_Q8_0, width));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), packed.data(), 0, 1, width, nullptr);
        }
    }
    return rows;
}

static std::vector<uint8_t> requantize_q4(const std::vector<uint8_t> & q8, int64_t width) {
    std::vector<float> values(width);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q8.data(), values.data(), width);
    std::vector<uint8_t> q4(ggml_row_size(GGML_TYPE_Q4_0, width));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), q4.data(), 0, 1, width, nullptr);
    return q4;
}

static size_t backing_bytes(const llama_kv_recency_store & store) {
    size_t bytes = 0;
    for (const auto & entry : store.memory_breakdown()) { bytes += entry.second; }
    return bytes;
}

// Every written cell is resident exactly once: oldest pages as Q4 at their logical offset, the rest in the Q8 ring.
static void verify_resident(const llama_kv_recency_store & store, const llama_kv_recency_plan_params & p,
                            const std::vector<std::vector<llama_kv_recency_row>> & sources, uint32_t n_cells, llama_seq_id seq) {
    const uint32_t cold = store.get_migration_stats().cold_pages*p.page_tokens;
    std::vector<int> seen(n_cells, 0);
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        for (uint32_t slab = 0; slab < store.get_slab_count(type); ++slab) {
            if (!store.get_tensor(0, true, type, slab)) { continue; }
            const auto rows = uint32_t(store.get_tensor(0, true, type, slab)->ne[1]);
            std::vector<int32_t> cells(rows);
            for (uint32_t row = 0; row < rows; ++row) {
                cells[row] = store.get_row_cell(type, slab, row);
                const uint32_t global = slab*p.slab_tokens + row;
                if (type == GGML_TYPE_Q4_0) { assert(cells[row] == (global < cold ? int32_t(global) : -1)); }
                if (cells[row] < 0) { continue; }
                assert(uint32_t(cells[row]) < n_cells && (type == GGML_TYPE_Q4_0) == (uint32_t(cells[row]) < cold));
                ++seen[cells[row]];
            }
            for (size_t layer = 0; layer < p.layers.size(); ++layer) {
                for (bool key : {true, false}) {
                    auto * tensor = store.get_tensor(layer, key, type, slab);
                    if (!tensor) {
                        assert(!key && !p.layers[layer].v_width);
                        continue;
                    }
                    std::vector<uint8_t> bytes(size_t(rows)*tensor->nb[1]);
                    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
                    const size_t size = ggml_row_size(type, tensor->ne[0]);
                    for (uint32_t row = 0; row < rows; ++row) {
                        const uint8_t * actual = bytes.data() + size_t(row)*tensor->nb[1];
                        if (cells[row] < 0) {
                            // Unpublished Q4 rows may hold aborted conversion output; free Q8 slots hold no history.
                            assert(type == GGML_TYPE_Q4_0 || std::all_of(actual, actual + size, [](uint8_t b) { return b == 0; }));
                            continue;
                        }
                        const auto & source = key ? sources[cells[row]][layer].k : sources[cells[row]][layer].v;
                        const auto expected = type == GGML_TYPE_Q8_0 ? source : requantize_q4(source, tensor->ne[0]);
                        assert(expected.size() == size && std::equal(expected.begin(), expected.end(), actual));
                    }
                }
            }
        }
    }
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        assert(seen[cell] == 1 && "each logical cell must have exactly one resident representation");
        assert(store.get_cells().pos_get(cell) == int(1000+cell) && store.get_cells().seq_has(cell, seq));
    }
}

// Expected backing when the first q8/q4 slab positions of each region are allocated.
static size_t slab_backing(const llama_kv_recency_plan_params & p, uint32_t q8, uint32_t q4) {
    const auto plan = llama_kv_recency_plan_create(p);
    size_t bytes = 0;
    for (uint32_t slab = 0; slab < q8; ++slab) {
        bytes += llama_kv_recency_slab_alloc_bytes(p, GGML_TYPE_Q8_0, std::min(p.slab_tokens, plan.ring_tokens - slab*p.slab_tokens));
    }
    for (uint32_t slab = 0; slab < q4; ++slab) {
        bytes += llama_kv_recency_slab_alloc_bytes(p, GGML_TYPE_Q4_0, std::min(p.slab_tokens, plan.cold_tokens - slab*p.slab_tokens));
    }
    return bytes;
}

// Remaining successful allocations of the failing CPU buffer type; -1 is unlimited.
static int failing_allocations = -1;

static ggml_backend_buffer_t failing_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (failing_allocations == 0) { return nullptr; }
    if (failing_allocations > 0) { --failing_allocations; }
    return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(buft, size);
}

static ggml_backend_buffer_type_t failing_buffer_type() {
    static ggml_backend_buffer_type buft = [] {
        ggml_backend_buffer_type result = *ggml_backend_cpu_buffer_type();
        result.iface.alloc_buffer = failing_alloc_buffer;
        return result;
    }();
    return &buft;
}

static void test_slab_allocation_failure() {
    llama_kv_recency_plan_params p;
    p.logical_ctx = 2048;
    p.hot_tokens = 256;
    p.migrate_tokens = 256;
    p.slab_tokens = 256;
    p.layers = {{128, 64, failing_buffer_type()}, {64, 128, failing_buffer_type()}};
    assert(llama_kv_recency_plan_create(p).ring_tokens == 768);
    llama_kv_recency_store store(p, llama_kv_recency_store::mode::committed_aging);
    std::vector<std::vector<llama_kv_recency_row>> sources(p.logical_ctx);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell); }
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    const auto rejected = [](auto fn) {
        try { fn(); } catch (const std::runtime_error &) { return true; }
        return false;
    };
    failing_allocations = 0;
    assert(rejected([&] { store.load_q8_row(0, 1000, {1}, sources[0]); }));
    assert(store.get_cells().get_used() == 0 && backing_bytes(store) == 0);
    failing_allocations = -1;
    for (uint32_t cell = 0; cell < 512; ++cell) { store.load_q8_row(cell, 1000+cell, {1}, sources[cell]); }
    assert(backing_bytes(store) == slab_backing(p, 2, 0));
    failing_allocations = 0;
    assert(rejected([&] { store.load_q8_row(512, 1512, {1}, sources[512]); }) && "third Q8 slab cannot be allocated");
    assert(store.get_cells().is_empty(512) && store.get_cells().get_used() == 512);
    assert(rejected([&] { store.migrate_oldest_pages({backend.get()}, 1); }) && "first Q4 slab cannot be allocated");
    const auto stats = store.get_migration_stats();
    assert(stats.cold_pages == 0 && stats.hot_pages == 2 && backing_bytes(store) == slab_backing(p, 2, 0));
    verify_resident(store, p, sources, 512, 1);
    failing_allocations = -1;
    assert(store.migrate_oldest_pages({backend.get()}, 1) == 1 && backing_bytes(store) == slab_backing(p, 2, 1));
    for (uint32_t cell = 512; cell < 768; ++cell) { store.load_q8_row(cell, 1000+cell, {1}, sources[cell]); }
    assert(backing_bytes(store) == slab_backing(p, 2, 1) && "page 2 reuses the freed slot without a new slab");
    for (uint32_t cell = 768; cell < 1024; ++cell) { store.load_q8_row(cell, 1000+cell, {1}, sources[cell]); }
    assert(backing_bytes(store) == slab_backing(p, 3, 1));
    verify_resident(store, p, sources, 1024, 1);
    std::puts("PASS failed Q8 and Q4 slab allocations change no logical state and retry");
}

// Bytes one buffer type needs for a full session: every ring and cold slab plus one migration step.
static size_t admission_need(llama_kv_recency_plan_params p, ggml_backend_buffer_type_t buft) {
    std::vector<llama_kv_recency_layer> layers;
    for (const auto & layer : p.layers) {
        if (layer.buft == buft) { layers.push_back(layer); }
    }
    p.layers = layers;
    const auto plan = llama_kv_recency_plan_create(p);
    const uint32_t q8 = (plan.ring_tokens + p.slab_tokens - 1)/p.slab_tokens;
    const uint32_t q4 = (plan.cold_tokens + p.slab_tokens - 1)/p.slab_tokens;
    const size_t workspace = plan.cold_tokens ? llama_kv_recency_slab_alloc_bytes(p, GGML_TYPE_F32, p.page_tokens) : 0;
    return slab_backing(p, q8, q4) + workspace;
}

static void test_admission(ggml_backend_dev_t device) {
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    ggml_backend_ptr host(ggml_backend_cpu_init());
    assert(backend && host);
    // Two buffer types: the device and host memory, or two host types on CPU runs.
    const bool cpu = ggml_backend_is_cpu(backend.get());
    const ggml_backend_buffer_type_t bufts[2] = {ggml_backend_dev_buffer_type(device), cpu ? failing_buffer_type() : ggml_backend_cpu_buffer_type()};
    llama_kv_recency_plan_params p;
    p.logical_ctx = 8292;
    p.migrate_tokens = 512;
    p.slab_tokens = 1024;
    p.layers = {{128, 64, bufts[0]}, {64, 128, bufts[1]}};
    assert(p.hot_tokens == 8192);
    const auto need = [&](uint32_t hot, ggml_backend_buffer_type_t buft) {
        auto trial = p;
        trial.hot_tokens = hot;
        return admission_need(trial, buft);
    };
    std::map<ggml_backend_buffer_type_t, size_t> budgets = {{bufts[0], need(2048, bufts[0])}, {bufts[1], need(2048, bufts[1])}};
    auto plan = llama_kv_recency_plan_admit(p, budgets);
    assert(plan.hot_tokens == 2048 && "admission must pick the largest hot window that fits");
    assert(need(2304, bufts[0]) > budgets[bufts[0]] || need(2304, bufts[1]) > budgets[bufts[1]]);
    assert(plan.migration_workspace_bytes == llama_kv_recency_slab_alloc_bytes(p, GGML_TYPE_F32, p.page_tokens));
    auto lopsided = budgets;
    lopsided[bufts[0]] *= 4;
    assert(llama_kv_recency_plan_admit(p, lopsided).hot_tokens == 2048 && "each buffer type must fit its own budget");
    auto smaller = budgets;
    --smaller[bufts[1]];
    assert(llama_kv_recency_plan_admit(p, smaller).hot_tokens == 1792);
    smaller[bufts[0]] = need(256, bufts[0]) - 1;
    bool rejected = false;
    try { llama_kv_recency_plan_admit(p, smaller); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected && "a context that cannot hold one hot page must be rejected, not shrunk");
    rejected = false;
    try { llama_kv_recency_plan_admit(p, {{bufts[0], SIZE_MAX}}); } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected && "every buffer type needs a budget");
    const std::map<ggml_backend_buffer_type_t, size_t> unlimited = {{bufts[0], SIZE_MAX/4}, {bufts[1], SIZE_MAX/4}};
    assert(llama_kv_recency_plan_admit(p, unlimited).hot_tokens == 8192);
    auto short_ctx = p;
    short_ctx.logical_ctx = 1000;
    const auto all_hot = llama_kv_recency_plan_admit(short_ctx, unlimited);
    assert(all_hot.hot_tokens == 1024 && all_hot.cold_tokens == 0 && all_hot.migration_workspace_bytes == 0);

    // A full session from the admitted plan never exceeds any budget.
    auto admitted = p;
    admitted.hot_tokens = plan.hot_tokens;
    llama_kv_recency_store store(admitted, llama_kv_recency_store::mode::committed_aging);
    const std::vector<ggml_backend_t> readers = {backend.get(), host.get()};
    std::vector<std::vector<llama_kv_recency_row>> sources(p.logical_ctx);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell); }
    const uint32_t ring_pages = plan.ring_tokens/p.page_tokens;
    std::map<ggml_backend_buffer_type_t, size_t> workspace;
    for (size_t layer = 0; layer < p.layers.size(); ++layer) {
        auto only = admitted;
        only.layers = {admitted.layers[layer]};
        workspace[only.layers[0].buft] = llama_kv_recency_slab_alloc_bytes(only, GGML_TYPE_F32, p.page_tokens);
    }
    size_t peak = 0;
    const auto check = [&]() {
        size_t total = 0;
        for (const auto & entry : store.memory_breakdown()) {
            assert(budgets.count(entry.first) && entry.second + workspace[entry.first] <= budgets[entry.first]);
            total += entry.second;
        }
        peak = std::max(peak, total);
    };
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) {
        if (cell % p.page_tokens == 0 && store.get_migration_stats().hot_pages == ring_pages) {
            assert(store.migrate_oldest_pages(readers, 100) > 0);
        }
        store.load_q8_row(cell, 1000+cell, {4}, sources[cell]);
        if (cell % p.page_tokens == 0) { check(); }
    }
    while (store.migrate_oldest_pages(readers, 100)) { check(); }
    check();
    assert(store.memory_breakdown().size() == 2 && workspace[bufts[0]] + workspace[bufts[1]] == plan.migration_workspace_bytes);
    assert(store.get_migration_stats().peak_workspace_bytes == plan.migration_workspace_bytes);
    verify_resident(store, p, sources, p.logical_ctx, 4);
    std::printf("PASS admission: hot=%u ring=%u, session peak backing=%zu within budgets %zu+%zu, workspace=%zu\n",
                plan.hot_tokens, plan.ring_tokens, peak, budgets[bufts[0]], budgets[bufts[1]], plan.migration_workspace_bytes);
}

static void test_append_truncate(ggml_backend_dev_t device, bool k_only = false) {
    llama_kv_recency_plan_params p;
    p.logical_ctx = 2048;
    p.hot_tokens = 256;
    p.migrate_tokens = 256;
    p.slab_tokens = 512;
    p.layers = {{128, 64, ggml_backend_dev_buffer_type(device)}, {64, 128, ggml_backend_dev_buffer_type(device)}};
    if (k_only) {
        // GLM-style MLA latents: one K row per token, V is a view of K
        p.layers = {{512, 0, ggml_backend_dev_buffer_type(device)}, {128, 0, ggml_backend_dev_buffer_type(device)}};
    }
    llama_kv_recency_store store(p, llama_kv_recency_store::mode::committed_aging);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    assert(backend);
    std::vector<std::vector<llama_kv_recency_row>> sources(p.logical_ctx);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell); }
    // Append like a decode loop: metadata first, then the graph writes the Q8 rows.
    const auto append = [&](uint32_t n) {
        const uint32_t first = store.get_n_cells();
        for (int attempt = 0;; ++attempt) {
            try {
                assert(store.append(1000 + first, 5, n) == first);
                break;
            } catch (const std::runtime_error &) {
                assert(attempt == 0 && store.get_cells().is_empty(first) && store.migrate_oldest_pages({backend.get()}, 100) > 0);
            }
        }
        for (uint32_t cell = first; cell < first + n; ++cell) {
            const auto dst = store.get_q8_row(cell);
            for (size_t layer = 0; layer < p.layers.size(); ++layer) {
                for (bool key : {true, false}) {
                    auto * tensor = store.get_tensor(layer, key, GGML_TYPE_Q8_0, dst.first);
                    if (!tensor) {
                        continue;
                    }
                    const auto & row = key ? sources[cell][layer].k : sources[cell][layer].v;
                    ggml_backend_tensor_set(tensor, row.data(), size_t(dst.second)*tensor->nb[1], row.size());
                }
            }
        }
    };
    for (uint32_t n : {100u, 156u, 1u, 255u, 256u, 200u, 200u, 132u, 200u}) { append(n); }
    assert(store.get_n_cells() == 1500 && store.get_migration_stats().cold_pages >= 2);
    verify_resident(store, p, sources, 1500, 5);
    const auto invalid = [&](auto fn) {
        try { fn(); } catch (const std::invalid_argument &) { return; }
        assert(false && "invalid append accepted");
    };
    invalid([&] { store.append(1000 + 1501, 5, 1); });
    invalid([&] { store.append(1000 + 1500, 4, 1); });
    invalid([&] { store.append(1000 + 1500, 5, 257); }); // more than one migration batch
    assert(store.get_n_cells() == 1500);
    assert(store.truncate(1400) && store.get_n_cells() == 1400);
    verify_resident(store, p, sources, 1400, 5);
    const auto cold = store.get_migration_stats().cold_pages;
    assert(!store.truncate(300) && store.get_n_cells() == 1400 && "truncation must not split a Q4 page");
    assert(store.truncate(512) && store.get_migration_stats().cold_pages == 2 && cold >= 2);
    assert(store.get_migration_stats().hot_pages == 0);
    verify_resident(store, p, sources, 512, 5);
    // Rewritten positions are new committed KV: they enter Q8 and age again.
    for (uint32_t cell = 512; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell + 7000); }
    for (uint32_t n : {256u, 256u, 256u, 256u}) { append(n); }
    while (store.migrate_oldest_pages({backend.get()}, 100)) {}
    assert(store.get_migration_stats().cold_pages == 5);
    verify_resident(store, p, sources, 1536, 5);
    assert(store.truncate(0) && store.get_n_cells() == 0 && store.get_cells().get_used() == 0);
    assert(store.get_migration_stats().cold_pages == 0 && store.get_migration_stats().hot_pages == 0);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell + 9000); }
    for (uint32_t n : {256u, 256u, 188u}) { append(n); }
    verify_resident(store, p, sources, 700, 5);
    if (k_only) {
        bool rejected = false;
        try { store.read_row(0, false, 0); } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected && store.get_tensor(0, false, GGML_TYPE_Q8_0, 0) == nullptr);
        const auto plan = llama_kv_recency_plan_create(p);
        assert(plan.q8_bytes_per_token == (512 + 128)/32*34 && plan.q4_bytes_per_token == (512 + 128)/32*18);
    }
    std::printf("PASS append, graph-style Q8 writes, tail truncation, cold-boundary rewrite and clear%s\n", k_only ? " (K-only)" : "");
}

static void test_committed_aging(ggml_backend_dev_t device) {
    llama_kv_recency_plan_params p;
    p.logical_ctx = 2048;
    p.hot_tokens = 256;
    p.migrate_tokens = 512;
    p.layers = {{128, 64, ggml_backend_dev_buffer_type(device)}, {64, 128, ggml_backend_dev_buffer_type(device)}};
    const auto plan = llama_kv_recency_plan_create(p);
    assert(plan.ring_tokens == 1024 && "Q8 ring holds the hot window, its crossing page and one migration batch");
    llama_kv_recency_store store(p, llama_kv_recency_store::mode::committed_aging);
    assert(backing_bytes(store) == 0 && "slabs are allocated on first use");
    std::vector<std::vector<llama_kv_recency_row>> sources(p.logical_ctx);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell); }
    const auto load = [&](uint32_t begin, uint32_t end) {
        for (uint32_t cell = begin; cell < end; ++cell) {
            store.load_q8_row(cell, 1000+cell, {2}, sources[cell]);
            assert(store.read_row(0, true, cell) == sources[cell][0].k);
        }
    };
    load(0, 1);
    assert(store.read_row(0, true, 0) == sources[0][0].k && "new committed KV must enter Q8 even in future cold positions");
    assert(backing_bytes(store) == slab_backing(p, 1, 0));
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    assert(backend);
    load(1, 511);
    assert(store.migrate_oldest_pages({backend.get()}, 2) == 0); // Page intersects protected hot tail.
    load(511, 512);
    if (ggml_backend_is_cpu(backend.get())) {
        unsigned nodes = 0;
        ggml_backend_cpu_set_abort_callback(backend.get(), [](void * pointer) {
            return ++*static_cast<unsigned *>(pointer) >= 3;
        }, &nodes);
        bool failed = false;
        try { store.migrate_oldest_pages({backend.get()}, 1); } catch (const std::runtime_error &) { failed = true; }
        ggml_backend_cpu_set_abort_callback(backend.get(), nullptr, nullptr);
        assert(failed && nodes == 3 && store.get_migration_stats().cold_pages == 0);
        assert(store.read_row(0, true, 0) == sources[0][0].k && store.read_row(1, false, 255) == sources[255][1].v);
        auto * unpublished = store.get_tensor(0, true, GGML_TYPE_Q4_0);
        std::vector<uint8_t> partial(ggml_row_size(GGML_TYPE_Q4_0, unpublished->ne[0]));
        ggml_backend_tensor_get(unpublished, partial.data(), 0, partial.size());
        assert(std::any_of(partial.begin(), partial.end(), [](uint8_t byte) { return byte != 0; }));
        std::puts("PASS conversion failure after Q4 write preserves Q8 mapping and supports retry");
    }
    assert(store.migrate_oldest_pages({backend.get()}, 1) == 1 && "oldest complete eligible Q8 page must migrate");
    assert(store.migrate_oldest_pages({backend.get()}, 2) == 0);
    assert(backing_bytes(store) == slab_backing(p, 1, 1));
    load(512, 1280);
    assert(store.get_migration_stats().hot_pages == 4);
    const auto before = store.get_migration_stats();
    bool full = false;
    try { store.load_q8_row(1280, 2280, {2}, sources[1280]); } catch (const std::runtime_error &) { full = true; }
    assert(full && store.get_cells().is_empty(1280) && "a full ring must reject the next page, never overwrite a slot");
    verify_resident(store, p, sources, 1280, 2);
    const auto invalid = [&](auto fn) {
        try { fn(); } catch (const std::invalid_argument &) { return; } catch (const std::out_of_range &) { return; }
        assert(false && "invalid committed aging operation accepted");
    };
    invalid([&] { store.migrate_oldest_pages({}, 1); });
    invalid([&] { store.migrate_oldest_pages({nullptr}, 1); });
    assert(store.get_migration_stats().cold_pages == before.cold_pages);
    assert(store.migrate_oldest_pages({backend.get()}, 0) == 0);
    assert(store.migrate_oldest_pages({backend.get()}, 100) == 2); // Session batch limit, not caller request.
    assert(store.get_migration_stats().cold_pages == 3);
    load(1280, 1792);
    assert(store.migrate_oldest_pages({backend.get()}, 100) == 2);
    load(1792, 2048);
    assert(store.migrate_oldest_pages({backend.get()}, 100) == 2);
    assert(store.migrate_oldest_pages({backend.get()}, 100) == 0);
    const auto stats = store.get_migration_stats();
    assert(stats.cold_pages == 7 && stats.hot_pages == 1 && stats.migrated_source_bytes == size_t(1792)*plan.q8_bytes_per_token);
    assert(stats.resident_payload_bytes == size_t(1792)*plan.q4_bytes_per_token + size_t(256)*plan.q8_bytes_per_token);
    assert(stats.peak_workspace_bytes > 0 && stats.peak_workspace_bytes <= plan.migration_reserve_bytes);
    verify_resident(store, p, sources, p.logical_ctx, 2);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) {
        for (size_t layer = 0; layer < p.layers.size(); ++layer) {
            for (bool key : {true, false}) {
                const auto & source = key ? sources[cell][layer].k : sources[cell][layer].v;
                const auto width = key ? p.layers[layer].k_width : p.layers[layer].v_width;
                assert(store.read_row(layer, key, cell) == (cell >= 1792 ? source : requantize_q4(source, width)));
            }
        }
    }
    assert(backing_bytes(store) == plan.committed_alloc_bytes);
    invalid([&] { store.load_q8_row(0, 1000, {2}, sources[0]); });
    llama_kv_recency_store rejecting(p, llama_kv_recency_store::mode::committed_aging);
    const auto & rows = sources[0];
    invalid([&] { rejecting.load_q8_row(1, 1000, {2}, rows); });
    invalid([&] { rejecting.load_q8_row(0, 1000, {0, 2}, rows); });
    auto bad_rows = rows;
    bad_rows.back().v.pop_back();
    invalid([&] { rejecting.load_q8_row(0, 1000, {2}, bad_rows); });
    assert(rejecting.get_cells().get_used() == 0);
    rejecting.load_q8_row(0, 1000, {2}, rows);
    invalid([&] { rejecting.load_q8_row(1, 1002, {2}, rows); });
    invalid([&] { rejecting.load_q8_row(1, 1001, {0}, rows); });
    assert(rejecting.get_cells().get_used() == 1);
    std::printf("PASS bounded committed aging: pages=%u, resident=%zu, peak workspace=%zu, backing=%zu\n",
                stats.cold_pages, stats.resident_payload_bytes, stats.peak_workspace_bytes, backing_bytes(store));
}

static void test_hot_ring(ggml_backend_dev_t device, bool k_only = false) {
    llama_kv_recency_plan_params p;
    p.logical_ctx = 8292; // Final page is partial.
    p.hot_tokens = 512;
    p.migrate_tokens = 512;
    p.slab_tokens = 1024;
    p.layers = {{128, 64, ggml_backend_dev_buffer_type(device)}, {64, 128, ggml_backend_dev_buffer_type(device)}};
    if (k_only) {
        p.layers = {{512, 0, ggml_backend_dev_buffer_type(device)}, {64, 0, ggml_backend_dev_buffer_type(device)}};
    }
    const auto plan = llama_kv_recency_plan_create(p);
    const uint32_t page = plan.page_tokens;
    const uint32_t slab_pages = p.slab_tokens/page;
    const uint32_t ring_pages = plan.ring_tokens/page;
    assert(plan.physical_tokens == 8448 && plan.cold_tokens == 7936 && plan.ring_tokens == 1280);
    auto all_q8 = p;
    all_q8.hot_tokens = p.logical_ctx;
    const size_t phase5_backing = llama_kv_recency_plan_create(all_q8).hot_alloc_bytes + plan.cold_alloc_bytes;
    llama_kv_recency_store store(p, llama_kv_recency_store::mode::committed_aging);
    assert(backing_bytes(store) == 0 && store.get_slab_count(GGML_TYPE_Q8_0) == 2 && store.get_slab_count(GGML_TYPE_Q4_0) == 8);
    assert(plan.committed_alloc_bytes == slab_backing(p, 2, 8) && plan.committed_alloc_bytes < phase5_backing);
    uint32_t max_slot = 0;
    const auto expected_backing = [&](uint32_t q4_pages) {
        return slab_backing(p, max_slot/slab_pages + 1, (q4_pages + slab_pages - 1)/slab_pages);
    };
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    ggml_backend_ptr host(ggml_backend_cpu_init()); // Extra reader: synchronized, but owns no KV on device runs.
    assert(backend && host);
    const std::vector<ggml_backend_t> readers = {backend.get(), host.get()};
    std::vector<std::vector<llama_kv_recency_row>> sources(p.logical_ctx);
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) { sources[cell] = make_rows(p, cell); }
    std::vector<uint32_t> slot_uses(ring_pages);
    uint32_t calls = 0;
    bool aborted = !ggml_backend_is_cpu(backend.get());
    for (uint32_t cell = 0; cell < p.logical_ctx; ++cell) {
        if (cell % page == 0 && store.get_migration_stats().hot_pages == ring_pages) {
            // Delayed migration: run only when the next page has no free slot.
            const auto before = store.get_migration_stats();
            bool rejected = false;
            try { store.load_q8_row(cell, 1000+cell, {3}, sources[cell]); } catch (const std::runtime_error &) { rejected = true; }
            assert(rejected && store.get_cells().is_empty(cell) && store.get_migration_stats().hot_pages == ring_pages);
            if (!aborted && cell >= 2*plan.ring_tokens) {
                unsigned nodes = 0;
                ggml_backend_cpu_set_abort_callback(backend.get(), [](void * pointer) {
                    return ++*static_cast<unsigned *>(pointer) >= 3;
                }, &nodes);
                bool failed = false;
                try { store.migrate_oldest_pages(readers, 1); } catch (const std::runtime_error &) { failed = true; }
                ggml_backend_cpu_set_abort_callback(backend.get(), nullptr, nullptr);
                assert(failed && store.get_migration_stats().cold_pages == before.cold_pages);
                assert(store.get_migration_stats().hot_pages == ring_pages);
                // The destination slab is allocated before conversion and kept for the retry.
                assert(backing_bytes(store) == expected_backing(before.cold_pages + 1));
                verify_resident(store, p, sources, cell, 3);
                aborted = true;
            }
            const uint32_t limit = ++calls % 2 ? 1 : 100;
            const uint32_t done = store.migrate_oldest_pages(readers, limit);
            assert(done == std::min(limit, plan.migrate_tokens/page) && "a full ring always holds a migratable batch");
            const auto after = store.get_migration_stats();
            assert(after.cold_pages == before.cold_pages + done && after.hot_pages == ring_pages - done);
            assert(after.cold_pages*page + plan.hot_tokens <= cell && "pages in the newest hot window stay Q8");
            if (calls % 4 == 0) { verify_resident(store, p, sources, cell, 3); }
            assert(backing_bytes(store) == expected_backing(after.cold_pages));
        }
        store.load_q8_row(cell, 1000+cell, {3}, sources[cell]);
        if (cell % page == 0) {
            uint32_t slot = 0;
            while (store.get_row_cell(GGML_TYPE_Q8_0, slot/slab_pages, slot%slab_pages*page) != int32_t(cell)) { assert(++slot < ring_pages); }
            ++slot_uses[slot];
            max_slot = std::max(max_slot, slot);
            assert(backing_bytes(store) == expected_backing(store.get_migration_stats().cold_pages));
        }
    }
    assert(aborted && calls > 2*ring_pages);
    while (store.migrate_oldest_pages(readers, 100)) {}
    const auto stats = store.get_migration_stats();
    assert(stats.cold_pages == 30 && stats.hot_pages == 3 && "final partial page and its hot predecessors stay Q8");
    verify_resident(store, p, sources, p.logical_ctx, 3);
    uint32_t pages = 0;
    for (const auto uses : slot_uses) { assert(uses >= 5 && "every ring slot must be reused across wraps"); pages += uses; }
    const size_t backing = backing_bytes(store);
    assert(pages == plan.physical_tokens/page && backing == plan.committed_alloc_bytes);
    std::printf("PASS hot ring: %u pages through %u slots, %u migration calls, backing=%zu vs full-Q8 Phase5 backing=%zu, resident payload=%zu\n",
                pages, ring_pages, calls, backing, phase5_backing, stats.resident_payload_bytes);
}

static void test_fixed_tiers(ggml_backend_buffer_type_t buft) {
    llama_kv_recency_plan_params p;
    p.logical_ctx = 512;
    p.hot_tokens = 256;
    p.layers = {{128, 64, buft}, {64, 128, buft}};
    llama_kv_recency_store store(p);
    std::vector<llama_kv_recency_row> rows;
    std::vector<llama_kv_recency_row> cold;
    for (const auto & layer : p.layers) {
        llama_kv_recency_row row, expected;
        for (bool key : {true, false}) {
            const size_t width = key ? layer.k_width : layer.v_width;
            std::vector<float> values(width);
            for (size_t i = 0; i < width; ++i) {
                values[i] = std::sin(float(i)*0.37f)*3.0f + float(i % 7)*0.1f;
            }
            auto & packed = key ? row.k : row.v;
            packed.resize(ggml_row_size(GGML_TYPE_Q8_0, width));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), packed.data(), 0, 1, width, nullptr);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(packed.data(), values.data(), width);
            auto & q4 = key ? expected.k : expected.v;
            q4.resize(ggml_row_size(GGML_TYPE_Q4_0, width));
            ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), q4.data(), 0, 1, width, nullptr);
        }
        rows.push_back(row);
        cold.push_back(expected);
    }
    // Absolute positions differ from cell indices on both sides of the physical boundary.
    store.load_q8_row(255, 1000, {0, 2}, rows);
    store.load_q8_row(256, 1001, {0}, rows);
    store.load_q8_row(511, 1002, {2}, rows);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        assert(store.read_row(layer, true, 255) == cold[layer].k);
        assert(store.read_row(layer, false, 255) == cold[layer].v);
        assert(store.read_row(layer, true, 256) == rows[layer].k);
        assert(store.read_row(layer, false, 511) == rows[layer].v);
        auto * k4 = store.get_tensor(layer, true, GGML_TYPE_Q4_0);
        auto * k8 = store.get_tensor(layer, true, GGML_TYPE_Q8_0);
        assert(k4->ne[1] == 256 && k8->ne[1] == 256);
        assert(k4->type == GGML_TYPE_Q4_0 && k8->type == GGML_TYPE_Q8_0);
    }
    const auto & cells = store.get_cells();
    assert(cells.get_used() == 3 && cells.pos_get(255) == 1000 && cells.pos_get(256) == 1001);
    assert(cells.seq_has(255, 2) && !cells.seq_has(256, 2));
    assert(cells.seq_pos_min(0) == 1000 && cells.seq_pos_max(0) == 1001);
    assert(cells.pos_get(511) > 1001); // Causal mask must exclude this row for query position1001.
    size_t bytes = 0;
    for (const auto & entry : store.memory_breakdown()) {
        assert(entry.first == buft);
        bytes += entry.second;
    }
    assert(bytes == llama_kv_recency_plan_create(p).hot_alloc_bytes + llama_kv_recency_plan_create(p).cold_alloc_bytes);
    std::printf("fixed-tier backing bytes = %zu, all-Q8 payload bytes = %zu\n", bytes, size_t(512)*llama_kv_recency_plan_create(p).q8_bytes_per_token);
    const auto invalid = [&](auto fn) {
        try { fn(); } catch (const std::invalid_argument &) { return; } catch (const std::out_of_range &) { return; }
        assert(false && "invalid fixed-tier row accepted");
    };
    auto bad_rows = rows;
    bad_rows[1].v.pop_back();
    invalid([&] { store.load_q8_row(0, 3, {0}, bad_rows); });
    invalid([&] { store.load_q8_row(0, 3, {LLAMA_MAX_SEQ}, rows); });
    invalid([&] { store.load_q8_row(0, -1, {0}, rows); });
    invalid([&] { store.load_q8_row(0, 3, {}, rows); });
    invalid([&] { store.load_q8_row(512, 3, {0}, rows); });
    invalid([&] { store.load_q8_row(255, 1000, {0}, rows); });
    invalid([&] { store.read_row(0, true, 0); });
    invalid([&] { store.read_row(2, true, 255); });
    invalid([&] { store.get_tensor(0, true, GGML_TYPE_F16); });
    assert(cells.is_empty(0) && cells.get_used() == 3);
    assert(store.read_row(0, true, 255) == cold[0].k);
    p.streams = 2;
    invalid([&] { llama_kv_recency_store unsupported(p); });
    p.streams = 1;
    p.logical_ctx = 100;
    llama_kv_recency_store all_hot(p);
    assert(all_hot.get_tensor(0, true, GGML_TYPE_Q4_0) == nullptr);
    all_hot.load_q8_row(99, 900, {1}, rows);
    assert(all_hot.read_row(0, true, 99) == rows[0].k);
    invalid([&] { all_hot.load_q8_row(100, 901, {1}, rows); });
    std::puts("PASS fixed Q4/Q8 arenas, packed rows, absolute positions, ownership and invalid-row publication");
}

static void test_merge_empty_and_imbalanced(ggml_backend_dev_t device = nullptr) {
    ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    ggml_backend_ptr backend(device ? ggml_backend_dev_init(device, nullptr) : ggml_backend_cpu_init());
    assert(backend);
    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 5, 2);
    auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 5, 2);
    auto * empty = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 5, 2);
    auto * state = ggml_attn_merge(ctx.get(), a, b, false);
    auto * out = ggml_attn_merge(ctx.get(), state, empty, true);
    auto * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    const float av[10] = {2, 4, 6, 1000, 2, NAN, NAN, NAN, -INFINITY, 0};
    const float bv[10] = {9, 12, 15, -1000, 3, NAN, NAN, NAN, -INFINITY, 0};
    const float ev[10] = {NAN, NAN, NAN, -INFINITY, 0, NAN, NAN, NAN, -INFINITY, 0};
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
    ggml_backend_tensor_set(empty, ev, 0, sizeof(ev));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) { assert(ggml_backend_supports_op(backend.get(), ggml_graph_node(gf, i))); }
    assert(ggml_backend_graph_compute(backend.get(), gf) == GGML_STATUS_SUCCESS);
    float actual[6];
    ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
    const float want[6] = {1, 2, 3, 0, 0, 0};
    for (int i = 0; i < 6; ++i) { assert(actual[i] == want[i]); }
    std::puts("PASS three-range merge, extreme imbalance and empty states");
}

static void test_partial_attention(int nq, bool extreme, bool all_masked, float softcap, bool with_sink = false,
                                   bool quantized = true, int empty_region = -1, float max_bias = 0, ggml_backend_dev_t device = nullptr,
                                   bool store_backed = false, bool aged_store = false, bool slabbed = false) {
    const int width = device ? 128 : 64;
    constexpr int heads = 2;
    // The aged store wraps a three-page Q8 ring, so physical page order differs from logical order and has gaps.
    const int nk = aged_store ? 768 : (device || store_backed) ? 256 : (quantized ? 32 : 512);
    const int n_cells = aged_store ? 1024 : 2*nk;
    const int q0 = aged_store ? n_cells - 32 : nk + 8;
    const ggml_type cold_type = quantized ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
    const ggml_type hot_type = quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
    ggml_context_ptr ctx(ggml_init({128*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    ggml_backend_ptr backend(device ? ggml_backend_dev_init(device, nullptr) : ggml_backend_cpu_init());
    assert(backend);
    if (!device) { ggml_backend_cpu_set_n_threads(backend.get(), 4); }
    auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, nq, heads, 1);
    const auto make_value = [&](bool key, int region, int t, int d) {
        return key ? float((d % 5) - 2)*(extreme ? 50.0f : 0.2f) + float(region*3+t)*0.07f :
                     std::sin(float(d+t)*0.1f) + float(region)*4.0f;
    };
    // Independent per-cell Q8 sources and Q4 requantizations; the store decides physical placement.
    std::unique_ptr<llama_kv_recency_store> store;
    std::vector<llama_kv_recency_row> sources(n_cells), cold(n_cells);
    if (store_backed) {
        llama_kv_recency_plan_params p;
        p.logical_ctx = n_cells;
        p.hot_tokens = aged_store ? 256 : nk;
        p.migrate_tokens = aged_store ? 256 : p.migrate_tokens;
        p.slab_tokens = slabbed ? 256 : p.slab_tokens;
        p.layers = {{uint64_t(width), uint64_t(width), ggml_backend_dev_buffer_type(ggml_backend_get_device(backend.get()))}};
        store.reset(new llama_kv_recency_store(p, aged_store ? llama_kv_recency_store::mode::committed_aging :
                                                             llama_kv_recency_store::mode::fixed_snapshot));
        for (int cell = 0; cell < n_cells; ++cell) {
            for (bool key : {true, false}) {
                std::vector<float> floats(width);
                for (int d = 0; d < width; ++d) { floats[d] = make_value(key, cell/nk, cell%nk, d); }
                auto & q8 = key ? sources[cell].k : sources[cell].v;
                q8.resize(ggml_row_size(GGML_TYPE_Q8_0, width));
                ggml_quantize_chunk(GGML_TYPE_Q8_0, floats.data(), q8.data(), 0, 1, width, nullptr);
                (key ? cold[cell].k : cold[cell].v) = requantize_q4(q8, width);
            }
        }
        for (int cell = 0; cell < n_cells; ++cell) {
            if (aged_store && cell == 768) { assert(store->migrate_oldest_pages({backend.get()}, 1) == 1); }
            store->load_q8_row(cell, 1000+cell, {aged_store ? 0 : ((cell%nk)%3 == 0 ? 2 : 0)}, {sources[cell]});
        }
        if (aged_store) { assert(store->migrate_oldest_pages({backend.get()}, 1) == 1); }
    }
    struct region_t {
        ggml_type type;
        uint32_t slab;
        ggml_tensor * k;
        ggml_tensor * v;
        ggml_tensor * mask = nullptr;
        ggml_tensor * state = nullptr;
        std::vector<float> keys, values, key_scales;
        std::vector<int> row_cell;
        std::vector<ggml_fp16_t> masks;
    };
    std::vector<region_t> regions;
    for (ggml_type type : {cold_type, hot_type}) {
        if (!store) {
            region_t region;
            region.type = type;
            region.slab = 0;
            region.k = ggml_new_tensor_4d(ctx.get(), type, width, nk, 1, 1);
            region.v = ggml_new_tensor_4d(ctx.get(), type, width, nk, 1, 1);
            regions.push_back(std::move(region));
            continue;
        }
        // One homogeneous region per allocated slab.
        for (uint32_t slab = 0; slab < store->get_slab_count(type); ++slab) {
            if (store->get_tensor(0, true, type, slab)) {
                region_t region;
                region.type = type;
                region.slab = slab;
                region.k = store->get_tensor(0, true, type, slab);
                region.v = store->get_tensor(0, false, type, slab);
                regions.push_back(std::move(region));
            }
        }
    }
    assert(regions.size() == (slabbed ? 5u : 2u));
    auto * sink = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, heads);
    for (auto & region : regions) {
        region.mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, region.k->ne[1], nq, 1, 1);
        region.state = ggml_flash_attn_ext_partial(ctx.get(), q, region.k, region.v, region.mask, 0.125f, max_bias, softcap);
    }
    if (with_sink) { ggml_flash_attn_ext_add_sinks(regions[0].state, sink); }
    assert(ggml_prec_set_acc(regions[0].state, GGML_PREC_F32));
    auto * out = regions[0].state;
    for (size_t i = 1; i < regions.size(); ++i) { out = ggml_attn_merge(ctx.get(), out, regions[i].state, i + 1 == regions.size()); }
    auto * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    assert(buffer);
    const float sink_values[2] = {50.0f, -2.0f};
    ggml_backend_tensor_set(sink, sink_values, 0, sizeof(sink_values));
    std::vector<float> queries(width*nq*heads);
    for (size_t i = 0; i < queries.size(); ++i) { queries[i] = float(int(i % 7) - 3)*(extreme ? 10.0f : 0.05f); }
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size()*sizeof(float));
    const bool gpu = device && (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                                ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU);
    std::vector<int> seen(n_cells);
    for (size_t r = 0; r < regions.size(); ++r) {
        auto & region = regions[r];
        const int rows = int(region.k->ne[1]);
        const size_t row_bytes = ggml_row_size(region.type, width);
        region.row_cell.assign(rows, -1);
        region.key_scales.assign(rows*(width/32), 0.0f);
        for (bool key : {true, false}) {
            auto & floats = key ? region.keys : region.values;
            floats.assign(width*rows, 0.0f);
            std::vector<uint8_t> packed(row_bytes*rows);
            for (int t = 0; t < rows; ++t) {
                const int cell = region.row_cell[t] = store ? store->get_row_cell(region.type, region.slab, t) : int(r)*nk + t;
                if (cell < 0) { continue; }
                uint8_t * row = packed.data() + t*row_bytes;
                if (store) {
                    const auto & bytes = region.type == GGML_TYPE_Q4_0 ? (key ? cold[cell].k : cold[cell].v) : (key ? sources[cell].k : sources[cell].v);
                    std::copy(bytes.begin(), bytes.end(), row);
                    ggml_get_type_traits(region.type)->to_float(row, floats.data()+t*width, width);
                } else {
                    for (int d = 0; d < width; ++d) { floats[t*width+d] = make_value(key, int(r), t, d); }
                    ggml_quantize_chunk(region.type, floats.data()+t*width, row, 0, 1, width, nullptr);
                    if (quantized) { ggml_get_type_traits(region.type)->to_float(row, floats.data()+t*width, width); }
                }
                for (int block = 0; key && region.type == GGML_TYPE_Q4_0 && block < width/32; ++block) {
                    ggml_fp16_t scale;
                    std::memcpy(&scale, row + block*ggml_type_size(region.type), sizeof(scale));
                    region.key_scales[t*(width/32)+block] = ggml_fp16_to_fp32(scale);
                }
                seen[cell] += key;
            }
            if (!store) { ggml_backend_tensor_set(key ? region.k : region.v, packed.data(), 0, packed.size()); }
        }
    }
    for (int cell = 0; cell < n_cells; ++cell) { assert(seen[cell] == 1 && "every logical cell must be resident exactly once"); }
    if (aged_store) {
        // Newest page reuses slot 0, slot 1 is free, slot 2 holds the older hot page and the Q4 tail is unpublished.
        const auto cell_at = [&](ggml_type type, int row) {
            for (const auto & region : regions) {
                const int base = int(region.slab)*(slabbed ? 256 : 768);
                if (region.type == type && row >= base && row < base + int(region.row_cell.size())) { return region.row_cell[row - base]; }
            }
            return -2;
        };
        assert(cell_at(GGML_TYPE_Q8_0, 0) == 768 && cell_at(GGML_TYPE_Q8_0, 256) == -1 && cell_at(GGML_TYPE_Q8_0, 512) == 512);
        assert(cell_at(GGML_TYPE_Q4_0, 256) == 256 && cell_at(GGML_TYPE_Q4_0, 512) == (slabbed ? -2 : -1));
    }
    for (size_t r = 0; r < regions.size(); ++r) {
        auto & region = regions[r];
        const int rows = int(region.row_cell.size());
        region.masks.resize(rows*nq);
        for (int iq = 0; iq < nq; ++iq) {
            for (int t = 0; t < rows; ++t) {
                // Sequence ownership and absolute-position causal visibility of the cell stored in this physical row.
                const int cell = region.row_cell[t];
                const int position = cell < 0 ? 0 : store ? store->get_cells().pos_get(cell) : 1000 + cell;
                const bool owned = cell >= 0 && (store ? store->get_cells().seq_has(cell, 0) : t % 3 != 0);
                const bool visible = !all_masked && int(r) != empty_region && owned && position <= 1000 + q0 + iq;
                const float bias = max_bias ? -0.1f*float(1000 + q0 + iq - position) : 0.0f;
                region.masks[iq*rows+t] = ggml_fp32_to_fp16(visible ? bias : -INFINITY);
            }
        }
        ggml_backend_tensor_set(region.mask, region.masks.data(), 0, region.masks.size()*sizeof(ggml_fp16_t));
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend.get(), ggml_graph_node(gf, i))) {
            std::fprintf(stderr, "FAIL required device op %s on %s\n", ggml_op_name(ggml_graph_node(gf, i)->op), ggml_backend_name(backend.get()));
            assert(false);
        }
    }
    assert(ggml_backend_graph_compute(backend.get(), gf) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(width*nq*heads);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
    for (int ih = 0; ih < heads; ++ih) {
        for (int iq = 0; iq < nq; ++iq) {
            std::vector<uint8_t> qq(ggml_row_size(GGML_TYPE_Q8_0, width));
            std::vector<float> qf(width);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, queries.data()+(ih*nq+iq)*width, qq.data(), 0, 1, width, nullptr);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(qq.data(), qf.data(), width);
            if (!quantized) { std::copy_n(queries.data()+(ih*nq+iq)*width, width, qf.data()); }
            if (gpu) {
                // ROCm uses Q8_1 queries: float scale and the sum of original scaled values.
                const float scale = softcap ? 0.125f/softcap : 0.125f;
                for (int block = 0; block < width; block += 32) {
                    float amax = 0;
                    for (int d = 0; d < 32; ++d) { amax = std::max(amax, std::abs(scale*queries[(ih*nq+iq)*width+block+d])); }
                    const float step = amax/127;
                    for (int d = 0; d < 32; ++d) {
                        const float value = scale*queries[(ih*nq+iq)*width+block+d];
                        qf[block+d] = step ? std::round(value/step)*step/scale : 0;
                    }
                }
            }
            std::vector<std::vector<double>> scores(regions.size());
            double m = with_sink ? sink_values[ih] : -INFINITY;
            for (size_t r = 0; r < regions.size(); ++r) {
                const auto & region = regions[r];
                const int rows = int(region.row_cell.size());
                scores[r].assign(rows, -INFINITY);
                for (int t = 0; t < rows; ++t) {
                    if (!std::isfinite(ggml_fp16_to_fp32(region.masks[iq*rows+t]))) { continue; }
                    double score = 0;
                    for (int d = 0; d < width; ++d) { score += double(qf[d])*region.keys[t*width+d]; }
                    if (gpu && region.type == GGML_TYPE_Q4_0) {
                        for (int block = 0; block < width/32; ++block) {
                            double correction = 0;
                            for (int d = 0; d < 32; ++d) {
                                correction += qf[block*32+d] - queries[(ih*nq+iq)*width+block*32+d];
                            }
                            score += 8*region.key_scales[t*(width/32)+block]*correction;
                        }
                    }
                    score *= 0.125;
                    if (softcap) { score = softcap*std::tanh(score/softcap); }
                    const double slope = max_bias ? std::pow(std::pow(2.0, -max_bias/heads), ih+1) : 1.0;
                    score += slope*ggml_fp16_to_fp32(region.masks[iq*rows+t]);
                    scores[r][t] = score;
                    m = std::max(m, score);
                }
            }
            double sum = with_sink ? std::exp(sink_values[ih] - m) : 0;
            std::vector<double> expected(width, 0);
            if (std::isfinite(m)) {
                for (size_t r = 0; r < regions.size(); ++r) {
                    for (size_t t = 0; t < scores[r].size(); ++t) {
                        if (!std::isfinite(scores[r][t])) { continue; }
                        const double weight = std::exp(scores[r][t] - m);
                        sum += weight;
                        for (int d = 0; d < width; ++d) { expected[d] += weight*regions[r].values[t*width+d]; }
                    }
                }
            }
            for (int d = 0; d < width; ++d) {
                const float want = sum ? float(expected[d]/sum) : 0.0f;
                const float got = actual[(iq*heads+ih)*width+d];
                if (!std::isfinite(got) || std::abs(got-want) > 0.0005f) {
                    std::fprintf(stderr, "FAIL partial merge nq=%d extreme=%d masked=%d softcap=%g q=%d head=%d d=%d got=%g want=%g\n",
                                 nq, extreme, all_masked, softcap, iq, ih, d, got, want);
                    assert(false);
                }
            }
        }
    }
    assert(regions[0].state->ne[0] == width+2 && out->ne[0] == width);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    if (argc == 3 && strcmp(argv[1], "--device") == 0) {
        auto * device = ggml_backend_dev_by_name(argv[2]);
        assert(device && "requested backend device unavailable");
        test_committed_aging(device);
        test_hot_ring(device);
        test_admission(device);
        test_append_truncate(device);
        test_append_truncate(device, true);
        test_hot_ring(device, true);
        for (int nq : {1, 8, 64}) {
            test_partial_attention(nq, false, false, 0, false, true, -1, 0, device);
            test_partial_attention(nq, true, false, 30, true, true, -1, 8, device);
            test_partial_attention(nq, false, true, 0, false, true, -1, 0, device);
            test_partial_attention(nq, false, true, 0, true, true, -1, 0, device);
            test_partial_attention(nq, false, false, 0, false, true, 0, 0, device);
            test_partial_attention(nq, false, false, 0, false, true, 1, 0, device);
        }
        for (int nq : {1, 64}) { test_partial_attention(nq, false, false, 30, true, true, -1, 8, device, true); }
        for (int nq : {1, 64}) { test_partial_attention(nq, false, false, 30, true, true, -1, 8, device, true, true); }
        for (int nq : {1, 64}) { test_partial_attention(nq, false, false, 30, true, true, -1, 8, device, true, true, true); }
        std::puts("PASS store-backed attention over logical positions and sequence masks");
        test_merge_empty_and_imbalanced(device);
        test_fixed_tiers(ggml_backend_dev_buffer_type(device));
        std::puts("PASS required-device combined-sequence oracle (18 cases), state merge and fixed tiers");
        return 0;
    }
    assert(argc == 1 && "usage: test-kv-recency-plan [--device NAME]");
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(dev);
    auto * buft = ggml_backend_dev_buffer_type(dev);
    test_committed_aging(dev);
    test_hot_ring(dev);
    test_slab_allocation_failure();
    test_admission(dev);
    test_append_truncate(dev);
    test_append_truncate(dev, true);
    test_hot_ring(dev, true);
    try { test_fixed_tiers(buft); } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL fixed tiers: %s\n", e.what()); return 1;
    }
    test_merge_empty_and_imbalanced();
    test_partial_attention(1, false, false, 0, false, true, -1, 0, nullptr, true);
    test_partial_attention(64, false, false, 30, true, true, -1, 8, nullptr, true);
    test_partial_attention(1, false, false, 30, true, true, -1, 8, nullptr, true, true);
    test_partial_attention(64, false, false, 30, true, true, -1, 8, nullptr, true, true);
    test_partial_attention(1, false, false, 30, true, true, -1, 8, nullptr, true, true, true);
    test_partial_attention(64, false, false, 30, true, true, -1, 8, nullptr, true, true, true);
    test_partial_attention(1, false, false, 0);
    test_partial_attention(8, false, false, 0);
    test_partial_attention(1, true, false, 0);
    test_partial_attention(8, true, false, 30);
    test_partial_attention(8, false, true, 0);
    test_partial_attention(8, false, false, 0, true);
    test_partial_attention(8, false, true, 0, true);
    test_partial_attention(1, false, false, 0, false, false);
    test_partial_attention(8, false, false, 0, true, false);
    test_partial_attention(64, false, false, 0, true, false);
    test_partial_attention(64, false, false, 30, true, false, -1, 8);
    test_partial_attention(1, false, false, 0, false, true, 0);
    test_partial_attention(8, false, false, 0, false, true, 1);
    std::puts("PASS partial attention combined-sequence oracle");
    llama_kv_recency_plan_params params;
    params.logical_ctx = 65536;
    params.layers.assign(32, {1024, 1024, buft});
    auto plan = llama_kv_recency_plan_create(params);
    assert(plan.q8_bytes_per_token == 68*1024);
    assert(plan.q4_bytes_per_token == 36*1024);
    assert(plan.logical_ctx == 65536 && plan.physical_tokens == 65536);
    assert(plan.hot_tokens == 8192 && plan.cold_tokens == 57344);
    assert(plan.hot_bytes == size_t(544)*1024*1024);
    assert(plan.cold_bytes == size_t(2016)*1024*1024);
    assert(plan.hot_alloc_bytes >= plan.hot_bytes && plan.cold_alloc_bytes >= plan.cold_bytes);
    assert(plan.migration_reserve_bytes >= size_t(292)*1024*1024);
    assert(plan.ring_tokens == 8192 + 256 + 1024 && plan.slab_tokens == 4096);
    assert(plan.committed_alloc_bytes >= size_t(plan.ring_tokens)*plan.q8_bytes_per_token + plan.cold_bytes);
    assert(plan.committed_alloc_bytes == 2*llama_kv_recency_slab_alloc_bytes(params, GGML_TYPE_Q8_0, 4096) +
           llama_kv_recency_slab_alloc_bytes(params, GGML_TYPE_Q8_0, 1280) + 14*llama_kv_recency_slab_alloc_bytes(params, GGML_TYPE_Q4_0, 4096));

    params.streams = 3;
    auto parallel = llama_kv_recency_plan_create(params);
    assert(parallel.hot_bytes == 3*plan.hot_bytes && parallel.cold_bytes == 3*plan.cold_bytes);
    params.streams = 1;
    params.logical_ctx = 513;
    plan = llama_kv_recency_plan_create(params);
    assert(plan.logical_ctx == 513 && plan.physical_tokens == 768);
    assert(plan.hot_tokens == 768 && plan.cold_tokens == 0 && plan.cold_alloc_bytes == 0);
    assert(plan.ring_tokens == 768 && plan.committed_alloc_bytes == plan.hot_alloc_bytes);
    params.logical_ctx = 65537;
    plan = llama_kv_recency_plan_create(params);
    assert(plan.physical_tokens == 65792 && plan.hot_tokens + plan.cold_tokens == plan.physical_tokens);

    const auto invalid = [](const llama_kv_recency_plan_params & p) {
        try {
            llama_kv_recency_plan_create(p);
        } catch (const std::invalid_argument &) {
            return;
        } catch (const std::overflow_error &) {
            return;
        }
        assert(false && "invalid or overflowing KV geometry accepted");
    };
    auto bad = params; bad.logical_ctx = 0; invalid(bad);
    bad = params; bad.page_tokens = 0; invalid(bad);
    bad = params; bad.page_tokens = 255; invalid(bad);
    bad = params; bad.migrate_tokens = 257; invalid(bad);
    bad = params; bad.hot_tokens = 0; invalid(bad);
    bad = params; bad.slab_tokens = 0; invalid(bad);
    bad = params; bad.slab_tokens = 384; invalid(bad);
    bad = params; bad.streams = 0; invalid(bad);
    bad = params; bad.layers.clear(); invalid(bad);
    bad = params; bad.layers[0].k_width = 31; invalid(bad);
    bad = params; bad.layers[0].k_width = 0; invalid(bad);
    auto k_only = params;
    k_only.layers[0].v_width = 0;
    assert(llama_kv_recency_plan_create(k_only).q8_bytes_per_token == plan.q8_bytes_per_token - 1024/32*34);
    bad = params; bad.layers[0].buft = nullptr; invalid(bad);
    bad = params; bad.logical_ctx = std::numeric_limits<uint32_t>::max(); invalid(bad);
    bad = params; bad.layers[0].k_width = uint64_t(1) << 62; invalid(bad);
    bad = params; bad.streams = std::numeric_limits<uint32_t>::max(); invalid(bad);
    bad = params;
    bad.logical_ctx = (uint32_t(1) << 30) + 8192;
    bad.layers = {{uint64_t(1) << 33, 32, buft}};
    invalid(bad); // Quantized bytes fit, but the GGML element count exceeds int64_t.
    std::puts("PASS recency sizing, padding, streams, validation and overflow");
}
