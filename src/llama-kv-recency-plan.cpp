#include "llama-kv-recency-plan.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace {
size_t checked_add(size_t a, size_t b) {
    const size_t limit = std::min<uint64_t>(SIZE_MAX, INT64_MAX);
    if (a > limit || b > limit - a) {
        throw std::overflow_error("KV recency size overflow");
    }
    return a + b;
}

size_t checked_mul(uint64_t a, uint64_t b) {
    const size_t limit = std::min<uint64_t>(SIZE_MAX, INT64_MAX);
    if (a > limit || (a && b > limit/a)) {
        throw std::overflow_error("KV recency size overflow");
    }
    return size_t(a*b);
}

size_t align_up(size_t value, size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) {
        throw std::invalid_argument("KV recency alignment must be a power of two");
    }
    return checked_add(value, alignment - 1) & ~(alignment - 1);
}

size_t row_bytes(ggml_type type, uint64_t width) {
    const uint64_t block = ggml_blck_size(type);
    if (!width || width % block) {
        throw std::invalid_argument("KV recency row width must match the GGML block size");
    }
    checked_mul(width/block, ggml_type_size(type));
    if (width > INT64_MAX) {
        throw std::overflow_error("KV recency row width overflow");
    }
    return ggml_row_size(type, int64_t(width));
}

std::map<ggml_backend_buffer_type_t, size_t> arena_bytes_by_buft(const llama_kv_recency_plan_params & params, ggml_type type, uint32_t tokens) {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    if (!tokens) {
        return result;
    }
    const size_t metadata_bytes = checked_mul(checked_mul(params.layers.size(), 2), ggml_tensor_overhead());
    const ggml_init_params init = {metadata_bytes, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(init), ggml_free);
    if (!ctx) {
        throw std::runtime_error("failed to allocate KV recency tensor metadata");
    }
    std::map<ggml_backend_buffer_type_t, std::vector<ggml_tensor *>> groups;
    size_t upper_bound = 0;
    for (const auto & layer : params.layers) {
        const size_t alignment = ggml_backend_buft_get_alignment(layer.buft);
        for (bool key : {true, false}) {
            // a K-only layer (MLA latent) has no V rows
            const uint64_t width = key ? layer.k_width : layer.v_width;
            if (!key && !width) {
                continue;
            }
            size_t elements = 1;
            for (const uint64_t dimension : {width, uint64_t(tokens), uint64_t(params.streams), uint64_t(1)}) {
                if (elements >= uint64_t(INT64_MAX)/dimension) {
                    throw std::overflow_error("KV recency tensor element count overflow");
                }
                elements = checked_mul(elements, dimension);
            }
            checked_mul(checked_mul(row_bytes(type, width), tokens), params.streams);
            auto * tensor = ggml_new_tensor_3d(ctx.get(), type, int64_t(width), tokens, params.streams);
            const size_t bytes = ggml_backend_buft_get_alloc_size(layer.buft, tensor);
            if (bytes > ggml_backend_buft_get_max_size(layer.buft)) {
                throw std::invalid_argument("KV recency tensor exceeds backend allocation limit");
            }
            upper_bound = checked_add(upper_bound, align_up(bytes, alignment));
            groups[layer.buft].push_back(tensor);
        }
    }
    for (auto & group : groups) {
        result[group.first] = ggml_backend_buft_get_alloc_size_n(group.first, group.second.data(), int(group.second.size()));
    }
    return result;
}

size_t arena_bytes(const llama_kv_recency_plan_params & params, ggml_type type, uint32_t tokens) {
    size_t result = 0;
    for (const auto & entry : arena_bytes_by_buft(params, type, tokens)) {
        result = checked_add(result, entry.second);
    }
    return result;
}

// Adds every slab of a region to per-buffer-type totals.
void add_slabs(std::map<ggml_backend_buffer_type_t, size_t> & total, const llama_kv_recency_plan_params & params, ggml_type type, uint32_t tokens, uint32_t slab) {
    const uint32_t full = tokens/slab;
    for (const auto & entry : arena_bytes_by_buft(params, type, slab)) {
        total[entry.first] = checked_add(total[entry.first], checked_mul(full, entry.second));
    }
    for (const auto & entry : arena_bytes_by_buft(params, type, tokens - full*slab)) {
        total[entry.first] = checked_add(total[entry.first], entry.second);
    }
}

uint32_t ring_slab(const llama_kv_recency_plan_params & params, const llama_kv_recency_plan & plan) {
    return params.ring_one_slab ? std::max(plan.ring_tokens, 1u) : params.slab_tokens;
}

// Full-session commitment per buffer type: all ring and cold slabs plus one migration step workspace.
std::map<ggml_backend_buffer_type_t, size_t> session_bytes(const llama_kv_recency_plan_params & params, const llama_kv_recency_plan & plan) {
    std::map<ggml_backend_buffer_type_t, size_t> total;
    add_slabs(total, params, GGML_TYPE_Q8_0, plan.ring_tokens, ring_slab(params, plan));
    add_slabs(total, params, GGML_TYPE_Q4_0, plan.cold_tokens, params.slab_tokens);
    for (const auto & entry : arena_bytes_by_buft(params, GGML_TYPE_F32, plan.cold_tokens ? plan.page_tokens : 0)) {
        total[entry.first] = checked_add(total[entry.first], entry.second);
    }
    return total;
}
}

llama_kv_recency_plan llama_kv_recency_plan_create(const llama_kv_recency_plan_params & params) {
    if (!params.logical_ctx || !params.streams || !params.hot_tokens || params.layers.empty() ||
            params.layers.size() > size_t(INT_MAX/2) || params.page_tokens < 32 ||
            (params.page_tokens & (params.page_tokens - 1)) ||
            !params.migrate_tokens || params.migrate_tokens % params.page_tokens ||
            !params.slab_tokens || params.slab_tokens % params.page_tokens) {
        throw std::invalid_argument("invalid KV recency page, migration or context geometry");
    }
    llama_kv_recency_plan plan;
    plan.logical_ctx = params.logical_ctx;
    plan.page_tokens = params.page_tokens;
    plan.migrate_tokens = params.migrate_tokens;
    plan.slab_tokens = params.slab_tokens;
    const size_t physical_tokens = align_up(params.logical_ctx, params.page_tokens);
    if (physical_tokens > UINT32_MAX) {
        throw std::overflow_error("KV recency padded context overflow");
    }
    plan.physical_tokens = uint32_t(physical_tokens);
    plan.hot_tokens = uint32_t(align_up(std::min(params.logical_ctx, params.hot_tokens), params.page_tokens));
    plan.cold_tokens = plan.physical_tokens - plan.hot_tokens;
    for (const auto & layer : params.layers) {
        if (!layer.buft) {
            throw std::invalid_argument("KV recency requires a backend buffer type");
        }
        for (bool key : {true, false}) {
            const uint64_t width = key ? layer.k_width : layer.v_width;
            if (!key && !width) {
                continue;
            }
            plan.q8_bytes_per_token = checked_add(plan.q8_bytes_per_token, row_bytes(GGML_TYPE_Q8_0, width));
            plan.q4_bytes_per_token = checked_add(plan.q4_bytes_per_token, row_bytes(GGML_TYPE_Q4_0, width));
        }
    }
    plan.hot_bytes = checked_mul(checked_mul(plan.q8_bytes_per_token, plan.hot_tokens), params.streams);
    plan.cold_bytes = checked_mul(checked_mul(plan.q4_bytes_per_token, plan.cold_tokens), params.streams);
    checked_add(plan.hot_bytes, plan.cold_bytes);
    plan.hot_alloc_bytes = arena_bytes(params, GGML_TYPE_Q8_0, plan.hot_tokens);
    plan.cold_alloc_bytes = arena_bytes(params, GGML_TYPE_Q4_0, plan.cold_tokens);
    const uint32_t migration_tokens = std::min(plan.cold_tokens, plan.migrate_tokens);
    // At most hot/page + 1 pages touch the hot window, so a full ring always holds one migratable batch.
    plan.ring_tokens = uint32_t(std::min<size_t>(plan.physical_tokens, size_t(plan.hot_tokens) + plan.page_tokens + migration_tokens));
    std::map<ggml_backend_buffer_type_t, size_t> slabs;
    add_slabs(slabs, params, GGML_TYPE_Q8_0, plan.ring_tokens, ring_slab(params, plan));
    add_slabs(slabs, params, GGML_TYPE_Q4_0, plan.cold_tokens, params.slab_tokens);
    for (const auto & entry : slabs) {
        plan.committed_alloc_bytes = checked_add(plan.committed_alloc_bytes, entry.second);
    }
    plan.migration_workspace_bytes = arena_bytes(params, GGML_TYPE_F32, plan.cold_tokens ? plan.page_tokens : 0);
    // Conservative prototype reserve: one Q4 destination batch plus a full F32 conversion workspace.
    plan.migration_reserve_bytes = checked_add(arena_bytes(params, GGML_TYPE_Q4_0, migration_tokens),
                                               arena_bytes(params, GGML_TYPE_F32, migration_tokens));
    checked_add(checked_add(plan.hot_alloc_bytes, plan.cold_alloc_bytes), plan.migration_reserve_bytes);
    return plan;
}

size_t llama_kv_recency_slab_alloc_bytes(const llama_kv_recency_plan_params & params, ggml_type type, uint32_t tokens) {
    llama_kv_recency_plan_create(params);
    return arena_bytes(params, type, tokens);
}

llama_kv_recency_plan llama_kv_recency_plan_admit(const llama_kv_recency_plan_params & params,
                                                  const std::map<ggml_backend_buffer_type_t, size_t> & budgets) {
    const auto cap = llama_kv_recency_plan_create(params);
    for (const auto & layer : params.layers) {
        if (!budgets.count(layer.buft)) {
            throw std::invalid_argument("KV recency admission requires a budget for every buffer type");
        }
    }
    auto trial = params;
    for (uint32_t hot = cap.hot_tokens; hot >= params.page_tokens; hot -= params.page_tokens) {
        trial.hot_tokens = hot;
        const auto plan = llama_kv_recency_plan_create(trial);
        bool fits = true;
        for (const auto & entry : session_bytes(trial, plan)) {
            fits = fits && entry.second <= budgets.at(entry.first);
        }
        if (fits) {
            return plan;
        }
    }
    throw std::runtime_error("KV recency budget cannot guarantee the requested context");
}
