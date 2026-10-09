#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <map>
#include <vector>

struct llama_kv_recency_layer {
    uint64_t k_width;
    uint64_t v_width; // 0 for a K-only layer
    ggml_backend_buffer_type_t buft;
};

struct llama_kv_recency_plan_params {
    uint32_t logical_ctx = 0;
    uint32_t page_tokens = 256;
    uint32_t migrate_tokens = 1024;
    uint32_t hot_tokens = 8192;
    uint32_t slab_tokens = 4096;
    bool ring_one_slab = false; // keep the whole Q8 ring in one slab, so any append writes one tensor
    uint32_t streams = 1;
    std::vector<llama_kv_recency_layer> layers;
};

struct llama_kv_recency_plan {
    uint32_t logical_ctx = 0;
    uint32_t physical_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t migrate_tokens = 0;
    uint32_t hot_tokens = 0;
    uint32_t cold_tokens = 0;
    uint32_t ring_tokens = 0;
    uint32_t slab_tokens = 0;
    size_t q8_bytes_per_token = 0;
    size_t q4_bytes_per_token = 0;
    size_t hot_bytes = 0;
    size_t cold_bytes = 0;
    size_t hot_alloc_bytes = 0;
    size_t cold_alloc_bytes = 0;
    size_t committed_alloc_bytes = 0; // every Q8 ring and Q4 cold slab allocated
    size_t migration_reserve_bytes = 0;
    size_t migration_workspace_bytes = 0; // F32 page conversion buffers of one migration step
};

LLAMA_API llama_kv_recency_plan llama_kv_recency_plan_create(const llama_kv_recency_plan_params & params);
// Largest page-aligned hot window, up to params.hot_tokens, whose full-session slabs and migration workspace fit every
// buffer type budget. Throws std::runtime_error when even one hot page does not fit.
LLAMA_API llama_kv_recency_plan llama_kv_recency_plan_admit(const llama_kv_recency_plan_params & params,
                                                            const std::map<ggml_backend_buffer_type_t, size_t> & budgets);
// Backend allocation for one slab of the given rows across all layers.
LLAMA_API size_t llama_kv_recency_slab_alloc_bytes(const llama_kv_recency_plan_params & params, ggml_type type, uint32_t tokens);
