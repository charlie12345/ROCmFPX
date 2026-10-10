#pragma once

#include "common.cuh"

// GLM-5.3-Flash DSA selection mask written in one pass (see dsa-mask.cu)
struct ggml_cuda_dsa_mask_match {
    const ggml_tensor * add      = nullptr;   // the mask (output written here)
    const ggml_tensor * idx      = nullptr;   // [n_sel, n_tokens] selected cells (dump rows >= n_kv for dead slots)
    const ggml_tensor * kq_mask  = nullptr;   // [n_kv, n_tokens] causal mask
    const ggml_tensor * rep_inf  = nullptr;   // skipped nodes
    const ggml_tensor * rep_zero = nullptr;
    const ggml_tensor * set_rows = nullptr;
    int64_t n_kv = 0, n_sel = 0, n_tok = 0;
};

bool ggml_cuda_dsa_mask_match_at_add(const ggml_cgraph * cgraph, int i, ggml_cuda_dsa_mask_match & m);
// a REPEAT or SET_ROWS of a matched chain: its work is done by the fused kernel at the ADD
bool ggml_cuda_dsa_mask_part_skippable(const ggml_cgraph * cgraph, int i);
void ggml_cuda_dsa_mask(ggml_backend_cuda_context & ctx, const ggml_cuda_dsa_mask_match & m);

// per-token attention lists of the last fused mask (see dsa-mask.cu)
void ggml_cuda_dsa_lists_reset(int device);
void ggml_cuda_dsa_lists_invalidate(int device, const ggml_tensor * t);
bool ggml_cuda_dsa_lists_lookup(int device, const ggml_tensor * mask, int n_q, int n_kv_max, const int32_t ** lists, const int32_t ** counts);
