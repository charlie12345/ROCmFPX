#pragma once
#include "common.cuh"
// [TAG_GDN_CONV_SNAP] one kernel for CONCAT(state, x) -> K snapshot CPYs -> SSM_CONV -> SILU of a delta-net layer
constexpr int GDN_CONV_MAX_SNAPSHOTS = 16;

struct ggml_cuda_gdn_conv_match {
    int concat_idx = -1, conv_idx = -1;   // node indices
    const ggml_tensor * x = nullptr;      // [C, T, S] or [C, T*S] F32 contiguous (root of the transpose view)
    const ggml_tensor * state = nullptr;  // [3, C, S] F32 contiguous
    const ggml_tensor * concat = nullptr; // [T+3, C, S] F32, never materialized
    const ggml_tensor * w = nullptr;
    ggml_tensor * conv_out = nullptr;
    int64_t C = 0, T = 0, S = 0;
    bool silu = false;
    int n_snap = 0;                                         // the state snapshots of the concat, K = n_rs_seq + 1
    const ggml_tensor * snap_dst[GDN_CONV_MAX_SNAPSHOTS];   // [3*C, S] F32
    int snap_col[GDN_CONV_MAX_SNAPSHOTS];                   // first concat column of the snapshot
};
bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_buffers_ok(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_set_folded(const ggml_tensor * concat);
bool ggml_cuda_gdn_conv_is_folded(const ggml_tensor * conv);
bool ggml_cuda_gdn_conv_folds(const ggml_tensor * node);
void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
