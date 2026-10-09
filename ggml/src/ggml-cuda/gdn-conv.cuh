#pragma once
#include "common.cuh"
struct ggml_cuda_gdn_conv_match {
    int concat_idx = -1, conv_idx = -1;   // node indices
    int cont_idx = -1;                    // CONT materializing the transposed input, or -1
    const ggml_tensor * x = nullptr;      // [C, T] F32 contiguous (root of the transpose view)
    const ggml_tensor * state = nullptr;  // [3, C] F32 contiguous
    const ggml_tensor * concat = nullptr; // [T+3, C] F32
    const ggml_tensor * w = nullptr;
    ggml_tensor * conv_out = nullptr;
    int64_t C = 0, T = 0, tail_from = 0;  // first concat column that must be materialized
    bool silu = false;
};
bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_cont(const ggml_cgraph * cgraph, int k, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);

#define GDN_CONV_DECODE_MAX_CPY 4
struct ggml_cuda_gdn_conv_decode_match {
    int concat_idx = -1, silu_idx = -1;
    const ggml_tensor * x = nullptr;      // [C, T] F32 contiguous (root of the transpose view)
    const ggml_tensor * state = nullptr;  // [3, C] F32 contiguous
    const ggml_tensor * w = nullptr;      // [4, C]
    ggml_tensor * out = nullptr;          // SILU output [C, T]
    int64_t C = 0, T = 0;
    struct cpy_args {
        float * dst[GDN_CONV_DECODE_MAX_CPY];
        int     col[GDN_CONV_DECODE_MAX_CPY];  // first concat column of each saved window
        int     n;
    } cp;
};
bool ggml_cuda_gdn_conv_decode_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_decode_match & m);
void ggml_cuda_gdn_conv_decode(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_decode_match & m);
