#pragma once
#include "common.cuh"

// GLM-5.3-Flash prefill sweep (288 experts, top 8): 32-token slices on hipBLAS beat 16, 64 and 128; from 1024
// tokens the fused BF16 tile kernel beats per-expert hipBLAS for MUL_MAT_ID (pp4096 at ubatch 1024/2048/4096:
// 192.8/237.9/271.8 vs 182.9/207.5/237.2 t/s), below that hipBLAS wins (ubatch 512: 160.7 vs 143.6). IQ3_S and
// IQ4_XS tiles are prefetched and stay on the tile kernel at every size (ubatch 512: 188.7 vs 161.2)
#define GGML_CUDA_MM_BLAS_MIN_TOKENS   32
#define GGML_CUDA_MMID_BLAS_MIN_TOKENS 512
#define GGML_CUDA_MMID_BLAS_MAX_TOKENS 1024
// Quantized-weight BF16 WMMA GEMM on gfx1151, from 512 tokens up.
bool ggml_cuda_mmb_supported_mm  (const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// gfx1151: large quantized GEMMs run faster as F16 dequantization plus hipBLAS than on the BF16 tile kernel.
// Dense matmuls take hipBLAS from this many tokens; MUL_MAT_ID splits into per-expert matmuls from mmid_tokens.
bool ggml_cuda_mm_prefer_blas  (const ggml_tensor * src0, int64_t n_tokens);
bool ggml_cuda_mmid_prefer_blas(const ggml_tensor * src0, int64_t n_tokens);
bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph();
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n);
void ggml_cuda_mmb_marks_clear();
size_t ggml_cuda_mmb_marks_count();
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_gatemix();
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_release_all();
