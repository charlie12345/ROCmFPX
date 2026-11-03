#pragma once

#include "common.cuh"

// ROCmFPX [TAG_WMMA_GEMM]: prompt-sized products of quantized weights (the fork's FP4/FP6 types and the stock
// quants listed in ggml_cuda_should_use_wmma_gemm) through an FP16 WMMA GEMM
// (rocm_wmma_gemm by Adel Johar, MIT, vendored in vendor/rocm_wmma_gemm and built per architecture by
// ggml-hip/wmma-gemm.cmake) instead of MMQ. The weight rows are unpacked to f16 into a reused scratch, the activations
// converted to f16, and the product accumulated in f32. Only on gfx11 GPUs (tuned for gfx1100 and gfx1151, the other
// chips use the nearest of those configurations), only when its arch library loads, only for batches of ROCMFPX_WMMA_GEMM_MIN_BATCH (512) tokens or more.
// ROCMFPX_WMMA_GEMM=0 turns it off.

bool ggml_cuda_should_use_wmma_gemm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_wmma_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
