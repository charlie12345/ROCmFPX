#pragma once
#include "common.cuh"

// Dense mul_mat whose weight has only a few rows (ne01 <= 8), e.g. Qwen3.8 HC inject (10240 -> 4) over a long batch:
// the GEMM tiles waste nearly every row, so one warp per token streams the activation row once instead.
bool ggml_cuda_mm_skinny_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mm_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
