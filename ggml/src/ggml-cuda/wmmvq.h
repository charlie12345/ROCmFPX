// WMMA-based tg GEMV for Q4_K / Q4_0_ROCMI4 on RDNA3 — see wmmvq.cu.
#pragma once

#include "common.cuh"

struct ggml_cuda_mm_fusion_args_host;

bool wmmvq_q4_k_eligible(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_mul_mat_wmmvq_q4_k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool wmmvq_rocmi4_eligible(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_mul_mat_wmmvq_rocmi4(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
