#pragma once
#include "common.cuh"

// GLM-5.3-Flash DSA prefill attention on gfx11 (RDNA3 WMMA), ordinary or partial (elastic latent regions):
// 512-wide K-only latents (V is a view of K), 64 query heads per KV head. Returns false without doing anything
// when the op does not have that shape.
bool ggml_cuda_flash_attn_ext_glm512(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
