#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// mHC post gate (gscale * sigmoid(gx * gs + gb) + gbias) and the comb node's Sinkhorn of the same mixes, one launch;
// false (nothing launched) when the shapes do not fit
bool ggml_cuda_dsv4_hc_post_gate_comb(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, ggml_tensor * gdst, float gscale, float gbias, ggml_tensor * comb);

// mHC pre with its stream weights computed from the pre gate chain's inputs (that chain's output is then not written);
// false (nothing launched) when the shapes do not fit
bool ggml_cuda_dsv4_hc_pre_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, const ggml_tensor * gate, float gscale, float gbias, ggml_tensor * pre);
