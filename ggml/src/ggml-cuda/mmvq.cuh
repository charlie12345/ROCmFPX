#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);
void ggml_cuda_mul_mat_vec_q_shared(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
    const ggml_tensor * src0_a, ggml_tensor * dst_a,
    const ggml_tensor * src0_b, ggml_tensor * dst_b,
    const ggml_tensor * src0_c, ggml_tensor * dst_c,
    const ggml_tensor * src0_d = nullptr, ggml_tensor * dst_d = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// F32 activation fusion: quantizes F32 to q8_1 in shared memory,
// eliminating the separate quantize_q8_1 dispatch.
void ggml_cuda_mul_mat_vec_q_rocmi4_f32_act(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
