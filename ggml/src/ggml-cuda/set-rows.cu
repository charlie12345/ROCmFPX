#include "set-rows.cuh"
#include "cpy-utils.cuh"

#include <cmath>

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// ============================================================
// TurboQuant constants for set-rows FWHT
// ============================================================
#define TURBO_HEAD_DIM_SR 128
#define TURBO_BLOCKS_PER_CHUNK_SR (TURBO_HEAD_DIM_SR / 32)  // 4

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// ============================================================
// TurboQuant specialized set-rows kernel with FWHT
// ============================================================
// Each CUDA block processes one 128-element chunk.
// 128 threads per block, one thread per element in the chunk.
// Steps:
//   1. Each thread reads one float from the source row
//   2. Cooperative norm computation via shared memory reduction
//   3. Normalize the chunk
//   4. FWHT butterfly in shared memory (7 stages for n=128)
//   5. Each thread scalar-quantizes its element and packs into blocks

// Device-side codebook references for turbo quantize (same as in cpy-utils.cuh)
__device__ static const float sr_codebook_3bit[8] = {
    -0.1883972972f, -0.1181399059f, -0.0665857641f, -0.0216044751f,
     0.0216041461f,  0.0665854520f,  0.1181396281f,  0.1883970748f
};

__device__ static const float sr_codebook_4bit[16] = {
    -0.2376389871f, -0.1808080141f, -0.1417777640f, -0.1102646123f,
    -0.0828112376f, -0.0577640422f, -0.0341540905f, -0.0113168380f,
     0.0112761586f,  0.0341139667f,  0.0577250301f,  0.0827738972f,
     0.1102295202f,  0.1417455465f,  0.1807794468f,  0.2376153882f
};

static __device__ uint8_t sr_nearest_codebook(float val, const float *codebook, int n) {
    float best_dist = fabsf(val - codebook[0]);
    uint8_t best_idx = 0;
    for (int i = 1; i < n; i++) {
        float dist = fabsf(val - codebook[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = (uint8_t)i;
        }
    }
    return best_idx;
}

// Turbo3 set-rows kernel: processes 128-element chunks with FWHT
template <typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne_total_chunks,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t ne00,
        const uint3   ne00_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    __shared__ float smem[TURBO_HEAD_DIM_SR];
    __shared__ float reduction[TURBO_HEAD_DIM_SR];

    const int64_t chunk_global = blockIdx.x;
    const int tid = threadIdx.x;  // 0..127

    if (chunk_global >= ne_total_chunks) return;

    // Map the global chunk index to i00 (element offset within a row) + row indices
    // Each chunk covers 128 elements, so the chunk's base element = chunk_global * 128
    const int64_t elem_base = chunk_global * TURBO_HEAD_DIM_SR;
    uint32_t tmp = (uint32_t)elem_base;
    uint2 div_mod;

    div_mod = fast_div_modulo(tmp, ne00_fd);
    const int64_t i00 = div_mod.y;  // offset within row (multiple of 128)
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    const float val = src0_row[i00 + tid];
    smem[tid] = val;

    // Step 1: Compute L2 norm via parallel reduction
    reduction[tid] = val * val;
    __syncthreads();

    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            reduction[tid] += reduction[tid + s];
        }
        __syncthreads();
    }

    float norm = sqrtf(reduction[0]);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Step 2: Normalize
    smem[tid] *= inv_norm;
    __syncthreads();

    // Step 3: FWHT butterfly stages (7 stages for n=128)
    for (int h = 1; h < TURBO_HEAD_DIM_SR; h *= 2) {
        if (tid < 64) {
            int group = tid / h;
            int pos = tid % h;
            int i = group * h * 2 + pos;
            float a = smem[i];
            float b = smem[i + h];
            smem[i]     = a + b;
            smem[i + h] = a - b;
        }
        __syncthreads();
    }

    // Apply 1/sqrt(128) normalization
    const float fwht_scale = 0.08838834764831844f;
    smem[tid] *= fwht_scale;
    __syncthreads();

    // Step 4: Scalar quantize and pack into turbo3 blocks
    // Each thread quantizes its element
    uint8_t my_idx = sr_nearest_codebook(smem[tid], sr_codebook_3bit, 8);

    // We need to pack 32 indices per block cooperatively
    // Use shared memory to collect indices, then pack
    // Reuse reduction[] as uint8 storage
    ((uint8_t *)reduction)[tid] = my_idx;
    __syncthreads();

    // Compute destination block pointer
    // dst layout: dst_row*s1 + i02*s2 + i03*s3 gives byte offset to row start
    // Then add block offset for i00
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int64_t dst_block_base = i00 / TURBO3_BLOCK_SIZE;

    // Only 4 threads (one per block) do the packing
    if (tid < TURBO_BLOCKS_PER_CHUNK_SR) {
        const int blk = tid;
        block_turbo3_0 * dst_block = dst_row_ptr + dst_block_base + blk;
        const uint8_t * indices = ((const uint8_t *)reduction) + blk * 32;

        // Store norm
        dst_block->d = __float2half(norm);

        // Pack 32 x 3-bit indices into 12 bytes
        memset(dst_block->qs, 0, 12);
        for (int j = 0; j < 32; j++) {
            int bit_off = j * 3;
            int byte_pos = bit_off / 8;
            int shift = bit_off % 8;
            dst_block->qs[byte_pos] |= (uint8_t)((indices[j] & 0x07) << shift);
            if (shift > 5 && byte_pos + 1 < 12) {
                dst_block->qs[byte_pos + 1] |= (uint8_t)((indices[j] & 0x07) >> (8 - shift));
            }
        }
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Turbo4 set-rows kernel: processes 128-element chunks with FWHT
template <typename idx_t>
static __global__ void k_set_rows_turbo4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne_total_chunks,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t ne00,
        const uint3   ne00_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    __shared__ float smem[TURBO_HEAD_DIM_SR];
    __shared__ float reduction[TURBO_HEAD_DIM_SR];

    const int64_t chunk_global = blockIdx.x;
    const int tid = threadIdx.x;  // 0..127

    if (chunk_global >= ne_total_chunks) return;

    // Map the global chunk index to i00 (element offset within a row) + row indices
    const int64_t elem_base = chunk_global * TURBO_HEAD_DIM_SR;
    uint32_t tmp = (uint32_t)elem_base;
    uint2 div_mod;

    div_mod = fast_div_modulo(tmp, ne00_fd);
    const int64_t i00 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    const float val = src0_row[i00 + tid];
    smem[tid] = val;

    // Step 1: Compute L2 norm via parallel reduction
    reduction[tid] = val * val;
    __syncthreads();

    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            reduction[tid] += reduction[tid + s];
        }
        __syncthreads();
    }

    float norm = sqrtf(reduction[0]);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Step 2: Normalize
    smem[tid] *= inv_norm;
    __syncthreads();

    // Step 3: FWHT butterfly stages (7 stages for n=128)
    for (int h = 1; h < TURBO_HEAD_DIM_SR; h *= 2) {
        if (tid < 64) {
            int group = tid / h;
            int pos = tid % h;
            int i = group * h * 2 + pos;
            float a = smem[i];
            float b = smem[i + h];
            smem[i]     = a + b;
            smem[i + h] = a - b;
        }
        __syncthreads();
    }

    // Apply 1/sqrt(128) normalization
    const float fwht_scale = 0.08838834764831844f;
    smem[tid] *= fwht_scale;
    __syncthreads();

    // Step 4: Scalar quantize and pack into turbo4 blocks
    uint8_t my_idx = sr_nearest_codebook(smem[tid], sr_codebook_4bit, 16);

    // Collect indices in shared memory
    ((uint8_t *)reduction)[tid] = my_idx;
    __syncthreads();

    // Compute destination block pointer
    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int64_t dst_block_base = i00 / TURBO4_BLOCK_SIZE;

    // Only 4 threads (one per block) do the packing
    if (tid < TURBO_BLOCKS_PER_CHUNK_SR) {
        const int blk = tid;
        block_turbo4_0 * dst_block = dst_row_ptr + dst_block_base + blk;
        const uint8_t * indices = ((const uint8_t *)reduction) + blk * 32;

        // Store norm
        dst_block->d = __float2half(norm);

        // Pack 32 x 4-bit indices into 16 bytes
        for (int j = 0; j < TURBO4_BLOCK_SIZE / 2; j++) {
            dst_block->qs[j] = (indices[2*j] & 0x0F) | ((indices[2*j + 1] & 0x0F) << 4);
        }
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Dispatch functions for turbo set-rows
template<typename idx_t>
static void set_rows_cuda_turbo3(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % TURBO_HEAD_DIM_SR == 0);
    const int64_t ne_total_chunks = (ne00 * ne01 * ne02 * ne03) / TURBO_HEAD_DIM_SR;
    const dim3 grid_size((int)ne_total_chunks);
    const dim3 block_size(TURBO_HEAD_DIM_SR);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total_chunks > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_turbo3<idx_t><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total_chunks, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo4(
        const float * src0_d, const idx_t * src1_d, block_turbo4_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % TURBO_HEAD_DIM_SR == 0);
    const int64_t ne_total_chunks = (ne00 * ne01 * ne02 * ne03) / TURBO_HEAD_DIM_SR;
    const dim3 grid_size((int)ne_total_chunks);
    const dim3 block_size(TURBO_HEAD_DIM_SR);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total_chunks > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_turbo4<idx_t><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total_chunks, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0_ROCMFP4) {
        set_rows_cuda_quant<idx_t, block_rocmfp4, QK_ROCMFP4, quantize_f32_rocmfp4_block>(
            src0_d, src1_d, (block_rocmfp4*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        // FWHT-aware 128-thread kernels for correct TurboQuant encoding
        set_rows_cuda_turbo3<idx_t>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        set_rows_cuda_quant<idx_t, block_rocmfp4_fast, QK_ROCMFP4, quantize_f32_rocmfp4_fast_block>(
            src0_d, src1_d, (block_rocmfp4_fast*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q3_0_ROCMFPX) {
        set_rows_cuda_quant<idx_t, block_rocmfp3, QK_ROCMFP3, quantize_f32_rocmfpx_fp3_block>(
            src0_d, src1_d, (block_rocmfp3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q6_0_ROCMFPX) {
        set_rows_cuda_quant<idx_t, block_rocmfp6_device, QK_ROCMFP6,
#if GGML_ROCMFP6_EXPANDED_DEVICE
            quantize_f32_rocmfpx_fp6_expanded_block
#else
            quantize_f32_rocmfpx_fp6_block
#endif
        >(
            src0_d, src1_d, (block_rocmfp6_device*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0_ROCMFPX) {
        set_rows_cuda_quant<idx_t, block_rocmfp8, QK_ROCMFP8, quantize_f32_rocmfpx_fp8_block>(
            src0_d, src1_d, (block_rocmfp8*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        // FWHT-aware 128-thread kernels for correct TurboQuant encoding
        set_rows_cuda_turbo4<idx_t>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
