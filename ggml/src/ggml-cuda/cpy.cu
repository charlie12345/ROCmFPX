#include "cpy.cuh"
#include "dequantize.cuh"
#include "cpy-utils.cuh"
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
#include "ggml-musa/mudnn.cuh"
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

static int64_t rocmfpx_fp6_device_stride(int64_t stride) {
#if GGML_ROCMFP6_EXPANDED_DEVICE
    GGML_ASSERT(stride % (int64_t) sizeof(block_rocmfp6) == 0);
    return (stride / (int64_t) sizeof(block_rocmfp6)) * (int64_t) sizeof(block_rocmfp6_expanded);
#else
    return stride;
#endif
}

const int CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

#ifndef GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE
#define GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE 128
#endif

template <cpy_kernel_t cpy_1>
static __global__ void cpy_scalar(const char * cx, char * cdst, const int64_t ne,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                  const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                  const int64_t nb12, const int64_t nb13) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <typename T>
static __global__ void cpy_scalar_transpose(const char * cx, char * cdst, const int64_t ne,
                               const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                               const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                               const int64_t nb12, const int64_t nb13) {

    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const int x = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int y = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    __shared__ float tile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];
    int cur_tile_buf = 0;

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {

        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM + i;
        if (imat >= nmat)
            break;

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if(x < ne01 && y + j < ne00){
                const int row = threadIdx.y+j;
                const int col = threadIdx.x * sizeof(float)/sizeof(T);
                T *tile2 = reinterpret_cast<T*>(tile[cur_tile_buf][row]);
                tile2[col] = src[imat*n + (y+j)*ne01 + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (ty + j < ne01 && tx < ne00) {
                const int col = (threadIdx.y+j)*sizeof(float)/sizeof(T);
                const T *tile2 = reinterpret_cast<const T*>(tile[cur_tile_buf][threadIdx.x]);
                dst[imat*n + (ty+j)*ne00 + tx] = tile2[col];
            }
        }

        cur_tile_buf = (cur_tile_buf + 1) % 2;
    }

    GGML_UNUSED_VARS(ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
        nb12, nb13);
}

static __device__ void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        float2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}

template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < qk/2; j++) {
        float2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + qk/2) = dq.y;
    }
}

static __device__ void cpy_blck_rocmfp4_f32(const char * cxi, char * cdsti) {
    const block_rocmfp4 * x = (const block_rocmfp4 *) cxi;
    float * cdstf = (float *) cdsti;

    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(x->e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(x->e[1]);

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const uint8_t q = x->qs[j];
        cdstf[j]                  = d0 * (float) rocmfp4_decode_i8(q);
        cdstf[j + QK_ROCMFP4/2]   = d1 * (float) rocmfp4_decode_i8(q >> 4);
    }
}

static __device__ void cpy_blck_rocmfp4_fast_f32(const char * cxi, char * cdsti) {
    const block_rocmfp4_fast * x = (const block_rocmfp4_fast *) cxi;
    float * cdstf = (float *) cdsti;

    const float d = rocmfp4_ue4m3_to_fp32_half_finite(x->e);

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const uint8_t q = x->qs[j];
        cdstf[j]                  = d * (float) rocmfp4_decode_i8(q);
        cdstf[j + QK_ROCMFP4/2]   = d * (float) rocmfp4_decode_i8(q >> 4);
    }
}

static __device__ void cpy_blck_rocmfpx_fp3_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) cdsti;

#pragma unroll
    for (int j = 0; j < QK_ROCMFP3; j += 2) {
        float2 dq;
        dequantize_rocmfpx_fp3(cxi, 0, j, dq);
        cdstf[j + 0] = dq.x;
        cdstf[j + 1] = dq.y;
    }
}

static __device__ void cpy_blck_rocmfpx_fp6_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) cdsti;

#pragma unroll
    for (int j = 0; j < QK_ROCMFP6; j += 2) {
        float2 dq;
        dequantize_rocmfpx_fp6(cxi, 0, j, dq);
        cdstf[j + 0] = dq.x;
        cdstf[j + 1] = dq.y;
    }
}

static __device__ void cpy_blck_rocmfpx_fp8_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) cdsti;

#pragma unroll
    for (int j = 0; j < QK_ROCMFP8; j += 2) {
        float2 dq;
        dequantize_rocmfpx_fp8(cxi, 0, j, dq);
        cdstf[j + 0] = dq.x;
        cdstf[j + 1] = dq.y;
    }
}

static __global__ void cpy_rocmfp4_f32_contiguous(const block_rocmfp4 * cx, float * cdst, const int64_t ne) {
    const int64_t packed_idx = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t packed_count = (ne / QK_ROCMFP4) * (QK_ROCMFP4/2);

    if (packed_idx >= packed_count) {
        return;
    }

    const int64_t ib = packed_idx >> 4;
    const int j = packed_idx & 0x0f;
    const int64_t base = ib*QK_ROCMFP4;
    const uint8_t q = cx[ib].qs[j];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(cx[ib].e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(cx[ib].e[1]);

    cdst[base + j]                  = d0 * (float) rocmfp4_decode_i8(q);
    cdst[base + j + QK_ROCMFP4/2]   = d1 * (float) rocmfp4_decode_i8(q >> 4);
}

static __global__ void cpy_rocmfp4_fast_f32_contiguous(const block_rocmfp4_fast * cx, float * cdst, const int64_t ne) {
    const int64_t packed_idx = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t packed_count = (ne / QK_ROCMFP4) * (QK_ROCMFP4/2);

    if (packed_idx >= packed_count) {
        return;
    }

    const int64_t ib = packed_idx >> 4;
    const int j = packed_idx & 0x0f;
    const int64_t base = ib*QK_ROCMFP4;
    const uint8_t q = cx[ib].qs[j];
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(cx[ib].e);

    cdst[base + j]                  = d * (float) rocmfp4_decode_i8(q);
    cdst[base + j + QK_ROCMFP4/2]   = d * (float) rocmfp4_decode_i8(q >> 4);
}

static __global__ void cpy_rocmfpx_fp3_f32_contiguous(const block_rocmfp3 * cx, float * cdst, const int64_t ne) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int64_t ib = i / QK_ROCMFP3;
    const int j = i % QK_ROCMFP3;
    const float d = rocmfpx_ue4m3_to_fp32_finite(cx[ib].e[j >= QK_ROCMFP3/2]);
    cdst[i] = d * (float) rocmfpx_decode_fp3_code_cuda(rocmfpx_get_fp3_code_cuda(cx[ib].qs, j));
}

static __global__ void cpy_rocmfpx_fp6_f32_contiguous(const block_rocmfp6_device * cx, float * cdst, const int64_t ne) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int64_t ib = i / QK_ROCMFP6;
    const int j = i % QK_ROCMFP6;
    const float d = rocmfpx_ue4m3_to_fp32_finite(cx[ib].e[j >= QK_ROCMFP6/2]);
#if GGML_ROCMFP6_EXPANDED_DEVICE
    cdst[i] = d * (float) cx[ib].qs[j];
#else
    cdst[i] = d * (float) rocmfpx_decode_fp6_code_cuda(rocmfpx_get_fp6_code_cuda(cx[ib].qs, j));
#endif
}

static __global__ void cpy_rocmfpx_fp8_f32_contiguous(const block_rocmfp8 * cx, float * cdst, const int64_t ne) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int64_t ib = i / QK_ROCMFP8;
    const int j = i % QK_ROCMFP8;
    const float d = rocmfpx_ue4m3_to_fp32_finite(cx[ib].e);
    cdst[i] = d * (float) cx[ib].qs[j];
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <int qk, int block_bytes>
static __global__ void cpy_q_q_block(const char * cx, char * cdst, const int64_t ne,
                                     const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                     const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                     const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t) blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < block_bytes; ++j) {
        cdst[dst_offset + j] = cx[x_offset + j];
    }
}

template<typename src_t, typename dst_t>
static __global__ void cpy_scalar_contiguous(const char * cx, char * cdst, const int64_t ne) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const src_t * x = (const src_t *) cx;
    dst_t *     dst = (dst_t *) cdst;

    dst[i] = ggml_cuda_cast<dst_t>(x[i]);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_scalar_contiguous_cuda(
    const char * cx, char * cdst, const int64_t ne,
cudaStream_t stream) {

    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_scalar_contiguous<src_t, dst_t><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne);
}

template<typename src_t, typename dst_t, bool transposed = false>
static void ggml_cpy_scalar_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    if (transposed) {
        GGML_ASSERT(ne == ne00*ne01*ne02);  // ne[3] is 1 assumed
        int64_t ne00n, ne01n, ne02n;
        if (nb00 <= nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = ne00;
            ne01n = ne01;
            ne02n = ne02;
        } else {
            ne00n = ne00;
            ne01n = ne01*ne02;
            ne02n = 1;
        }

        int64_t grid_x = (ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_y = (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_z = (ne/(ne01n*ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM;
        GGML_ASSERT(grid_x < UINT_MAX);
        GGML_ASSERT(grid_y < USHRT_MAX);
        GGML_ASSERT(grid_z < USHRT_MAX);
        dim3 dimGrid(grid_x, grid_y, grid_z);
        dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
        cpy_scalar_transpose<dst_t><<<dimGrid, dimBlock, 0, stream>>>
            (cx, cdst, ne, ne00n, ne01n, ne02n, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    } else {
        const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        GGML_ASSERT(num_blocks < UINT_MAX);
        cpy_scalar<cpy_1_scalar<src_t, dst_t>><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
            (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    }
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int64_t num_blocks = ne / QK8_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q8_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int64_t num_blocks = ne / QK4_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int64_t num_blocks = ne / QK4_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_rocmfp4_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_rocmfp4, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_rocmfp4_fast_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_rocmfp4_fast, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_rocmfp4_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f16_rocmfp4, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_rocmfp4_fast_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f16_rocmfp4_fast, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_bf16_rocmfp4_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_bf16_rocmfp4, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_bf16_rocmfp4_fast_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_bf16_rocmfp4_fast, QK_ROCMFP4><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

template <cpy_kernel_t cpy_blck, int qk>
static void ggml_cpy_to_rocmfpx_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % qk == 0);
    const int64_t num_qblocks = ne / qk;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck, qk><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_rocmfpx_fp3_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_f32_rocmfpx_fp3, QK_ROCMFP3>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}

static void ggml_cpy_f16_rocmfpx_fp3_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_f16_rocmfpx_fp3, QK_ROCMFP3>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}

static void ggml_cpy_bf16_rocmfpx_fp3_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_bf16_rocmfpx_fp3, QK_ROCMFP3>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}

static void ggml_cpy_f32_rocmfpx_fp6_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_f32_rocmfpx_fp6, QK_ROCMFP6>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
        rocmfpx_fp6_device_stride(nb10), rocmfpx_fp6_device_stride(nb11),
        rocmfpx_fp6_device_stride(nb12), rocmfpx_fp6_device_stride(nb13), stream);
}

static void ggml_cpy_f16_rocmfpx_fp6_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_f16_rocmfpx_fp6, QK_ROCMFP6>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
        rocmfpx_fp6_device_stride(nb10), rocmfpx_fp6_device_stride(nb11),
        rocmfpx_fp6_device_stride(nb12), rocmfpx_fp6_device_stride(nb13), stream);
}

static void ggml_cpy_bf16_rocmfpx_fp6_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {
    ggml_cpy_to_rocmfpx_hip<cpy_blck_bf16_rocmfpx_fp6, QK_ROCMFP6>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
        rocmfpx_fp6_device_stride(nb10), rocmfpx_fp6_device_stride(nb11),
        rocmfpx_fp6_device_stride(nb12), rocmfpx_fp6_device_stride(nb13), stream);
}

static void ggml_cpy_f32_rocmfpx_fp8_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP8 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP8;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_rocmfpx_fp8, QK_ROCMFP8><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_rocmfpx_fp8_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP8 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP8;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f16_rocmfpx_fp8, QK_ROCMFP8><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_bf16_rocmfpx_fp8_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK_ROCMFP8 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP8;
    const int64_t num_blocks = (num_qblocks + GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE - 1) / GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_bf16_rocmfpx_fp8, QK_ROCMFP8><<<num_blocks, GGML_ROCMFP4_CPY_QUANT_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_rocmfp4_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_rocmfp4_f32, QK_ROCMFP4><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_rocmfp4_f32_contiguous_hip(
    const char * cx, char * cdst, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t packed_count = (ne / QK_ROCMFP4) * (QK_ROCMFP4/2);
    const int64_t num_blocks = (packed_count + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_rocmfp4_f32_contiguous<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
            (const block_rocmfp4 *) cx, (float *) cdst, ne);
}

static void ggml_cpy_rocmfp4_fast_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_rocmfp4_fast_f32, QK_ROCMFP4><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_rocmfp4_fast_f32_contiguous_hip(
    const char * cx, char * cdst, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t packed_count = (ne / QK_ROCMFP4) * (QK_ROCMFP4/2);
    const int64_t num_blocks = (packed_count + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_rocmfp4_fast_f32_contiguous<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
            (const block_rocmfp4_fast *) cx, (float *) cdst, ne);
}

template <cpy_kernel_t cpy_blck, int qk>
static void ggml_cpy_rocmfpx_to_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    GGML_ASSERT(ne % qk == 0);
    const int64_t num_qblocks = ne / qk;
    const int64_t num_blocks = (num_qblocks + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck, qk><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_rocmfpx_fp3_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    ggml_cpy_rocmfpx_to_f32_hip<cpy_blck_rocmfpx_fp3_f32, QK_ROCMFP3>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}

static void ggml_cpy_rocmfpx_fp3_f32_contiguous_hip(
    const char * cx, char * cdst, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP3 == 0);
    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_rocmfpx_fp3_f32_contiguous<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
            (const block_rocmfp3 *) cx, (float *) cdst, ne);
}

static void ggml_cpy_rocmfpx_fp6_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    ggml_cpy_rocmfpx_to_f32_hip<cpy_blck_rocmfpx_fp6_f32, QK_ROCMFP6>(
        cx, cdst, ne, ne00, ne01, ne02,
        rocmfpx_fp6_device_stride(nb00), rocmfpx_fp6_device_stride(nb01),
        rocmfpx_fp6_device_stride(nb02), rocmfpx_fp6_device_stride(nb03),
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}

static void ggml_cpy_rocmfpx_fp6_f32_contiguous_hip(
    const char * cx, char * cdst, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP6 == 0);
    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_rocmfpx_fp6_f32_contiguous<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
            (const block_rocmfp6_device *) cx, (float *) cdst, ne);
}

static void ggml_cpy_rocmfpx_fp8_f32_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP8 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP8;
    const int64_t num_blocks = (num_qblocks + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_rocmfpx_fp8_f32, QK_ROCMFP8><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_rocmfpx_fp8_f32_contiguous_hip(
    const char * cx, char * cdst, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP8 == 0);
    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_rocmfpx_fp8_f32_contiguous<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
            (const block_rocmfp8 *) cx, (float *) cdst, ne);
}

template <int block_bytes>
static void ggml_cpy_rocmfp4_rocmfp4_hip(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    GGML_ASSERT(ne % QK_ROCMFP4 == 0);
    const int64_t num_qblocks = ne / QK_ROCMFP4;
    const int64_t num_blocks = (num_qblocks + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_q_block<QK_ROCMFP4, block_bytes><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int64_t num_blocks = ne / QK5_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int64_t num_blocks = ne / QK5_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int64_t num_blocks = ne / QK4_NL;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
    const bool can_be_transposed = nb01 == (int64_t)ggml_element_size(src0) &&
        src0->ne[3] == 1 && nb02 == ne00 * ne01 * (int64_t)ggml_element_size(src0);

    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        } else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
        }
    } else if (src0->type == GGML_TYPE_Q4_0_ROCMFP4 && src1->type == GGML_TYPE_Q4_0_ROCMFP4) {
        ggml_cpy_rocmfp4_rocmfp4_hip<sizeof(block_rocmfp4)>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0_ROCMFP4_FAST && src1->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        ggml_cpy_rocmfp4_rocmfp4_hip<sizeof(block_rocmfp4_fast)>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q3_0_ROCMFPX && src1->type == GGML_TYPE_Q3_0_ROCMFPX) {
        ggml_cpy_rocmfp4_rocmfp4_hip<sizeof(block_rocmfp3)>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q6_0_ROCMFPX && src1->type == GGML_TYPE_Q6_0_ROCMFPX) {
        ggml_cpy_rocmfp4_rocmfp4_hip<sizeof(block_rocmfp6_device)>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                 rocmfpx_fp6_device_stride(nb00), rocmfpx_fp6_device_stride(nb01),
                 rocmfpx_fp6_device_stride(nb02), rocmfpx_fp6_device_stride(nb03),
                 ne10, ne11, ne12,
                 rocmfpx_fp6_device_stride(nb10), rocmfpx_fp6_device_stride(nb11),
                 rocmfpx_fp6_device_stride(nb12), rocmfpx_fp6_device_stride(nb13), main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0_ROCMFPX && src1->type == GGML_TYPE_Q8_0_ROCMFPX) {
        ggml_cpy_rocmfp4_rocmfp4_hip<sizeof(block_rocmfp8)>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<float, float, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0_ROCMFP4) {
        ggml_cpy_f32_rocmfp4_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        ggml_cpy_f32_rocmfp4_fast_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0_ROCMFP4) {
        ggml_cpy_f16_rocmfp4_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        ggml_cpy_f16_rocmfp4_fast_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_Q4_0_ROCMFP4) {
        ggml_cpy_bf16_rocmfp4_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        ggml_cpy_bf16_rocmfp4_fast_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q3_0_ROCMFPX) {
        ggml_cpy_f32_rocmfpx_fp3_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q3_0_ROCMFPX) {
        ggml_cpy_f16_rocmfpx_fp3_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_Q3_0_ROCMFPX) {
        ggml_cpy_bf16_rocmfpx_fp3_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q6_0_ROCMFPX) {
        ggml_cpy_f32_rocmfpx_fp6_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q6_0_ROCMFPX) {
        ggml_cpy_f16_rocmfpx_fp6_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_Q6_0_ROCMFPX) {
        ggml_cpy_bf16_rocmfpx_fp6_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0_ROCMFPX) {
        ggml_cpy_f32_rocmfpx_fp8_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q8_0_ROCMFPX) {
        ggml_cpy_f16_rocmfpx_fp8_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_Q8_0_ROCMFPX) {
        ggml_cpy_bf16_rocmfpx_fp8_hip
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0_ROCMFP4 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_rocmfp4_f32_contiguous_hip(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_rocmfp4_f32_hip
                    (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_Q4_0_ROCMFP4_FAST && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_rocmfp4_fast_f32_contiguous_hip(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_rocmfp4_fast_f32_hip
                    (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_Q3_0_ROCMFPX && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_rocmfpx_fp3_f32_contiguous_hip(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_rocmfpx_fp3_f32_hip
                    (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_Q6_0_ROCMFPX && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_rocmfpx_fp6_f32_contiguous_hip(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_rocmfpx_fp6_f32_hip
                    (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_Q8_0_ROCMFPX && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_rocmfpx_fp8_f32_contiguous_hip(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_rocmfpx_fp8_f32_hip
                    (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<half, half, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<int32_t, int32_t, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}
