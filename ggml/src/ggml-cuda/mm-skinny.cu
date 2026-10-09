#include "mm-skinny.cuh"
#include "convert.cuh"

#define MM_SKINNY_MAX_M  8
#define MM_SKINNY_WARPS  8

template <typename T>
static __device__ __forceinline__ float4 mm_skinny_w4(const T * w) {
    if constexpr (std::is_same_v<T, float>) {
        return *(const float4 *) w;
    } else {
        const uint2 u = *(const uint2 *) w;
        const T * h = (const T *) &u;
        return make_float4(ggml_cuda_cast<float>(h[0]), ggml_cuda_cast<float>(h[1]), ggml_cuda_cast<float>(h[2]), ggml_cuda_cast<float>(h[3]));
    }
}

template <typename T, int M>
static __global__ void __launch_bounds__(32 * MM_SKINNY_WARPS)
mm_skinny_kernel(const T * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, const int K, const int n_tok,
                 const int64_t sw, const int64_t sx, const int64_t sy) {
    const int lane = threadIdx.x % 32;
    const int t    = blockIdx.x * MM_SKINNY_WARPS + threadIdx.x / 32;
    if (t >= n_tok) {
        return;
    }
    const float * xr = x + t * sx;
    float acc[M] = {};
    for (int k = 4 * lane; k < K; k += 4 * 32) {
        const float4 xv = *(const float4 *) (xr + k);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float4 wv = mm_skinny_w4(w + m * sw + k);
            acc[m] = fmaf(xv.x, wv.x, fmaf(xv.y, wv.y, fmaf(xv.z, wv.z, fmaf(xv.w, wv.w, acc[m]))));
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = warp_reduce_sum<32>(acc[m]);
    }
    if (lane < M) {
        float v = acc[0];
#pragma unroll
        for (int m = 1; m < M; ++m) {
            v = lane == m ? acc[m] : v;
        }
        y[t * sy + lane] = v;
    }
}

bool ggml_cuda_mm_skinny_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // a long batch only: short ones are the vector kernels' case
    const int64_t K = src0->ne[0], M = src0->ne[1], T = src1->ne[1];
    return M >= 2 && M <= MM_SKINNY_MAX_M && K % 128 == 0 && T >= 64 && T <= INT32_MAX && K <= INT32_MAX &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->ne[0] == K &&
        dst->ne[0] == M && dst->ne[1] == T;
}

template <typename T>
static void mm_skinny_launch(const T * w, const float * x, float * y, const int M, const int K, const int n_tok, cudaStream_t stream) {
    const dim3 grid((n_tok + MM_SKINNY_WARPS - 1) / MM_SKINNY_WARPS);
    const dim3 block(32 * MM_SKINNY_WARPS);
    switch (M) {
        case 2: mm_skinny_kernel<T, 2><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 3: mm_skinny_kernel<T, 3><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 4: mm_skinny_kernel<T, 4><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 5: mm_skinny_kernel<T, 5><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 6: mm_skinny_kernel<T, 6><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 7: mm_skinny_kernel<T, 7><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        case 8: mm_skinny_kernel<T, 8><<<grid, block, 0, stream>>>(w, x, y, K, n_tok, K, K, M); break;
        default: GGML_ABORT("mm_skinny: unsupported row count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mm_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], T = (int) src1->ne[1];
    const float * x = (const float *) src1->data;
    float       * y = (float *) dst->data;
    cudaStream_t stream = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_F32:  mm_skinny_launch((const float *)       src0->data, x, y, M, K, T, stream); break;
        case GGML_TYPE_F16:  mm_skinny_launch((const half *)        src0->data, x, y, M, K, T, stream); break;
        case GGML_TYPE_BF16: mm_skinny_launch((const nv_bfloat16 *) src0->data, x, y, M, K, T, stream); break;
        default: GGML_ABORT("mm_skinny: unsupported type");
    }
}
