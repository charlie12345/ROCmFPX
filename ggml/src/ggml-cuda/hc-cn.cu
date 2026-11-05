#include "hc-cn.cuh"
#include "mmq.cuh"
#include <cstdlib>

#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

#define HC_CN_BLOCK 1024
#define HC_CN_MAX_EMB 3072

__device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
__device__ __forceinline__ uint16_t hc_f2bf32(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_f32(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ __align__(16) float s_sum[32];

    const int c   = blockIdx.x;   // stream
    const int t   = blockIdx.y;   // token
    const int hc  = gridDim.x;
    const int tid = threadIdx.x;

    // identical expressions to scale_f32 / op_sigmoid / scale_f32
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;

    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16  = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst    = out_res      + row * n_embd;
    uint16_t *       dst16  = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    block_out += (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    float xs[3];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        xs[k] = 0.0f;
        if (col < n_embd) {
            const float m  = hc_mul_rn(blk16 ? hc_bf2f32(blk16[col]) : block_out[col], w);
            const float xi = hc_add_rn(res16 ? hc_bf2f32(res16[col]) : res[col], m);
            if (dst16) dst16[col] = hc_f2bf32(xi); else dst[col] = xi;
            xs[k]    = xi;
            tmp += xi * xi;
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK>(tmp, s_sum);

    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        if (col < n_embd) {
            const float v = scale * xs[k] * g[col];
            if (store_xn_f32) xn[col] = v;
            if (xh) { uint32_t u = __float_as_uint(v); u += 0x7fffu + ((u >> 16) & 1u); xh[col] = (uint16_t)(u >> 16); }
        }
    }
}

#define HC_CN_BLOCK2 256
__device__ __forceinline__ float hc_lo(const uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hc_hi(const uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t hc_pack2(const float a, const float b) {
    return (uint32_t) hc_f2bf32(a) | ((uint32_t) hc_f2bf32(b) << 16);
}

// two elements per thread per iteration, packed 32-bit accesses (Halogen's k_hc_scatter_norm layout)
static __global__ void __launch_bounds__(HC_CN_BLOCK2, 4) hc_combine_norm_f32_b256(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        block_q8_1_mmq * out_xn_q8,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ __align__(16) float s_sum[32];
    const int c = blockIdx.x, t = blockIdx.y, hc = gridDim.x, tid = threadIdx.x;
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;
    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst   = out_res      + row * n_embd;
    uint16_t *       dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    const float *    blk   = block_out + (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    constexpr int KP = (HC_CN_MAX_EMB / 2 + HC_CN_BLOCK2 - 1) / HC_CN_BLOCK2;
    float xs[2 * KP];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        xs[2 * k] = 0.0f; xs[2 * k + 1] = 0.0f;
        if (col + 1 < n_embd) {
            float r0, r1, m0, m1;
            if (res16) { const uint32_t u = *(const uint32_t *)(res16 + col); r0 = hc_lo(u); r1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(res + col);     r0 = v.x;      r1 = v.y; }
            if (blk16) { const uint32_t u = *(const uint32_t *)(blk16 + col); m0 = hc_lo(u); m1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(blk + col);     m0 = v.x;      m1 = v.y; }
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
            if (dst16) *(uint32_t *)(dst16 + col) = hc_pack2(a0, a1);
            else       *(float2 *)(dst + col)     = make_float2(a0, a1);
            xs[2 * k] = a0; xs[2 * k + 1] = a1;
            tmp += a0 * a0 + a1 * a1;
        } else if (col < n_embd) {
            const float r0 = res16 ? hc_bf2f32(res16[col]) : res[col];
            const float m0 = blk16 ? hc_bf2f32(blk16[col]) : blk[col];
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            if (dst16) dst16[col] = hc_f2bf32(a0); else dst[col] = a0;
            xs[2 * k] = a0;
            tmp += a0 * a0;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK2>(tmp, s_sum);
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        if (col + 1 < n_embd) {
            const float2 gv = *(const float2 *)(g + col);
            const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
            if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
            if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
            if (out_xn_q8) {
                // same arithmetic as quantize_mmq_q8_1 (D4 layout): one scale per 32 values, held by 16 lanes
                float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
                }
                const float d_inv = 127.0f / amax;
                char2 q;
                q.x = roundf(v0 * d_inv);
                q.y = roundf(v1 * d_inv);
                // gather 16 consecutive int8 values into every 8th lane so the copy is written with
                // 16-byte stores instead of 2-byte ones (all lanes take part: n_embd % 512 == 0)
                const uint32_t b2 = (uint32_t) (uint8_t) q.x | ((uint32_t) (uint8_t) q.y << 8);
                const uint32_t w0 = b2 | (__shfl_xor_sync(0xFFFFFFFF, b2, 1, WARP_SIZE) << 16);
                const uint32_t w1 = __shfl_xor_sync(0xFFFFFFFF, w0, 2, WARP_SIZE);
                const uint32_t w2 = __shfl_xor_sync(0xFFFFFFFF, w0, 4, WARP_SIZE);
                const uint32_t w3 = __shfl_xor_sync(0xFFFFFFFF, w1, 4, WARP_SIZE);
                const int64_t kk = (int64_t) c * n_embd + col;
                block_q8_1_mmq & yb = out_xn_q8[(kk / QK8_1_MMQ) * gridDim.y + t];
                const int iqs = kk % QK8_1_MMQ;
                if (iqs % 16 == 0) {
                    *(uint4 *) &yb.qs[iqs] = make_uint4(w0, w1, w2, w3);
                }
                if (iqs % 32 == 0) {
                    yb.d4[iqs / 32] = 1.0f / d_inv;
                }
            }
        } else if (col < n_embd) {
            const float v0 = scale * xs[2 * k] * g[col];
            if (store_xn_f32) xn[col] = v0;
            if (xh) xh[col] = hc_f2bf32(v0);
        }
    }
}

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & a, const int warp_size) {
    const int64_t n_embd = a.out_res->ne[0];
    const int64_t hc     = a.out_res->ne[1];
    return warp_size == 32 && n_embd >= 1024 && n_embd <= HC_CN_MAX_EMB && hc >= 1 && hc <= 16 &&
        ggml_nelements(a.gamma) == n_embd * hc;
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a) {
    const int64_t n_embd   = a.out_res->ne[0];
    const int64_t hc       = a.out_res->ne[1];
    const int64_t n_tokens = a.out_res->ne[2] * a.out_res->ne[3];

    GGML_ASSERT(a.out_res->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(a.out_res) == ggml_nelements(a.out_xn) && ggml_are_same_shape(a.out_res, a.residual));
    GGML_ASSERT(ggml_nelements(a.block_out) == n_embd * n_tokens && ggml_nelements(a.gamma) == n_embd * hc &&
                ggml_nelements(a.inject) == hc * n_tokens);

    {
        const ggml_cuda_kernel_launch_params lp2(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK2, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_combine_norm_f32_b256, lp2,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
            a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16, (block_q8_1_mmq *) a.out_xn_q8,
            (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps);
        return;
    }
    const ggml_cuda_kernel_launch_params launch_params(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32, launch_params,
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
        a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
        (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps);
}
