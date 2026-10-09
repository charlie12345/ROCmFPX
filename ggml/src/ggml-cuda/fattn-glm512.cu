#include "fattn-glm512.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"

#include <cstring>

// One block per query token and all 64 heads. Warp w owns heads 16*(w/4).. and latent dims 128*(w%4).., keeping its
// Q slice and its O slice in registers. Per block of 16 keys: the K rows are staged in LDS (plus a transposed copy
// for P*V, since V is K), each warp computes a partial 16x16 score tile over its 128 dims, the four partials of a
// head group are summed by the softmax threads, and each warp then adds P*V for its dims. Key blocks whose 16 mask
// entries are all -inf are skipped before any K is loaded: the DSA selection keeps whole pools, so at depth most
// blocks are invisible to a query.

#define G5_D   512
#define G5_H   64
#define G5_KB  16
#define G5_NW  16
#define G5_NT  (G5_NW*32)
#define G5_MIN_TOKENS 64

namespace {

#if defined(GGML_USE_HIP) && defined(RDNA3)
typedef _Float16 g5_v16h __attribute__((ext_vector_type(16)));
typedef float    g5_v8f  __attribute__((ext_vector_type(8)));

// 16 halves as two 16-byte LDS reads (rows are only 16-byte aligned)
static __device__ __forceinline__ g5_v16h g5_ld16(const _Float16 * p) {
    const uint4 a = *(const uint4 *) p;
    const uint4 b = *(const uint4 *) (p + 8);
    return __builtin_bit_cast(g5_v16h, (uint4[2]){a, b});
}
#endif

// partial: rows of D+2 values per head, the unnormalized output then (max, sum), as FLASH_ATTN_EXT_PARTIAL defines
template <bool partial>
__launch_bounds__(G5_NT, 1)
__global__ void flash_attn_glm512(
        const float * __restrict__ Q, const half * __restrict__ K, const half * __restrict__ mask, float * __restrict__ dst,
        const int n_kv, const int64_t sq_tok, const int64_t sq_head, const int64_t sk, const int64_t sm, const float scale,
        const int32_t * __restrict__ lists = nullptr, const int32_t * __restrict__ counts = nullptr, const int n_kv_max = 0) {
#if defined(GGML_USE_HIP) && defined(RDNA3)
    __shared__ __align__(16) _Float16 Ks[G5_KB][G5_D + 8];   // key rows
    __shared__ __align__(16) _Float16 Kt[G5_D][G5_KB];       // the same rows transposed (built in the softmax phase), B of P*V
    __shared__ float Sp[4][G5_H][G5_KB + 1];                  // partial scores, one per dim quarter
    __shared__ __align__(16) _Float16 Ps[G5_H][G5_KB];        // probabilities
    __shared__ float m_s[G5_H];
    __shared__ float l_s[G5_H];
    __shared__ float a_s[G5_H];

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int w    = tid / 32;
    const int lane = tid % 32;
    const int r    = lane % 16;
    const int hi   = lane / 16;
    const int g    = w / 4;     // head group: heads 16*g..16*g+15
    const int q    = w % 4;     // dim quarter: dims 128*q..128*q+127

    if (tid < G5_H) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }

    // Q slice in registers: 8 k-steps of 16 dims, lane r holds head 16*g + r (lanes 16..31 repeat lanes 0..15)
    g5_v16h qa[8];
    {
        const float * Qh = Q + t*sq_tok + (16*g + r)*sq_head + 128*q;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                qa[s][e] = (_Float16) (Qh[16*s + e]*scale);
            }
        }
    }

    g5_v8f o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            o[j][e] = 0.0f;
        }
    }

    const half * mrow = mask + t*sm;
    // list mode: walk this token's selected cells (a sorted list) instead of every key block
    const int32_t * lst = lists ? lists + (int64_t) t*n_kv_max : nullptr;
    const int n_iter = lists ? counts[t] : n_kv;
    auto mask_at = [&](const int kk) -> float {
        if (lists) {
            return kk < n_iter ? __half2float(mrow[lst[kk]]) : -INFINITY;
        }
        return __half2float(mrow[kk]);
    };
    __syncthreads();

    for (int kb = 0; kb < n_iter; kb += G5_KB) {
        // every warp sees the same 16 mask values, so the skip is uniform without a barrier
        if (!lists) {
            const float mv = __half2float(mrow[kb + r]);
            if (__ballot(mv != -INFINITY) == 0) {
                continue;
            }
        }

        // stage 16 key rows: thread t copies 16 halves of row t/32
        {
            const int row = tid / 32;
            const int col = (tid % 32)*16;
            const int idx = lists ? (kb + row < n_iter ? lst[kb + row] : -1) : kb + row;
            if (idx >= 0) {
                const uint4 * src = (const uint4 *) (K + (int64_t) idx*sk + col);
                *(uint4 *) &Ks[row][col]     = src[0];
                *(uint4 *) &Ks[row][col + 8] = src[1];
            } else {
                *(uint4 *) &Ks[row][col]     = make_uint4(0, 0, 0, 0);
                *(uint4 *) &Ks[row][col + 8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // partial scores over this warp's 128 dims: rows = heads, columns = keys
        {
            g5_v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[e] = 0.0f;
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const g5_v16h b = g5_ld16(&Ks[r][128*q + 16*s]);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[s], b, acc);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                Sp[q][16*g + 2*e + hi][r] = acc[e];
            }
        }
        __syncthreads();

        // transposed copy for P*V (Ks is not written again until the next block): thread t builds dim t
        {
            __align__(16) _Float16 col[G5_KB];
#pragma unroll
            for (int k = 0; k < G5_KB; ++k) {
                col[k] = Ks[k][tid];
            }
            *(uint4 *) &Kt[tid][0] = *(const uint4 *) &col[0];
            *(uint4 *) &Kt[tid][8] = *(const uint4 *) &col[8];
        }

        // online softmax: 8 threads per head, 2 keys each
        {
            const int h  = tid / 8;
            const int k0 = (tid % 8)*2;
            float s0 = Sp[0][h][k0]     + Sp[1][h][k0]     + Sp[2][h][k0]     + Sp[3][h][k0];
            float s1 = Sp[0][h][k0 + 1] + Sp[1][h][k0 + 1] + Sp[2][h][k0 + 1] + Sp[3][h][k0 + 1];
            s0 += mask_at(kb + k0);
            s1 += mask_at(kb + k0 + 1);
            float bmax = fmaxf(s0, s1);
#pragma unroll
            for (int off = 1; off < 8; off *= 2) {
                bmax = fmaxf(bmax, __shfl_xor(bmax, off, 32));
            }
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, bmax);
            // the normalizer sums the same f16 values that P*V accumulates
            const _Float16 p0 = (_Float16) (m_new == -INFINITY ? 0.0f : expf(s0 - m_new));
            const _Float16 p1 = (_Float16) (m_new == -INFINITY ? 0.0f : expf(s1 - m_new));
            float psum = (float) p0 + (float) p1;
#pragma unroll
            for (int off = 1; off < 8; off *= 2) {
                psum += __shfl_xor(psum, off, 32);
            }
            Ps[h][k0]     = p0;
            Ps[h][k0 + 1] = p1;
            if (tid % 8 == 0) {
                const float alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                a_s[h] = alpha;
                l_s[h] = alpha*l_s[h] + psum;
                m_s[h] = m_new;
            }
        }
        __syncthreads();

        // O = alpha*O + P*V over this warp's 128 dims
        {
            float alpha[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                alpha[e] = a_s[16*g + 2*e + hi];
            }
            const g5_v16h pa = g5_ld16(&Ps[16*g + r][0]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o[j][e] *= alpha[e];
                }
                const g5_v16h b = g5_ld16(&Kt[128*q + 16*j + r][0]);
                o[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[j]);
            }
        }
        __syncthreads();
    }

    // store: dst is [512 (+2 when partial), 64 heads, tokens]
    constexpr int row = partial ? G5_D + 2 : G5_D;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int h = 16*g + 2*e + hi;
        const float l = l_s[h];
        const float inv = partial ? 1.0f : (l > 0.0f ? 1.0f/l : 0.0f);
        float * d = dst + ((int64_t) t*G5_H + h)*row + 128*q + r;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            d[16*j] = o[j][e]*inv;
        }
        if (partial && q == 0 && r == 0) {
            dst[((int64_t) t*G5_H + h)*row + G5_D]     = m_s[h];
            dst[((int64_t) t*G5_H + h)*row + G5_D + 1] = l;
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, mask, dst, n_kv, sq_tok, sq_head, sk, sm, scale);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA3)
}

// Two query tokens per block (128 rows: token-major, then head), 32 warps: warp w owns rows 16*(w/4).. and dims
// 128*(w%4)... Each staged key tile now serves both tokens. The four dim-quarter partial scores stay in LDS (summed
// in a fixed order, so results do not depend on scheduling); to make room, P*V reads its B operand straight from the
// row-major key tile instead of a transposed copy. A key block is skipped only when both tokens mask all of it.
#define G5_QT  2
#define G5_NW2 32
#define G5_NT2 (G5_NW2*32)
#define G5_R2  (G5_QT*G5_H)

template <bool partial>
__launch_bounds__(G5_NT2, 1)
__global__ void flash_attn_glm512_x2(
        const float * __restrict__ Q, const half * __restrict__ K, const half * __restrict__ mask, float * __restrict__ dst,
        const int n_q, const int n_kv, const int64_t sq_tok, const int64_t sq_head, const int64_t sk, const int64_t sm,
        const float scale) {
#if defined(GGML_USE_HIP) && defined(RDNA3)
    __shared__ __align__(16) _Float16 Ks[G5_KB][G5_D + 8];   // key rows
    __shared__ float Sp[4][G5_R2][G5_KB + 1];                 // partial scores, one per dim quarter
    __shared__ __align__(16) _Float16 Ps[G5_R2][G5_KB];       // probabilities
    __shared__ float m_s[G5_R2];
    __shared__ float l_s[G5_R2];
    __shared__ float a_s[G5_R2];

    const int t0   = blockIdx.x * G5_QT;
    const int tid  = threadIdx.x;
    const int w    = tid / 32;
    const int lane = tid % 32;
    const int r    = lane % 16;
    const int hi   = lane / 16;
    const int g    = w / 4;              // row group: rows 16*g..16*g+15
    const int q    = w % 4;              // dim quarter: dims 128*q..128*q+127
    const int tt   = g / 4;              // token of this row group
    const int hb   = 16 * (g % 4);       // its first head
    const int t    = min(t0 + tt, n_q - 1);

    if (tid < G5_R2) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }

    g5_v16h qa[8];
    {
        const float * Qh = Q + t*sq_tok + (hb + r)*sq_head + 128*q;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                qa[s][e] = (_Float16) (Qh[16*s + e]*scale);
            }
        }
    }

    g5_v8f o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            o[j][e] = 0.0f;
        }
    }

    const half * mrow0 = mask + (int64_t) t0*sm;
    const half * mrow1 = mask + (int64_t) min(t0 + 1, n_q - 1)*sm;
    __syncthreads();

    for (int kb = 0; kb < n_kv; kb += G5_KB) {
        // uniform skip: every warp sees the same 32 mask values
        const float mv0 = __half2float(mrow0[kb + r]);
        const float mv1 = __half2float(mrow1[kb + r]);
        if (__ballot(mv0 != -INFINITY || mv1 != -INFINITY) == 0) {
            continue;
        }

        // stage 16 key rows: thread t copies 8 halves of row t/64
        {
            const int row = tid / 64;
            const int col = (tid % 64)*8;
            *(uint4 *) &Ks[row][col] = *(const uint4 *) (K + (int64_t) (kb + row)*sk + col);
        }
        __syncthreads();

        {
            g5_v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[e] = 0.0f;
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[s], g5_ld16(&Ks[r][128*q + 16*s]), acc);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                Sp[q][16*g + 2*e + hi][r] = acc[e];
            }
        }
        __syncthreads();

        // online softmax: 8 threads per row, 2 keys each
        {
            const int h  = tid / 8;
            const int k0 = (tid % 8)*2;
            const half * mrow = h < G5_H ? mrow0 : mrow1;
            float s0 = Sp[0][h][k0]     + Sp[1][h][k0]     + Sp[2][h][k0]     + Sp[3][h][k0];
            float s1 = Sp[0][h][k0 + 1] + Sp[1][h][k0 + 1] + Sp[2][h][k0 + 1] + Sp[3][h][k0 + 1];
            s0 += __half2float(mrow[kb + k0]);
            s1 += __half2float(mrow[kb + k0 + 1]);
            float bmax = fmaxf(s0, s1);
#pragma unroll
            for (int off = 1; off < 8; off *= 2) {
                bmax = fmaxf(bmax, __shfl_xor(bmax, off, 32));
            }
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, bmax);
            const _Float16 p0 = (_Float16) (m_new == -INFINITY ? 0.0f : expf(s0 - m_new));
            const _Float16 p1 = (_Float16) (m_new == -INFINITY ? 0.0f : expf(s1 - m_new));
            float psum = (float) p0 + (float) p1;
#pragma unroll
            for (int off = 1; off < 8; off *= 2) {
                psum += __shfl_xor(psum, off, 32);
            }
            Ps[h][k0]     = p0;
            Ps[h][k0 + 1] = p1;
            if (tid % 8 == 0) {
                const float alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                a_s[h] = alpha;
                l_s[h] = alpha*l_s[h] + psum;
                m_s[h] = m_new;
            }
        }
        __syncthreads();

        // O = alpha*O + P*V over this warp's 128 dims; B gathered from the key rows (lane r: dim column r, 16 keys)
        {
            float alpha[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                alpha[e] = a_s[16*g + 2*e + hi];
            }
            const g5_v16h pa = g5_ld16(&Ps[16*g + r][0]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                g5_v16h b;
#pragma unroll
                for (int k = 0; k < G5_KB; ++k) {
                    b[k] = Ks[k][128*q + 16*j + r];
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o[j][e] *= alpha[e];
                }
                o[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[j]);
            }
        }
        __syncthreads();
    }

    if (t0 + tt >= n_q) {
        return;
    }
    constexpr int row = partial ? G5_D + 2 : G5_D;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int rr = 16*g + 2*e + hi;
        const int h  = hb + 2*e + hi;
        const float l = l_s[rr];
        const float inv = partial ? 1.0f : (l > 0.0f ? 1.0f/l : 0.0f);
        float * d = dst + ((int64_t) t*G5_H + h)*row + 128*q + r;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            d[16*j] = o[j][e]*inv;
        }
        if (partial && q == 0 && r == 0) {
            dst[((int64_t) t*G5_H + h)*row + G5_D]     = m_s[rr];
            dst[((int64_t) t*G5_H + h)*row + G5_D + 1] = l;
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, mask, dst, n_q, n_kv, sq_tok, sq_head, sk, sm, scale);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA3)
}

} // namespace

bool ggml_cuda_flash_attn_ext_glm512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool partial = dst->op == GGML_OP_FLASH_ATTN_EXT_PARTIAL;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || (dst->op != GGML_OP_FLASH_ATTN_EXT && !partial) || !mask || dst->src[4] || dst->src[5]) {
        return false;
    }
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // shape: 512-wide K-only latents (V is K), one KV head, 64 query heads, one sequence, enough queries to fill the GPU
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || mask->type != GGML_TYPE_F16 ||
            Q->ne[0] != G5_D || V->ne[0] != G5_D || K->ne[0] != G5_D || Q->ne[2] != G5_H || K->ne[2] != 1 ||
            Q->ne[3] != 1 || K->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 || Q->ne[1] < G5_MIN_TOKENS ||
            K->ne[1] % G5_KB != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            V->data != K->data || V->nb[1] != K->nb[1] || V->type != K->type || !ggml_is_contiguous(dst) ||
            Q->nb[0] != sizeof(float) || mask->nb[0] != sizeof(half)) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 && (!ggml_get_to_fp16_cuda(K->type) || K->nb[0] != ggml_type_size(K->type) ||
            K->nb[1] != ggml_row_size(K->type, K->ne[0]))) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const half * K_h = (const half *) K->data;
    int64_t sk = K->nb[1] / sizeof(half);
    ggml_cuda_pool_alloc<half> K_f16(ctx.pool());
    if (K->type != GGML_TYPE_F16) {
        // the node's allocation already holds an f16 copy of K for the tile kernel: convert into it when present
        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
        half * K_dst = nullptr;
        if (dst->buffer && ggml_backend_buffer_get_alloc_size(dst->buffer, dst) >= f16_extra.end - (uintptr_t) dst->data) {
            K_dst = (half *) f16_extra.K;
        } else {
            K_dst = K_f16.alloc(K->ne[1]*G5_D);
        }
        ggml_get_to_fp16_cuda(K->type)(K->data, K_dst, K->ne[1]*G5_D, stream);
        K_h = K_dst;
        sk  = G5_D;
    } else if (K->nb[0] != sizeof(half) || K->nb[1] % 16 != 0) {
        return false;
    }

    // two tokens per block while most keys are visible anyway; deep in the context the DSA selection is sparse and a
    // shared key block is skipped less often (GLM pp4096 ub4096: depth 0 368.7 -> 372.6 t/s, depth 16384 305.1 -> 302.2)
    // once the selection is at most half of the keys, each token walks its own sorted list on the one-token kernel:
    // the two-token kernel shares staged keys but skips only blocks masked for both tokens, and selected pools land
    // in most 16-key blocks (pp4096 at depth 4096: 405.5 -> 426.2, at 12288: 375.3 -> 421.2; depth 0 unchanged)
    const int32_t n_kv_max = partial ? 0 : ggml_get_op_params_i32(dst, 4);
    const bool use_lists = n_kv_max > 0 && 2*int64_t(n_kv_max) <= K->ne[1] && mask->ne[3] == 1;
    if (K->ne[1] <= 8192 && !use_lists) {
        const dim3 grid((unsigned) ((Q->ne[1] + G5_QT - 1) / G5_QT), 1, 1);
        const auto kernel = partial ? flash_attn_glm512_x2<true> : flash_attn_glm512_x2<false>;
        kernel<<<grid, dim3(G5_NT2, 1, 1), 0, stream>>>((const float *) Q->data, K_h, (const half *) mask->data, (float *) dst->data,
            (int) Q->ne[1], (int) K->ne[1], Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float), sk, mask->nb[1]/sizeof(half), scale);
    } else {
    const dim3 grid((unsigned) Q->ne[1], 1, 1);
    const dim3 block(G5_NT, 1, 1);
    const auto kernel = partial ? flash_attn_glm512<true> : flash_attn_glm512<false>;
    // a list visits only the selected rows; masked-block skipping visits every key block holding one selected pool
    ggml_cuda_pool_alloc<int32_t> lists(ctx.pool());
    if (use_lists) {
        const int64_t n_q = Q->ne[1];
        const int64_t n_scratch = int64_t(ggml_cuda_flash_attn_ext_compact_mask_scratch(mask))*n_q;
        lists.alloc(n_q*n_kv_max + n_q + n_scratch);
        int32_t * counts = lists.ptr + n_q*n_kv_max;
        ggml_cuda_flash_attn_ext_compact_mask(mask, lists.ptr, counts, (int) n_q, 1, n_kv_max, stream,
            n_scratch ? counts + n_q : nullptr);
        kernel<<<grid, block, 0, stream>>>((const float *) Q->data, K_h, (const half *) mask->data, (float *) dst->data,
            (int) K->ne[1], Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float), sk, mask->nb[1]/sizeof(half), scale,
            lists.ptr, counts, n_kv_max);
    } else {
        kernel<<<grid, block, 0, stream>>>((const float *) Q->data, K_h, (const half *) mask->data, (float *) dst->data,
            (int) K->ne[1], Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float), sk, mask->nb[1]/sizeof(half), scale,
            nullptr, nullptr, 0);
    }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
