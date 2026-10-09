#include "gated_delta_net.cuh"
#include "common.cuh"
#include "ggml-impl.h"

#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}
static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}
static __device__ __forceinline__ float gdn_warp_reduce_sum32(float x) {
    x += gdn_permlanex16_swap(x);
    x += gdn_dpp_row_xmask<8>(x);
    x += gdn_dpp_row_xmask<4>(x);
    x += gdn_dpp_row_xmask<2>(x);
    x += gdn_dpp_row_xmask<1>(x);
    return x;
}
#define GDN_DPP_REDUCE 1
#endif

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
#if defined(GDN_DPP_REDUCE)
    if constexpr (width == 32) {
        return gdn_warp_reduce_sum32(x);
    }
#endif
    return warp_reduce_sum<width>(x);
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_ids,
                                     int64_t       state_row) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    if (state_ids != nullptr) {
        // state row of this sequence in the recurrent cache instead of a gathered copy
        curr_state += (int64_t) state_ids[sequence] * state_row - (int64_t) sequence * H * S_v * S_v;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// KDA gates every state row by its own exp(g[i]); exp(g) is staged once per tile and shared by all columns
template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NUM_WARPS, 1)
gated_delta_net_tiled_cuda(const float * q,
                           const float * k,
                           const float * v,
                           const float * g,
                           const float * beta,
                           const float * curr_state,
                           float *       dst,
                           float *       state,
                           int64_t       H,
                           int64_t       n_tokens,
                           int64_t       sq1,
                           int64_t       sq2,
                           int64_t       sq3,
                           int64_t       sv1,
                           int64_t       sv2,
                           int64_t       sv3,
                           int64_t       sb1,
                           int64_t       sb2,
                           int64_t       sb3,
                           const uint3   neqk1_magic,
                           const uint3   rq3_magic,
                           float         scale,
                           int64_t       state_slot_stride,
                           int           K) {
    constexpr int warp_size     = 32;
    constexpr int rows_per_lane = S_v / warp_size;
    constexpr int block_cols    = NUM_WARPS * COLS;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of the warp size");
    static_assert(S_v % block_cols == 0, "block columns must divide S_v");

    __shared__ float q_shared[TOKEN_TILE][S_v];
    __shared__ float k_shared[TOKEN_TILE][S_v];
    __shared__ float v_shared[TOKEN_TILE][block_cols];
    __shared__ float g_shared[TOKEN_TILE * (KDA ? S_v : 1)];
    __shared__ float beta_shared[TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = blockIdx.z * block_cols;          // first column of the block
    const int      colw     = threadIdx.y * COLS;               // first column of this warp inside the block
    const int      thread   = threadIdx.y * warp_size + lane;
    constexpr int  nthreads = NUM_WARPS * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + (col0 + colw) * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colw;

    float s_shard[COLS][rows_per_lane];

#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);

        for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
            const int tt = idx / S_v;
            const int i  = idx % S_v;
            const int t  = t0 + tt;
            q_shared[tt][i] = q[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            k_shared[tt][i] = k[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
        }
        for (int idx = thread; idx < tile_size * block_cols; idx += nthreads) {
            const int tt = idx / block_cols;
            const int c  = idx % block_cols;
            const int t  = t0 + tt;
            v_shared[tt][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col0 + c];
        }
        if (thread < tile_size) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            if constexpr (!KDA) {
                g_shared[thread] = g[gb_offset];
            }
            beta_shared[thread] = beta[gb_offset];
        }
        if constexpr (KDA) {
            for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
                const int tt = idx / S_v;
                const int64_t gb_offset = sequence * sb3 + (t0 + tt) * sb2 + h_idx * sb1;
                g_shared[idx] = expf(g[gb_offset * S_v + idx % S_v]);
            }
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
                g_reg[r] = KDA ? g_shared[tt * S_v + i] : 0.0f;
            }

            const float g_val    = KDA ? 1.0f : expf(g_shared[tt]);
            const float beta_val = beta_shared[tt];

            float attn_col[COLS];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                // KDA: kv[col] = sum_i g[i] * S[i][col] * k[i], delta[col] = (v[col] - kv[col]) * beta
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard = fmaf(KDA ? g_reg[r] * s_shard[c][r] : s_shard[c][r], k_reg[r], kv_shard);
                }
                const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = fmaf(-g_val, kv_col, v_shared[tt][colw + c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(KDA ? g_reg[r] : g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
                }
                attn_col[c] = gdn_warp_reduce_sum<warp_size>(attn_partial);
            }

            if (lane < COLS) {
                float a = attn_col[0];
#pragma unroll
                for (int c = 1; c < COLS; c++) {
                    a = lane == c ? attn_col[c] : a;
                }
                attn_data[(int64_t) t * S_v * H + lane] = a * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride + (col0 + colw) * S_v;
#pragma unroll
                    for (int c = 0; c < COLS; c++) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            snapshot[c * S_v + r * warp_size + lane] = s_shard[c][r];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + colw + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define GDN_CHUNK_WMMA 1
typedef _Float16 gdn_v16h __attribute__((ext_vector_type(16)));
typedef float    gdn_v8f  __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ gdn_v16h gdn_ld16(const _Float16 * p) {
    const uint4 a = *(const uint4 *) p;
    const uint4 b = *(const uint4 *) (p + 8);
    return __builtin_bit_cast(gdn_v16h, (uint4[2]){a, b});
}

// LDS barrier that leaves global loads in flight (__syncthreads also waits for them)
static __device__ __forceinline__ void gdn_lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// WMMA accumulator (lane r, half hi holds rows 2e+hi of column r) -> B operand (lane r holds rows 0..15 of column r)
static __device__ __forceinline__ gdn_v16h gdn_acc_to_b(const gdn_v8f acc, const int hi) {
    gdn_v16h b;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float other = gdn_permlanex16_swap(acc[e]);
        b[2*e]     = (_Float16) (hi ? other  : acc[e]);
        b[2*e + 1] = (_Float16) (hi ? acc[e] : other);
    }
    return b;
}

#endif // defined(GGML_USE_HIP) && defined(RDNA3)

// Chunked gated delta rule (scalar gate, 128x128 state) for prefill with RDNA3 WMMA, using the chunkwise WY/UT form
// of Yang et al., "Gated Delta Networks" (2024), as flash-linear-attention computes it. For a chunk of 16 tokens with
// in-chunk cumulative log gate b and G = exp(b):
//   A = strict_lower(beta_t exp(b_t - b_s) k_t.k_s),  T = (I + A)^-1,  P = lower(exp(b_t - b_s) q_t.k_s)
//   D = T (beta v - beta G (K S)),  O = scale (G (Q S) + P D),  S' = G_last S + (exp(b_last - b) k)^T D
// T, P and b need no state, so gdn_chunk_prep builds them for every chunk at once. The scan then walks the chunks of
// one head with 8 warps; D, O and S' are column-wise in the value dim, so warp w keeps value columns 16w..16w+15 of S
// as fp32 WMMA accumulators.
#define GDN_CHUNK      16
#define GDN_CHUNK_REC  1152   // per chunk and value head: T, P (f16 16x16), b, beta (f32 x16)
#define GDN_CHUNK_QK   8192   // per chunk and K head: q then k rows, f16 16x128

static __global__ void __launch_bounds__(128)
gdn_chunk_prep(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ g,
               const float * __restrict__ beta, char * __restrict__ qkrec, char * __restrict__ rec,
               const int64_t H, const int64_t n_kh, const int64_t n_tokens, const int64_t n_chunks,
               const int64_t sq1, const int64_t sq2, const int64_t sb1, const int64_t sb2) {
#if defined(GDN_CHUNK_WMMA)
    constexpr int S = 128, C = GDN_CHUNK;
    __shared__ __align__(16) _Float16 Qs[4][C][S + 8];
    __shared__ __align__(16) _Float16 Ks[4][C][S + 8];
    __shared__ float As[4][C][C + 1];
    __shared__ float bs[4][C];
    __shared__ float betas[4][C];

    const int w    = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r    = lane % 16;
    const int hi   = lane / 16;
    // K heads fastest, so neighbouring blocks read the same token rows; warp w takes one chunk
    const int     kh = blockIdx.x;
    const int64_t c  = (int64_t) blockIdx.y * 4 + w;
    if (c >= n_chunks) {
        return;   // only wave-level syncs below
    }
    const int64_t t0 = c * C;
    const int nt = (int) min((int64_t) C, n_tokens - t0);

    // q, k to f16 in LDS and, chunk-contiguous, for the scan
    {
        const float * qb = q + kh * sq1;
        const float * kb = k + kh * sq1;
        uint2 * qkg = (uint2 *) (qkrec + (kh * n_chunks + c) * GDN_CHUNK_QK);
#pragma unroll
        for (int u = 0; u < C * S / 4 / 32; ++u) {
            const int idx = lane + 32 * u;
            const int t   = idx / (S / 4);
            const int i4  = (idx % (S / 4)) * 4;
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 kv = qv;
            if (t < nt) {
                qv = *(const float4 *) (qb + (t0 + t) * sq2 + i4);
                kv = *(const float4 *) (kb + (t0 + t) * sq2 + i4);
            }
            _Float16 qh[4] = { (_Float16) qv.x, (_Float16) qv.y, (_Float16) qv.z, (_Float16) qv.w };
            _Float16 kh4[4] = { (_Float16) kv.x, (_Float16) kv.y, (_Float16) kv.z, (_Float16) kv.w };
            const uint2 qp = *(const uint2 *) qh;
            const uint2 kp = *(const uint2 *) kh4;
            *(uint2 *) &Qs[w][t][i4] = qp;
            *(uint2 *) &Ks[w][t][i4] = kp;
            qkg[idx]               = qp;
            qkg[C * S / 4 + idx]   = kp;
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

    gdn_v8f kk = {}, qk = {};
#pragma unroll
    for (int s = 0; s < S / 16; ++s) {
        const gdn_v16h kr = gdn_ld16(&Ks[w][r][16 * s]);
        kk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kr, kr, kk);
        qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Qs[w][r][16 * s]), kr, qk);
    }

    // the value heads sharing this K head differ only in their gates
    for (int h = kh; h < H; h += (int) n_kh) {
        // inclusive in-chunk cumsum of the gate; padded tokens get g = beta = 0, so they change nothing
        float gv = lane < nt ? g[h * sb1 + (t0 + lane) * sb2] : 0.0f;
        const float bv = lane < nt ? beta[h * sb1 + (t0 + lane) * sb2] : 0.0f;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) {
            const float y = __shfl_up(gv, o, 32);
            if (r >= o) {
                gv += y;
            }
        }
        if (lane < C) {
            bs[w][lane]    = gv;
            betas[w][lane] = bv;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        _Float16 * Tg = (_Float16 *) (rec + (h * n_chunks + c) * GDN_CHUNK_REC);
        _Float16 * Pg = Tg + C * C;
        float    * bg = (float *) (Pg + C * C);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = 2 * e + hi;
            const int s = r;
            const float d = expf(fminf(bs[w][t] - bs[w][s], 0.0f));
            As[w][t][s]   = s < t ? betas[w][t] * d * kk[e] : 0.0f;
            Pg[t * C + s] = (_Float16) (s <= t ? d * qk[e] : 0.0f);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // T = (I + A)^-1 by forward substitution, lane r builds column r
        if (hi == 0) {
            float x[C];
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float a = i == r ? 1.0f : 0.0f;
#pragma unroll
                for (int s = 0; s < i; ++s) {
                    a = fmaf(-As[w][i][s], x[s], a);
                }
                x[i] = a;
                Tg[i * C + r] = (_Float16) a;
            }
            bg[r] = bs[w][r];
        } else {
            bg[C + r] = betas[w][r];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
#else
    GGML_UNUSED_VARS(q, k, g, beta, qkrec, rec, H, n_kh, n_tokens, n_chunks, sq1, sq2, sb1, sb2);
    NO_DEVICE_CODE;
#endif // defined(GDN_CHUNK_WMMA)
}

// KDA gates every key channel on its own, so exp(b_t - b_s) sits inside the K K^T and Q K^T sums. They are formed
// exactly in fp32 (a factored exp(b_t) exp(-b_s) overflows for sharp gates); q, k and the transposed k leave
// already scaled, so the scan needs no exp. One block of 4 warps per (value head, chunk).
#define GDN_CHUNK_REC_KDA 13888   // Qg, Kg (f16 16x128), Kt (f16 128x16), T, P (f16 16x16), G_last (f32 x128), beta (f32 x16)

static __global__ void __launch_bounds__(128)
gdn_chunk_prep_kda(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ g,
                   const float * __restrict__ beta, char * __restrict__ rec, const int64_t n_tokens,
                   const int64_t n_chunks, const int64_t sq1, const int64_t sq2, const int64_t sb1, const int64_t sb2,
                   const uint3 neqk1_magic) {
#if defined(GDN_CHUNK_WMMA)
    constexpr int S = 128, C = GDN_CHUNK, NT = 128;
    // rows padded by 4 floats: the pair loop reads 16 different rows at one offset, which would all hit one bank
    __shared__ __align__(16) float qf[C][S + 4];
    __shared__ __align__(16) float kf[C][S + 4];
    __shared__ __align__(16) float bf[C][S + 4];   // in-chunk cumulative gate per channel
    __shared__ float As[C][C + 1];
    __shared__ float betas[C];

    const int tid = threadIdx.x;
    const int h   = blockIdx.x;
    const int64_t c  = blockIdx.y;
    const int64_t t0 = c * C;
    const int nt = (int) min((int64_t) C, n_tokens - t0);
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const float * qb = q + iq1 * sq1;
    const float * kb = k + iq1 * sq1;

    // thread i owns channel i; padded tokens get q = k = g = beta = 0
    {
        float bacc = 0.0f;
#pragma unroll
        for (int t = 0; t < C; ++t) {
            const bool ok = t < nt;
            qf[t][tid] = ok ? qb[(t0 + t) * sq2 + tid] : 0.0f;
            kf[t][tid] = ok ? kb[(t0 + t) * sq2 + tid] : 0.0f;
            bacc += ok ? g[(h * sb1 + (t0 + t) * sb2) * S + tid] : 0.0f;
            bf[t][tid] = bacc;
        }
        if (tid < C) {
            betas[tid] = tid < nt ? beta[h * sb1 + (t0 + tid) * sb2] : 0.0f;
        }
    }
    __syncthreads();

    char * rc = rec + (h * n_chunks + c) * GDN_CHUNK_REC_KDA;
    _Float16 * Qg = (_Float16 *) rc;
    _Float16 * Kg = Qg + C * S;
    _Float16 * Ktg = Kg + C * S;
    _Float16 * Tg = Ktg + S * C;
    _Float16 * Pg = Tg + C * C;
    float    * Gl = (float *) (Pg + C * C);
    float    * bg = Gl + S;

    // pairs s <= t of the lower triangle, row by row: P over s <= t, A over s < t
    for (int p = tid; p < C * (C + 1) / 2; p += NT) {
        int t = 0;
        while ((t + 1) * (t + 2) / 2 <= p) {
            ++t;
        }
        const int s = p - t * (t + 1) / 2;
        float aq = 0.0f, ak = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; i += 4) {
            const float4 bt = *(const float4 *) &bf[t][i];
            const float4 bs = *(const float4 *) &bf[s][i];
            const float4 qt = *(const float4 *) &qf[t][i];
            const float4 kt = *(const float4 *) &kf[t][i];
            const float4 ks = *(const float4 *) &kf[s][i];
            // exponents are <= 0 here; __expf's ulp-level error is far below the f16 the results go to
            const float ex = __expf(bt.x - bs.x) * ks.x, ey = __expf(bt.y - bs.y) * ks.y;
            const float ez = __expf(bt.z - bs.z) * ks.z, ew = __expf(bt.w - bs.w) * ks.w;
            aq = fmaf(qt.x, ex, fmaf(qt.y, ey, fmaf(qt.z, ez, fmaf(qt.w, ew, aq))));
            ak = fmaf(kt.x, ex, fmaf(kt.y, ey, fmaf(kt.z, ez, fmaf(kt.w, ew, ak))));
        }
        Pg[t * C + s] = (_Float16) aq;
        As[t][s]      = s < t ? betas[t] * ak : 0.0f;
        if (s < t) {
            Pg[s * C + t] = (_Float16) 0.0f;
        }
    }

    // scaled q, k and the transposed k decayed to the chunk end; thread i keeps channel i
    {
        const float bl = bf[C - 1][tid];
        __align__(16) _Float16 kt[C];   // this channel's row of Kt is contiguous
#pragma unroll
        for (int t = 0; t < C; ++t) {
            const float b  = bf[t][tid];
            const float eb = __expf(b);
            Qg[t * S + tid] = (_Float16) (qf[t][tid] * eb);
            Kg[t * S + tid] = (_Float16) (kf[t][tid] * eb);
            kt[t] = (_Float16) (kf[t][tid] * __expf(bl - b));
        }
        ((uint4 *) (Ktg + tid * C))[0] = ((const uint4 *) kt)[0];
        ((uint4 *) (Ktg + tid * C))[1] = ((const uint4 *) kt)[1];
        Gl[tid] = __expf(bl);
        if (tid < C) {
            bg[tid] = betas[tid];
        }
    }
    __syncthreads();

    // T = (I + A)^-1 by forward substitution, thread r < 16 builds column r
    if (tid < C) {
        float x[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            float a = i == tid ? 1.0f : 0.0f;
#pragma unroll
            for (int s = 0; s < i; ++s) {
                a = fmaf(-As[i][s], x[s], a);
            }
            x[i] = a;
            Tg[i * C + tid] = (_Float16) a;
        }
    }
#else
    GGML_UNUSED_VARS(q, k, g, beta, rec, n_tokens, n_chunks, sq1, sq2, sb1, sb2, neqk1_magic);
    NO_DEVICE_CODE;
#endif // defined(GDN_CHUNK_WMMA)
}

template <bool KDA>
__global__ void __launch_bounds__(256, 1)
gated_delta_net_chunked_cuda(const char * __restrict__ qkrec, const float * __restrict__ v,
                             const char * __restrict__ rec, const float * __restrict__ curr_state,
                             float * __restrict__ dst, float * __restrict__ state,
                             const int64_t H, const int64_t n_tokens, const int64_t n_chunks,
                             const int64_t sv1, const int64_t sv2, const int64_t sv3,
                             const uint3 neqk1_magic, const float scale) {
#if defined(GDN_CHUNK_WMMA)
    constexpr int S = 128, C = GDN_CHUNK, NT = 256;
    static_assert(C * S / 8 == NT, "one 8-half granule of q and of k per thread");

    // double buffered: chunk c+1 is staged while slow warps may still read chunk c
    __shared__ __align__(16) _Float16 Qh[2][C][S + 8];
    __shared__ __align__(16) _Float16 Kh[2][C][S + 8];
    __shared__ __align__(16) _Float16 Kt[2][S][C + 8];   // exp(b_last - b_t) k_t, transposed
    __shared__ __align__(16) _Float16 Th[2][C][C + 8];
    __shared__ __align__(16) _Float16 Ph[2][C][C + 8];
    __shared__ float G_s[2][C];
    __shared__ float beta_s[2][C];
    __shared__ float Gl_s[2][KDA ? S : 1];   // KDA: per-channel decay over the chunk

    const int h    = blockIdx.x;
    const int seq  = blockIdx.y;
    const int tid  = threadIdx.x;
    const int w    = tid / 32;
    const int lane = tid % 32;
    const int r    = lane % 16;
    const int hi   = lane / 16;
    const int col0 = 16 * w;

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const char  * qkb = qkrec + iq1 * n_chunks * GDN_CHUNK_QK;
    const float * vb = v + seq * sv3 + h * sv1 + col0 + r;
    constexpr int REC = KDA ? GDN_CHUNK_REC_KDA : GDN_CHUNK_REC;
    const char  * rb = rec + (seq * H + h) * n_chunks * REC;
    float * attn = dst + (seq * n_tokens * H + h) * S + col0 + r;

    // S[i][col] lives at col*S + i; tile j holds rows 16j..16j+15
    gdn_v8f Sacc[8];
    {
        const float * s_in = curr_state + (seq * H + h) * S * S + (col0 + r) * S;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                Sacc[j][e] = s_in[16 * j + 2 * e + hi];
            }
        }
    }

    // the next chunk is loaded into registers while this one computes: the chunks of a head run one after another,
    // so loading at the point of use would expose the whole global latency every chunk
    const int gt = tid / (S / 8);         // token and first dim of this thread's q/k granule
    const int gi = (tid % (S / 8)) * 8;
    uint4    qn, kn, ktn;
    float    bt, bl;           // b of the granule's token and of the chunk end
    float    gln;              // KDA: G_last of channel tid
    float    vn[8];
    uint32_t tp;               // two f16 of T or P
    float    gb;               // b (lanes 0..15) or beta (16..31) for warp 0
    auto fetch = [&](const int64_t c) {
        const int64_t c0 = c * C;
        const int ntn = (int) min((int64_t) C, n_tokens - c0);
        const char * rc = rb + c * REC;
        if constexpr (KDA) {
            // Qg, Kg, Kt, then T and P, G_last and beta
            const uint4 * r4 = (const uint4 *) rc;
            qn  = r4[tid];
            kn  = r4[NT + tid];
            ktn = r4[2 * NT + tid];
            const float * gl = (const float *) (rc + 3 * C * S * sizeof(uint16_t) + 2 * C * C * sizeof(uint16_t));
            gln = tid < S ? gl[tid] : 0.0f;
            gb  = w == 0 && lane < C ? gl[S + lane] : 0.0f;
            tp  = ((const uint32_t *) (rc + 3 * C * S * sizeof(uint16_t)))[tid];
        } else {
        const float * bg = (const float *) (rc + 2 * C * C * sizeof(uint16_t));
        const uint4 * qk = (const uint4 *) (qkb + c * GDN_CHUNK_QK);
        qn = qk[tid];
        kn = qk[NT + tid];
        bt = bg[gt];
        bl = bg[C - 1];
        tp = ((const uint32_t *) rc)[tid];
        gb = w == 0 ? bg[lane] : 0.0f;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = 2 * e + hi;
            vn[e] = t < ntn ? vb[(c0 + t) * sv2] : 0.0f;
        }
    };
    fetch(0);

    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t t0 = c * C;
        const int nt  = (int) min((int64_t) C, n_tokens - t0);
        const int buf = c & 1;

        // stage this chunk from the registers
        *(uint4 *) &Qh[buf][gt][gi] = qn;
        *(uint4 *) &Kh[buf][gt][gi] = kn;
        if constexpr (KDA) {
            *(uint4 *) &Kt[buf][tid / 2][(tid % 2) * 8] = ktn;
            if (tid < S) {
                Gl_s[buf][tid] = gln;
            }
        } else {
            const _Float16 * kv = (const _Float16 *) &kn;
            const float d = expf(bl - bt);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                Kt[buf][gi + m][gt] = (_Float16) (d * (float) kv[m]);
            }
        }
        {
            // tid < 128: T, else P; each thread holds two f16
            const int e2 = (tid % 128) * 2;
            _Float16 (*dstm)[C + 8] = tid < 128 ? Th[buf] : Ph[buf];
            *(uint32_t *) &dstm[e2 / C][e2 % C] = tp;
        }
        if (w == 0) {
            if constexpr (KDA) {
                if (lane < C) {
                    beta_s[buf][lane] = gb;
                }
            } else if (lane < C) {
                G_s[buf][lane] = expf(gb);
            } else {
                beta_s[buf][lane - C] = gb;
            }
        }
        const float g_last = KDA ? 1.0f : expf(bl);
        float vr[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vr[e] = vn[e];
        }
        if (c + 1 < n_chunks) {
            fetch(c + 1);
        }
        gdn_lds_barrier();

        // X = K S, Y = Q S for this warp's columns
        gdn_v8f X = {}, Y = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const gdn_v16h sb = gdn_acc_to_b(Sacc[j], hi);
            X = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Kh[buf][r][16 * j]), sb, X);
            Y = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Qh[buf][r][16 * j]), sb, Y);
        }
        // D = T beta (v - G X)
        gdn_v8f R;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = 2 * e + hi;
            R[e] = beta_s[buf][t] * (KDA ? vr[e] - X[e] : fmaf(-G_s[buf][t], X[e], vr[e]));
        }
        const gdn_v8f  D  = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Th[buf][r][0]), gdn_acc_to_b(R, hi), gdn_v8f{});
        const gdn_v16h DB = gdn_acc_to_b(D, hi);

        // O = scale (G Y + P D)
        gdn_v8f O;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            O[e] = KDA ? Y[e] : G_s[buf][2 * e + hi] * Y[e];
        }
        O = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Ph[buf][r][0]), DB, O);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = 2 * e + hi;
            if (t < nt) {
                attn[(t0 + t) * S * H] = scale * O[e];
            }
        }

        // S = G_last S + Kt D
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                Sacc[j][e] *= KDA ? Gl_s[buf][16 * j + 2 * e + hi] : g_last;
            }
            Sacc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn_ld16(&Kt[buf][16 * j + r][0]), DB, Sacc[j]);
        }
    }

    float * s_out = state + (seq * H + h) * S * S + (col0 + r) * S;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            s_out[16 * j + 2 * e + hi] = Sacc[j][e];
        }
    }
#else
    GGML_UNUSED_VARS(qkrec, v, rec, curr_state, dst, state, H, n_tokens, n_chunks, sv1, sv2, sv3, neqk1_magic, scale);
    NO_DEVICE_CODE;
#endif // defined(GDN_CHUNK_WMMA)
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, ggml_cuda_pool & pool, cudaStream_t stream,
        const int32_t * state_ids = nullptr, int64_t state_row = 0) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        // chunked WMMA prefill; rollback snapshots come from the recurrent kernel on the last K tokens
        const int64_t n_tail = keep_rs_t ? K : 0;
        if (GGML_CUDA_CC_IS_RDNA3(cc) && S_v == 128 && n_seqs == 1 && H % neqk1 == 0 && n_tokens - n_tail >= 64 &&
                (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 && (uintptr_t) g_d % 16 == 0 &&
                sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0) {
            const int64_t n_chunked = n_tokens - n_tail;
            const int64_t n_chunks  = (n_chunked + GDN_CHUNK - 1) / GDN_CHUNK;
            if constexpr (KDA) {
                ggml_cuda_pool_alloc<char> rec(pool, (size_t) (n_chunks * H * GDN_CHUNK_REC_KDA));
                gdn_chunk_prep_kda<<<dim3(H, n_chunks, 1), 128, 0, stream>>>(q_d, k_d, g_d, b_d, rec.get(), n_chunked,
                    n_chunks, sq1, sq2, sb1, sb2, neqk1_magic);
                CUDA_CHECK(cudaGetLastError());
                gated_delta_net_chunked_cuda<true><<<dim3(H, 1, 1), 256, 0, stream>>>(nullptr, v_d, rec.get(), s_d, dst_d,
                    state_d, H, n_chunked, n_chunks, sv1, sv2, sv3, neqk1_magic, scale);
                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_pool_alloc<char> rec(pool, (size_t) (n_chunks * H * GDN_CHUNK_REC));
                ggml_cuda_pool_alloc<char> qkrec(pool, (size_t) (n_chunks * neqk1 * GDN_CHUNK_QK));
                const dim3 pgrid((unsigned) neqk1, (unsigned) ((n_chunks + 3) / 4), 1);
                gdn_chunk_prep<<<pgrid, 128, 0, stream>>>(q_d, k_d, g_d, b_d, qkrec.get(), rec.get(), H, neqk1,
                    n_chunked, n_chunks, sq1, sq2, sb1, sb2);
                CUDA_CHECK(cudaGetLastError());
                gated_delta_net_chunked_cuda<false><<<dim3(H, 1, 1), 256, 0, stream>>>(qkrec.get(), v_d, rec.get(), s_d,
                    dst_d, state_d, H, n_chunked, n_chunks, sv1, sv2, sv3, neqk1_magic, scale);
                CUDA_CHECK(cudaGetLastError());
            }
            if (n_tail == 0) {
                return;
            }
            // continue from the chunked state over the tail, writing the snapshots
            q_d += n_chunked * sq2;
            k_d += n_chunked * sq2;
            v_d += n_chunked * sv2;
            g_d += n_chunked * sb2 * (KDA ? S_v : 1);   // KDA gates are per channel
            b_d += n_chunked * sb2;
            s_d = state_d;
            dst_d += n_chunked * S_v * H;
            n_tokens = n_tail;
        }

        // the scalar gate was tuned at 48 heads (Qwen3.5); the KDA tile serves GLM-5.3-Flash's 64
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && (KDA || H == 48) && n_seqs == 1 && n_tokens >= 16 && n_tokens <= 32768) {
            const dim3 tiled_grid(H, n_seqs, 2);
            const dim3 tiled_block(warp_size, 8, 1);
            const ggml_cuda_kernel_launch_params tiled_params(tiled_grid, tiled_block, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<128, 8, 8, 16, KDA, keep_rs_t>, tiled_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

const ggml_tensor * ggml_cuda_gdn_state_gather(const ggml_tensor * gdn) {
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->src[2]->ne[2] >= 16) {
        return nullptr;   // longer batches take the tiled or chunked kernels, which read a gathered state
    }
    const ggml_tensor * st = gdn->src[5];
    const ggml_tensor * gr = (st->op == GGML_OP_RESHAPE || st->op == GGML_OP_VIEW) ? st->view_src : st;
    if (!gr || gr->op != GGML_OP_GET_ROWS || (st != gr && st->view_offs != 0) || !ggml_is_contiguous(st)) {
        return nullptr;
    }
    const ggml_tensor * cache = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    const int64_t n_seqs = gdn->src[2]->ne[3];
    if (cache->type != GGML_TYPE_F32 || cache->nb[0] != sizeof(float) || cache->ne[2] != 1 || cache->ne[3] != 1 ||
            gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) || gr->ne[0] != cache->ne[0] || gr->ne[1] != n_seqs ||
            ggml_nelements(gr) != ggml_nelements(st) || ids->type != GGML_TYPE_I32 || ids->ne[0] != n_seqs ||
            !ggml_is_contiguous(ids) || cache->nb[1] % sizeof(float) != 0) {
        return nullptr;
    }
    return gr;
}

static bool gdn_is_view_node(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

bool ggml_cuda_gdn_state_gather_skippable(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * gr = cgraph->nodes[i];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32) {
        return false;
    }
    // build_rs gathers all rows once and reads views of it: the ubatch's states (the delta net, through a reshape)
    // and the extra states copied back to the cache, an empty CPY at one-sequence decode. Every reader of the gather or
    // of a view of it must be one of those, and the use counts must show no reader outside the window.
    constexpr int window = 512;
    const int end = std::min(cgraph->n_nodes, i + window);
    std::vector<int> group = { i };   // node indices of gr and its views
    for (int n = i + 1; n < end; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (gdn_is_view_node(t) && t->view_src == gr) {
            group.push_back(n);
        }
    }
    int n_gdn = 0;
    for (const int gi : group) {
        const ggml_tensor * x = cgraph->nodes[gi];
        int readers = 0;
        for (int n = gi + 1; n < end; ++n) {
            const ggml_tensor * t = cgraph->nodes[n];
            bool reads = false;
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                reads |= t->src[s] == x;
            }
            if (!reads) {
                continue;
            }
            ++readers;
            if (gdn_is_view_node(t) && t->view_src == gr) {
                continue;
            }
            if (t->op == GGML_OP_CPY && ggml_nelements(t->src[0]) == 0) {
                continue;
            }
            if (ggml_cuda_gdn_state_gather(t) == gr) {
                ++n_gdn;
                continue;
            }
            return false;
        }
        if (readers != ggml_node_get_use_count(cgraph, gi)) {
            return false;
        }
    }
    return n_gdn > 0;
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    // short batches read the state straight from the recurrent cache (the gathering GET_ROWS may not have run)
    const int32_t * state_ids = nullptr;
    int64_t         state_row = 0;
    if (const ggml_tensor * gr = ggml_cuda_gdn_state_gather(dst)) {
        s_d       = (const float *) gr->src[0]->data;
        state_ids = (const int32_t *) gr->src[1]->data;
        state_row = gr->src[0]->nb[1] / sizeof(float);
    }
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ctx.pool(), stream, state_ids, state_row);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ctx.pool(), stream, state_ids, state_row);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ctx.pool(), stream, state_ids, state_row);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ctx.pool(), stream, state_ids, state_row);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
