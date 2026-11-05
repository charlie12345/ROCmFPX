#include "gated_delta_net.cuh"
#include "common.cuh"

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

#if defined(GGML_USE_HIP)
// ROCmFPX: one state column per warp on HIP. The <16-token kernel must stay bit-identical to the RDNA3.5
// tiled kernel that takes over at 16+ tokens, or a 1-4 token decode step and a 16+ token speculative verify
// batch disagree in the last bit and n-gram replays of a cached answer stop being accepted (measured on
// Qwen3.8-Flash-Next: json replay 136 -> 46 t/s). Four columns per warp changes the reduction width.
constexpr int gdn_cols_per_warp = 1;
#else
constexpr int gdn_cols_per_warp = 4;
#endif

// [TAG_RS_REPLAY] GGML_OP_GATED_DELTA_NET_REPLAY (see ggml_gated_delta_net_replay); hist offsets in floats
struct gdn_replay_args {
    const int32_t * rep      = nullptr; // [3, n_seqs]: tokens to replay, hist row to read, hist row to write
    float *         hist     = nullptr;
    int64_t         hist_row = 0;       // floats per hist row
    int64_t         rec      = 0;       // floats per token record [k | v | g | beta]
    int64_t         off_v    = 0;
    int64_t         off_g    = 0;
    int64_t         off_b    = 0;
    bool            prologue = true;    // false: the replay already ran (the tail after a chunked prefix)
};

// replay mode: the state first takes the seq's recorded tokens (same arithmetic as the token loop, so the
// result equals the snapshot the plain op would have kept), and the token loop keeps 2 states (slot 0 the
// final one, slot 1 the base W - 1 tokens back) plus the records of the W - 1 tokens after the base
template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp, bool replay = false>
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
                                     const gdn_replay_args ra) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

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

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    if constexpr (replay) {
        const int32_t n_rep = ra.prologue ? ra.rep[3 * sequence + 0] : 0;
        const float * h_rd  = ra.hist + (int64_t) ra.rep[3 * sequence + 1] * ra.hist_row;
        for (int j = 0; j < n_rep; j++) {
            const float * rec_j    = h_rd + j * ra.rec;
            const float * k_t      = rec_j + h_idx * S_v;
            const float * g_t      = rec_j + ra.off_g + h_idx * (KDA ? S_v : 1);
            const float   v_col    = rec_j[ra.off_v + h_idx * S_v + col];
            const float   beta_val = rec_j[ra.off_b + h_idx];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * lanes_per_col + lane_in_col];
            }

            if constexpr (!KDA) {
                const float g_val = expf(*g_t);

                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

                float delta_col = (v_col - g_val * kv_col) * beta_val;

#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                }
            } else {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
                }

                float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

                float delta_col = (v_col - kv_col) * beta_val;

#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    s_shard[r] = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                }
            }
        }
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
            const int i = r * lanes_per_col + lane_in_col;
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
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

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

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (replay && keep_rs_t) {
            // slot 0 = final state, slot 1 = the replay base W - 1 tokens back; records for the tokens after it
            const int W      = n_tokens < K ? (int) n_tokens : K;
            const int t_base = (int) n_tokens - W;
            if (t == n_tokens - 1 || (W > 1 && t == t_base)) {
                float * curr_state = state + (t == n_tokens - 1 ? 0 : state_slot_stride);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
            if (W > 1 && t > t_base) {
                float * rec_t = ra.hist + (int64_t) ra.rep[3 * sequence + 2] * ra.hist_row + (t - t_base - 1) * ra.rec;
                if (lane_in_col == 0) {
                    rec_t[ra.off_v + h_idx * S_v + col] = v_t[col];
                }
                if (col == 0) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes_per_col + lane_in_col;
                        rec_t[h_idx * S_v + i] = k_reg[r];
                        if constexpr (KDA) {
                            rec_t[ra.off_g + h_idx * S_v + i] = g_t[i];
                        }
                    }
                    if (lane_in_col == 0) {
                        if constexpr (!KDA) {
                            rec_t[ra.off_g + h_idx] = *g_t;
                        }
                        rec_t[ra.off_b + h_idx] = beta_val;
                    }
                }
            }
        } else if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>
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

    __shared__ __align__(16) float q_shared[TOKEN_TILE][S_v];
    __shared__ __align__(16) float k_shared[TOKEN_TILE][S_v];
    __shared__ __align__(16) float v_shared[TOKEN_TILE][block_cols];
    __shared__ __align__(16) float g_shared[TOKEN_TILE];
    __shared__ __align__(16) float beta_shared[TOKEN_TILE];

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
            g_shared[thread]    = g[gb_offset];
            beta_shared[thread] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
            }

            const float g_val    = expf(g_shared[tt]);
            const float beta_val = beta_shared[tt];

            float attn_col[COLS];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard = fmaf(s_shard[c][r], k_reg[r], kv_shard);
                }
                const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = fmaf(-g_val, kv_col, v_shared[tt][colw + c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
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

#if defined(GGML_USE_HIP)
// Chunked prefill for the scalar-gate delta rule, S_v == S_k == 128, one sequence, RDNA3 WMMA.
// Per chunk of C tokens with cumulative log decay gc and a = exp(gc):
//   A[t][j] = beta_t exp(gc_t - gc_j) k_t.k_j (j < t), T = (I + A)^-1
//   W = T diag(beta a) K, U = T diag(beta) V, d = U - W S0
//   o = scale (diag(a) Q S0 + M d), M[t][j] = exp(gc_t - gc_j) q_t.k_j (j <= t)
//   S1 = a_C S0 + K^T diag(exp(gc_C - gc)) d
// One workgroup per (head, 64 state columns). The state stays in F32 WMMA accumulators; it enters the
// products as an f16 hi + lo pair, so only W, q and the decayed k are rounded to f16.
#define GDN_CHUNK      16
#define GDN_CHUNK_D    128
#define GDN_CHUNK_COLS 64

struct gdn_chunk_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    int64_t sq1, sq2, sv1, sv2, sb1, sb2;
    int64_t neqk1;
    int64_t n_tokens;
};

typedef _Float16 gdn_v16h __attribute__((ext_vector_type(16)));
typedef float    gdn_v8f  __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ gdn_v16h gdn_frag(const _Float16 * p) {
    const uint4 a = *(const uint4 *) p, b = *(const uint4 *) (p + 8);
    return __builtin_bit_cast(gdn_v16h, (uint4[2]){a, b});
}

// acc[r][c] += sum_k P[r][k] * Q[c][k]; lane L supplies row (L & 15) of each operand, result row 2e + (L >> 4), column L & 15
static __device__ __forceinline__ void gdn_mma(gdn_v8f & acc, const gdn_v16h & a, const gdn_v16h & b) {
#if defined(RDNA3)
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
#else
    GGML_UNUSED(acc); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void gdn_wave_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__global__ void __launch_bounds__(256, 1) gdn_chunk_fused(const gdn_chunk_args p,
        const float * __restrict__ s0, float * __restrict__ dst, float * __restrict__ state_out, const int64_t H, const float scale) {
    constexpr int C = GDN_CHUNK, D = GDN_CHUNK_D, NC = GDN_CHUNK_COLS, DP = D + 4, DH = D + 8, CH = C + 8, NP = NC + 4, NT = NC / 16;
    static_assert(C == 16 && NC == 64, "tile layout assumes 16-token chunks and 64-column blocks");
    __shared__ __align__(16) float kf[C][DP];
    __shared__ __align__(16) float qf[C][DP];
    __shared__ __align__(16) float vq[C][NP];       // v slice
    __shared__ __align__(16) float as[C][C + 1], ms[C][C + 1], ts[C][C + 1];
    __shared__ __align__(16) float uf[C][NC + 1];
    __shared__ __align__(16) float gcs[C], bts[C], bas[C];
    __shared__ __align__(16) _Float16 wh[C][DH];
    __shared__ __align__(16) _Float16 qh[C][DH];
    __shared__ __align__(16) _Float16 kdt[D][CH];
    __shared__ __align__(16) _Float16 dth[NC][CH];
    __shared__ __align__(16) _Float16 dtl[NC][CH];
    __shared__ __align__(16) _Float16 trs[2][8][16][CH];   // per-wave transpose scratch (hi, lo); also holds the K K^T / Q K^T partials
    _Float16 (*trh)[16][CH] = trs[0];
    _Float16 (*trl)[16][CH] = trs[1];

    const int h = blockIdx.x, col0 = blockIdx.y * NC, tid = threadIdx.x;
    const int lane = tid % 32, wave = tid / 32, r16 = lane & 15, hi16 = lane >> 4;
    // wave w holds state columns (w & 3) * 16 .. + 15 and rows (w >> 2) * 64 .. + 63
    const int ct = wave & 3, ih = wave >> 2;
    const int64_t hq = h % p.neqk1;
    constexpr int NI = D / 32;   // row tiles per wave

    gdn_v8f acc[NI];
#pragma unroll
    for (int it = 0; it < NI; ++it) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int c = ct * 16 + 2 * e + hi16, i = (ih * NI + it) * 16 + r16;
            acc[it][e] = s0[(int64_t) h * D * D + (int64_t) (col0 + c) * D + i];
        }
    }

    // inputs of the next chunk are loaded into registers while the current one is processed
    float4 pk[2], pq[2];
    float  pv[4], pg = 0.0f, pb = 0.0f;
    auto prefetch = [&](const int64_t c0) {
        const int nv = (int) min((int64_t) C, p.n_tokens - c0);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int idx = tid + r * 256, t = idx / (D / 4), i = (idx % (D / 4)) * 4;
            const bool ok = t < nv;
            pk[r] = ok ? *(const float4 *) (p.k + (c0 + t) * p.sq2 + hq * p.sq1 + i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            pq[r] = ok ? *(const float4 *) (p.q + (c0 + t) * p.sq2 + hq * p.sq1 + i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = tid + r * 256, t = idx / NC, d = idx % NC;
            pv[r] = t < nv ? p.v[(c0 + t) * p.sv2 + h * p.sv1 + col0 + d] : 0.0f;
        }
        if (tid < C) {
            const bool ok = tid < nv;
            pg = ok ? p.g[(c0 + tid) * p.sb2 + h * p.sb1] : 0.0f;
            pb = ok ? p.beta[(c0 + tid) * p.sb2 + h * p.sb1] : 0.0f;
        }
    };
    prefetch(0);

    const int64_t n_chunks = (p.n_tokens + C - 1) / C;
    for (int64_t ch = 0; ch < n_chunks; ++ch) {
        const int64_t t0 = ch * C;
        const int nvalid = (int) min((int64_t) C, p.n_tokens - t0);

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int idx = tid + r * 256, t = idx / (D / 4), i = (idx % (D / 4)) * 4;
            *(float4 *) &kf[t][i] = pk[r];
            *(float4 *) &qf[t][i] = pq[r];
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = tid + r * 256;
            vq[idx / NC][idx % NC] = pv[r];
        }
        if (tid < C) {
            // inclusive prefix sum of the log decays over the 16 lanes
            float sum = pg;
#pragma unroll
            for (int off = 1; off < C; off *= 2) {
                const float o = __shfl_up(sum, off, C);
                sum += tid >= off ? o : 0.0f;
            }
            gcs[tid] = sum;
            bts[tid] = pb;
            bas[tid] = pb * expf(sum);
        }
        __syncthreads();
        if (ch + 1 < n_chunks) {
            prefetch(t0 + C);
        }

        // K K^T (waves 0-3) and Q K^T (waves 4-7) on WMMA with f16 hi/lo splits, each wave one quarter of K;
        // the partial tiles meet in the (idle) transpose scratch. q is also converted to f16 here.
        {
            static_assert(sizeof(trs) >= 2 * 4 * 16 * 17 * sizeof(float), "partial tiles do not fit");
            float (*pp)[4][16][17] = (float (*)[4][16][17]) &trs[0][0][0][0];
            const float (*xr)[DP] = wave < 4 ? kf : qf;
            const int k0 = (wave & 3) * (D / 4);
            gdn_v8f r = {};
#pragma unroll
            for (int kk = 0; kk < D / 4; kk += 16) {
                gdn_v16h xh, xl, kh, kl;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const float x = xr[r16][k0 + kk + k], y = kf[r16][k0 + kk + k];
                    xh[k] = (_Float16) x; xl[k] = (_Float16) (x - (float) xh[k]);
                    kh[k] = (_Float16) y; kl[k] = (_Float16) (y - (float) kh[k]);
                }
                gdn_mma(r, xh, kh);
                gdn_mma(r, xh, kl);
                gdn_mma(r, xl, kh);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pp[wave >> 2][wave & 3][2 * e + hi16][r16] = r[e];
            }
            for (int idx = tid; idx < C * D; idx += blockDim.x) {
                qh[idx / D][idx % D] = (_Float16) qf[idx / D][idx % D];
            }
            __syncthreads();
            {
                const int t = tid / C, j = tid % C;
                const float kk = pp[0][0][t][j] + pp[0][1][t][j] + pp[0][2][t][j] + pp[0][3][t][j];
                const float qk = pp[1][0][t][j] + pp[1][1][t][j] + pp[1][2][t][j] + pp[1][3][t][j];
                const float dec = j <= t ? expf(gcs[t] - gcs[j]) : 0.0f;
                as[t][j] = j < t ? bts[t] * dec * kk : 0.0f;
                ms[t][j] = dec * qk;
            }
        }
        __syncthreads();

        // T = (I + A)^-1 by forward substitution; lane c owns column c
        if (tid < C) {
            const int c = tid;
            float tc[C];
#pragma unroll
            for (int t = 0; t < C; ++t) {
                float sum = t == c ? 1.0f : 0.0f;
#pragma unroll
                for (int j = 0; j < t; ++j) {
                    sum = j >= c ? fmaf(-as[t][j], tc[j], sum) : sum;
                }
                tc[t] = t < c ? 0.0f : sum;
                ts[t][c] = tc[t];
            }
        }
        __syncthreads();

        // W = T diag(beta a) K (f16) and U = T diag(beta) V on WMMA (T, K, V split into f16 hi + lo), Kd^T (f16)
        {
            gdn_v16h th, tl, kh, kl;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float x = ts[r16][j] * bas[j];
                th[j] = (_Float16) x; tl[j] = (_Float16) (x - (float) th[j]);
                const float y = kf[j][wave * 16 + r16];
                kh[j] = (_Float16) y; kl[j] = (_Float16) (y - (float) kh[j]);
            }
            gdn_v8f r = {};
            gdn_mma(r, th, kh);
            gdn_mma(r, th, kl);
            gdn_mma(r, tl, kh);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                wh[2 * e + hi16][wave * 16 + r16] = (_Float16) r[e];
            }
            if (wave < NT) {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const float x = ts[r16][j] * bts[j];
                    th[j] = (_Float16) x; tl[j] = (_Float16) (x - (float) th[j]);
                    const float y = vq[j][wave * 16 + r16];
                    kh[j] = (_Float16) y; kl[j] = (_Float16) (y - (float) kh[j]);
                }
                gdn_v8f u = {};
                gdn_mma(u, th, kh);
                gdn_mma(u, th, kl);
                gdn_mma(u, tl, kh);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    uf[2 * e + hi16][wave * 16 + r16] = u[e];
                }
            }
        }
        for (int idx = tid; idx < C * D; idx += blockDim.x) {
            const int i = idx / C, t = idx % C;
            kdt[i][t] = (_Float16) (expf(gcs[C - 1] - gcs[t]) * kf[t][i]);
        }
        __syncthreads();

        // d = U - W S0 and q S0: each wave multiplies its 64 state rows, the two halves are summed through LDS
        // (qf is free at this point). S^T tiles are transposed through a per-wave scratch.
        float (*part)[2][16][16] = (float (*)[2][16][16]) &qf[0][0];
        gdn_v8f dw = {}, qs = {};
#pragma unroll
        for (int it = 0; it < NI; ++it) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float x = acc[it][e];
                const _Float16 xh = (_Float16) x;
                trh[wave][2 * e + hi16][r16] = xh;
                trl[wave][2 * e + hi16][r16] = (_Float16) (x - (float) xh);
            }
            gdn_wave_lds_sync();
            const int k0 = (ih * NI + it) * 16;
            const gdn_v16h sh = gdn_frag(&trh[wave][r16][0]), sl = gdn_frag(&trl[wave][r16][0]);
            const gdn_v16h pw = gdn_frag(&wh[r16][k0]), pq = gdn_frag(&qh[r16][k0]);
            gdn_wave_lds_sync();
            gdn_mma(dw, pw, sh);
            gdn_mma(dw, pw, sl);
            gdn_mma(qs, pq, sh);
            gdn_mma(qs, pq, sl);
        }
        // decays used below, read now: wave 0 may already be writing the next chunk's gc after the last barrier
        const float al = expf(gcs[C - 1]);
        float ag[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            ag[e] = expf(gcs[2 * e + hi16]);
        }
        if (ih == 1) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                part[ct][0][2 * e + hi16][r16] = dw[e];
                part[ct][1][2 * e + hi16][r16] = qs[e];
            }
        }
        __syncthreads();
        if (ih == 0) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = 2 * e + hi16, c = ct * 16 + r16;
                qs[e] += part[ct][1][t][r16];
                const float d = uf[t][c] - (dw[e] + part[ct][0][t][r16]);
                const _Float16 dh = (_Float16) d;
                dth[c][t] = dh;
                dtl[c][t] = (_Float16) (d - (float) dh);
            }
            gdn_wave_lds_sync();
            // o = scale (a_t q S + M d) for this wave's 16 columns
            gdn_v16h mh, ml;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float x = ms[r16][j];
                mh[j] = (_Float16) x; ml[j] = (_Float16) (x - (float) mh[j]);
            }
            const gdn_v16h dh = gdn_frag(&dth[ct * 16 + r16][0]), dl = gdn_frag(&dtl[ct * 16 + r16][0]);
            gdn_v8f o = {};
            gdn_mma(o, mh, dh);
            gdn_mma(o, mh, dl);
            gdn_mma(o, ml, dh);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = 2 * e + hi16, c = ct * 16 + r16;
                if (t < nvalid) {
                    dst[(t0 + t) * D * H + (int64_t) h * D + col0 + c] = (ag[e] * qs[e] + o[e]) * scale;
                }
            }
        }
        __syncthreads();

        // S^T = a_C S^T + d^T Kd. No barrier after it: the next chunk only rewrites dth/dtl/kdt after its first barrier.
        {
            const gdn_v16h ph = gdn_frag(&dth[ct * 16 + r16][0]), pl = gdn_frag(&dtl[ct * 16 + r16][0]);
#pragma unroll
            for (int it = 0; it < NI; ++it) {
                const gdn_v16h kq = gdn_frag(&kdt[(ih * NI + it) * 16 + r16][0]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[it][e] *= al;
                }
                gdn_mma(acc[it], ph, kq);
                gdn_mma(acc[it], pl, kq);
            }
        }
    }

#pragma unroll
    for (int it = 0; it < NI; ++it) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int c = ct * 16 + 2 * e + hi16, i = (ih * NI + it) * 16 + r16;
            state_out[(int64_t) h * D * D + (int64_t) (col0 + c) * D + i] = acc[it][e];
        }
    }
}
#endif // defined(GGML_USE_HIP)

template <bool KDA, bool keep_rs_t, bool replay = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        const gdn_replay_args & ra = gdn_replay_args()) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // gdn_cols_per_warp columns per warp (four on CUDA, one on HIP; see the kernel); shrink the CTA when the
    // wider CTA would leave SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // the tiled kernel keeps K snapshots; the replay op runs on the per-token kernel below
    if constexpr (!KDA && !replay) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && H == 48 && n_seqs == 1 && n_tokens >= 16 && n_tokens <= 32768) {
            const dim3 tiled_grid(H, n_seqs, 2);
            const dim3 tiled_block(warp_size, 8, 1);
            const ggml_cuda_kernel_launch_params tiled_params(tiled_grid, tiled_block, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<128, 8, 8, 16, keep_rs_t>, tiled_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ra);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ra);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ra);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ra);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#if defined(GGML_USE_HIP)
static bool gdn_chunked_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// returns the number of leading tokens handled by the chunked path (0 = not used)
static int64_t gdn_chunked_prefix(ggml_backend_cuda_context & ctx, const gdn_chunk_args & p0,
        const float * s_d, float * dst_d, float * state_d, const int64_t H, const int64_t n_seqs, const int64_t rq3,
        const int K, const float scale, const int64_t state_slot_stride, cudaStream_t stream,
        const gdn_replay_args * ra) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t tail = K > 1 ? K : 0;
    const int64_t n_prefix = p0.n_tokens - tail;
    if (!gdn_chunked_enabled() || !GGML_CUDA_CC_IS_RDNA3(cc) || n_seqs != 1 || rq3 != 1 || n_prefix < 64) {
        return 0;
    }
    if ((uintptr_t) p0.q % 16 != 0 || (uintptr_t) p0.k % 16 != 0 || p0.sq1 % 4 != 0 || p0.sq2 % 4 != 0) {
        return 0;
    }
    gdn_chunk_args p = p0;
    p.n_tokens = n_prefix;
    // with a snapshot tail the chunked part ends in a temporary state that the sequential kernel continues from
    ggml_cuda_pool_alloc<float> mid(ctx.pool(), tail > 0 ? (size_t) H * GDN_CHUNK_D * GDN_CHUNK_D : 1);
    float * s_out = tail > 0 ? mid.get() : state_d;

    // [TAG_RS_REPLAY] the replayed tokens advance the state before the chunked part, in a temporary state
    ggml_cuda_pool_alloc<float> replayed(ctx.pool(), ra != nullptr ? (size_t) H * GDN_CHUNK_D * GDN_CHUNK_D : 1);
    if (ra != nullptr) {
        launch_gated_delta_net<false, false, true>(p0.q, p0.k, p0.v, p0.g, p0.beta, s_d, dst_d, replayed.get(),
            GDN_CHUNK_D, H, 0, 1, p0.sq1, p0.sq2, 0, p0.sv1, p0.sv2, 0, p0.sb1, p0.sb2, 0,
            p0.neqk1, 1, scale, 0, K, stream, *ra);
        CUDA_CHECK(cudaGetLastError());
        s_d = replayed.get();
    }
    gdn_chunk_fused<<<dim3(H, GDN_CHUNK_D / GDN_CHUNK_COLS, 1), 256, 0, stream>>>(p, s_d, dst_d, s_out, H, scale);

    if (tail > 0) {
        CUDA_CHECK(cudaGetLastError());
        const int64_t P = n_prefix;
        if (ra != nullptr) {
            gdn_replay_args ra_tail = *ra;
            ra_tail.prologue = false;
            launch_gated_delta_net<false, true, true>(p0.q + P * p0.sq2, p0.k + P * p0.sq2, p0.v + P * p0.sv2,
                p0.g + P * p0.sb2, p0.beta + P * p0.sb2, s_out, dst_d + P * GDN_CHUNK_D * H, state_d,
                GDN_CHUNK_D, H, tail, 1, p0.sq1, p0.sq2, 0, p0.sv1, p0.sv2, 0, p0.sb1, p0.sb2, 0,
                p0.neqk1, 1, scale, state_slot_stride, K, stream, ra_tail);
        } else {
            launch_gated_delta_net<false, true>(p0.q + P * p0.sq2, p0.k + P * p0.sq2, p0.v + P * p0.sv2,
                p0.g + P * p0.sb2, p0.beta + P * p0.sb2, s_out, dst_d + P * GDN_CHUNK_D * H, state_d,
                GDN_CHUNK_D, H, tail, 1, p0.sq1, p0.sq2, 0, p0.sv1, p0.sv2, 0, p0.sb1, p0.sb2, 0,
                p0.neqk1, 1, scale, state_slot_stride, K, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return p0.n_tokens;
}
#endif // defined(GGML_USE_HIP)

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

    // [TAG_RS_REPLAY] the replay op keeps 2 states (slot 0 final, slot 1 the replay base) and writes token records
    const bool replay = dst->op == GGML_OP_GATED_DELTA_NET_REPLAY;
    gdn_replay_args ra;
    if (replay) {
        const ggml_tensor * src_hist = dst->src[6];
        GGML_ASSERT(ggml_is_contiguous(src_hist) && ggml_is_contiguous(dst->src[7]));
        ra.rep      = (const int32_t *) dst->src[7]->data;
        ra.hist     = (float *) src_hist->data;
        ra.hist_row = src_hist->nb[1] / sizeof(float);
        ra.rec      = src_hist->ne[0] / (K - 1);
        ra.off_v    = S_v * H;
        ra.off_g    = 2 * S_v * H;
        ra.off_b    = ra.off_g + src_g->ne[0] * H;
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if defined(GGML_USE_HIP)
    if (!kda && S_v == GDN_CHUNK_D) {
        const gdn_chunk_args p = { q_d, k_d, v_d, g_d, b_d, sq1, sq2, sv1, sv2, sb1, sb2, neqk1, n_tokens };
        if (gdn_chunked_prefix(ctx, p, s_d, dst_d, state_d, H, n_seqs, rq3, K, scale, state_slot_stride, stream,
                    replay ? &ra : nullptr) > 0) {
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    if (replay) {
        if (kda) {
            launch_gated_delta_net<true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, ra);
        } else {
            launch_gated_delta_net<false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, ra);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
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
