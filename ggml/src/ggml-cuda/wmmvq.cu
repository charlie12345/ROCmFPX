// WMMA-based pure-autoregressive GEMV for Q4_K on RDNA3 (gfx1100).
//
// Targets the tied lm_head GEMV (vocab-sized rows), which is compute/
// latency bound in the dp4a MMVQ kernel (~270 GB/s) while the machine
// streams ~640 GB/s on bandwidth-limited GEMVs. WMMA F16 tiles move the
// heavy multiply-accumulate into the AI accelerators:
//   - A fragment: 16 output rows x 16 k, one row per lane (RDNA3 WMMA A
//     is column-major: lane l holds row l, k = 0..15 = half a 32-element
//     Q4_K sub-block), raw 4-bit codes as F16 (no per-element scale).
//   - B fragment: the activation row (read as F32, converted in
//     registers) replicated across all 16 N columns, so every D column
//     holds the same dot product (D[2*ele + lane/16][lane%16], F32).
//   - Per-tile scale fixup in F32: value = d*sc*(q - m), so
//     dot = (d*sc)*sum(q*a) - (d*sc*m)*sum(a), accumulated per row.
//
// Work decomposition: one 256-thread block (8 warps) per 16-row tile,
// Q4_K blocks split across the 8 warps; partials reduced through LDS.
//
// Numerics: F16 activation/codes with F32 accumulation. This is a
// numerics change vs the exact int8 MMVQ path; validated by prompt and
// backend-op gates with documented divergence.
//
// Reference: https://gpuopen.com/learn/wmma_on_rdna3/
// Fragment layouts cross-checked with ROCm/amd_matrix_instruction_calculator.

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdlib>

#define WMMVQ_K_SPLIT 8

// Q4_K sub-block scale decode (6-bit sc and min from the 12 packed bytes).
__device__ __forceinline__ void wmmvq_get_scale_min_k4(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) {
        sc = q[j + 0] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4) | ((q[j - 4] >> 6) << 4);
    }
}

// One 256-thread block per 16 output rows; 8 warps split the Q4_K blocks.
__global__ void wmmvq_q4_k_f16(
        const char * __restrict__ vx,
        const float * __restrict__ yf,
        float * __restrict__ dst,
        const int nrows, const int blocks_per_row, const int64_t row_bytes) {

#if defined(RDNA3)
    typedef __attribute__((ext_vector_type(16))) _Float16 halfx16_t;
    typedef __attribute__((ext_vector_type(8)))  float   floatx8_t;

    __shared__ float lds[WMMVQ_K_SPLIT][16];

    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;   // K-split warp id
    const int row0 = blockIdx.x * 16;
    if (row0 >= nrows) {
        return;
    }

    const block_q4_K * lane_row =
        (const block_q4_K *)(vx + (int64_t)(row0 + (lane % 16)) * row_bytes);

    halfx16_t a;
    halfx16_t b;
    _Float16 * av = (_Float16 *) &a;
    _Float16 * bv = (_Float16 *) &b;
    floatx8_t c;      // fresh per K tile
    floatx8_t czero;  // zero accumulator for a single tile

    // Per-lane row partials for this warp's K chunk (rows 2*e + lane/16).
    float res[8] = {};

    for (int kb = w; kb < blocks_per_row; kb += WMMVQ_K_SPLIT) {
        const block_q4_K * blk = lane_row + kb;
        const float d   = __low2float(blk->dm);
        const float dmn = __high2float(blk->dm);

        for (int tt = 0; tt < 16; ++tt) {   // 16 K tiles of 16 elements
            const int sb = tt / 2;          // 32-element sub-block
            const int h  = tt % 2;          // first/second half of sub-block
            int sc, m;
            wmmvq_get_scale_min_k4(sb, blk->scales, sc, m);
            const float dsc = d * (float) sc;
            const float dsm = d * (float) sc * (float) m;
            const int base  = 32 * (sb / 2);

            float sum_a = 0.0f;
            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                const float act = yf[256*kb + 32*sb + h*16 + ele];
                const uint8_t byte = blk->qs[base + h*16 + ele];
                const int qv = (sb % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                av[ele] = __float2half((float) qv);
                bv[ele] = __float2half(act);
                sum_a  += act;
            }

            czero = floatx8_t{};
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, czero);

            #pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                res[ele] += dsc * c[ele] - dsm * sum_a;
            }
        }
    }

    // D[i][j]: i = 2*ele + lane/16, j = lane%16; all columns identical.
    if (lane == 0) {
        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            lds[w][2*ele] = res[ele];
        }
    } else if (lane == 16) {
        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            lds[w][2*ele + 1] = res[ele];
        }
    }

    __syncthreads();

    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < WMMVQ_K_SPLIT; ++s) {
            sum += lds[s][threadIdx.x];
        }
        dst[row0 + threadIdx.x] = sum;
    }
#else
    GGML_UNUSED_VARS(vx, yf, dst, nrows, blocks_per_row, row_bytes);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

// WMMA-based tg GEMV for Q4_0_ROCMI4 on RDNA3 (gfx1100).
// ROCMI4 has simpler dequant than Q4_K: signed 4-bit nibbles (-8..7) with
// a single UE4M3 scale per 32-element block. No sub-block scales or min.
// Each block (QK=32) yields 2 WMMA tiles of 16 K-elements.
__global__ void wmmvq_rocmi4_f16(
        const char * __restrict__ vx,
        const float * __restrict__ yf,
        float * __restrict__ dst,
        const int nrows, const int blocks_per_row, const int64_t row_bytes) {

#if defined(RDNA3)
    typedef __attribute__((ext_vector_type(16))) _Float16 halfx16_t;
    typedef __attribute__((ext_vector_type(8)))  float   floatx8_t;

    __shared__ float lds[WMMVQ_K_SPLIT][16];

    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    const int row0 = blockIdx.x * 16;
    if (row0 >= nrows) {
        return;
    }

    const block_rocmi4 * lane_row =
        (const block_rocmi4 *)(vx + (int64_t)(row0 + (lane % 16)) * row_bytes);

    halfx16_t a;
    halfx16_t b;
    _Float16 * av = (_Float16 *) &a;
    _Float16 * bv = (_Float16 *) &b;
    floatx8_t c;
    floatx8_t czero;

    float res[8] = {};

    for (int kb = w; kb < blocks_per_row; kb += WMMVQ_K_SPLIT) {
        const block_rocmi4 * blk = lane_row + kb;
        const float d = rocmfpx_ue4m3_to_fp32_finite(blk->e);

        // 2 WMMA tiles of 16 elements per 32-element block
        #pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
            const int base = tt * 8; // 8 bytes = 16 nibbles per tile

            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                const uint8_t byte = blk->qs[base + ele/2];
                int qv = (ele % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                // Sign-extend 4-bit to 8-bit signed
                qv = (qv & 0x8) ? (qv | 0xF0) : qv;
                const float act = yf[32*kb + tt*16 + ele];
                av[ele] = __float2half((float)(int8_t)qv);
                bv[ele] = __float2half(act);
            }

            czero = floatx8_t{};
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, czero);

            #pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                res[ele] += d * c[ele];
            }
        }
    }

    // D[i][j]: i = 2*ele + lane/16, j = lane%16; all columns identical.
    if (lane == 0) {
        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            lds[w][2*ele] = res[ele];
        }
    } else if (lane == 16) {
        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            lds[w][2*ele + 1] = res[ele];
        }
    }

    __syncthreads();

    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < WMMVQ_K_SPLIT; ++s) {
            sum += lds[s][threadIdx.x];
        }
        dst[row0 + threadIdx.x] = sum;
    }
#else
    GGML_UNUSED_VARS(vx, yf, dst, nrows, blocks_per_row, row_bytes);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

bool wmmvq_q4_k_eligible(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    // Experimental path, off by default: measured at parity with the dp4a
    // MMVQ head kernel on gfx1100 (149.2 vs 149.6 t/s tg128 median).
    static const bool enabled = getenv("GGML_CUDA_WMMVQ_Q4K") != nullptr;
    if (!enabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // Pure autoregressive decode: a single activation row.
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // Fused GLU epilogues stay on the MMVQ path.
    if (fusion && (fusion->gate || fusion->x_bias || fusion->gate_bias)) {
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src0->ne[1] % 16 != 0) {
        return false;
    }
    const size_t ts0 = ggml_type_size(src0->type);
    if (src0->nb[0] != ts0 || src0->nb[1] != ts0 * (src0->ne[0]/256)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_wmmvq_q4_k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    const int warps = (int) (src0->ne[1] / 16);
    wmmvq_q4_k_f16<<<warps, 32*WMMVQ_K_SPLIT, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) src0->ne[1], (int) (src0->ne[0] / 256), src0->nb[1]);
}

bool wmmvq_rocmi4_eligible(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    static const bool enabled = getenv("GGML_CUDA_WMMVQ_ROCMI4") != nullptr;
    if (!enabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_0_ROCMI4) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (fusion && (fusion->gate || fusion->x_bias || fusion->gate_bias)) {
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src0->ne[1] % 16 != 0) {
        return false;
    }
    const size_t ts0 = ggml_type_size(src0->type);
    if (src0->nb[0] != ts0 || src0->nb[1] != ts0 * (src0->ne[0]/QK_ROCMI4)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_wmmvq_rocmi4(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0_ROCMI4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    const int warps = (int) (src0->ne[1] / 16);
    wmmvq_rocmi4_f16<<<warps, 32*WMMVQ_K_SPLIT, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) src0->ne[1], (int) (src0->ne[0] / QK_ROCMI4), src0->nb[1]);
}
