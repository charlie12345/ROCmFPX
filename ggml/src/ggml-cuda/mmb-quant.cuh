#pragma once
#include "dequantize.cuh"
#include <type_traits>

struct mmb_quant_slice {
    uint16_t * dst;
    int begin;
    int offset = 0;
    struct element {
        uint16_t * dst;
        int index;
        __device__ void operator=(float value) const {
            if (index >= 0 && index < 64) dst[index] = mmb_f2bf(value);
        }
    };
    __device__ mmb_quant_slice operator+(int64_t n) const { return {dst, begin, offset + (int) n}; }
    __device__ element operator[](int64_t n) const { return {dst, offset + (int) n - begin}; }
};

template <ggml_type TYPE>
__device__ __forceinline__ void mmb_decode_slice(const void * row, const int k0, uint16_t * dst, const int lane) {
    constexpr int QK = ggml_cuda_type_traits<TYPE>::qk;
    if constexpr (TYPE == GGML_TYPE_Q1_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q1_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q2_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q2_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q8_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q8_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q2_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q2_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q3_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q3_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const mmb_quant_slice out{dst, k0 % QK};
        const int tid = (k0 % QK) / 64 * 8 + lane;
        dequantize_q4_K<float>(row, k0 / QK, out, tid);
    }
    // q5_K tid covers 4 values of 64-value group tid/16: a slice needs only its group's 16 tids
    else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const mmb_quant_slice out{dst, k0 % QK};
        const int t0 = (k0 % QK)/64*16 + lane;
        dequantize_q5_K<float>(row, k0 / QK, out, t0);
        dequantize_q5_K<float>(row, k0 / QK, out, t0 + 8);
    }
    // q6_K tid covers 4 values spread over its 128-value half tid/32: a slice needs that half's 32 tids
    else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dequantize_q6_K<float>(row, k0 / QK, out, (k0 % QK)/128*32 + lane + 8*j);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_M) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_m<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_XXS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_xxs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_XS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_xs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ3_XXS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq3_xxs<float>(row, k0 / QK, out, tid);
    }
    // dequantize tid covers 8 values of sub-block tid%8: a 64-value slice needs only the 8 tids of its two sub-blocks
    else if constexpr (TYPE == GGML_TYPE_IQ3_S) {
        const mmb_quant_slice out{dst, k0 % QK};
        dequantize_iq3_s<float>(row, k0 / QK, out, (lane % 4)*8 + (k0 % QK)/32 + lane/4);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_XS) {
        const mmb_quant_slice out{dst, k0 % QK};
        dequantize_iq4_xs<float>(row, k0 / QK, out, (lane % 4)*8 + (k0 % QK)/32 + lane/4);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / 32, q = (pos % 32) / 2;
            const block_iq4_nl & b = ((const block_iq4_nl *) row)[ib];
            const int o = ib * 32 + q - k0;
            dst[o] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] & 15]);
            dst[o + 16] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] >> 4]);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_MXFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, q = (pos % QK) / 2;
            const block_mxfp4 & b = ((const block_mxfp4 *) row)[ib];
            const float d = ggml_cuda_e8m0_to_fp32(b.e);
            const int o = ib * QK + q - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] & 15] * 0.5f);
            dst[o + QK / 2] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] >> 4] * 0.5f);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_NVFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, in = pos % QK;
            const int sub = in / QK_NVFP4_SUB, j = (in % QK_NVFP4_SUB) / 2;
            const block_nvfp4 & b = ((const block_nvfp4 *) row)[ib];
            const float d = ggml_cuda_ue4m3_to_fp32(b.d[sub]);
            const uint8_t q = b.qs[sub * (QK_NVFP4_SUB / 2) + j];
            const int o = ib * QK + sub * QK_NVFP4_SUB + j - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[q & 15]);
            dst[o + QK_NVFP4_SUB / 2] = mmb_f2bf(d * kvalues_mxfp4[q >> 4]);
        }
    }
}

// IQ3_S / IQ4_XS slices prefetched into registers one K step ahead: thread t covers row t/8 + 32*p of the tile and
// the 8 values of dequantize thread (lane%4)*8 + ib, ib the slice's sub-block lane/4. Same arithmetic as dequantize_*.
// The fetch keeps every field as loaded (no wait before the step's math); the store assembles the bits and looks
// values up without memory: IQ4_XS through byte permutes of the table in four constants, IQ3_S in an LDS grid.
template <int WTYPE>
constexpr bool mmb_iq_prefetch() { return WTYPE == 32 + GGML_TYPE_IQ3_S || WTYPE == 32 + GGML_TYPE_IQ4_XS; }

struct mmb_iq_raw { uint32_t a, b, c, e, f; };

// the field at p from the aligned word holding it, shifted down (higher bits are other fields; the store masks them).
// Bytes and even-offset 16-bit fields never straddle a word; whole-word loads avoid the serialized sub-word loads.
static __device__ __forceinline__ uint32_t mmb_ld_field(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    return *(const uint32_t *) (a & ~(uintptr_t) 3) >> (8 * (a & 3));
}

template <int WTYPE, int BM>
__device__ __forceinline__ void mmb_iq_fetch(const uint8_t * weights, const size_t row_bytes, const int rows, const int ks, mmb_iq_raw * raw) {
    const int lane = threadIdx.x % 8, il = lane % 4;
    const int k0 = ks * 64, ibs = k0 / QK_K, ib = (k0 % QK_K) / 32 + lane / 4;
#pragma unroll
    for (int p = 0; p < BM / 32; ++p) {
        const int row = threadIdx.x / 8 + 32 * p;
        if (row >= rows) { raw[p] = {0, 0, 0, 0, 0}; continue; }
        if constexpr (WTYPE == 32 + GGML_TYPE_IQ3_S) {
            const block_iq3_s * x = (const block_iq3_s *) (weights + (size_t) row * row_bytes) + ibs;
            raw[p].a = mmb_ld_field(x->qs + 8 * ib + 2 * il);
            raw[p].b = mmb_ld_field(x->qh + ib);
            raw[p].c = mmb_ld_field(x->signs + 4 * ib + il);
            raw[p].e = mmb_ld_field((const uint8_t *) &x->d);
            raw[p].f = mmb_ld_field(x->scales + ib / 2);
        } else {
            // the 136-byte block is 4-byte aligned: whole words for d | scales_h << 16 and the four scales_l bytes
            const block_iq4_xs * x = (const block_iq4_xs *) (weights + (size_t) row * row_bytes) + ibs;
            raw[p].a = *(const uint32_t *) (x->qs + 16 * ib + 4 * il);
            raw[p].b = ((const uint32_t *) x)[0];
            raw[p].c = ((const uint32_t *) x)[1];
            raw[p].e = 0;
            raw[p].f = 0;
        }
    }
}

// IQ3_S fields packed into two words per slice (qs | qh << 16 | signs << 24, d | scale << 16), as the staged
// gate/up tile assembles them
template <int BM, int STRIDE>
__device__ __forceinline__ void mmb_iq3s_store_packed(const mmb_iq_raw * raw, uint16_t * tile, const uint32_t * grid) {
    const int lane = threadIdx.x % 8, il = lane % 4, sb = lane / 4;
#pragma unroll
    for (int p = 0; p < BM / 32; ++p) {
        uint16_t * dst = tile + (threadIdx.x / 8 + 32 * p) * STRIDE + 32 * sb;
        const uint32_t qh = (raw[p].a >> 16) & 0xff, signs = raw[p].a >> 24;
        const uint32_t g1 = grid[(raw[p].a & 0xff) | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t g2 = grid[((raw[p].a >> 8) & 0xff) | ((qh << (7 - 2 * il)) & 256)];
        const float d = mmb_h2f((uint16_t) (raw[p].b & 0xffff)) * (1 + 2 * (raw[p].b >> 16));
        float v[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j]     = d * (float) ((g1 >> 8 * j) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
            v[j + 4] = d * (float) ((g2 >> 8 * j) & 0xff) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
        }
        *(uint4 *) (dst + 8 * il) = make_uint4(mmb_pack2(v[0], v[1]), mmb_pack2(v[2], v[3]), mmb_pack2(v[4], v[5]), mmb_pack2(v[6], v[7]));
    }
}

// Block staging for IQ3_S / IQ4_XS tiles: a 256-value block covers 4 K steps, so every 4th step each tile row's
// block is loaded into LDS with coalesced word loads, and the dequantization reads its fields there. Fetching the
// fields of every step separately took about 20 scattered sub-word loads per thread and was the largest cost of the
// routed tiles. Rows of IQ3_S blocks (110 bytes) are only 2-byte aligned: the staged words start at the aligned word
// before the block, and the field offsets add the block's misalignment.
template <int WTYPE>
constexpr int mmb_stage_words() { return WTYPE == 32 + GGML_TYPE_IQ3_S ? 28 : WTYPE == 32 + GGML_TYPE_IQ4_XS ? 34 : 0; }

template <int WTYPE>
constexpr int mmb_stage_block_bytes() { return WTYPE == 32 + GGML_TYPE_IQ3_S ? (int) sizeof(block_iq3_s) : (int) sizeof(block_iq4_xs); }

template <int WTYPE>
static __device__ __forceinline__ int mmb_stage_off(const uint8_t * w, const size_t row_bytes, const int row, const int bi) {
    return (int) ((uintptr_t) (w + (size_t) row * row_bytes + (size_t) bi * mmb_stage_block_bytes<WTYPE>()) & 3);
}

// words of block bi of rows [0, BM) of NS weight streams into registers (thread t takes words t, t + NT, ...)
template <int WTYPE, int BM, int NS, int NT>
static __device__ __forceinline__ void mmb_stage_load(const uint8_t * const * w, const size_t row_bytes, const int rows, const int bi,
        uint32_t * reg) {
    constexpr int RW = mmb_stage_words<WTYPE>(), N = NS * BM * RW, ITEMS = (N + NT - 1) / NT;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int idx = threadIdx.x + it * NT;
        const int st = idx / (BM * RW), rem = idx % (BM * RW), row = rem / RW, wd = rem % RW;
        reg[it] = 0;
        if (idx < N && row < rows) {
            const uint8_t * base = w[st] + (size_t) row * row_bytes + (size_t) bi * mmb_stage_block_bytes<WTYPE>();
            if (4 * wd < (int) ((uintptr_t) base & 3) + mmb_stage_block_bytes<WTYPE>()) {
                reg[it] = *(const uint32_t *) (((uintptr_t) base & ~(uintptr_t) 3) + 4 * wd);
            }
        }
    }
}

template <int WTYPE, int BM, int NS, int NT>
static __device__ __forceinline__ void mmb_stage_store(const uint32_t * reg, uint32_t * lds) {
    constexpr int RW = mmb_stage_words<WTYPE>(), N = NS * BM * RW, ITEMS = (N + NT - 1) / NT;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int idx = threadIdx.x + it * NT;
        if (idx < N) lds[idx] = reg[it];
    }
}

// the fields of K step ks of the staged rows of one stream, in the layouts mmb_iq_store (IQ4_XS) and
// mmb_iq3s_store_packed (IQ3_S) take
template <int WTYPE, int BM>
static __device__ __forceinline__ void mmb_stage_fields(const uint32_t * lds, const uint8_t * w, const size_t row_bytes, const int rows,
        const int ks, mmb_iq_raw * raw) {
    constexpr int RW = mmb_stage_words<WTYPE>();
    const int lane = threadIdx.x % 8, il = lane % 4, sb = lane / 4;
    const int bi = ks / 4, ib = (ks * 64 % QK_K) / 32 + sb;
#pragma unroll
    for (int p = 0; p < BM / 32; ++p) {
        const int row = threadIdx.x / 8 + 32 * p;
        const uint8_t * rb = (const uint8_t *) (lds + row * RW) + mmb_stage_off<WTYPE>(w, row_bytes, row, bi);
        if constexpr (WTYPE == 32 + GGML_TYPE_IQ3_S) {
            const uint32_t qs = rb[2 + 8 * ib + 2 * il] | ((uint32_t) rb[3 + 8 * ib + 2 * il] << 8);
            raw[p].a = qs | ((uint32_t) rb[66 + ib] << 16) | ((uint32_t) rb[74 + 4 * ib + il] << 24);
            raw[p].b = (rb[0] | ((uint32_t) rb[1] << 8)) | ((uint32_t) ((rb[106 + ib / 2] >> 4 * (ib % 2)) & 0xf) << 16);
        } else {
            // the 136-byte IQ4_XS block is word aligned
            const uint32_t * rw = (const uint32_t *) rb;
            raw[p].a = rw[(8 + 16 * ib + 4 * il) / 4];
            raw[p].b = rw[0];
            raw[p].c = rw[1];
        }
        if (row >= rows) { raw[p].a = 0; raw[p].b = 0; raw[p].c = 0; }
    }
}

// kvalues_iq4nl as four little-endian words, for byte permutes
#define MMB_IQ4NL_W0 0xbfad9881u
#define MMB_IQ4NL_W1 0xf6eaddcfu
#define MMB_IQ4NL_W2 0x26190d01u
#define MMB_IQ4NL_W3 0x71594535u

// 4 table bytes for the 4 nibbles in the low half of each byte of q
static __device__ __forceinline__ uint32_t mmb_iq4nl_lookup4(const uint32_t q) {
    const uint32_t sel = q & 0x07070707u;
    const uint32_t lo  = __builtin_amdgcn_perm(MMB_IQ4NL_W1, MMB_IQ4NL_W0, sel);
    const uint32_t hi  = __builtin_amdgcn_perm(MMB_IQ4NL_W3, MMB_IQ4NL_W2, sel);
    return __builtin_amdgcn_perm(hi, lo, 0x03020100u | ((q & 0x08080808u) >> 1));
}

// grid: iq3s_grid in LDS for IQ3_S (unused otherwise); ks: the K step the raw values were fetched for
template <int WTYPE, int BM, int STRIDE>
__device__ __forceinline__ void mmb_iq_store(const mmb_iq_raw * raw, uint16_t * tile, const int ks, const uint32_t * grid) {
    const int lane = threadIdx.x % 8, il = lane % 4, sb = lane / 4;
    const int ib = (ks * 64 % QK_K) / 32 + sb;
#pragma unroll
    for (int p = 0; p < BM / 32; ++p) {
        uint16_t * dst = tile + (threadIdx.x / 8 + 32 * p) * STRIDE + 32 * sb;
        if constexpr (WTYPE == 32 + GGML_TYPE_IQ3_S) {
            const uint32_t qs = raw[p].a & 0xffff, qh = raw[p].b & 0xff, signs = raw[p].c & 0xff;
            const uint32_t g1 = grid[(qs & 0xff) | ((qh << (8 - 2 * il)) & 256)];
            const uint32_t g2 = grid[((qs >> 8) & 0xff) | ((qh << (7 - 2 * il)) & 256)];
            const float d = mmb_h2f((uint16_t) (raw[p].e & 0xffff)) * (1 + 2 * ((raw[p].f >> 4 * (ib % 2)) & 0xf));
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j]     = d * (float) ((g1 >> 8 * j) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[j + 4] = d * (float) ((g2 >> 8 * j) & 0xff) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
            *(uint4 *) (dst + 8 * il) = make_uint4(mmb_pack2(v[0], v[1]), mmb_pack2(v[2], v[3]), mmb_pack2(v[4], v[5]), mmb_pack2(v[6], v[7]));
        } else {
            const uint32_t sl = (raw[p].c >> 8 * (ib / 2)) & 0xff, sh = raw[p].b >> 16;
            const uint32_t sc = ((sl >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            const float d = mmb_h2f((uint16_t) (raw[p].b & 0xffff)) * ((int) sc - 32);
            const uint32_t tlo = mmb_iq4nl_lookup4(raw[p].a), thi = mmb_iq4nl_lookup4(raw[p].a >> 4);
            float lo[4], hi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo[j] = d * (float) (int8_t) (tlo >> 8 * j);
                hi[j] = d * (float) (int8_t) (thi >> 8 * j);
            }
            *(uint2 *) (dst + 4 * il)      = make_uint2(mmb_pack2(lo[0], lo[1]), mmb_pack2(lo[2], lo[3]));
            *(uint2 *) (dst + 4 * il + 16) = make_uint2(mmb_pack2(hi[0], hi[1]), mmb_pack2(hi[2], hi[3]));
        }
    }
}

template <int WTYPE>
__host__ __device__ constexpr size_t mmb_row_bytes(int k) {
    if constexpr (WTYPE == 0) return (size_t)(k / 32) * 18;
    else if constexpr (WTYPE == 1) return (size_t)(k / 32) * 34;
    else if constexpr (WTYPE == 2) return (size_t)k * 2;
    else {
        using traits = ggml_cuda_type_traits<(ggml_type)(WTYPE - 32)>;
        return (size_t)(k / traits::qk) * traits::bs;
    }
}

template <int WTYPE, int BM, int STRIDE>
__device__ __forceinline__ void mmb_load_quant_tile(const uint8_t * weights, size_t row_bytes, int rows, int ks, uint16_t * tile) {
    for (int row = threadIdx.x / 8; row < BM; row += blockDim.x / 8) {
        uint16_t * dst = tile + row * STRIDE;
        const int lane = threadIdx.x % 8;
        if (row < rows) mmb_decode_slice<(ggml_type)(WTYPE - 32)>(weights + row * row_bytes, ks * 64, dst, lane);
        else {
#pragma unroll
            for (int k = lane; k < 64; k += 8) dst[k] = 0;
        }
    }
}

static bool mmb_quant_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default: return false;
    }
}

template <typename Fn>
static void mmb_dispatch_quant(ggml_type type, Fn fn) {
    switch (type) {
        case GGML_TYPE_Q1_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q1_0>{}); break;
        case GGML_TYPE_Q2_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_0>{}); break;
        case GGML_TYPE_Q4_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_0>{}); break;
        case GGML_TYPE_Q4_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_1>{}); break;
        case GGML_TYPE_Q5_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_0>{}); break;
        case GGML_TYPE_Q5_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_1>{}); break;
        case GGML_TYPE_Q8_0: fn(std::integral_constant<int, 1>{}); break;
        case GGML_TYPE_Q2_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_K>{}); break;
        case GGML_TYPE_Q3_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q3_K>{}); break;
        case GGML_TYPE_Q4_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_IQ1_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_S>{}); break;
        case GGML_TYPE_IQ1_M: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_M>{}); break;
        case GGML_TYPE_IQ2_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XXS>{}); break;
        case GGML_TYPE_IQ2_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XS>{}); break;
        case GGML_TYPE_IQ2_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_S>{}); break;
        case GGML_TYPE_IQ3_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ3_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_S>{}); break;
        case GGML_TYPE_IQ4_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ4_XS>{}); break;
        case GGML_TYPE_IQ4_NL: fn(std::integral_constant<int, 0>{}); break;
        case GGML_TYPE_MXFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_MXFP4>{}); break;
        case GGML_TYPE_NVFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_NVFP4>{}); break;
        default: GGML_ABORT("unsupported MMB quant type");
    }
}

__device__ __forceinline__ void mmb_dq_q4k_slice(uint4 q0, uint4 q1, uint4 meta, int slice, uint32_t * out) {
    const float d = mmb_h2f((uint16_t) meta.x), dm = mmb_h2f((uint16_t)(meta.x >> 16));
    auto byte = [&](int i) { const uint32_t word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w; return (word >> (8 * (i & 3))) & 255; };
    auto scale_min = [&](int i, float & ds, float & ms) {
        const uint32_t sc = i < 4 ? byte(i) & 63 : (byte(i + 4) & 15) | ((byte(i - 4) >> 6) << 4);
        const uint32_t mn = i < 4 ? byte(i + 4) & 63 : (byte(i + 4) >> 4) | ((byte(i) >> 6) << 4);
        ds = d * sc; ms = dm * mn;
    };
    float d0, d1, m0, m1;
    scale_min(2 * slice, d0, m0); scale_min(2 * slice + 1, d1, m1);
    const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t q = words[j];
        out[2*j] = mmb_pack2(d0 * (q & 15) - m0, d0 * ((q >> 8) & 15) - m0);
        out[2*j + 1] = mmb_pack2(d0 * ((q >> 16) & 15) - m0, d0 * ((q >> 24) & 15) - m0);
        out[16 + 2*j] = mmb_pack2(d1 * ((q >> 4) & 15) - m1, d1 * ((q >> 12) & 15) - m1);
        out[16 + 2*j + 1] = mmb_pack2(d1 * ((q >> 20) & 15) - m1, d1 * (q >> 28) - m1);
    }
}


__device__ __forceinline__ void mmb_dq_q51_pair(uint4 a, uint4 b, uint4 c, uint32_t * out) {
    const uint32_t words[12] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x,c.y,c.z,c.w};
#pragma unroll
    for (int block = 0; block < 2; ++block) {
        const uint32_t dm = words[block * 6], high = words[block * 6 + 1];
        const float d = mmb_h2f((uint16_t)dm), m = mmb_h2f((uint16_t)(dm >> 16));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = words[block * 6 + 2 + j];
            float lo[4], hi[4];
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int low = ((q >> (8 * lane)) & 15) | (((high >> (4 * j + lane)) & 1) << 4);
                const int upper = ((q >> (8 * lane + 4)) & 15) | (((high >> (16 + 4 * j + lane)) & 1) << 4);
                lo[lane] = d * low + m; hi[lane] = d * upper + m;
            }
            out[block * 16 + 2*j] = mmb_pack2(lo[0],lo[1]);
            out[block * 16 + 2*j + 1] = mmb_pack2(lo[2],lo[3]);
            out[block * 16 + 8 + 2*j] = mmb_pack2(hi[0],hi[1]);
            out[block * 16 + 8 + 2*j + 1] = mmb_pack2(hi[2],hi[3]);
        }
    }
}
