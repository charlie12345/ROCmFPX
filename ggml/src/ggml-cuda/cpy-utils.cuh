#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "../../rocmfp4/rocmfp4_hip_scale.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

static __device__ __forceinline__ uint8_t rocmfp4_best_index_scaled_cuda(float x, float inv_scale_half) {
    const float a = fabsf(x * inv_scale_half);
    if (a <= 0.5f) {
        return 0;
    }

    const bool neg = x < 0.0f;
    if (a <= 1.5f) {
        return neg ?  9 : 1;
    }
    if (a <= 2.5f) {
        return neg ? 10 : 2;
    }
    if (a <= 3.5f) {
        return neg ? 11 : 3;
    }
    if (a <= 5.0f) {
        return neg ? 12 : 4;
    }
    if (a <= 7.0f) {
        return neg ? 13 : 5;
    }
    if (a <= 9.0f) {
        return neg ? 14 : 6;
    }

    return neg ? 15 : 7;
}

static __device__ __forceinline__ int8_t rocmfp4_best_value_scaled_cuda(float x, float inv_scale_half) {
    const float a = fabsf(x * inv_scale_half);
    if (a <= 0.5f) {
        return 0;
    }

    const bool neg = x < 0.0f;
    int mag;
    if (a <= 1.5f) {
        mag = 1;
    } else if (a <= 2.5f) {
        mag = 2;
    } else if (a <= 3.5f) {
        mag = 3;
    } else if (a <= 5.0f) {
        mag = 4;
    } else if (a <= 7.0f) {
        mag = 6;
    } else if (a <= 9.0f) {
        mag = 8;
    } else {
        mag = 10;
    }

    return neg ? -mag : mag;
}

static __device__ __forceinline__ uint8_t rocmfp4_nearest_scale_ue4m3_cuda(float target_scale_half) {
    if (!(target_scale_half > 0.0f) || !isfinite(target_scale_half)) {
        return 1;
    }

    int lo = 1;
    int hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) mid) < target_scale_half) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) lo);
    const float lo_scale = rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) (lo - 1));

    // Keep midpoint ties on the lower scale byte to match the CPU reference.
    return (target_scale_half - lo_scale <= hi_scale - target_scale_half) ? (uint8_t) (lo - 1) : (uint8_t) lo;
}

template<int start, int n>
static __device__ __forceinline__ float rocmfp4_block_mse_for_scale_cuda(
        const float * __restrict__ x, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        const int8_t qv = rocmfp4_best_value_scaled_cuda(xi, inv_scale_half);
        const float yi = (float) qv * scale_half;
        const float d = xi - yi;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

template<int start, int n>
static __device__ __forceinline__ uint8_t rocmfp4_choose_scale_ue4m3_cuda(const float * __restrict__ x) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        const float ax = fabsf(xi);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const int start_e = rocmfp4_nearest_scale_ue4m3_cuda(max_abs / 10.0f);
    int best_e = 0;
    float best_err = FLT_MAX;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale_half = rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) e0);
            const float clip_delta = max_abs - 10.0f*scale_half;
            if (clip_delta > 0.0f && clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfp4_block_mse_for_scale_cuda<start, n>(x, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = e0;
                }
            }
        }

        const int e1 = start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfp4_block_mse_for_scale_cuda<start, n>(x, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return (uint8_t) best_e;
}

static __device__ void quantize_f32_rocmfp4_block(const float * __restrict__ x, block_rocmfp4 * __restrict__ y) {
    const uint8_t e0 = rocmfp4_choose_scale_ue4m3_cuda<0, QK_ROCMFP4/2>(x);
    const uint8_t e1 = rocmfp4_choose_scale_ue4m3_cuda<QK_ROCMFP4/2, QK_ROCMFP4/2>(x);
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(e0);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(e1);
    const float id0 = d0 > 0.0f ? 1.0f/d0 : 0.0f;
    const float id1 = d1 > 0.0f ? 1.0f/d1 : 0.0f;

    y->e[0] = e0;
    y->e[1] = e1;

    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const float v0 = x[j];
        const float v1 = x[j + QK_ROCMFP4/2];
        const uint8_t q0 = rocmfp4_best_index_scaled_cuda(v0, id0);
        const uint8_t q1 = rocmfp4_best_index_scaled_cuda(v1, id1);
        y->qs[j] = q0 | (q1 << 4);
    }
}

static __device__ void quantize_f32_rocmfp4_fast_block(const float * __restrict__ x, block_rocmfp4_fast * __restrict__ y) {
    const uint8_t e = rocmfp4_choose_scale_ue4m3_cuda<0, QK_ROCMFP4>(x);
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(e);
    const float id = d > 0.0f ? 1.0f/d : 0.0f;

    y->e = e;

    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const float v0 = x[j];
        const float v1 = x[j + QK_ROCMFP4/2];
        const uint8_t q0 = rocmfp4_best_index_scaled_cuda(v0, id);
        const uint8_t q1 = rocmfp4_best_index_scaled_cuda(v1, id);
        y->qs[j] = q0 | (q1 << 4);
    }
}

static __device__ __forceinline__ int8_t rocmfpx_fp8_quantize_code_cuda(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) roundf(x * inv_scale);
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    return (int8_t) q;
}

static __device__ __forceinline__ float rocmfpx_max_abs_range_cuda(const float * __restrict__ x, const int offset, const int n) {
    float max_abs = 0.0f;

#pragma unroll
    for (int j = 0; j < n; ++j) {
        const float xj = x[offset + j];
        if (isfinite(xj)) {
            max_abs = fmaxf(max_abs, fabsf(xj));
        }
    }

    return max_abs;
}

static __device__ __forceinline__ void rocmfpx_set_bits_cuda(uint8_t * dst, const int bit_pos, const int nbits, const uint32_t code) {
#pragma unroll
    for (int bit = 0; bit < nbits; ++bit) {
        const int dst_bit = bit_pos + bit;
        const uint8_t mask = (uint8_t) (1u << (dst_bit & 7));

        if (code & (1u << bit)) {
            dst[dst_bit >> 3] |= mask;
        } else {
            dst[dst_bit >> 3] &= (uint8_t) ~mask;
        }
    }
}

static __device__ __forceinline__ uint8_t rocmfpx_fp3_quantize_code_cuda(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float ax = fabsf(x * inv_scale);
    uint8_t mag;
    if (ax <= 0.5f) {
        mag = 0;
    } else if (ax <= 1.5f) {
        mag = 1;
    } else if (ax <= 3.0f) {
        mag = 2;
    } else {
        mag = 3;
    }

    return mag == 0 ? 0 : (uint8_t) ((x < 0.0f ? 4u : 0u) | mag);
}

static __device__ __forceinline__ int rocmfpx_fp3_decode_value_cuda(uint8_t code) {
    const int mag = (code & 3u) == 3u ? 4 : (int) (code & 3u);
    return (code & 4u) ? -mag : mag;
}

template<int start, int n>
static __device__ __forceinline__ float rocmfpx_fp3_block_mse_for_scale_cuda(
        const float * __restrict__ x, int e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) e);
    const float inv_scale = scale > 0.0f ? 1.0f/scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        if (!isfinite(xi)) {
            continue;
        }

        const uint8_t code = rocmfpx_fp3_quantize_code_cuda(xi, inv_scale);
        const float yi = (float) rocmfpx_fp3_decode_value_cuda(code) * scale;
        const float d = xi - yi;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

template<int start, int n>
static __device__ __forceinline__ uint8_t rocmfpx_choose_scale_fp3_mse_cuda(const float * __restrict__ x) {
    const float max_abs = rocmfpx_max_abs_range_cuda(x, start, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const int start_e = rocmfpx_nearest_scale_ue4m3_cuda(max_abs / 4.0f);
    int best_e = start_e;
    float best_err = FLT_MAX;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f*scale;
            if (clip_delta > 0.0f && clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_fp3_block_mse_for_scale_cuda<start, n>(x, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = e0;
                }
            }
        }

        const int e1 = start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_fp3_block_mse_for_scale_cuda<start, n>(x, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return (uint8_t) best_e;
}

static __device__ __forceinline__ uint8_t rocmfpx_fp6_quantize_code_cuda(float x, float inv_scale) {
    if (!isfinite(x) || !isfinite(inv_scale) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) roundf(x * inv_scale);
    q = q > 31 ? 31 : (q < -32 ? -32 : q);
    return q == 0 ? 0 : (uint8_t) (q < 0 ? (32u | ((uint8_t) -q & 31u)) : (uint8_t) q);
}

static __device__ __forceinline__ int rocmfpx_fp6_decode_value_cuda(uint8_t code) {
    const int mag = (int) (code & 31u);
    return (code & 32u) ? -(mag == 0 ? 32 : mag) : mag;
}

template<int start, int n>
static __device__ __forceinline__ float rocmfpx_fp6_block_mse_for_scale_cuda(
        const float * __restrict__ x, int e, float best_err) {
    const float scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) e);
    const float inv_scale = scale > 0.0f ? 1.0f/scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        if (!isfinite(xi)) {
            continue;
        }

        const uint8_t code = rocmfpx_fp6_quantize_code_cuda(xi, inv_scale);
        const float yi = (float) rocmfpx_fp6_decode_value_cuda(code) * scale;
        const float d = xi - yi;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

template<int start, int n>
static __device__ __forceinline__ uint8_t rocmfpx_choose_scale_fp6_mse_cuda(const float * __restrict__ x) {
    const float max_abs = rocmfpx_max_abs_range_cuda(x, start, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const int start_e = rocmfpx_nearest_scale_ue4m3_cuda(max_abs / 31.0f);
    int best_e = start_e;
    float best_err = FLT_MAX;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) e0);
            const float clip_delta = max_abs - 31.0f*scale;
            if (clip_delta > 0.0f && clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_fp6_block_mse_for_scale_cuda<start, n>(x, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = e0;
                }
            }
        }

        const int e1 = start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_fp6_block_mse_for_scale_cuda<start, n>(x, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return (uint8_t) best_e;
}

static __device__ void quantize_f32_rocmfpx_fp3_block(const float * __restrict__ x, block_rocmfp3 * __restrict__ y) {
#pragma unroll
    for (int j = 0; j < QS_ROCMFP3; ++j) {
        y->qs[j] = 0;
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int offset = half * (QK_ROCMFP3/2);
        y->e[half] = half == 0 ?
            rocmfpx_choose_scale_fp3_mse_cuda<0, QK_ROCMFP3/2>(x) :
            rocmfpx_choose_scale_fp3_mse_cuda<QK_ROCMFP3/2, QK_ROCMFP3/2>(x);

        const float d = rocmfpx_ue4m3_to_fp32_finite(y->e[half]);
        const float id = d > 0.0f ? 1.0f/d : 0.0f;

#pragma unroll
        for (int j = 0; j < QK_ROCMFP3/2; ++j) {
            const int i = offset + j;
            rocmfpx_set_bits_cuda(y->qs, i*3, 3, rocmfpx_fp3_quantize_code_cuda(x[i], id));
        }
    }
}

static __device__ void quantize_f32_rocmfpx_fp6_block(const float * __restrict__ x, block_rocmfp6 * __restrict__ y) {
#pragma unroll
    for (int j = 0; j < QS_ROCMFP6; ++j) {
        y->qs[j] = 0;
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int offset = half * (QK_ROCMFP6/2);
        y->e[half] = half == 0 ?
            rocmfpx_choose_scale_fp6_mse_cuda<0, QK_ROCMFP6/2>(x) :
            rocmfpx_choose_scale_fp6_mse_cuda<QK_ROCMFP6/2, QK_ROCMFP6/2>(x);

        const float d = rocmfpx_ue4m3_to_fp32_finite(y->e[half]);
        const float id = d > 0.0f ? 1.0f/d : 0.0f;

#pragma unroll
        for (int j = 0; j < QK_ROCMFP6/2; ++j) {
            const int i = offset + j;
            rocmfpx_set_bits_cuda(y->qs, i*6, 6, rocmfpx_fp6_quantize_code_cuda(x[i], id));
        }
    }
}

static __device__ void quantize_f32_rocmfpx_fp8_block(const float * __restrict__ x, block_rocmfp8 * __restrict__ y) {
    y->e = rocmfpx_nearest_scale_ue4m3_cuda(rocmfpx_max_abs_range_cuda(x, 0, QK_ROCMFP8) / 127.0f);
    const float d = rocmfpx_ue4m3_to_fp32_finite(y->e);
    const float id = d > 0.0f ? 1.0f/d : 0.0f;

#pragma unroll
    for (int j = 0; j < QK_ROCMFP8; ++j) {
        y->qs[j] = rocmfpx_fp8_quantize_code_cuda(x[j], id);
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfp4(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP4];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfp4_block(tmp, (block_rocmfp4 *) cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfp4_fast(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP4];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfp4_fast_block(tmp, (block_rocmfp4_fast *) cdsti);
}

static __device__ void cpy_blck_f32_rocmfp4(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4<float>(cxi, cdsti);
}

static __device__ void cpy_blck_f32_rocmfp4_fast(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4_fast<float>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfp4(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4<half>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfp4_fast(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4_fast<half>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfp4(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4<nv_bfloat16>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfp4_fast(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4_fast<nv_bfloat16>(cxi, cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfpx_fp3(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP3];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP3; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfpx_fp3_block(tmp, (block_rocmfp3 *) cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfpx_fp6(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP6];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP6; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfpx_fp6_block(tmp, (block_rocmfp6 *) cdsti);
}

static __device__ void cpy_blck_f32_rocmfpx_fp3(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp3<float>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfpx_fp3(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp3<half>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfpx_fp3(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp3<nv_bfloat16>(cxi, cdsti);
}

static __device__ void cpy_blck_f32_rocmfpx_fp6(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp6<float>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfpx_fp6(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp6<half>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfpx_fp6(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp6<nv_bfloat16>(cxi, cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfpx_fp8(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP8];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP8; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfpx_fp8_block(tmp, (block_rocmfp8 *) cdsti);
}

static __device__ void cpy_blck_f32_rocmfpx_fp8(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp8<float>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfpx_fp8(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp8<half>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfpx_fp8(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfpx_fp8<nv_bfloat16>(cxi, cdsti);
}

__device__ static const float tq_codebook_3bit_q[8] = {
    -0.1883972972f, -0.1181399059f, -0.0665857641f, -0.0216044751f,
     0.0216041461f,  0.0665854520f,  0.1181396281f,  0.1883970748f
};

__device__ static const float tq_codebook_4bit_q[16] = {
    -0.2376389871f, -0.1808080141f, -0.1417777640f, -0.1102646123f,
    -0.0828112376f, -0.0577640422f, -0.0341540905f, -0.0113168380f,
     0.0112761586f,  0.0341139667f,  0.0577250301f,  0.0827738972f,
     0.1102295202f,  0.1417455465f,  0.1807794468f,  0.2376153882f
};

static __device__ uint8_t tq_nearest_codebook(float val, const float * codebook, int n) {
    float best_dist = fabsf(val - codebook[0]);
    uint8_t best_idx = 0;
    for (int i = 1; i < n; i++) {
        float dist = fabsf(val - codebook[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = (uint8_t) i;
        }
    }
    return best_idx;
}

static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ x, block_turbo3_0 * __restrict__ y) {
    float sum_sq = 0.0f;
    for (int j = 0; j < TURBO3_BLOCK_SIZE; j++) {
        sum_sq += x[j] * x[j];
    }
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    uint8_t indices[32];
    for (int j = 0; j < TURBO3_BLOCK_SIZE; j++) {
        indices[j] = tq_nearest_codebook(x[j] * inv_norm, tq_codebook_3bit_q, 8);
    }

    memset(y->qs, 0, 12);
    for (int j = 0; j < 32; j++) {
        int bit_off = j * 3;
        int byte_idx = bit_off / 8;
        int shift = bit_off % 8;
        y->qs[byte_idx] |= (uint8_t) ((indices[j] & 0x07) << shift);
        if (shift > 5 && byte_idx + 1 < 12) {
            y->qs[byte_idx + 1] |= (uint8_t) ((indices[j] & 0x07) >> (8 - shift));
        }
    }
}

static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ x, block_turbo4_0 * __restrict__ y) {
    float sum_sq = 0.0f;
    for (int j = 0; j < TURBO4_BLOCK_SIZE; j++) {
        sum_sq += x[j] * x[j];
    }
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    for (int j = 0; j < TURBO4_BLOCK_SIZE / 2; j++) {
        uint8_t idx0 = tq_nearest_codebook(x[2*j]     * inv_norm, tq_codebook_4bit_q, 16);
        uint8_t idx1 = tq_nearest_codebook(x[2*j + 1] * inv_norm, tq_codebook_4bit_q, 16);
        y->qs[j] = (idx0 & 0x0F) | ((idx1 & 0x0F) << 4);
    }
}

static __device__ void cpy_blck_f32_turbo3_0(const char * cxi, char * cdsti) {
    quantize_f32_turbo3_0_block((const float *) cxi, (block_turbo3_0 *) cdsti);
}

static __device__ void cpy_blck_f32_turbo4_0(const char * cxi, char * cdsti) {
    quantize_f32_turbo4_0_block((const float *) cxi, (block_turbo4_0 *) cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
