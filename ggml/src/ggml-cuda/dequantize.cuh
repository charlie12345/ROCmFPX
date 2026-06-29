#include "common.cuh"
#include "../../rocmfp4/rocmfp4_hip_scale.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_rocmfp4(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp4 * x = (const block_rocmfp4 *) vx;

    const int q = x[ib].qs[iqs];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[1]);

    v.x = d0 * rocmfp4_decode_i8(q);
    v.y = d1 * rocmfp4_decode_i8(q >> 4);
}

static __device__ __forceinline__ void dequantize_rocmfp4_fast(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp4_fast * x = (const block_rocmfp4_fast *) vx;

    const int q = x[ib].qs[iqs];
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e);

    v.x = d * rocmfp4_decode_i8(q);
    v.y = d * rocmfp4_decode_i8(q >> 4);
}

template<int qs>
static __device__ __forceinline__ uint32_t rocmfpx_load_qs_window_cuda(const uint8_t * src, const int byte_pos) {
    uint32_t v = (uint32_t) src[byte_pos + 0];

    if (byte_pos + 1 < qs) {
        v |= (uint32_t) src[byte_pos + 1] << 8;
    }
    if (byte_pos + 2 < qs) {
        v |= (uint32_t) src[byte_pos + 2] << 16;
    }

    return v;
}

static __device__ __forceinline__ uint32_t rocmfpx_get_fp3_code_cuda(const uint8_t * src, const int i) {
    const int bit_pos  = i * 3;
    const int byte_pos = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    return (rocmfpx_load_qs_window_cuda<QS_ROCMFP3>(src, byte_pos) >> shift) & 7u;
}

static __device__ __forceinline__ uint32_t rocmfpx_get_fp6_code_cuda(const uint8_t * src, const int i) {
    const int bit_pos  = i * 6;
    const int byte_pos = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    return (rocmfpx_load_qs_window_cuda<QS_ROCMFP6>(src, byte_pos) >> shift) & 63u;
}

static __device__ __forceinline__ int rocmfpx_decode_fp3_code_cuda(const uint32_t code) {
    const uint32_t mag_code = code & 3u;
    const int mag = mag_code == 3u ? 4 : (int) mag_code;
    return (code & 4u) ? -mag : mag;
}

static __device__ __forceinline__ int rocmfpx_decode_fp6_code_cuda(const uint32_t code) {
    const int mag = (int) (code & 31u);
    return (code & 32u) ? -mag : mag;
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp3(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp3 * x = (const block_rocmfp3 *) vx;

    const int i0 = iqs + 0;
    const int i1 = iqs + 1;
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP3/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP3/2]);

    v.x = d0 * (float) rocmfpx_decode_fp3_code_cuda(rocmfpx_get_fp3_code_cuda(x[ib].qs, i0));
    v.y = d1 * (float) rocmfpx_decode_fp3_code_cuda(rocmfpx_get_fp3_code_cuda(x[ib].qs, i1));
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp6(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp6_device * x = (const block_rocmfp6_device *) vx;

    const int i0 = iqs + 0;
    const int i1 = iqs + 1;
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP6/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP6/2]);

#if GGML_ROCMFP6_EXPANDED_DEVICE
    v.x = d0 * (float) x[ib].qs[i0];
    v.y = d1 * (float) x[ib].qs[i1];
#else
    v.x = d0 * (float) rocmfpx_decode_fp6_code_cuda(rocmfpx_get_fp6_code_cuda(x[ib].qs, i0));
    v.y = d1 * (float) rocmfpx_decode_fp6_code_cuda(rocmfpx_get_fp6_code_cuda(x[ib].qs, i1));
#endif
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp8(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp8 * x = (const block_rocmfp8 *) vx;

    const float d = rocmfpx_ue4m3_to_fp32_finite(x[ib].e);
    v.x = d * (float) x[ib].qs[iqs + 0];
    v.y = d * (float) x[ib].qs[iqs + 1];
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// ============================================================
// TurboQuant GPU dequantize device functions
// ============================================================

__device__ __constant__ static float dc_codebook_3bit[8] = {
    -0.1883972972f, -0.1181399059f, -0.0665857641f, -0.0216044751f,
     0.0216041461f,  0.0665854520f,  0.1181396281f,  0.1883970748f
};

__device__ __constant__ static float dc_codebook_4bit[16] = {
    -0.2376389871f, -0.1808080141f, -0.1417777640f, -0.1102646123f,
    -0.0828112376f, -0.0577640422f, -0.0341540905f, -0.0113168380f,
     0.0112761586f,  0.0341139667f,  0.0577250301f,  0.0827738972f,
     0.1102295202f,  0.1417455465f,  0.1807794468f,  0.2376153882f
};

static __device__ __forceinline__ void dequantize_turbo3_0(
    const void * vx, const int64_t ib, const int iqs, float2 & v)
{
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx + ib;
    const uint8_t * qs = x->qs;

    // Unpack two consecutive 3-bit indices
    int elem0 = iqs * 2;
    int elem1 = iqs * 2 + 1;

    // Extract 3-bit value for elem0
    int bit_off0 = elem0 * 3;
    int byte0 = bit_off0 / 8;
    int shift0 = bit_off0 % 8;
    uint16_t raw0 = (uint16_t)qs[byte0] >> shift0;
    if (shift0 > 5 && byte0 + 1 < 12)
        raw0 |= (uint16_t)qs[byte0 + 1] << (8 - shift0);
    uint8_t idx0 = (uint8_t)(raw0 & 0x07);

    // Extract 3-bit value for elem1
    int bit_off1 = elem1 * 3;
    int byte1 = bit_off1 / 8;
    int shift1 = bit_off1 % 8;
    uint16_t raw1 = (uint16_t)qs[byte1] >> shift1;
    if (shift1 > 5 && byte1 + 1 < 12)
        raw1 |= (uint16_t)qs[byte1 + 1] << (8 - shift1);
    uint8_t idx1 = (uint8_t)(raw1 & 0x07);

    const float norm = __half2float(x->d);
    v.x = dc_codebook_3bit[idx0] * norm;
    v.y = dc_codebook_3bit[idx1] * norm;
}

static __device__ __forceinline__ void dequantize_turbo4_0(
    const void * vx, const int64_t ib, const int iqs, float2 & v)
{
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx + ib;

    // 4-bit: 2 values per byte, simple nibble extraction
    uint8_t packed = x->qs[iqs];
    uint8_t idx0 = packed & 0x0F;
    uint8_t idx1 = (packed >> 4) & 0x0F;

    const float norm = __half2float(x->d);
    v.x = dc_codebook_4bit[idx0] * norm;
    v.y = dc_codebook_4bit[idx1] * norm;
}
