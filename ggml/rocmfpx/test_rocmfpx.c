#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void fill_row(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        const float wave = 0.75f*sinf((float) i * 0.37f) + 0.25f*cosf((float) i * 0.13f);
        const float ramp = ((float) (i % 11) - 5.0f) * 0.035f;
        x[i] = wave + ramp;
    }

    x[7]  =  3.25f;
    x[19] = -2.75f;
    x[43] =  1.875f;
}

static float mse(const float * a, const float * b, int n) {
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += d*d;
    }

    return err / (float) n;
}

static float weighted_mse(const float * a, const float * b, const float * w, int n) {
    float err = 0.0f;
    float sum_w = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += w[i]*d*d;
        sum_w += w[i];
    }

    return sum_w > 0.0f ? err / sum_w : 0.0f;
}

static void fill_imatrix_case(float * src, float * imatrix, int n, float base, float outlier) {
    for (int i = 0; i < n; ++i) {
        const float sign = (i % 2) ? 1.0f : -1.0f;
        src[i] = sign * base * (1.0f + 0.03f*(float)(i % 5));
        imatrix[i] = 100.0f;
    }

    src[0] = outlier;
    imatrix[0] = 0.0f;
}

static void check_weighted_imatrix_fp3(void) {
    enum { N = QK_ROCMFP3 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp3 q_plain[N / QK_ROCMFP3];
    block_rocmfp3 q_weighted[N / QK_ROCMFP3];

    fill_imatrix_case(src, imatrix, N, 0.21f, 9.0f);

    rocmfpx_quantize_fp3(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp3(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp3(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp3(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP3 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_fp6(void) {
    enum { N = QK_ROCMFP6 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp6 q_plain[N / QK_ROCMFP6];
    block_rocmfp6 q_weighted[N / QK_ROCMFP6];

    fill_imatrix_case(src, imatrix, N, 0.045f, 6.0f);

    rocmfpx_quantize_fp6(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp6(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp6(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp6(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP6 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_fp8(void) {
    enum { N = QK_ROCMFP8 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp8 q_plain[N / QK_ROCMFP8];
    block_rocmfp8 q_weighted[N / QK_ROCMFP8];

    fill_imatrix_case(src, imatrix, N, 0.008f, 8.0f);

    rocmfpx_quantize_fp8(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp8(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp8(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp8(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP8 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_i4(void) {
    enum { N = QK_ROCMI4 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmi4 q_plain[N / QK_ROCMI4];
    block_rocmi4 q_weighted[N / QK_ROCMI4];

    fill_imatrix_case(src, imatrix, N, 0.21f, 9.0f);

    rocmfpx_quantize_i4(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_i4(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_i4(q_plain,    plain,    N);
    rocmfpx_dequantize_row_i4(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmI4 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_fp2_swar_codebook(void) {
    static const int codebook[4] = {-4, -1, 1, 4};

    for (uint32_t packed = 0; packed <= UINT8_MAX; ++packed) {
        // Keep this arithmetic identical to rocmfpx_vq_fp2_pack4() in the
        // Vulkan Q8_1 shader so every possible packed byte is covered.
        uint32_t codes = (packed | (packed << 12u)) & 0x000f000fu;
        codes = (codes | (codes << 6u)) & 0x03030303u;
        const uint32_t high = (codes >> 1u) & 0x01010101u;
        const uint32_t lanes = 0xfcfcfcfcu + 3u * codes - high - (high << 8u);

        for (int lane = 0; lane < 4; ++lane) {
            const int got_u8 = (int) ((lanes >> (8 * lane)) & 0xffu);
            const int got = got_u8 >= 128 ? got_u8 - 256 : got_u8;
            const int expected = codebook[(packed >> (2 * lane)) & 3u];
            assert(got == expected);
        }

        block_rocmfp2 block = {{0}, {0x40, 0x40}};
        float dequantized[QK_ROCMFP2];
        for (int i = 0; i < QS_ROCMFP2; ++i) {
            block.qs[i] = (uint8_t) packed;
        }
        rocmfpx_dequantize_row_fp2(&block, dequantized, QK_ROCMFP2);

        for (int i = 0; i < QK_ROCMFP2; ++i) {
            const int expected = codebook[(packed >> (2 * (i % 4))) & 3u];
            assert(dequantized[i] == (float) expected);
        }
    }

    printf("ROCmFP2 Vulkan SWAR: all 256 packed bytes match {-4,-1,1,4}\n");
}

int main(void) {
    enum { N = 64 };

    float src[N];
    float fp3[N];
    float fp6[N];
    float fp8[N];
    float i4[N];

    block_rocmfp3 q3[N / QK_ROCMFP3];
    block_rocmfp6 q6[N / QK_ROCMFP6];
    block_rocmfp8 q8[N / QK_ROCMFP8];
    block_rocmi4 qi4[N / QK_ROCMI4];

    fill_row(src, N);

    rocmfpx_quantize_row_fp3_ref(src, q3, N);
    rocmfpx_quantize_row_fp6_ref(src, q6, N);
    rocmfpx_quantize_row_fp8_ref(src, q8, N);
    rocmfpx_quantize_row_i4_ref(src, qi4, N);

    assert(rocmfpx_validate_row_data_fp3(q3, sizeof(q3)));
    assert(rocmfpx_validate_row_data_fp6(q6, sizeof(q6)));
    assert(rocmfpx_validate_row_data_fp8(q8, sizeof(q8)));
    assert(rocmfpx_validate_row_data_i4(qi4, sizeof(qi4)));

    rocmfpx_dequantize_row_fp3(q3, fp3, N);
    rocmfpx_dequantize_row_fp6(q6, fp6, N);
    rocmfpx_dequantize_row_fp8(q8, fp8, N);
    rocmfpx_dequantize_row_i4(qi4, i4, N);

    const float mse3 = mse(src, fp3, N);
    const float mse6 = mse(src, fp6, N);
    const float mse8 = mse(src, fp8, N);
    const float msei4 = mse(src, i4, N);

    printf("ROCmFP3: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp3), rocmfpx_row_size_fp3(N),
            8.0f*(float) sizeof(block_rocmfp3)/(float) QK_ROCMFP3, mse3);
    printf("ROCmFP6: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp6), rocmfpx_row_size_fp6(N),
            8.0f*(float) sizeof(block_rocmfp6)/(float) QK_ROCMFP6, mse6);
    printf("ROCmFP8: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp8), rocmfpx_row_size_fp8(N),
            8.0f*(float) sizeof(block_rocmfp8)/(float) QK_ROCMFP8, mse8);
    printf("ROCmI4: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmi4), rocmfpx_row_size_i4(N),
            8.0f*(float) sizeof(block_rocmi4)/(float) QK_ROCMI4, msei4);

    assert(isfinite(mse3));
    assert(isfinite(mse6));
    assert(isfinite(mse8));
    assert(isfinite(msei4));
    assert(msei4 < mse3);
    assert(mse8 < mse6);
    assert(mse6 < mse3);

    check_weighted_imatrix_fp3();
    check_weighted_imatrix_fp6();
    check_weighted_imatrix_fp8();
    check_weighted_imatrix_i4();
    check_fp2_swar_codebook();

    return 0;
}
