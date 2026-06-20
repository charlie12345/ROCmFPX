// TurboQuant CPU reference tests: FWHT self-inverse, roundtrip MSE, bit-packing
//
// Validates that the quantize/dequantize pipeline in ggml-quants.c produces
// MSE*d values consistent with the paper (Zandieh et al., ICLR 2026).

#include "ggml.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int HEAD_DIM          = 128;
static constexpr int BLOCK_SIZE        = 32;
static constexpr int BLOCKS_PER_CHUNK  = HEAD_DIM / BLOCK_SIZE;
static constexpr int N_VECTORS         = 10000;
static constexpr int N_ELEMENTS        = N_VECTORS * HEAD_DIM;

// Expected MSE*d ranges (paper: TQ3 ~0.034, TQ4 ~0.009 for d=128)
static constexpr float TQ3_MSE_D_MIN = 0.025f;
static constexpr float TQ3_MSE_D_MAX = 0.045f;
static constexpr float TQ4_MSE_D_MIN = 0.005f;
static constexpr float TQ4_MSE_D_MAX = 0.015f;

// ============================================================
// FWHT reference (must match ggml-quants.c)
// ============================================================

static void fwht_f32(float * x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

// ============================================================
// Test 1: FWHT self-inverse (FWHT(FWHT(x)) == x)
// ============================================================

static int test_fwht_self_inverse(void) {
    printf("  FWHT self-inverse (d=%d)... ", HEAD_DIM);

    float orig[HEAD_DIM];
    float work[HEAD_DIM];

    for (int i = 0; i < HEAD_DIM; i++) {
        orig[i] = sinf((float)(i + 1) * 0.7f) * 2.0f;
    }
    memcpy(work, orig, sizeof(orig));

    fwht_f32(work, HEAD_DIM);
    fwht_f32(work, HEAD_DIM);

    float max_err = 0.0f;
    for (int i = 0; i < HEAD_DIM; i++) {
        float err = fabsf(work[i] - orig[i]);
        if (err > max_err) { max_err = err; }
    }

    bool pass = max_err < 1e-5f;
    printf("max_err=%.2e %s\n", max_err, pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Test 2 & 3: Roundtrip MSE for TQ3 and TQ4
// ============================================================

static void generate_random_vectors(float * dst, int n_elements, unsigned int seed) {
    unsigned int state = seed;
    for (int i = 0; i < n_elements; i++) {
        state = state * 1664525u + 1013904223u;
        dst[i] = ((float)(state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
    }
}

static int test_roundtrip_mse(ggml_type type, float mse_d_min, float mse_d_max) {
    const char * name = ggml_type_name(type);
    printf("  %s roundtrip MSE*d (n=%d, d=%d)... ", name, N_VECTORS, HEAD_DIM);

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    assert(traits->from_float_ref != nullptr);
    assert(traits->to_float != nullptr);

    std::vector<float> src(N_ELEMENTS);
    std::vector<float> dst(N_ELEMENTS);
    size_t quant_size = (size_t)N_ELEMENTS / BLOCK_SIZE * ggml_type_size(type);
    std::vector<uint8_t> quant(quant_size);

    generate_random_vectors(src.data(), N_ELEMENTS, 42);

    traits->from_float_ref(src.data(), quant.data(), N_ELEMENTS);
    traits->to_float(quant.data(), dst.data(), N_ELEMENTS);

    // MSE*d = E[ ||x - x̃||² / ||x||² ] (normalized reconstruction error)
    double total_nmse = 0.0;
    for (int v = 0; v < N_VECTORS; v++) {
        double err_sq = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) {
            double diff = (double)src[v * HEAD_DIM + i] - (double)dst[v * HEAD_DIM + i];
            err_sq += diff * diff;
            norm_sq += (double)src[v * HEAD_DIM + i] * (double)src[v * HEAD_DIM + i];
        }
        if (norm_sq > 1e-20) {
            total_nmse += err_sq / norm_sq;
        }
    }
    float mse_d = (float)(total_nmse / N_VECTORS);

    bool pass = mse_d >= mse_d_min && mse_d <= mse_d_max;
    printf("MSE*d=%.4f [%.3f..%.3f] %s\n", mse_d, mse_d_min, mse_d_max, pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Test 4: Bit-pack determinism and sanity
// ============================================================

static int test_bitpack_deterministic(ggml_type type) {
    const char * name = ggml_type_name(type);
    printf("  %s pack determinism... ", name);

    const ggml_type_traits * traits = ggml_get_type_traits(type);

    float src[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        src[i] = cosf((float)i * 0.31415f);
    }

    size_t qsize = BLOCKS_PER_CHUNK * ggml_type_size(type);
    std::vector<uint8_t> q1(qsize);
    std::vector<uint8_t> q2(qsize);

    traits->from_float_ref(src, q1.data(), HEAD_DIM);
    traits->from_float_ref(src, q2.data(), HEAD_DIM);

    bool pass = memcmp(q1.data(), q2.data(), qsize) == 0;
    printf("%s\n", pass ? "ok" : "FAILED (non-deterministic)");
    return pass ? 0 : 1;
}

static int test_bitpack_sanity(ggml_type type) {
    const char * name = ggml_type_name(type);
    printf("  %s dequantize sanity... ", name);

    const ggml_type_traits * traits = ggml_get_type_traits(type);

    float src[HEAD_DIM];
    float dst[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        src[i] = cosf((float)i * 0.31415f);
    }

    size_t qsize = BLOCKS_PER_CHUNK * ggml_type_size(type);
    std::vector<uint8_t> q(qsize);

    traits->from_float_ref(src, q.data(), HEAD_DIM);
    traits->to_float(q.data(), dst, HEAD_DIM);

    bool all_finite = true;
    bool any_nonzero = false;
    for (int i = 0; i < HEAD_DIM; i++) {
        if (!std::isfinite(dst[i])) { all_finite = false; }
        if (fabsf(dst[i]) > 1e-10f) { any_nonzero = true; }
    }

    bool pass = all_finite && any_nonzero;
    printf("finite=%s nonzero=%s %s\n",
           all_finite ? "yes" : "NO",
           any_nonzero ? "yes" : "NO",
           pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("TurboQuant CPU reference tests\n");
    printf("==============================\n\n");

    int n_fail = 0;

    printf("Test 1: FWHT self-inverse\n");
    n_fail += test_fwht_self_inverse();

    printf("\nTest 2: TQ3 roundtrip MSE\n");
    n_fail += test_roundtrip_mse(GGML_TYPE_TURBO3_0, TQ3_MSE_D_MIN, TQ3_MSE_D_MAX);

    printf("\nTest 3: TQ4 roundtrip MSE\n");
    n_fail += test_roundtrip_mse(GGML_TYPE_TURBO4_0, TQ4_MSE_D_MIN, TQ4_MSE_D_MAX);

    printf("\nTest 4: Bit-pack tests\n");
    n_fail += test_bitpack_deterministic(GGML_TYPE_TURBO3_0);
    n_fail += test_bitpack_deterministic(GGML_TYPE_TURBO4_0);
    n_fail += test_bitpack_sanity(GGML_TYPE_TURBO3_0);
    n_fail += test_bitpack_sanity(GGML_TYPE_TURBO4_0);

    printf("\n==============================\n");
    printf("%d/%d tests passed\n", 7 - n_fail, 7);

    return n_fail > 0 ? 1 : 0;
}
