#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_DECL_HIP
#include "ggml-common.h"

#include <cstdio>
#include <initializer_list>
#include <tuple>

// Compile the real configuration declarations and tables without a GPU runtime.
#define __host__
#define __device__
#define AMD_WMMA_AVAILABLE
#define GGML_ROCMI4_W4A4 0
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K 256
#define MMQ_ITER_K_FP4 512
static bool amd_mfma_available(int) { return false; }
static bool amd_wmma_available(int) { return true; }
static bool turing_mma_available(int) { return false; }
#include "rocmfpx-mmq-config-extract.h"

// Disable only the downstream fallback to obtain the vanilla table as a control.
#define ggml_cuda_mmq_get_config_rdna3 ggml_cuda_mmq_get_config_rdna3_vanilla
#define ggml_rocmfpx_mmq_get_config_rdna3(type, J, fallback) \
    ggml_cuda_mmq_config(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true)
#define CASE(type_, nt_, occ_, I_, J_, layout_, K_, stream_, fallback_) \
    if (type == type_ && J == J_ && fallback == fallback_) { \
        return ggml_cuda_mmq_config(type_, nt_, occ_, I_, J_, layout_, K_, stream_, fallback_); \
    }
#include "mmq-config-rdna3.cuh"
#undef CASE
#undef ggml_rocmfpx_mmq_get_config_rdna3
#undef ggml_cuda_mmq_get_config_rdna3

static bool equal(const ggml_cuda_mmq_config & a, const ggml_cuda_mmq_config & b) {
    return std::tie(a.type, a.nthreads, a.occupancy, a.I, a.J, a.sram_layout, a.K_vram, a.stream_k, a.fallback) ==
           std::tie(b.type, b.nthreads, b.occupancy, b.I, b.J, b.sram_layout, b.K_vram, b.stream_k, b.fallback);
}

static bool custom(ggml_type type) {
    return type == GGML_TYPE_Q4_0_ROCMFP4 || type == GGML_TYPE_Q4_0_ROCMFP4_FAST ||
           type == GGML_TYPE_Q4_0_ROCMI4 || type == GGML_TYPE_Q2_0_ROCMFPX ||
           type == GGML_TYPE_Q3_0_ROCMFPX || type == GGML_TYPE_Q6_0_ROCMFPX || type == GGML_TYPE_Q8_0_ROCMFPX;
}

int main() {
    int controls = 0;
    int configs = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const auto type = static_cast<ggml_type>(t);
        for (bool fallback : {false, true}) {
            for (int J = 0; J <= 136; J += 8) {
                const auto got = ggml_cuda_mmq_get_config_rdna3(type, J, fallback);
                const auto ref = ggml_cuda_mmq_get_config_rdna3_vanilla(type, J, fallback);
                if (!custom(type) || J < 16 || J > 128 || J % 16 != 0) {
                    if (!equal(got, ref)) {
                        std::fprintf(stderr, "FAIL vanilla control: type=%d J=%d fallback=%d\n", t, J, fallback);
                        return 1;
                    }
                    ++controls;
                    continue;
                }
                const bool dual_scale = type == GGML_TYPE_Q4_0_ROCMFP4 || type == GGML_TYPE_Q2_0_ROCMFPX ||
                                        type == GGML_TYPE_Q3_0_ROCMFPX || type == GGML_TYPE_Q6_0_ROCMFPX;
                const auto layout = dual_scale ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K : GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;
                const auto expected = ggml_cuda_mmq_config(type, 128, 2, 64, J, layout, MMQ_ITER_K, false, fallback);
                // Match the RDNA3 MMA allocation: X tile + Q8_1 Y tile + row IDs.
                const int y_bytes = J * (128 + 16);
                const int alignment = got.nthreads * 4;
                const int lds = got.I * ggml_cuda_mmq_get_sram_stride(got.sram_layout) * 4 +
                                (y_bytes + alignment - 1) / alignment * alignment + J * 4;
                if (!equal(got, expected) || lds > 65536 || got.nthreads / 32 * got.rows_per_warp() != got.I) {
                    std::fprintf(stderr, "FAIL ROCmFPX configuration: type=%d J=%d fallback=%d LDS=%d\n", t, J, fallback, lds);
                    return 1;
                }
                ++configs;
            }
        }
    }
    std::printf("PASS RDNA3 MMQ: %d ROCmFPX configurations, %d unchanged vanilla/unsupported controls\n", configs, controls);
}
