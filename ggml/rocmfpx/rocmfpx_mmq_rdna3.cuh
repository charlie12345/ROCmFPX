#pragma once

// ROCmFPX tile loaders use Q8_0 staging, with Q3_K-sized scale storage for dual-scale formats.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_rocmfpx_mmq_get_config_rdna3(
        ggml_type type, int J, bool fallback) {
    auto layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;
    bool supported = true;
    switch (type) {
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
        case GGML_TYPE_Q4_0_ROCMI4:
        case GGML_TYPE_Q8_0_ROCMFPX:
            break;
        case GGML_TYPE_Q4_0_ROCMFP4:
        case GGML_TYPE_Q2_0_ROCMFPX:
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q6_0_ROCMFPX:
            layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K;
            break;
        default:
            supported = false;
            break;
    }

    // Four WMMA warps cover 64 rows. These tiles stay below the RDNA3 64 KiB LDS limit.
    if (supported && J >= 16 && J <= 128 && J % 16 == 0) {
        return ggml_cuda_mmq_config(type, 128, 2, 64, J, layout, MMQ_ITER_K, false, fallback);
    }

    return ggml_cuda_mmq_config(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}
