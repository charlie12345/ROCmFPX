#!/usr/bin/env bash

rocmfp4_decode_tune_known_profiles() {
    echo "stable, strix-moe-rpb1, strix-moe-rpb2, strix-moe-rpb3, strix-moe-rpb4, strix-nwarps1, strix-nwarps2, strix-nwarps4, strix-mmid3, strix-mmid4, rocmfpx-strix-moe-rpb1, rocmfpx-strix-moe-rpb2, rocmfpx-strix-moe-rpb3, rocmfpx-strix-moe-rpb4, rocmfpx-strix-nwarps1, rocmfpx-strix-nwarps2, rocmfpx-strix-nwarps4, rocmfpx-strix-rpb2, rocmfpx-strix-mmid1, rocmfpx-strix-mmid2, rocmfpx-strix-mmid3, rocmfpx-strix-mmid4, rocmfpx-strix-vdr2, rocmfpx-strix-vdr8"
}

rocmfp4_decode_tune_flags() {
    local profile="${1:-stable}"

    case "$profile" in
        stable|"")
            return 0
            ;;
        strix-moe-rpb1)
            echo "-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=1"
            ;;
        strix-moe-rpb2)
            echo "-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=2"
            ;;
        strix-moe-rpb3)
            echo "-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=3"
            ;;
        strix-moe-rpb4)
            echo "-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=4"
            ;;
        strix-nwarps1)
            echo "-DGGML_ROCMFP4_RDNA35_NWARPS=1"
            ;;
        strix-nwarps2)
            echo "-DGGML_ROCMFP4_RDNA35_NWARPS=2"
            ;;
        strix-nwarps4)
            echo "-DGGML_ROCMFP4_RDNA35_NWARPS=4"
            ;;
        strix-mmid3)
            echo "-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3"
            ;;
        strix-mmid4)
            echo "-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4"
            ;;
        rocmfpx-strix-moe-rpb1)
            echo "-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=1"
            ;;
        rocmfpx-strix-moe-rpb2)
            echo "-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=2"
            ;;
        rocmfpx-strix-moe-rpb3)
            echo "-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=3"
            ;;
        rocmfpx-strix-moe-rpb4)
            echo "-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4"
            ;;
        rocmfpx-strix-nwarps1)
            echo "-DGGML_ROCMFPX_RDNA35_NWARPS=1"
            ;;
        rocmfpx-strix-nwarps2)
            echo "-DGGML_ROCMFPX_RDNA35_NWARPS=2"
            ;;
        rocmfpx-strix-nwarps4)
            echo "-DGGML_ROCMFPX_RDNA35_NWARPS=4"
            ;;
        rocmfpx-strix-rpb2)
            echo "-DGGML_ROCMFPX_RDNA35_RPB_WIDE=2"
            ;;
        rocmfpx-strix-mmid1)
            echo "-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=1"
            ;;
        rocmfpx-strix-mmid2)
            echo "-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=2"
            ;;
        rocmfpx-strix-mmid3)
            echo "-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=3"
            ;;
        rocmfpx-strix-mmid4)
            echo "-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=4"
            ;;
        rocmfpx-strix-vdr2)
            echo "-DGGML_ROCMFP6_Q8_1_MMVQ_VDR=2"
            ;;
        rocmfpx-strix-vdr8)
            echo "-DGGML_ROCMFP6_Q8_1_MMVQ_VDR=8"
            ;;
        *)
            return 2
            ;;
    esac
}
