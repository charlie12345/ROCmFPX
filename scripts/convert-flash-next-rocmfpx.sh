#!/usr/bin/env bash
# convert-flash-next-rocmfpx.sh — Qwen3.8-Flash-Next FP8 safetensors -> ROCmFP4_FAST GGUF
#
# Pipeline:
#   1. convert_hf_to_gguf.py (port-qwen4exp branch, PLE streaming converter) -> F16 GGUF
#   2. llama-quantize -> Q4_0_ROCMFP4_FAST (MoE/FFN experts 4.25 bpw;
#      per_layer_token_embd auto-protected to Q8_0 by the quantizer fix in PR #98)
#   3. smoke: load + 1-token inference with timer
#
# Requirements: ~250 GiB free on SRC_DIR filesystem, 128 GiB RAM for quantize.
set -eo pipefail

SRC_DIR="${SRC_DIR:-/mnt/ssd2/models/qwen38-flash-next-fp8}"
WORK_DIR="${WORK_DIR:-/mnt/ssd2/models/qwen38-flash-next-build}"
ENGINE="${ENGINE:-/home/user/source/ROCmFPX}"
OUT_F16="${WORK_DIR}/Qwen3.8-Flash-Next-F16.gguf"
OUT_QUANT="${WORK_DIR}/Qwen3.8-Flash-Next-ROCmFP4_FAST.gguf"

mkdir -p "${WORK_DIR}"

if [ ! -f "${SRC_DIR}/config.json" ]; then
    echo "ERROR: ${SRC_DIR}/config.json missing - download not complete?" >&2
    exit 1
fi

echo "=== [1/3] convert FP8 safetensors -> F16 GGUF ==="
if [ ! -f "${OUT_F16}" ]; then
    cd "${ENGINE}"
    python3 convert_hf_to_gguf.py "${SRC_DIR}" \
        --outfile "${OUT_F16}" \
        --outtype f16
else
    echo "F16 GGUF exists, skipping: ${OUT_F16}"
fi

echo "=== [2/3] quantize -> Q4_0_ROCMFP4_FAST ==="
if [ ! -f "${OUT_QUANT}" ]; then
    "${ENGINE}/build-strix-rocmfp4/bin/llama-quantize" \
        "${OUT_F16}" \
        "${OUT_QUANT}" \
        Q4_0_ROCMFP4_FAST
else
    echo "quant exists, skipping: ${OUT_QUANT}"
fi

echo "=== [3/3] smoke: load + generate with timer ==="
/usr/bin/time -v timeout 300 "${ENGINE}/build-strix-rocmfp4/bin/llama-cli" \
    -m "${OUT_QUANT}" \
    -dev Vulkan0 -ngl 99 -ot ple_ngram_embd=CPU -fa on -ub 512 \
    -p "The capital of France is" -n 8 --temp 0 -no-cnv --simple-io 2>&1 | tail -40

ls -lah "${OUT_F16}" "${OUT_QUANT}"
echo "DONE"
