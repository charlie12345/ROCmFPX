#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
TEST_BACKEND_OPS_BIN="${TEST_BACKEND_OPS_BIN:-$BUILD_DIR/bin/test-backend-ops}"
LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-$BUILD_DIR/bin/llama-cli}"

cd "$ROOT"

echo "== ROCmFP4 quantization guard =="
scripts/check-rocmfp4-quant-regression.sh

echo
echo "== ROCmFP4 Vulkan runtime guard =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    BIN="$TEST_BACKEND_OPS_BIN" \
    scripts/check-rocmfp4-vulkan-runtime-regression.sh

# Off since 2026-11-07: the 2026-09-15 merge dropped Vulkan's ROCmFP copy pipelines (the shaders remain), and no
# ROCmFP KV cache type is selectable, so the check has nothing to measure.
# echo
# echo "== ROCmFP4 Vulkan CPY guard =="
# env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
#     BIN="$TEST_BACKEND_OPS_BIN" \
#     scripts/check-rocmfp4-vulkan-cpy-regression.sh

echo
echo "== ROCmFP4 ROCm runtime guard =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    BIN="$TEST_BACKEND_OPS_BIN" \
    scripts/check-rocmfp4-rocm-runtime-regression.sh

echo
echo "== ROCmFP4 ROCm FlashAttention guard =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    BIN="$TEST_BACKEND_OPS_BIN" \
    scripts/check-rocmfp4-rocm-fattn-regression.sh

echo
echo "== ROCmFP4 ROCm CPY guard =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    BIN="$TEST_BACKEND_OPS_BIN" \
    scripts/check-rocmfp4-rocm-cpy-regression.sh

if [[ "${INCLUDE_DEEPSEEK_SMOKE:-0}" == "1" ]]; then
    echo
    echo "== ROCmFP4 DeepSeek compatibility smoke guard =="
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        BIN="$LLAMA_CLI_BIN" \
        scripts/check-rocmfp4-deepseek-regression.sh
fi

echo
echo "== ROCmFP4 Qwen MTP guard =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    BIN="$LLAMA_CLI_BIN" \
    scripts/check-rocmfp4-qwen-mtp-regression.sh

if [[ "${INCLUDE_QWEN35_A3B_GUARD:-0}" == "1" ]]; then
    echo
    echo "== ROCmFP4 Qwen 35B A3B MTP guard =="
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        BIN="$LLAMA_CLI_BIN" \
        scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
fi

echo
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" rocm-smi --showpids
