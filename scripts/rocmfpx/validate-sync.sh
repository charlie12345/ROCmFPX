#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfpx-sync-check}"
JOBS="${JOBS:-$(nproc)}"

echo "Validating ROCmFPX core in $ROOT"

BUILD_DIR="$BUILD_DIR/reference" "$ROOT/scripts/check-rocmfpx-reference.sh"
BUILD_DIR="$BUILD_DIR/reference-fp2" "$ROOT/scripts/check-rocmfp2-reference.sh"

PYTHONPATH="$ROOT/gguf-py" python3 - <<'PY'
import gguf

expected = {
    "Q4_0_ROCMFP4": (100, (32, 18)),
    "Q4_0_ROCMFP4_FAST": (101, (32, 17)),
    "Q6_0_ROCMFPX": (102, (32, 26)),
    "Q8_0_ROCMFPX": (103, (32, 33)),
    "Q3_0_ROCMFPX": (104, (32, 14)),
    "TURBO3_0": (105, (32, 14)),
    "TURBO4_0": (106, (32, 18)),
    "Q2_0_ROCMFPX_LEGACY_AMBIGUOUS": (107, (32, 10)),
    "Q4_0_ROCMI4": (108, (32, 17)),
    "Q5_0_ROCMFPX": (109, (32, 22)),
    "Q7_0_ROCMFPX": (110, (32, 30)),
    "Q2_0_ROCMFPX": (111, (32, 10)),
}
for name, (value, size) in expected.items():
    qtype = getattr(gguf.GGMLQuantizationType, name)
    assert qtype.value == value, (name, qtype.value, value)
    assert gguf.GGML_QUANT_SIZES[qtype] == size, (name, gguf.GGML_QUANT_SIZES[qtype], size)
print("GGUF ROCmFPX IDs and block sizes: OK")
PY

cmake -S "$ROOT" -B "$BUILD_DIR/cpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_TESTS=ON
cmake --build "$BUILD_DIR/cpu" -j "$JOBS" --target llama-quantize test-backend-ops

if [[ "${ROCMFPX_VALIDATE_VULKAN:-0}" == "1" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR/vulkan" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=OFF \
        -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_TESTS=ON
    cmake --build "$BUILD_DIR/vulkan" -j "$JOBS" --target llama-bench test-backend-ops
fi

if [[ "${ROCMFPX_VALIDATE_HIP:-0}" == "1" ]]; then
    cmake --preset strix-rocmfpx
    cmake --build "$ROOT/build-strix-rocmfpx" -j "$JOBS" --target llama-bench llama-quantize test-backend-ops
fi

if [[ "${ROCMFPX_VALIDATE_W4A4:-0}" == "1" ]]; then
    cmake --preset strix-rocmfpx-w4a4
    cmake --build "$ROOT/build-strix-rocmfpx-w4a4" -j "$JOBS" --target llama-bench llama-quantize test-backend-ops
fi

echo "ROCmFPX sync validation: PASS"
