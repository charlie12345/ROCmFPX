#!/usr/bin/env bash
# Build the isolated Strix Halo ROCmFP4 + MTP llama.cpp tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
JOBS="${JOBS:-$(nproc)}"
HIP_ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1151}"
HIP_EXTRA_FLAGS="${CMAKE_HIP_FLAGS:-}"
HIP_COMPILER_ARGS=()
if [[ -n "${CMAKE_HIP_COMPILER:-}" ]]; then
    HIP_COMPILER_ARGS+=("-DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}")
fi
ROCMFP4_DECODE_TUNE="${ROCMFP4_DECODE_TUNE:-stable}"
DECODE_TUNE_PROFILE="${ROCMFPX_DECODE_TUNE:-$ROCMFP4_DECODE_TUNE}"
source "$ROOT/scripts/rocmfp4-decode-tune-flags.sh"

if ! tune_flags="$(rocmfp4_decode_tune_flags "$DECODE_TUNE_PROFILE")"; then
    echo "Unknown decode tuning profile '$DECODE_TUNE_PROFILE'" >&2
    echo "Known profiles: $(rocmfp4_decode_tune_known_profiles)" >&2
    exit 2
fi

if [[ -n "$tune_flags" ]]; then
    HIP_EXTRA_FLAGS="$tune_flags ${HIP_EXTRA_FLAGS}"
    echo "Decode tuning profile: $DECODE_TUNE_PROFILE"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
    -DGPU_TARGETS="$HIP_ARCH" \
    -DCMAKE_HIP_FLAGS="$HIP_EXTRA_FLAGS" \
    "${HIP_COMPILER_ARGS[@]}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF

if [[ $# -gt 0 ]]; then
    cmake --build "$BUILD_DIR" -j "$JOBS" --target "$@"
else
    cmake --build "$BUILD_DIR" -j "$JOBS" --target \
        llama-cli \
        llama-server \
        llama-completion \
        llama-quantize \
        llama-bench \
        llama-perplexity \
        test-backend-ops \
        test-quantize-fns \
        test-quantize-perf
fi

echo "Built:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-completion"
echo "  $BUILD_DIR/bin/llama-quantize"
echo "  $BUILD_DIR/bin/llama-bench"
echo "  $BUILD_DIR/bin/test-backend-ops"
echo "  $BUILD_DIR/bin/test-quantize-fns"
echo "  $BUILD_DIR/bin/test-quantize-perf"
