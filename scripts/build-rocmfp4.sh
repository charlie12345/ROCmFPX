#!/usr/bin/env bash
# Generic ROCmFP4 build for a single AMD GPU architecture.
#
# Examples:
#   CMAKE_HIP_ARCHITECTURES=gfx1030 scripts/build-rocmfp4.sh
#   CMAKE_HIP_ARCHITECTURES=gfx1100 BUILD_DIR=build-rdna3 scripts/build-rocmfp4.sh
#   CMAKE_HIP_ARCHITECTURES=gfx1151 scripts/build-rocmfp4.sh
#   CMAKE_HIP_ARCHITECTURES=gfx1200 scripts/build-rocmfp4.sh
#
# Prefer the thin wrappers when possible:
#   scripts/build-rdna2.sh   (gfx1030)
#   scripts/build-rdna3.sh   (gfx1100)
#   scripts/build-rdna4.sh   (gfx1200)
#   scripts/build-strix-rocmfp4-mtp.sh (gfx1151, validated default)
#
# See docs/BUILD-AMD-ARCHITECTURES.md for the full architecture table.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp4}"
JOBS="${JOBS:-$(nproc)}"
HIP_ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1151}"
HIP_EXTRA_FLAGS="${CMAKE_HIP_FLAGS:-}"
ROCMFP4_DECODE_TUNE="${ROCMFP4_DECODE_TUNE:-stable}"
DECODE_TUNE_PROFILE="${ROCMFPX_DECODE_TUNE:-$ROCMFP4_DECODE_TUNE}"
source "$ROOT/scripts/rocmfp4-decode-tune-flags.sh"
source "$ROOT/scripts/rocmfpx-hip-arch.sh"

if [[ -z "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
    echo "Using default CMAKE_HIP_ARCHITECTURES=${HIP_ARCH}" >&2
    echo "Set CMAKE_HIP_ARCHITECTURES to your GPU target (for example gfx1030, gfx1100, gfx1151, gfx1200)." >&2
fi

if ! tune_flags="$(rocmfp4_decode_tune_flags "$DECODE_TUNE_PROFILE")"; then
    echo "Unknown decode tuning profile '$DECODE_TUNE_PROFILE'" >&2
    echo "Known profiles: $(rocmfp4_decode_tune_known_profiles)" >&2
    exit 2
fi

if [[ -n "$tune_flags" ]]; then
    HIP_EXTRA_FLAGS="$tune_flags ${HIP_EXTRA_FLAGS}"
    echo "Decode tuning profile: $DECODE_TUNE_PROFILE"
fi

rocmfpx_reset_stale_build_dir "$BUILD_DIR" "$HIP_ARCH"

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
    -DGPU_TARGETS="${HIP_ARCH}" \
    -DCMAKE_HIP_FLAGS="$HIP_EXTRA_FLAGS" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" -j "$JOBS" --target \
    llama-cli \
    llama-server \
    llama-completion \
    llama-quantize \
    llama-bench \
    test-backend-ops \
    test-quantize-fns \
    test-quantize-perf

rocmfpx_verify_hip_arch "$BUILD_DIR" "$HIP_ARCH"

echo "Built for ${HIP_ARCH}:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-quantize"
echo "  $BUILD_DIR/bin/test-backend-ops"
