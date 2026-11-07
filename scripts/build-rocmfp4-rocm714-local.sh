#!/usr/bin/env bash
# Build ROCmFPX for RDNA4 (gfx1200/gfx1201) using a local ROCm 7.14.0a20260624 toolchain.
#
# Downloads the TheRock nightly tarball for gfx120X-all and extracts it to a local path,
# then uses that for compilation. No system-wide ROCm installation required.
#
# Usage:
#   env JOBS=16 scripts/build-rocmfp4-rocm714-local.sh
#   env JOBS=16 ROCM_VERSION=7.14.0a20260624 scripts/build-rocmfp4-rocm714-local.sh
#
# The build output goes to build-rdna4-rocm714/ by default. Can be overridden:
#   env BUILD_DIR=build-custom ROCM_VERSION=7.14.0a20260624 scripts/build-rocmfp4-rocm714-local.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rdna4-rocm714}"
JOBS="${JOBS:-$(nproc)}"
ROCM_VERSION="${ROCM_VERSION:-7.14.0a20260624}"
ROCM_LOCAL_PATH="${ROCM_LOCAL_PATH:-${XDG_DATA_HOME:-$HOME/.local/share}/rocm-${ROCM_VERSION}}"
HIP_ARCH="gfx1200;gfx1201"
DECODE_TUNE_PROFILE="${ROCMFPX_DECODE_TUNE:-stable}"

# Decode tuning flags
source "$ROOT/scripts/rocmfp4-decode-tune-flags.sh"

if ! tune_flags="$(rocmfp4_decode_tune_flags "$DECODE_TUNE_PROFILE")"; then
    echo "Unknown decode tuning profile '$DECODE_TUNE_PROFILE'" >&2
    echo "Known profiles: $(rocmfp4_decode_tune_known_profiles)" >&2
    exit 2
fi

HIP_EXTRA_FLAGS="$tune_flags ${CMAKE_HIP_FLAGS:-}"

# ── Download and extract ROCm if needed ──────────────────────────────
if [[ ! -x "$ROCM_LOCAL_PATH/llvm/bin/clang" ]]; then
    echo "ROCm toolchain not found at $ROCM_LOCAL_PATH. Downloading..."

    TARBALL_URL="https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-gfx120X-all-${ROCM_VERSION}.tar.gz"
    TARBALL="${ROOT}/rocm-${ROCM_VERSION}.tar.gz"

    if [[ ! -s "$TARBALL" ]]; then
        echo "Downloading ROCm ${ROCM_VERSION} from TheRock nightlies..."
        curl -L -o "$TARBALL" "$TARBALL_URL"
    fi

    echo "Extracting to $ROCM_LOCAL_PATH..."
    mkdir -p "$ROCM_LOCAL_PATH"
    tar --use-compress-program=gzip -xf "$TARBALL" -C "$ROCM_LOCAL_PATH" --strip-components=1
    echo "ROCm toolchain ready at $ROCM_LOCAL_PATH"
fi

# ── Verify toolchain ────────────────────────────────────────────────
CLANG="$ROCM_LOCAL_PATH/llvm/bin/clang"
CLANGXX="$ROCM_LOCAL_PATH/llvm/bin/clang++"

if [[ ! -x "$CLANG" || ! -x "$CLANGXX" ]]; then
    echo "Error: clang/clang++ not found under $ROCM_LOCAL_PATH/llvm/bin" >&2
    exit 1
fi

echo "Using clang: $CLANG"
echo "Using clang++: $CLANGXX"
"$CLANG" --version | head -1

# ── Configure ────────────────────────────────────────────────────────
echo "Configuring build..."

# ggml-hip's CMakeLists falls back to $ENV{ROCM_PATH} / /opt/rocm when this is
# unset, which pulls in a second, mismatched HIP runtime alongside the local
# toolchain at link time. Pin it so only $ROCM_LOCAL_PATH is ever resolved.
export ROCM_PATH="$ROCM_LOCAL_PATH"
export HIP_PATH="$ROCM_LOCAL_PATH"

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CLANG" \
    -DCMAKE_CXX_COMPILER="$CLANGXX" \
    -DCMAKE_CXX_FLAGS="-I${ROCM_LOCAL_PATH}/include" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DGGML_HIP=ON \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
    -DGPU_TARGETS="$HIP_ARCH" \
    -DCMAKE_HIP_FLAGS="$HIP_EXTRA_FLAGS" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF \
    -DCMAKE_PREFIX_PATH="$ROCM_LOCAL_PATH" \
    -DROCM_DIR="$ROCM_LOCAL_PATH" \
    -DBUILD_SHARED_LIBS=OFF

# ── Build ────────────────────────────────────────────────────────────
echo "Building..."
cmake --build "$BUILD_DIR" -j "$JOBS" \
    --target llama-cli llama-server llama-quantize llama-bench \
    test-backend-ops test-quantize-fns

# ── Bundle ROCm runtime libs alongside the binaries ──────────────────
# Copy the required shared libraries into build-rdna4-rocm714/lib/ and set
# RPATH to $ORIGIN/../lib so the distribution is self-contained (no
# LD_LIBRARY_PATH needed at runtime).

if ! command -v patchelf &>/dev/null; then
    echo "patchelf not found — install it (e.g. 'sudo apt install patchelf')" >&2
    echo "Without patchelf you will need LD_LIBRARY_PATH at runtime."
else
    echo "Bundling ROCm runtime libraries..."
    BUNDLED_LIB_DIR="$BUILD_DIR/lib"
    mkdir -p "$BUNDLED_LIB_DIR"

    # Copy ROCm runtime libs
    cp -v "$ROCM_LOCAL_PATH/lib"/libamdhip64.so*          "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/librocblas.so*           "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/libhipblas.so*           "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/libhipblaslt.so*         "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/librocsolver.so*         "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/libroctx64.so*           "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/libamd_comgr.so*         "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/libhsa-runtime64.so*     "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/librocm_kpack.so*        "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/librocroller.so*         "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib"/librocprofiler-register.so* "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib/llvm/lib"/libLLVM.so*     "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib/llvm/lib"/libclang-cpp.so* "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib/llvm/lib"/libomp.so*      "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib/llvm/lib"/libomptarget.so* "$BUNDLED_LIB_DIR/" 2>/dev/null || true
    cp -v "$ROCM_LOCAL_PATH/lib/rocm_sysdeps/lib"/librocm_sysdeps_*.so* "$BUNDLED_LIB_DIR/" 2>/dev/null || true

    # Copy rocblas/hipblaslt kernel libraries
    if [[ -d "$ROCM_LOCAL_PATH/lib/rocblas/library" ]]; then
        cp -rv "$ROCM_LOCAL_PATH/lib/rocblas/library" "$BUNDLED_LIB_DIR/rocblas/" 2>/dev/null || true
    fi
    if [[ -d "$ROCM_LOCAL_PATH/lib/hipblaslt/library" ]]; then
        cp -rv "$ROCM_LOCAL_PATH/lib/hipblaslt/library" "$BUNDLED_LIB_DIR/hipblaslt/" 2>/dev/null || true
    fi

    echo "Setting RPATH on binaries to use bundled libs..."
    for bin in "$BUILD_DIR"/bin/llama-* "$BUILD_DIR"/bin/test-*; do
        [[ -f "$bin" ]] && patchelf --set-rpath '$ORIGIN/../lib' "$bin" 2>/dev/null || true
    done
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "Build complete for ${HIP_ARCH}:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-quantize"
echo "  $BUILD_DIR/bin/llama-bench"
echo ""
if command -v patchelf &>/dev/null; then
    echo "Rochm runtime libraries bundled in $BUILD_DIR/lib/"
    echo "Binaries are self-contained — no LD_LIBRARY_PATH needed."
    echo ""
    echo "To run:"
    echo "  $BUILD_DIR/bin/llama-cli -m model.gguf -dev ROCm0 -ngl 999"
    echo ""
    echo "To distribute, zip the entire build directory:"
    echo "  tar -czf rocmfpx-rdna4-rocm714.tar.gz -C $ROOT build-rdna4-rocm714/"
else
    echo "patchelf was not available; set LD_LIBRARY_PATH at runtime:"
    echo "  LD_LIBRARY_PATH=${ROCM_LOCAL_PATH}/lib/llvm/lib:${ROCM_LOCAL_PATH}/lib:${ROCM_LOCAL_PATH}/lib/rocm_sysdeps/lib $BUILD_DIR/bin/llama-cli -m model.gguf -dev ROCm0 -ngl 999"
fi

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "Verification:"
ldd "$BUILD_DIR/bin/llama-cli" 2>/dev/null | grep -E "not found|libomp|libamdhip|librocblas|libhipblas" | head -10 || echo "  (no issues)"
