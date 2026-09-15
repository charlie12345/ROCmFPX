#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfpx-reference}"
CC_BIN="${CC:-cc}"

mkdir -p "$BUILD_DIR"

"$CC_BIN" \
    -std=c11 \
    -Wall \
    -Wextra \
    -pedantic \
    -I"$ROOT/ggml/include" \
    -I"$ROOT/ggml/rocmfpx" \
    "$ROOT/ggml/rocmfpx/rocmfpx.c" \
    "$ROOT/ggml/rocmfpx/test_rocmfpx.c" \
    -lm \
    -o "$BUILD_DIR/test-rocmfpx"

"$BUILD_DIR/test-rocmfpx"

# Exercise the real HIP selector tables on CPU-only CI workers.
sed -n '/^enum ggml_cuda_mmq_sram_layout {/,/^#undef CASE/p' \
    "$ROOT/ggml/src/ggml-cuda/mmq.cuh" > "$BUILD_DIR/rocmfpx-mmq-config-extract.h"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
    -I"$ROOT/ggml/include" -I"$ROOT/ggml/src" -I"$ROOT/ggml/src/ggml-cuda" -I"$BUILD_DIR" \
    "$ROOT/ggml/rocmfpx/test_rocmfpx_mmq.cpp" -o "$BUILD_DIR/test-rocmfpx-mmq"
"$BUILD_DIR/test-rocmfpx-mmq"
