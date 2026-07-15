#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp2-reference}"
CC_BIN="${CC:-cc}"
EXTRA_CFLAGS="${EXTRA_CFLAGS:-}"
read -r -a EXTRA_CFLAGS_ARRAY <<< "$EXTRA_CFLAGS"

mkdir -p "$BUILD_DIR"

echo "ROCmFP2 Phase-1 CPU reference check"
echo "source_root=$ROOT"
echo "compiler=$CC_BIN"

"$CC_BIN" \
    -std=c11 \
    -O2 \
    -g \
    -Wall \
    -Wextra \
    -Werror \
    -pedantic \
    -ffp-contract=off \
    -fno-fast-math \
    "${EXTRA_CFLAGS_ARRAY[@]}" \
    "$ROOT/ggml/rocmfpx/rocmfp2_reference.c" \
    "$ROOT/ggml/rocmfpx/test_rocmfp2_reference.c" \
    -lm \
    -o "$BUILD_DIR/test-rocmfp2-reference"

"$BUILD_DIR/test-rocmfp2-reference"
