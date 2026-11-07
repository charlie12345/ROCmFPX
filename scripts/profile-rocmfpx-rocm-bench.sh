#!/usr/bin/env bash
# Profile the ROCmFPX llama-bench hot path with rocprofv3 when available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/llama-bench}"
MODEL="${MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-ROCmFP4-Agentic-GGUF/Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-rocprof}"
DEVICE="${DEVICE:-ROCm0}"

PROMPT="${PROMPT:-512}"
GEN="${GEN:-0}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
DEPTH="${DEPTH:-1}"
REPEAT="${REPEAT:-1}"
POLL="${POLL:-50}"

ROCPROF_BIN="${ROCPROF_BIN:-$(command -v rocprofv3 || true)}"
ROCPROF_ARGS="${ROCPROF_ARGS:---stats --hip-trace --kernel-trace}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

if [[ -z "$ROCPROF_BIN" || ! -x "$ROCPROF_BIN" ]]; then
    echo "rocprofv3 not found. Install ROCm profiler tools or set ROCPROF_BIN=/path/to/rocprofv3." >&2
    exit 2
fi

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "missing llama-bench: $BENCH_BIN" >&2
    exit 1
fi

if [[ ! -s "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

echo "out: $OUT_DIR"
echo "profiler: $ROCPROF_BIN"
echo "model: $MODEL"
echo "device: $DEVICE"

env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
    "$ROCPROF_BIN" $ROCPROF_ARGS \
    --output-directory "$OUT_DIR" \
    -- \
    "$BENCH_BIN" \
    -m "$MODEL" \
    -dev "$DEVICE" \
    -ngl 999 \
    -fa on \
    -ctk q4_0 \
    -ctv q4_0 \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -lm none \
    -t "$THREADS" \
    --poll "$POLL" \
    -p "$PROMPT" \
    -n "$GEN" \
    -d "$DEPTH" \
    -r "$REPEAT" \
    --no-warmup \
    -o json >"$OUT_DIR/llama-bench.json"

echo "profile output: $OUT_DIR"
