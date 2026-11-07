#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-eagle3-tests/target-Q3_0_ROCMFPX.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/caf/strix-fp4/models/rocmfpx-eagle3-tests/eagle3-draft.gguf}"
BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
N_PREDICT="${N_PREDICT:-48}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
PROMPT="${PROMPT:-Write one short sentence about GPU inference.}"

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-cli binary: $BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" || ! -f "$DRAFT_MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        echo "SKIP: ROCmFPX EAGLE3 smoke models not found: MODEL=$MODEL DRAFT_MODEL=$DRAFT_MODEL"
        exit 0
    fi
    echo "missing ROCmFPX EAGLE3 smoke models: MODEL=$MODEL DRAFT_MODEL=$DRAFT_MODEL" >&2
    exit 1
fi

for backend in $BACKENDS; do
    tmp_out="$(mktemp)"

    echo "=== ROCmFPX EAGLE3 smoke: $backend ==="
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        timeout --kill-after=30s 5m "$BIN" \
            -m "$MODEL" \
            -dev "$backend" \
            -ngl 999 \
            --spec-draft-model "$DRAFT_MODEL" \
            --spec-draft-device "$backend" \
            --spec-draft-ngl all \
            --spec-type draft-eagle3 \
            --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-4}" \
            --spec-draft-n-min "${SPEC_DRAFT_N_MIN:-0}" \
            -fa on \
            --load-mode none \
            -c "$CTX_SIZE" \
            -b "$BATCH_SIZE" \
            -ub "$UBATCH_SIZE" \
            --temp 0 \
            --seed 123 \
            --no-display-prompt \
            --simple-io \
            --no-warmup \
            -st \
            -no-cnv \
            -n "$N_PREDICT" \
            -p "$PROMPT" >"$tmp_out" 2>&1

    cat "$tmp_out"
    if ! rg -q "Generation:|decoded|tokens per second|tok/s" "$tmp_out"; then
        echo "FAIL: could not find decode/perf evidence in EAGLE3 smoke output for $backend" >&2
        rm -f "$tmp_out"
        exit 1
    fi

    rm -f "$tmp_out"
done

trap - EXIT

echo "ROCmFPX EAGLE3 smoke passed for ${MODEL}"
