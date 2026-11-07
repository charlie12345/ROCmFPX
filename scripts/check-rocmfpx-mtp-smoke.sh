#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-mtp-tests/Qwen-MTP-Q3_0_ROCMFPX.gguf}"
BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
N_PREDICT="${N_PREDICT:-48}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
PROMPT="${PROMPT:-Answer in one concise sentence: what is speculative decoding?}"

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-cli binary: $BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        echo "SKIP: ROCmFPX MTP smoke model not found: $MODEL"
        exit 0
    fi
    echo "missing ROCmFPX MTP smoke model: $MODEL" >&2
    exit 1
fi

for backend in $BACKENDS; do
    tmp_out="$(mktemp)"

    echo "=== ROCmFPX MTP smoke: $backend ==="
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        timeout --kill-after=30s 5m "$BIN" \
            -m "$MODEL" \
            -dev "$backend" \
            --spec-draft-device "$backend" \
            -ngl 999 \
            --spec-draft-ngl all \
            -fa on \
            --load-mode none \
            -t "$THREADS" \
            -tb "$THREADS_BATCH" \
            -ctk "$CACHE_TYPE_K" \
            -ctv "$CACHE_TYPE_V" \
            --spec-draft-type-k "$CACHE_TYPE_K" \
            --spec-draft-type-v "$CACHE_TYPE_V" \
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
            --spec-type draft-mtp \
            --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
            --spec-draft-n-min "$SPEC_DRAFT_N_MIN" \
            --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
            --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT" \
            -n "$N_PREDICT" \
            -p "$PROMPT" >"$tmp_out" 2>&1

    cat "$tmp_out"
    if ! rg -q "Generation:|decoded|tokens per second|tok/s" "$tmp_out"; then
        echo "FAIL: could not find decode/perf evidence in MTP smoke output for $backend" >&2
        rm -f "$tmp_out"
        exit 1
    fi

    rm -f "$tmp_out"
done

trap - EXIT

echo "ROCmFPX MTP smoke passed for ${MODEL}"
