#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
N_PREDICT="${N_PREDICT:-48}"
CTX_SIZE="${CTX_SIZE:-4096}"
PROMPT="${PROMPT:-Return one concise sentence about regression tests.}"

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-cli binary: $BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        echo "SKIP: ROCmFPX speculative smoke model not found: $MODEL"
        exit 0
    fi
    echo "missing ROCmFPX speculative smoke model: $MODEL" >&2
    exit 1
fi

run_case() {
    local label="$1"
    shift

    local tmp_out
    tmp_out="$(mktemp)"

    echo "=== ROCmFPX speculative smoke: $label ==="
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        timeout --kill-after=30s 5m "$BIN" \
            -m "$MODEL" \
            -dev "$BACKEND" \
            -ngl 999 \
            -fa on \
            --load-mode none \
            -c "$CTX_SIZE" \
            --temp 0 \
            --seed 123 \
            --no-display-prompt \
            --simple-io \
            --no-warmup \
            -st \
            -no-cnv \
            -n "$N_PREDICT" \
            -p "$PROMPT" \
            "$@" >"$tmp_out" 2>&1

    cat "$tmp_out"
    if ! rg -q "Generation:|decoded|tokens per second|tok/s" "$tmp_out"; then
        echo "FAIL: could not find decode/perf evidence in speculative smoke output for $label" >&2
        rm -f "$tmp_out"
        exit 1
    fi
    if ! rg -qi "accept|spec|sampled" "$tmp_out"; then
        echo "WARN: speculative smoke did not report spec-metric counters for $label" >&2
    fi
    rm -f "$tmp_out"
}

run_case ngram-cache \
    --spec-type ngram-cache \
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-4}" \
    --spec-draft-n-min "${SPEC_DRAFT_N_MIN:-0}"

if [[ -n "$DRAFT_MODEL" ]]; then
    if [[ ! -f "$DRAFT_MODEL" ]]; then
        echo "missing ROCmFPX draft model: $DRAFT_MODEL" >&2
        exit 1
    fi

    run_case draft-simple \
        --spec-type draft-simple \
        --spec-draft-model "$DRAFT_MODEL" \
        --spec-draft-device "$BACKEND" \
        --spec-draft-ngl all \
        --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-4}" \
        --spec-draft-n-min "${SPEC_DRAFT_N_MIN:-0}"
else
    echo "SKIP: draft-simple speculative smoke requires DRAFT_MODEL"
fi

echo "ROCmFPX speculative smoke passed for ${MODEL}"
