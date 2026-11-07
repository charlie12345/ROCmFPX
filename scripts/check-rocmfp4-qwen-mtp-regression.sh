#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
STALE_BINARY_CHECK="${STALE_BINARY_CHECK:-1}"
MIN_DECODE_TPS="${MIN_DECODE_TPS:-12.0}"
MIN_SUSTAINED_DECODE_TPS="${MIN_SUSTAINED_DECODE_TPS:-11.5}"
RUN_SUSTAINED="${RUN_SUSTAINED:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-8m}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CTX_SIZE="${CTX_SIZE:-262144}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
SPEC_DRAFT_THREADS="${SPEC_DRAFT_THREADS:-$THREADS}"
SPEC_DRAFT_THREADS_BATCH="${SPEC_DRAFT_THREADS_BATCH:-$THREADS_BATCH}"
SAMPLERS="${SAMPLERS:-}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-$CACHE_TYPE_K}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-$CACHE_TYPE_V}"
REASONING_MODE="${REASONING_MODE:-off}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SPEC_EXTRA_ARGS="${SPEC_EXTRA_ARGS:-}"

PROMPT_SHORT="Answer in one concise sentence: what is 17 plus 25?"
PROMPT_SUSTAINED="Write eight short bullet points explaining why a regression guard matters for an experimental quantized LLM backend."

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-cli binary: $BIN" >&2
    exit 1
fi

if [[ "$STALE_BINARY_CHECK" == "1" ]]; then
    for src in common/speculative.cpp src/llama-context.cpp src/llama-model.cpp common/arg.cpp; do
        if [[ -f "$src" && "$BIN" -ot "$src" ]]; then
            echo "FAIL: $BIN is older than $src; rebuild before running Qwen MTP regression" >&2
            exit 1
        fi
    done
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        echo "SKIP: Qwen MTP regression model not found: $MODEL"
        exit 0
    fi
    echo "missing Qwen MTP regression model: $MODEL" >&2
    exit 1
fi

run_case() {
    local prompt="$1"
    local n_predict="$2"
    local sampler_args=()
    local extra_args=()
    local spec_extra_args=()

    if [[ -n "$SAMPLERS" ]]; then
        sampler_args=(--samplers "$SAMPLERS")
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        read -r -a extra_args <<< "$EXTRA_ARGS"
    fi
    if [[ -n "$SPEC_EXTRA_ARGS" ]]; then
        read -r -a spec_extra_args <<< "$SPEC_EXTRA_ARGS"
    fi

    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    timeout --kill-after=60s "$TIMEOUT_SEC" "$BIN" \
        -m "$MODEL" \
        -dev "$BACKEND" \
        --spec-draft-device "$BACKEND" \
        -ngl 999 \
        -fa on \
        --load-mode none \
        -t "$THREADS" \
        -tb "$THREADS_BATCH" \
        --spec-draft-threads "$SPEC_DRAFT_THREADS" \
        --spec-draft-threads-batch "$SPEC_DRAFT_THREADS_BATCH" \
        -ctk "$CACHE_TYPE_K" \
        -ctv "$CACHE_TYPE_V" \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        --temp 0.2 \
        --top-k 20 \
        --top-p 0.9 \
        "${sampler_args[@]}" \
        "${extra_args[@]}" \
        --seed 123 \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -cnv \
        --jinja \
        --reasoning "$REASONING_MODE" \
        --spec-type draft-mtp \
        --spec-draft-ngl all \
        --spec-draft-type-k "$SPEC_DRAFT_TYPE_K" \
        --spec-draft-type-v "$SPEC_DRAFT_TYPE_V" \
        "${spec_extra_args[@]}" \
        --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
        --spec-draft-n-min "$SPEC_DRAFT_N_MIN" \
        --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
        --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT" \
        -n "$n_predict" \
        -p "$prompt" 2>&1
}

check_decode_floor() {
    local label="$1"
    local output="$2"
    local min_tps="$3"

    printf "%s\n" "$output"

    local line decode_tps
    line="$(printf "%s\n" "$output" | rg "Prompt: .*Generation:" | tail -n 1 || true)"
    decode_tps="$(printf "%s\n" "$line" | sed -n 's/.*Generation: \([0-9.]*\) t\/s.*/\1/p')"

    if [[ -z "$decode_tps" ]]; then
        echo "FAIL: could not parse Qwen MTP ${label} decode speed" >&2
        exit 1
    fi

    awk -v label="$label" -v got="$decode_tps" -v min="$min_tps" 'BEGIN {
        if (got + 0 < min + 0) {
            printf("FAIL: Qwen MTP %s decode %.2f tok/s is below floor %.2f tok/s\n", label, got, min) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: Qwen MTP %s decode %.2f tok/s meets floor %.2f tok/s\n", label, got, min);
    }'
}

short_output="$(run_case "$PROMPT_SHORT" 96)"
check_decode_floor "short" "$short_output" "$MIN_DECODE_TPS"

if [[ "$RUN_SUSTAINED" == "1" ]]; then
    sustained_output="$(run_case "$PROMPT_SUSTAINED" 160)"
    check_decode_floor "sustained" "$sustained_output" "$MIN_SUSTAINED_DECODE_TPS"
fi
