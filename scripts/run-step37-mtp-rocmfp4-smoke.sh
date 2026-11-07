#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp4-deepseek-clean-upstream}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/Step-3.7-Flash-MTP-GGUF/Step-3.7-flash-BF16-MTP-to-ROCmFP4-STRIX_LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
CTX_SIZE="${CTX_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UBATCH_SIZE="${UBATCH_SIZE:-32}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
N_PREDICT="${N_PREDICT:-8}"
PROMPT="${PROMPT:-Answer in one concise sentence: what is 17 plus 25?}"
TOKENIZER_PRE_OVERRIDE="${TOKENIZER_PRE_OVERRIDE:-}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-q4_0}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-q4_0}"
CACHE_RAM="${CACHE_RAM:-0}"
SPEC_DRAFT_BACKEND_SAMPLING="${SPEC_DRAFT_BACKEND_SAMPLING:-1}"
LOG_VERBOSITY="${LOG_VERBOSITY:-3}"
USE_MEMORY_SCOPE="${USE_MEMORY_SCOPE:-1}"
MEMORY_MAX="${MEMORY_MAX:-112G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"

cd "$ROOT"

"$SCRIPT_DIR/check-step37-mtp-gguf-metadata.sh" "$MODEL"

override_args=()
if [[ -n "$TOKENIZER_PRE_OVERRIDE" ]]; then
    override_args=(--override-kv "tokenizer.ggml.pre=str:$TOKENIZER_PRE_OVERRIDE")
fi

scope_args=()
if [[ "$USE_MEMORY_SCOPE" == "1" ]]; then
    scope_args=(
        systemd-run --user --scope --quiet
        -p "MemoryMax=$MEMORY_MAX"
        -p "MemorySwapMax=$MEMORY_SWAP_MAX"
    )
fi

backend_sampling_args=(--no-spec-draft-backend-sampling)
if [[ "$SPEC_DRAFT_BACKEND_SAMPLING" == "1" ]]; then
    backend_sampling_args=(--spec-draft-backend-sampling)
fi

set +e
output="$(
    "${scope_args[@]}" env \
    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
    timeout --kill-after=30s 15m "$BIN" \
        -m "$MODEL" \
        -dev "$BACKEND" \
        --spec-draft-device "$BACKEND" \
        -ngl 999 \
        --spec-draft-ngl all \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -fa on \
        --fit off \
        --load-mode mmap \
        -t "$THREADS" \
        -tb "$THREADS_BATCH" \
        --spec-draft-threads "$THREADS" \
        --spec-draft-threads-batch "$THREADS_BATCH" \
        -ctk "$CACHE_TYPE_K" \
        -ctv "$CACHE_TYPE_V" \
        --spec-draft-type-k "$SPEC_DRAFT_TYPE_K" \
        --spec-draft-type-v "$SPEC_DRAFT_TYPE_V" \
        --spec-type draft-mtp \
        --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
        --spec-draft-n-min "$SPEC_DRAFT_N_MIN" \
        --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
        --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT" \
        "${backend_sampling_args[@]}" \
        --cache-ram "$CACHE_RAM" \
        "${override_args[@]}" \
        --temp 0.0 \
        --seed 123 \
        -lv "$LOG_VERBOSITY" \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -cnv \
        --jinja \
        --reasoning off \
        -n "$N_PREDICT" \
        -p "$PROMPT" 2>&1
)"
rc=$?
set -e

printf "%s\n" "$output"

if (( rc != 0 )); then
    echo "FAIL: StepFun smoke llama-cli exited with status $rc" >&2
    exit "$rc"
fi

if ! rg -q "adding speculative implementation 'draft-mtp'" <<< "$output"; then
    echo "FAIL: StepFun smoke did not initialize native draft-mtp" >&2
    exit 1
fi

if ! rg -q "Prompt: .*Generation:" <<< "$output"; then
    echo "FAIL: StepFun smoke did not report generation speed" >&2
    exit 1
fi

echo "PASS: StepFun ROCmFP4 native-MTP smoke completed"
