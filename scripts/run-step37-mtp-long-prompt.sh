#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp4-deepseek-clean-upstream}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/Step-3.7-Flash-MTP-GGUF/Step-3.7-flash-BF16-MTP-to-ROCmFP4-STRIX_LEAN.gguf}"
BACKEND="${BACKEND:-Vulkan0}"
CTX_SIZE="${CTX_SIZE:-32768}"
PROMPT_WORDS="${PROMPT_WORDS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UBATCH_SIZE="${UBATCH_SIZE:-32}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-2}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-q8_0}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-q8_0}"
N_PREDICT="${N_PREDICT:-64}"
LOG_VERBOSITY="${LOG_VERBOSITY:-2}"
MEMORY_MAX="${MEMORY_MAX:-112G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
MAX_SWAP_USED_MIB="${MAX_SWAP_USED_MIB:-64}"
TIMEOUT="${TIMEOUT:-60m}"
RUN_DIR="${RUN_DIR:-/home/caf/strix-fp4/ROCMFP4-HANDOFF/results/step37-long-prompt}"
RUN_ID="${RUN_ID:-${BACKEND,,}-ctx${CTX_SIZE}-words${PROMPT_WORDS}-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$RUN_DIR/$RUN_ID.log}"

swap_kib="$(
    awk '
        /^SwapTotal:/ { total = $2 }
        /^SwapFree:/  { free = $2 }
        END { print total - free }
    ' /proc/meminfo
)"
if (( swap_kib > MAX_SWAP_USED_MIB * 1024 )); then
    echo "Refusing to start: swap still has $((swap_kib / 1024)) MiB in use." >&2
    echo "Run: sudo swapoff -a && sudo swapon -a" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
prompt_file="$(mktemp --tmpdir step37-long-prompt.XXXXXX.txt)"
trap 'rm -f "$prompt_file"' EXIT

awk -v words="$PROMPT_WORDS" '
    BEGIN {
        n = split("the quick brown fox jumps over lazy dog and then walks through green field while sun rises above quiet hills near river under clear sky", filler, " ")
        printf "The following varied words are context filler."
        for (i = 0; i < words; ++i) {
            printf " %s", filler[(i % n) + 1]
        }
        print " Now list the integers from one through sixty-four in order, separated by spaces."
    }
' > "$prompt_file"

"$SCRIPT_DIR/check-step37-mtp-gguf-metadata.sh" "$MODEL"

echo "Running StepFun long-prompt fill: backend=$BACKEND ctx=$CTX_SIZE words=$PROMPT_WORDS"
echo "Log: $LOG"

set +e
systemd-run --user --scope --quiet \
    -p "MemoryMax=$MEMORY_MAX" \
    -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
    env \
    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
    timeout --kill-after=30s "$TIMEOUT" "$BIN" \
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
        --spec-draft-backend-sampling \
        --cache-ram 0 \
        --temp 0.0 \
        --seed 123 \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -lv "$LOG_VERBOSITY" \
        -st \
        -cnv \
        -n "$N_PREDICT" \
        -f "$prompt_file" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
set -e

if (( rc != 0 )); then
    echo "FAIL: StepFun long-prompt fill exited with status $rc" >&2
    exit "$rc"
fi

echo "PASS: StepFun long-prompt fill completed"
