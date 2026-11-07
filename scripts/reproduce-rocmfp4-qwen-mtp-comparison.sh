#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-cli}"
ROCMFP4_MODEL="${ROCMFP4_MODEL:-/home/caf/strix-fp4/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf}"
BASELINE_MODEL="${BASELINE_MODEL:-/home/caf/llm-builds/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf}"
REPORT_DIR="${REPORT_DIR:-$ROOT/bench-reports}"
BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
CTX_SIZE="${CTX_SIZE:-262144}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
N_PREDICT="${N_PREDICT:-160}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-$CACHE_TYPE_K}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-$CACHE_TYPE_V}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
REASONING="${REASONING:-off}"
TOOLS="${TOOLS:-off}"
PROMPT="${PROMPT:-Write eight short bullet points explaining why a regression guard matters for an experimental quantized LLM backend.}"
RUN_BASELINE="${RUN_BASELINE:-1}"

cd "$ROOT"

mkdir -p "$REPORT_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
report="$REPORT_DIR/rocmfp4-qwen-mtp-comparison-${timestamp}.md"

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

file_size() {
    stat -c '%s bytes' "$1" 2>/dev/null || echo "unknown"
}

model_hash() {
    if [[ "${HASH_MODE:-0}" == "1" ]]; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "set HASH_MODE=1 to compute sha256"
    fi
}

extract_metric() {
    local output="$1"
    local pattern="$2"
    local line
    line="$(printf "%s\n" "$output" | rg "Prompt: .*Generation:" | tail -n 1 || true)"
    case "$pattern" in
        prompt)
            printf "%s\n" "$line" | sed -n 's/.*Prompt: \([0-9.]*\) t\/s.*/\1/p'
            ;;
        decode)
            printf "%s\n" "$line" | sed -n 's/.*Generation: \([0-9.]*\) t\/s.*/\1/p'
            ;;
    esac
}

run_case() {
    local model="$1"
    local backend="$2"

    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    timeout --kill-after=30s "${TIMEOUT_SECONDS:-10m}" "$BIN" \
        -m "$model" \
        -dev "$backend" \
        --spec-draft-device "$backend" \
        -ngl 999 \
        -fa on \
        --load-mode none \
        -t "$THREADS" \
        -tb "$THREADS_BATCH" \
        -ctk "$CACHE_TYPE_K" \
        -ctv "$CACHE_TYPE_V" \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        --temp 0.2 \
        --top-k 20 \
        --top-p 0.9 \
        --seed 123 \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -cnv \
        --jinja \
        --reasoning "$REASONING" \
        --spec-type draft-mtp \
        --spec-draft-ngl all \
        --spec-draft-type-k "$SPEC_DRAFT_TYPE_K" \
        --spec-draft-type-v "$SPEC_DRAFT_TYPE_V" \
        --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
        --spec-draft-n-min "$SPEC_DRAFT_N_MIN" \
        --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
        --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT" \
        -n "$N_PREDICT" \
        -p "$PROMPT" 2>&1
}

append_result() {
    local model_name="$1"
    local model_path="$2"
    local backend="$3"
    local output="$4"

    local prompt_tps decode_tps status
    prompt_tps="$(extract_metric "$output" prompt)"
    decode_tps="$(extract_metric "$output" decode)"
    status="ok"
    if [[ -z "$decode_tps" ]]; then
        prompt_tps="parse_failed"
        decode_tps="parse_failed"
        status="failed"
    fi

    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$model_name" "$backend" "$CTX_SIZE" "on" "$REASONING" "$TOOLS" \
        "$prompt_tps" "$decode_tps" "$status" >> "$report"

    local safe_backend
    safe_backend="$(printf "%s" "$backend" | tr -c 'A-Za-z0-9_.-' '_')"
    local log_file="$REPORT_DIR/${timestamp}-${model_name}-${safe_backend}.log"
    printf "%s\n" "$output" > "$log_file"
    printf "Saved raw log: %s\n" "$log_file"
}

require_file "llama-cli binary" "$BIN"
require_file "ROCmFP4 model" "$ROCMFP4_MODEL"
if [[ "$RUN_BASELINE" == "1" ]]; then
    require_file "baseline model" "$BASELINE_MODEL"
fi

{
    echo "# ROCmFP4 Qwen MTP Reproduction Report"
    echo
    echo "Date: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Kernel: $(uname -srmo)"
    echo "Root: $ROOT"
    echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Binary: $BIN"
    echo "Binary version: $("$BIN" --version 2>&1 | head -n 1 || echo unknown)"
    echo "Hardware note: Framework AMD Strix Halo 395+ desktop, 128 GB unified RAM"
    echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
    echo
    echo "## Models"
    echo
    echo "- ROCmFP4: \`$ROCMFP4_MODEL\`"
    echo "  - size: $(file_size "$ROCMFP4_MODEL")"
    echo "  - sha256: $(model_hash "$ROCMFP4_MODEL")"
    if [[ "$RUN_BASELINE" == "1" ]]; then
        echo "- Baseline: \`$BASELINE_MODEL\`"
        echo "  - size: $(file_size "$BASELINE_MODEL")"
        echo "  - sha256: $(model_hash "$BASELINE_MODEL")"
    fi
    echo
    echo "## Shared Flags"
    echo
    echo "\`\`\`bash"
    echo "-ngl 999 -c $CTX_SIZE -b $BATCH_SIZE -ub $UBATCH_SIZE -fa on \\"
    echo "-ctk $CACHE_TYPE_K -ctv $CACHE_TYPE_V --load-mode none --jinja -cnv -st \\"
    echo "--reasoning $REASONING \\"
    echo "--spec-type draft-mtp --spec-draft-ngl all \\"
    echo "--spec-draft-type-k $SPEC_DRAFT_TYPE_K --spec-draft-type-v $SPEC_DRAFT_TYPE_V \\"
    echo "--spec-draft-n-max $SPEC_DRAFT_N_MAX --spec-draft-n-min $SPEC_DRAFT_N_MIN \\"
    echo "--spec-draft-p-min $SPEC_DRAFT_P_MIN --spec-draft-p-split $SPEC_DRAFT_P_SPLIT \\"
    echo "--seed 123 --temp 0.2 --top-k 20 --top-p 0.9 -n $N_PREDICT"
    echo "\`\`\`"
    echo
    echo "Tools: $TOOLS"
    echo
    echo "## Results"
    echo
    echo "| Model | Backend | Context | MTP | Reasoning | Tools | Prompt tok/s | Decode tok/s | Status |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|---|"
} > "$report"

for backend in $BACKENDS; do
    echo "Running ROCmFP4 on ${backend}..."
    rocmfp4_output="$(run_case "$ROCMFP4_MODEL" "$backend" || true)"
    append_result "ROCmFP4_STRIX_LEAN" "$ROCMFP4_MODEL" "$backend" "$rocmfp4_output"

    if [[ "$RUN_BASELINE" == "1" ]]; then
        echo "Running baseline on ${backend}..."
        baseline_output="$(run_case "$BASELINE_MODEL" "$backend" || true)"
        append_result "UD-Q5_K_XL" "$BASELINE_MODEL" "$backend" "$baseline_output"
    fi
done

{
    echo
    echo "## VRAM Cleanup Check"
    echo
    echo "\`\`\`text"
    if command -v rocm-smi >/dev/null 2>&1; then
        env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" rocm-smi --showpids || true
    else
        echo "rocm-smi not found"
    fi
    echo "\`\`\`"
} >> "$report"

echo "Wrote report: $report"
