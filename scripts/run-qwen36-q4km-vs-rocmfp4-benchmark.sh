#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/llama-bench}"
CLI_BIN="${CLI_BIN:-$BUILD_DIR/bin/llama-cli}"
Q4KM_MODEL="${Q4KM_MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-Q4_K_M-GGUF/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf}"
ROCMFP4_MODEL="${ROCMFP4_MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-ROCmFP4-Agentic-GGUF/Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf}"
OUT_ROOT="${OUT_ROOT:-/home/caf/llm-builds/qwen36-q4km-vs-rocmfp4-benchmarks}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$STAMP}"

BENCH_PROMPT="${BENCH_PROMPT:-512}"
BENCH_GEN="${BENCH_GEN:-128}"
BENCH_REPS="${BENCH_REPS:-3}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
TIMEOUT_SEC="${TIMEOUT_SEC:-8m}"
HERMES_N_PREDICT="${HERMES_N_PREDICT:-512}"
HERMES_MAX_PROMPTS="${HERMES_MAX_PROMPTS:-20}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"

mkdir -p "$OUT_DIR"/raw

SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
COMMANDS="$OUT_DIR/commands.sh"
META="$OUT_DIR/meta.txt"

cat > "$SUMMARY" <<'EOF'
kind	model	backend	status	prompt_tok_s	gen_tok_s	seconds	log
EOF

cat > "$COMMANDS" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF

{
    echo "date=$(date -Is)"
    echo "root=$ROOT"
    echo "build_dir=$BUILD_DIR"
    echo "bench_bin=$BENCH_BIN"
    echo "cli_bin=$CLI_BIN"
    echo "q4km_model=$Q4KM_MODEL"
    echo "rocmfp4_model=$ROCMFP4_MODEL"
    echo "q4km_size_bytes=$(stat -c '%s' "$Q4KM_MODEL")"
    echo "rocmfp4_size_bytes=$(stat -c '%s' "$ROCMFP4_MODEL")"
    "$BENCH_BIN" --list-devices
} > "$META" 2>&1

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "missing file: $path" >&2
        exit 1
    fi
}

require_file "$BENCH_BIN"
require_file "$CLI_BIN"
require_file "$Q4KM_MODEL"
require_file "$ROCMFP4_MODEL"

extract_perf() {
    local log="$1"
    local prompt gen
    prompt="$(sed -n 's/.*Prompt: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log" | tail -n 1)"
    gen="$(sed -n 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log" | tail -n 1)"
    printf '%s\t%s\n' "${prompt:-}" "${gen:-}"
}

append_result() {
    local kind="$1"
    local model="$2"
    local backend="$3"
    local status="$4"
    local seconds="$5"
    local log="$6"
    local prompt=""
    local gen=""

    if [[ "$kind" == "hermes20" ]]; then
        IFS=$'\t' read -r prompt gen < <(extract_perf "$log")
    fi

    if [[ "$kind" == "llama-bench" ]]; then
        prompt="$(awk -F'|' '/ pp[0-9]+ / { gsub(/^[ \t]+|[ \t]+$/, "", $14); split($14, a, " "); print a[1] }' "$log" | tail -n 1)"
        gen="$(awk -F'|' '/ tg[0-9]+ / { gsub(/^[ \t]+|[ \t]+$/, "", $14); split($14, a, " "); print a[1] }' "$log" | tail -n 1)"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kind" "$model" "$backend" "$status" "${prompt:-}" "${gen:-}" "$seconds" "$log" >> "$SUMMARY"
}

run_and_log() {
    local label="$1"
    shift
    local log="$OUT_DIR/raw/${label}.log"
    local cmd_file="$OUT_DIR/raw/${label}.cmd"
    local start end status

    printf '=== %s ===\n' "$label" >&2
    printf '%q ' "$@" > "$cmd_file"
    printf '\n' >> "$cmd_file"
    printf '\n# %s\n' "$label" >> "$COMMANDS"
    printf '%q ' "$@" >> "$COMMANDS"
    printf '\n' >> "$COMMANDS"

    start="$(date +%s)"
    set +e
    "$@" > "$log" 2>&1
    status="$?"
    set -e
    end="$(date +%s)"
    printf '%s\t%s\n' "$status" "$((end - start))"
}

bench_case() {
    local model_label="$1"
    local model_path="$2"
    local backend="$3"
    local label="bench-${model_label}-${backend}"
    local status seconds

    IFS=$'\t' read -r status seconds < <(run_and_log "$label" \
        timeout --kill-after=60s "$TIMEOUT_SEC" "$BENCH_BIN" \
            -m "$model_path" \
            -dev "$backend" \
            -ngl 999 \
            -fa on \
            -ctk q4_0 \
            -ctv q4_0 \
            -p "$BENCH_PROMPT" \
            -n "$BENCH_GEN" \
            -b "$BATCH_SIZE" \
            -ub "$UBATCH_SIZE" \
            -t "$THREADS" \
            -lm none \
            -r "$BENCH_REPS")

    append_result "llama-bench" "$model_label" "$backend" "$status" "$seconds" "$OUT_DIR/raw/${label}.log"
}

hermes_prompt() {
    local idx="$1"
    case "$idx" in
        1)  printf 'Answer in one concise sentence: what is 17 plus 25?' ;;
        2)  printf 'Write a JSON object with keys status and reason. Status must be "ok".' ;;
        3)  printf 'Give three bullet points explaining why regression tests matter.' ;;
        4)  printf 'Rewrite this sentence to be clearer: The backend thing is slow when it does the tokens.' ;;
        5)  printf 'Classify the sentiment as positive, neutral, or negative: The benchmark completed without errors.' ;;
        6)  printf 'Extract the file path from this text: error in /tmp/demo/main.py at line 42.' ;;
        7)  printf 'Return only the next number in the sequence: 2, 4, 8, 16,' ;;
        8)  printf 'Summarize in one sentence why GPU memory bandwidth matters for LLM inference.' ;;
        9)  printf 'Write a tiny Python function named add that returns the sum of two inputs.' ;;
        10) printf 'Choose the better command for searching source code quickly: grep or rg. Answer with one word.' ;;
        11) printf 'Explain what a quantized model is in two short sentences.' ;;
        12) printf 'Convert this to lowercase: ROCmFP4 SHOULD STAY COHERENT.' ;;
        13) printf 'Return a valid shell command that prints disk usage for the current directory.' ;;
        14) printf 'What is the main risk of a stale benchmark binary? Answer in one sentence.' ;;
        15) printf 'List two differences between prompt processing and token generation.' ;;
        16) printf 'Write a concise commit message for adding MTP crash guards.' ;;
        17) printf 'Return only true or false: 128 is greater than 64.' ;;
        18) printf 'Make this polite and direct: send the logs now.' ;;
        19) printf 'Name one reason Vulkan and ROCm performance can differ on the same GPU.' ;;
        20) printf 'Give a one-line final verdict template for comparing two model benchmarks.' ;;
    esac
}

hermes20_case() {
    local model_label="$1"
    local model_path="$2"
    local backend="$3"
    local label="hermes20-${model_label}-${backend}"
    local log="$OUT_DIR/raw/${label}.log"
    local cmd_file="$OUT_DIR/raw/${label}.cmd"
    local start end status prompt_file prompt_text

    prompt_file="$OUT_DIR/raw/${label}.prompts.txt"
    {
        echo "Complete the following 20 short Hermes-style agent benchmark tasks. Keep each answer concise and number the answers 1 through 20."
        echo
    } > "$prompt_file"
    for i in $(seq 1 "$HERMES_MAX_PROMPTS"); do
        printf '%02d. ' "$i" >> "$prompt_file"
        hermes_prompt "$i" >> "$prompt_file"
        printf '\n' >> "$prompt_file"
    done
    prompt_text="$(<"$prompt_file")"

    {
        printf 'timeout --kill-after=60s %q %q ' "$TIMEOUT_SEC" "$CLI_BIN"
        printf '%q ' \
            -m "$model_path" \
            -dev "$backend" \
            --spec-draft-device "$backend" \
            -ngl 999 \
            --spec-draft-ngl all \
            -fa on \
            --load-mode none \
            -ctk q4_0 \
            -ctv q4_0 \
            --spec-draft-type-k q4_0 \
            --spec-draft-type-v q4_0 \
            -c "$CTX_SIZE" \
            -b "$BATCH_SIZE" \
            -ub "$UBATCH_SIZE" \
            -t "$THREADS" \
            -tb "$THREADS_BATCH" \
            --temp 0 \
            --seed 123 \
            --no-display-prompt \
            --simple-io \
            --no-warmup \
            -st \
            -cnv \
            --jinja \
            --reasoning off \
            --spec-type draft-mtp \
            --spec-draft-n-max 4 \
            --spec-draft-n-min 0 \
            --spec-draft-p-min 0.0 \
            --spec-draft-p-split 0.10 \
            -n "$HERMES_N_PREDICT" \
            -p "$prompt_text"
        printf '\n'
    } > "$cmd_file"
    printf '\n# %s\n' "$label" >> "$COMMANDS"
    cat "$cmd_file" >> "$COMMANDS"

    printf '=== %s ===\n' "$label" >&2
    start="$(date +%s)"
    set +e
    timeout --kill-after=60s "$TIMEOUT_SEC" "$CLI_BIN" \
        -m "$model_path" \
        -dev "$backend" \
        --spec-draft-device "$backend" \
        -ngl 999 \
        --spec-draft-ngl all \
        -fa on \
        --load-mode none \
        -ctk q4_0 \
        -ctv q4_0 \
        --spec-draft-type-k q4_0 \
        --spec-draft-type-v q4_0 \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -t "$THREADS" \
        -tb "$THREADS_BATCH" \
        --temp 0 \
        --seed 123 \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -cnv \
        --jinja \
        --reasoning off \
        --spec-type draft-mtp \
        --spec-draft-n-max 4 \
        --spec-draft-n-min 0 \
        --spec-draft-p-min 0.0 \
        --spec-draft-p-split 0.10 \
        -n "$HERMES_N_PREDICT" \
        -p "$prompt_text" > "$log" 2>&1
    status="$?"
    set -e
    end="$(date +%s)"

    append_result "hermes20" "$model_label" "$backend" "$status" "$((end - start))" "$log"
}

bench_case q4km "$Q4KM_MODEL" ROCm0
bench_case rocmfp4 "$ROCMFP4_MODEL" ROCm0
bench_case q4km "$Q4KM_MODEL" Vulkan0
bench_case rocmfp4 "$ROCMFP4_MODEL" Vulkan0

hermes20_case q4km "$Q4KM_MODEL" ROCm0
hermes20_case rocmfp4 "$ROCMFP4_MODEL" ROCm0
hermes20_case q4km "$Q4KM_MODEL" Vulkan0
hermes20_case rocmfp4 "$ROCMFP4_MODEL" Vulkan0

{
    echo "# Qwen 3.6 35B A3B MTP Q4_K_M vs ROCmFP4 Benchmark"
    echo
    echo "- Date: $(date -Is)"
    echo "- Build: $BUILD_DIR"
    echo "- Q4_K_M: $Q4KM_MODEL ($(du -h "$Q4KM_MODEL" | awk '{print $1}'))"
    echo "- ROCmFP4: $ROCMFP4_MODEL ($(du -h "$ROCMFP4_MODEL" | awk '{print $1}'))"
    echo "- Bench flags: prompt=$BENCH_PROMPT gen=$BENCH_GEN reps=$BENCH_REPS ctk=q4_0 ctv=q4_0 fa=on batch=$BATCH_SIZE ubatch=$UBATCH_SIZE"
    echo "- Hermes-20 flags: n_predict=$HERMES_N_PREDICT prompts=$HERMES_MAX_PROMPTS draft-mtp enabled for both models"
    echo
    echo "## Summary"
    echo
    echo '| kind | model | backend | status | prompt tok/s | gen tok/s | seconds | log |'
    echo '|---|---|---|---:|---:|---:|---:|---|'
    tail -n +2 "$SUMMARY" | while IFS=$'\t' read -r kind model backend status prompt gen seconds log; do
        printf '| %s | %s | %s | %s | %s | %s | %s | `%s` |\n' \
            "$kind" "$model" "$backend" "$status" "${prompt:-}" "${gen:-}" "$seconds" "$log"
    done
    echo
    echo "Raw logs and exact commands are in \`$OUT_DIR/raw\` and \`$COMMANDS\`."
} > "$REPORT"

echo "$OUT_DIR"
