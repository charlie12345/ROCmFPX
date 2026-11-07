#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/llama-bench}"
CLI_BIN="${CLI_BIN:-$BUILD_DIR/bin/llama-cli}"
PPL_BIN="${PPL_BIN:-$BUILD_DIR/bin/llama-perplexity}"
MODEL="${MODEL:-/mnt/seconddrive/models/Qwen3.6-27B-Q4_K_M-GGUF/Qwen3.6-27B-Q4_K_M.gguf}"
MODEL_LABEL="${MODEL_LABEL:-qwen36-27b-q4km}"
OUT_ROOT="${OUT_ROOT:-/home/caf/llm-builds/qwen36-27b-benchmarks}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$STAMP-$MODEL_LABEL}"

BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
TG_TOKENS="${TG_TOKENS:-128}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
PPL_CTX="${PPL_CTX:-512}"
PPL_CHUNKS="${PPL_CHUNKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10m}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"

mkdir -p "$OUT_DIR/raw"

SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
COMMANDS="$OUT_DIR/commands.sh"
CORPUS="$OUT_DIR/ppl-corpus.txt"
VRAM_LOG="$OUT_DIR/vram.log"

cat > "$SUMMARY" <<'EOF'
model	backend	test	status	metric	value	seconds	log
EOF

cat > "$COMMANDS" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF

cat > "$CORPUS.base" <<'EOF'
Regression benchmarks should measure the same behavior with the same flags every time. Prompt processing tests how quickly a model ingests context. Token generation tests sustained autoregressive decoding. Perplexity checks whether the model assigns reasonable probability to ordinary text. A reliable quantized backend should preserve coherent outputs while improving memory use and throughput.

GPU inference performance depends on memory bandwidth, kernel fusion, cache layout, scheduling overhead, and how much work can be kept on the accelerator. ROCm and Vulkan can behave differently on the same hardware because their drivers, shader compilers, and available matrix instructions differ. A fair benchmark keeps the model, prompt shape, cache type, batch size, and generation length fixed.

The agent should answer directly, preserve facts, follow instructions, and avoid surprising changes in behavior after quantization. Small smoke prompts are useful for catching crashes, but longer prompt and generation tests are needed to expose throughput and coherence problems. Logs should include exact commands so future runs can reproduce the result.
EOF
: > "$CORPUS"
for _ in $(seq 1 32); do
    cat "$CORPUS.base" >> "$CORPUS"
    printf '\n' >> "$CORPUS"
done

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "missing file: $path" >&2
        exit 1
    fi
}

require_file "$BENCH_BIN"
require_file "$CLI_BIN"
require_file "$PPL_BIN"
require_file "$MODEL"

snapshot_vram() {
    local label="$1"
    {
        echo "=== $label $(date -Is) ==="
        rocm-smi --showmeminfo vram || true
    } >> "$VRAM_LOG" 2>&1
}

clear_between_cases() {
    local label="$1"
    snapshot_vram "before-clear-$label"
    sleep 2
    snapshot_vram "after-clear-$label"
}

record_command() {
    local label="$1"
    shift
    local cmd_file="$OUT_DIR/raw/${label}.cmd"
    printf '%q ' "$@" > "$cmd_file"
    printf '\n' >> "$cmd_file"
    printf '\n# %s\n' "$label" >> "$COMMANDS"
    cat "$cmd_file" >> "$COMMANDS"
}

run_case() {
    local label="$1"
    shift
    local log="$OUT_DIR/raw/${label}.log"
    local start end status

    clear_between_cases "$label"
    record_command "$label" "$@"
    echo "=== $label ===" >&2

    start="$(date +%s)"
    set +e
    "$@" > "$log" 2>&1
    status="$?"
    set -e
    end="$(date +%s)"
    snapshot_vram "after-run-$label"
    printf '%s\t%s\t%s\n' "$status" "$((end - start))" "$log"
}

bench_metric() {
    local log="$1"
    local pattern="$2"
    awk -F'|' -v pattern="$pattern" '$13 ~ pattern {
        gsub(/^[ \t]+|[ \t]+$/, "", $14);
        split($14, a, " ");
        print a[1];
    }' "$log" | tail -n 1
}

decode_metric() {
    sed -n 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$1" | tail -n 1
}

ppl_metric() {
    sed -n 's/.*Final estimate: PPL = \([0-9.][0-9.]*\).*/\1/p; s/.*PPL = \([0-9.][0-9.]*\).*/\1/p' "$1" | tail -n 1
}

append_summary() {
    local backend="$1"
    local test="$2"
    local status="$3"
    local metric="$4"
    local value="$5"
    local seconds="$6"
    local log="$7"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODEL_LABEL" "$backend" "$test" "$status" "$metric" "$value" "$seconds" "$log" >> "$SUMMARY"
}

run_pp() {
    local backend="$1"
    local label="${MODEL_LABEL}-${backend}-pp${PROMPT_TOKENS}"
    local status seconds log value
    IFS=$'\t' read -r status seconds log < <(run_case "$label" \
        timeout --kill-after=60s "$TIMEOUT_SEC" "$BENCH_BIN" \
            -m "$MODEL" -dev "$backend" -ngl 999 -fa on \
            -ctk q4_0 -ctv q4_0 \
            -p "$PROMPT_TOKENS" -n 0 \
            -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" -t "$THREADS" \
            -lm none -r 3)
    value="$(bench_metric "$log" "pp${PROMPT_TOKENS}")"
    append_summary "$backend" "prompt-fill" "$status" "tok_s" "$value" "$seconds" "$log"
}

run_tg() {
    local backend="$1"
    local label="${MODEL_LABEL}-${backend}-tg${TG_TOKENS}"
    local status seconds log value
    IFS=$'\t' read -r status seconds log < <(run_case "$label" \
        timeout --kill-after=60s "$TIMEOUT_SEC" "$BENCH_BIN" \
            -m "$MODEL" -dev "$backend" -ngl 999 -fa on \
            -ctk q4_0 -ctv q4_0 \
            -p 0 -n "$TG_TOKENS" \
            -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" -t "$THREADS" \
            -lm none -r 3)
    value="$(bench_metric "$log" "tg${TG_TOKENS}")"
    append_summary "$backend" "tg128" "$status" "tok_s" "$value" "$seconds" "$log"
}

run_decode() {
    local backend="$1"
    local label="${MODEL_LABEL}-${backend}-decode${DECODE_TOKENS}"
    local status seconds log value
    local prompt="Answer these three items concisely: 1. define quantization, 2. name one GPU inference bottleneck, 3. say why benchmarks need fixed flags."
    IFS=$'\t' read -r status seconds log < <(run_case "$label" \
        timeout --kill-after=60s "$TIMEOUT_SEC" "$CLI_BIN" \
            -m "$MODEL" -dev "$backend" -ngl 999 -fa on \
            --load-mode none -ctk q4_0 -ctv q4_0 \
            -c 4096 -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" \
            -t "$THREADS" -tb "$THREADS_BATCH" \
            --temp 0 --seed 123 --no-display-prompt --simple-io \
            --no-warmup -st -no-cnv -n "$DECODE_TOKENS" -p "$prompt")
    value="$(decode_metric "$log")"
    append_summary "$backend" "decode" "$status" "tok_s" "$value" "$seconds" "$log"
}

run_ppl() {
    local backend="$1"
    local label="${MODEL_LABEL}-${backend}-ppl"
    local status seconds log value
    IFS=$'\t' read -r status seconds log < <(run_case "$label" \
        timeout --kill-after=60s "$TIMEOUT_SEC" "$PPL_BIN" \
            -m "$MODEL" -dev "$backend" -ngl 999 -fa on \
            --load-mode none -ctk q4_0 -ctv q4_0 \
            -c "$PPL_CTX" -b "$BATCH_SIZE" -ub "$UBATCH_SIZE" \
            -t "$THREADS" -tb "$THREADS_BATCH" \
            --no-warmup --chunks "$PPL_CHUNKS" -f "$CORPUS")
    value="$(ppl_metric "$log")"
    append_summary "$backend" "ppl" "$status" "ppl" "$value" "$seconds" "$log"
}

for backend in $BACKENDS; do
    run_pp "$backend"
    run_ppl "$backend"
    run_tg "$backend"
    run_decode "$backend"
done

{
    echo "# Qwen 3.6 27B Benchmark: $MODEL_LABEL"
    echo
    echo "- Date: $(date -Is)"
    echo "- Model: $MODEL"
    echo "- Size: $(du -h "$MODEL" | awk '{print $1}')"
    echo "- Backends: $BACKENDS"
    echo "- Tests: prompt-fill pp${PROMPT_TOKENS}, ppl chunks=${PPL_CHUNKS}, tg${TG_TOKENS}, decode${DECODE_TOKENS}"
    echo "- Cache: q4_0/q4_0, FA on, batch=$BATCH_SIZE, ubatch=$UBATCH_SIZE"
    echo
    echo "## Summary"
    echo
    echo "| backend | test | status | metric | value | seconds | log |"
    echo "|---|---|---:|---|---:|---:|---|"
    tail -n +2 "$SUMMARY" | while IFS=$'\t' read -r model backend test status metric value seconds log; do
        printf '| %s | %s | %s | %s | %s | %s | `%s` |\n' "$backend" "$test" "$status" "$metric" "${value:-}" "$seconds" "$log"
    done
    echo
    echo "Raw logs: \`$OUT_DIR/raw\`"
    echo "Commands: \`$COMMANDS\`"
    echo "VRAM snapshots: \`$VRAM_LOG\`"
} > "$REPORT"

echo "$OUT_DIR"
