#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

DFLASH_USE_ROCMFPX_ENV="${DFLASH_USE_ROCMFPX_ENV:-0}"
if [[ "$DFLASH_USE_ROCMFPX_ENV" != "1" ]]; then
    unset HSA_OVERRIDE_GFX_VERSION
    unset GGML_HIP_ENABLE_UNIFIED_MEMORY
fi

BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/bin/llama-server}"
BENCH_PY="${BENCH_PY:-$SCRIPT_DIR/server-bench.py}"
if [[ -x "/home/caf/llm-builds/venvs/rocmfpx-bench/bin/python" ]]; then
    BENCH_PYTHON="${BENCH_PYTHON:-/home/caf/llm-builds/venvs/rocmfpx-bench/bin/python}"
else
    BENCH_PYTHON="${BENCH_PYTHON:-python3}"
fi

MODEL="${MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18186}"
DEVICE="${DEVICE:-ROCm0}"
SPEC_DRAFT_DEVICE="${SPEC_DRAFT_DEVICE:-$DEVICE}"
ALIAS="${ALIAS:-qwen36-27b-rocmfp4-dflash-b16}"

OUT_ROOT="${OUT_ROOT:-/home/caf/llm-builds/qwen36-dflash-rocmfp4-batch16}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$STAMP}"

CTX_SIZE="${CTX_SIZE:-32768}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
PARALLEL="${PARALLEL:-16}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
REASONING="${REASONING:-off}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"

CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
CACHE_TYPE_K_DRAFT="${CACHE_TYPE_K_DRAFT:-q4_0}"
CACHE_TYPE_V_DRAFT="${CACHE_TYPE_V_DRAFT:-q4_0}"

SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-6}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"

N_PROMPTS="${N_PROMPTS:-64}"
PROMPT_SOURCE="${PROMPT_SOURCE:-rng-512-2048}"
N_PREDICT="${N_PREDICT:-512}"
N_PREDICT_MIN="${N_PREDICT_MIN:-256}"
SEED_OFFSET="${SEED_OFFSET:-36}"
NO_MMPROJ="${NO_MMPROJ:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-120}"

if [[ -z "$MODEL" ]]; then
    echo "MODEL must point to the target Qwen3.6 ROCmFP4 GGUF" >&2
    exit 2
fi
if [[ -z "$DRAFT_MODEL" ]]; then
    echo "DRAFT_MODEL must point to the DFlash draft GGUF" >&2
    exit 2
fi

rocmfpx_require_binary "$SERVER_BIN"
rocmfpx_require_binary "$BENCH_PY"
SKIP_MISSING_MODEL=0 rocmfpx_require_model "$MODEL"
SKIP_MISSING_MODEL=0 rocmfpx_require_model "$DRAFT_MODEL"

mkdir -p "$OUT_DIR"
SERVER_LOG="$OUT_DIR/server.log"
BENCH_LOG="$OUT_DIR/bench.log"
COMMANDS="$OUT_DIR/commands.sh"
ADDRESS="http://$HOST:$PORT"

server_args=(
    -m "$MODEL"
    --model-draft "$DRAFT_MODEL"
    --spec-type draft-dflash
    --spec-draft-device "$SPEC_DRAFT_DEVICE"
    --spec-draft-ngl all
    --spec-draft-type-k "$CACHE_TYPE_K_DRAFT"
    --spec-draft-type-v "$CACHE_TYPE_V_DRAFT"
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    --spec-draft-n-min "$SPEC_DRAFT_N_MIN"
    --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
    --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT"
    --no-spec-draft-backend-sampling
    --spec-draft-poll 1
    --spec-draft-poll-batch 1
    --alias "$ALIAS"
    --host "$HOST"
    --port "$PORT"
    --jinja
    --reasoning "$REASONING"
    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
    -c "$CTX_SIZE"
    -dev "$DEVICE"
    -ngl "$N_GPU_LAYERS"
    -fa "$FLASH_ATTN"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -t "$THREADS"
    -tb "$THREADS_BATCH"
    -ctk "$CACHE_TYPE_K"
    -ctv "$CACHE_TYPE_V"
    --parallel "$PARALLEL"
    --poll 50
    --poll-batch 1
    --no-cache-prompt
    --slot-prompt-similarity 0.0
    --metrics
    --no-webui
    --temp 0
    --seed 123
)

if [[ "$NO_MMPROJ" == "1" ]]; then
    server_args+=(--no-mmproj)
fi

cat > "$COMMANDS" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$ROOT"
MODEL=$(printf '%q' "$MODEL") \\
DRAFT_MODEL=$(printf '%q' "$DRAFT_MODEL") \\
OUT_DIR=$(printf '%q' "$OUT_DIR") \\
CHAT_TEMPLATE_KWARGS=$(printf '%q' "$CHAT_TEMPLATE_KWARGS") \\
REASONING=$(printf '%q' "$REASONING") \\
DFLASH_USE_ROCMFPX_ENV=$(printf '%q' "$DFLASH_USE_ROCMFPX_ENV") \\
BENCH_PYTHON=$(printf '%q' "$BENCH_PYTHON") \\
$(printf '%q ' "$0")

# Server command:
$(printf '%q ' "$SERVER_BIN" "${server_args[@]}")

# Benchmark command:
LLAMA_ARG_N_PARALLEL=$(printf '%q' "$PARALLEL") $(printf '%q ' "$BENCH_PYTHON" "$BENCH_PY" --path_server "$ADDRESS" --path_log "$OUT_DIR/server-bench-{port}.log" --name "$ALIAS" --prompt_source "$PROMPT_SOURCE" --n_prompts "$N_PROMPTS" --n_predict "$N_PREDICT" --n_predict_min "$N_PREDICT_MIN" --seed_offset "$SEED_OFFSET")
EOF
chmod +x "$COMMANDS"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$ROOT"
"$SERVER_BIN" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + SERVER_START_TIMEOUT))
while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "llama-server exited early; see $SERVER_LOG" >&2
        exit 1
    fi
    if curl -fsS "$ADDRESS/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -fsS "$ADDRESS/health" >/dev/null 2>&1; then
    echo "llama-server did not become healthy within ${SERVER_START_TIMEOUT}s; see $SERVER_LOG" >&2
    exit 1
fi

LLAMA_ARG_N_PARALLEL="$PARALLEL" "$BENCH_PYTHON" "$BENCH_PY" \
    --path_server "$ADDRESS" \
    --path_log "$OUT_DIR/server-bench-{port}.log" \
    --name "$ALIAS" \
    --prompt_source "$PROMPT_SOURCE" \
    --n_prompts "$N_PROMPTS" \
    --n_predict "$N_PREDICT" \
    --n_predict_min "$N_PREDICT_MIN" \
    --seed_offset "$SEED_OFFSET" \
    > "$BENCH_LOG" 2>&1

{
    echo "# Qwen3.6 27B ROCmFP4 + DFlash Batch-16 Server Bench"
    echo
    echo "- Date: $(date -Is)"
    echo "- Target model: \`$MODEL\`"
    echo "- DFlash draft: \`$DRAFT_MODEL\`"
    echo "- Device: \`$DEVICE\`"
    echo "- DFlash legacy ROCm env: \`$DFLASH_USE_ROCMFPX_ENV\`"
    echo "- Parallel slots: \`$PARALLEL\`"
    echo "- Batch / ubatch: \`$BATCH_SIZE / $UBATCH_SIZE\`"
    echo "- Target KV: \`$CACHE_TYPE_K / $CACHE_TYPE_V\`"
    echo "- Draft KV: \`$CACHE_TYPE_K_DRAFT / $CACHE_TYPE_V_DRAFT\`"
    echo "- Draft n-max: \`$SPEC_DRAFT_N_MAX\`"
    echo "- Reasoning: \`$REASONING\`"
    echo "- Chat template kwargs: \`$CHAT_TEMPLATE_KWARGS\`"
    echo "- Prompt source: \`$PROMPT_SOURCE\`"
    echo "- Prompts: \`$N_PROMPTS\`"
    echo "- Predict range: \`$N_PREDICT_MIN..$N_PREDICT\`"
    echo
    echo "## Benchmark Output"
    echo
    sed -n '/Benchmark duration:/,$p' "$BENCH_LOG"
    echo
    echo "## Logs"
    echo
    echo "- Server log: \`$SERVER_LOG\`"
    echo "- Benchmark log: \`$BENCH_LOG\`"
    echo "- Commands: \`$COMMANDS\`"
} > "$OUT_DIR/report.md"

echo "$OUT_DIR"
