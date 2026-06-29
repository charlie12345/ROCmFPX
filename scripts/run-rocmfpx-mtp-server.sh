#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18180}"
ALIAS="${ALIAS:-rocmfpx-mtp}"
DEVICE_USER_SET="${DEVICE+x}"
DEVICE="${DEVICE:-Vulkan0}"
SPEC_DRAFT_DEVICE="${SPEC_DRAFT_DEVICE:-$DEVICE}"
CTX_SIZE="${CTX_SIZE:-32768}"
BATCH_SIZE_USER_SET="${BATCH_SIZE+x}"
UBATCH_SIZE_USER_SET="${UBATCH_SIZE+x}"
PARALLEL_USER_SET="${PARALLEL+x}"
POLL_USER_SET="${POLL+x}"
POLL_BATCH_USER_SET="${POLL_BATCH+x}"
CACHE_TYPE_K_USER_SET="${CACHE_TYPE_K+x}"
CACHE_TYPE_V_USER_SET="${CACHE_TYPE_V+x}"
CACHE_TYPE_K_DRAFT_USER_SET="${CACHE_TYPE_K_DRAFT+x}"
CACHE_TYPE_V_DRAFT_USER_SET="${CACHE_TYPE_V_DRAFT+x}"
FLASH_ATTN_USER_SET="${FLASH_ATTN+x}"
SPEC_DRAFT_N_MAX_USER_SET="${SPEC_DRAFT_N_MAX+x}"
SPEC_DRAFT_N_MIN_USER_SET="${SPEC_DRAFT_N_MIN+x}"
SPEC_DRAFT_P_MIN_USER_SET="${SPEC_DRAFT_P_MIN+x}"
SPEC_DRAFT_P_SPLIT_USER_SET="${SPEC_DRAFT_P_SPLIT+x}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
PERF_PRESET="${PERF_PRESET:-balanced}"
PARALLEL="${PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
POLL="${POLL:-50}"
POLL_BATCH="${POLL_BATCH:-1}"
PRIO="${PRIO:-0}"
PRIO_BATCH="${PRIO_BATCH:-0}"
SPLIT_MODE="${SPLIT_MODE:-}"
MAIN_GPU="${MAIN_GPU:-}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
FIT_TARGET="${FIT_TARGET:-}"
FIT_CTX="${FIT_CTX:-}"
NO_HOST="${NO_HOST:-0}"
NO_OP_OFFLOAD="${NO_OP_OFFLOAD:-0}"
SPEC_DRAFT_THREADS="${SPEC_DRAFT_THREADS:-$THREADS}"
SPEC_DRAFT_THREADS_BATCH="${SPEC_DRAFT_THREADS_BATCH:-$THREADS_BATCH}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-$CACHE_TYPE_K}"
CACHE_TYPE_K_DRAFT="${CACHE_TYPE_K_DRAFT:-$CACHE_TYPE_K}"
CACHE_TYPE_V_DRAFT="${CACHE_TYPE_V_DRAFT:-$CACHE_TYPE_V}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CACHE_RAM="${CACHE_RAM:-0}"
STRICT_BENCH="${STRICT_BENCH:-1}"
NO_MMPROJ="${NO_MMPROJ:-1}"
AUTO_DETECT_MTP="${AUTO_DETECT_MTP:-1}"
REQUIRE_MTP="${REQUIRE_MTP:-0}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"

if [[ -z "$MODEL" ]]; then
    echo "MODEL must point to a ROCmFPX/ROCmFP4 GGUF" >&2
    exit 2
fi

case "$PERF_PRESET" in
    balanced)
        ;;
    prompt-fast|prompt_fast)
        if [[ -z "$DEVICE_USER_SET" ]]; then DEVICE=Vulkan0; fi
        if [[ -z "$BATCH_SIZE_USER_SET" ]]; then BATCH_SIZE=4096; fi
        if [[ -z "$UBATCH_SIZE_USER_SET" ]]; then UBATCH_SIZE=512; fi
        if [[ -z "$PARALLEL_USER_SET" ]]; then PARALLEL=1; fi
        if [[ -z "$POLL_USER_SET" ]]; then POLL=50; fi
        if [[ -z "$POLL_BATCH_USER_SET" ]]; then POLL_BATCH=1; fi
        if [[ -z "$CACHE_TYPE_K_USER_SET" ]]; then CACHE_TYPE_K=q4_0; fi
        if [[ -z "$CACHE_TYPE_V_USER_SET" ]]; then CACHE_TYPE_V=q4_0; fi
        if [[ -z "$CACHE_TYPE_K_DRAFT_USER_SET" ]]; then CACHE_TYPE_K_DRAFT=q4_0; fi
        if [[ -z "$CACHE_TYPE_V_DRAFT_USER_SET" ]]; then CACHE_TYPE_V_DRAFT=q4_0; fi
        if [[ -z "$FLASH_ATTN_USER_SET" ]]; then FLASH_ATTN=on; fi
        SPEC_DRAFT_DEVICE="${SPEC_DRAFT_DEVICE:-$DEVICE}"
        ;;
    latency)
        if [[ -z "$POLL_USER_SET" ]]; then POLL=100; fi
        if [[ -z "$POLL_BATCH_USER_SET" ]]; then POLL_BATCH=1; fi
        if [[ -z "$PARALLEL_USER_SET" ]]; then PARALLEL=1; fi
        ;;
    decode-fast|decode_fast)
        if [[ -z "$DEVICE_USER_SET" ]]; then DEVICE=Vulkan0; fi
        if [[ -z "$BATCH_SIZE_USER_SET" ]]; then BATCH_SIZE=512; fi
        if [[ -z "$UBATCH_SIZE_USER_SET" ]]; then UBATCH_SIZE=512; fi
        if [[ -z "$PARALLEL_USER_SET" ]]; then PARALLEL=1; fi
        if [[ -z "$POLL_USER_SET" ]]; then POLL=100; fi
        if [[ -z "$POLL_BATCH_USER_SET" ]]; then POLL_BATCH=1; fi
        if [[ -z "$CACHE_TYPE_K_USER_SET" ]]; then CACHE_TYPE_K=q8_0; fi
        if [[ -z "$CACHE_TYPE_V_USER_SET" ]]; then CACHE_TYPE_V=q8_0; fi
        if [[ -z "$CACHE_TYPE_K_DRAFT_USER_SET" ]]; then CACHE_TYPE_K_DRAFT=q4_0; fi
        if [[ -z "$CACHE_TYPE_V_DRAFT_USER_SET" ]]; then CACHE_TYPE_V_DRAFT=q4_0; fi
        if [[ -z "$FLASH_ATTN_USER_SET" ]]; then FLASH_ATTN=on; fi
        if [[ -z "$SPEC_DRAFT_N_MAX_USER_SET" ]]; then SPEC_DRAFT_N_MAX=4; fi
        if [[ -z "$SPEC_DRAFT_N_MIN_USER_SET" ]]; then SPEC_DRAFT_N_MIN=0; fi
        if [[ -z "$SPEC_DRAFT_P_MIN_USER_SET" ]]; then SPEC_DRAFT_P_MIN=0.75; fi
        if [[ -z "$SPEC_DRAFT_P_SPLIT_USER_SET" ]]; then SPEC_DRAFT_P_SPLIT=0.10; fi
        SPEC_DRAFT_DEVICE="${SPEC_DRAFT_DEVICE:-$DEVICE}"
        ;;
    throughput)
        if [[ -z "$BATCH_SIZE_USER_SET" ]]; then BATCH_SIZE=2048; fi
        if [[ -z "$UBATCH_SIZE_USER_SET" ]]; then UBATCH_SIZE=512; fi
        if [[ -z "$PARALLEL_USER_SET" ]]; then PARALLEL=2; fi
        ;;
    *)
        echo "unknown PERF_PRESET=$PERF_PRESET; use balanced, prompt-fast, decode-fast, latency, or throughput" >&2
        exit 2
        ;;
esac

rocmfpx_require_binary "$BIN"
SKIP_MISSING_MODEL=0 rocmfpx_require_model "$MODEL"

spec_args=(
    --spec-type draft-mtp
    --spec-draft-device "$SPEC_DRAFT_DEVICE"
    --spec-draft-ngl all
    --spec-draft-threads "$SPEC_DRAFT_THREADS"
    --spec-draft-threads-batch "$SPEC_DRAFT_THREADS_BATCH"
    --spec-draft-type-k "$CACHE_TYPE_K_DRAFT"
    --spec-draft-type-v "$CACHE_TYPE_V_DRAFT"
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    --spec-draft-n-min "$SPEC_DRAFT_N_MIN"
    --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
    --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT"
    --no-spec-draft-backend-sampling
    --spec-draft-poll 1
    --spec-draft-poll-batch 1
)

if [[ "$AUTO_DETECT_MTP" == "1" ]]; then
    if ! "$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$MODEL" --has-mtp --quiet; then
        if [[ "$REQUIRE_MTP" == "1" ]]; then
            echo "MODEL does not expose MTP metadata/tensors; refusing to start with REQUIRE_MTP=1" >&2
            exit 3
        fi
        echo "warning: MODEL does not expose MTP metadata/tensors; starting without draft-mtp" >&2
        spec_args=()
    fi
fi

cache_args=(--cache-ram "$CACHE_RAM")
if [[ "$STRICT_BENCH" == "1" ]]; then
    cache_args=(--no-cache-prompt --cache-ram "$CACHE_RAM" --slot-prompt-similarity 0.0)
fi

mmproj_args=()
if [[ "$NO_MMPROJ" == "1" ]]; then
    mmproj_args=(--no-mmproj)
fi

chat_template_args=()
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    if [[ ! -f "$CHAT_TEMPLATE_FILE" ]]; then
        echo "missing CHAT_TEMPLATE_FILE: $CHAT_TEMPLATE_FILE" >&2
        exit 1
    fi
    chat_template_args=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

perf_args=(
    --poll "$POLL"
    --poll-batch "$POLL_BATCH"
    --prio "$PRIO"
    --prio-batch "$PRIO_BATCH"
)

fit_args=()
if [[ -n "$FIT_TARGET" ]]; then
    fit_args+=(-fitt "$FIT_TARGET")
fi
if [[ -n "$FIT_CTX" ]]; then
    fit_args+=(-fitc "$FIT_CTX")
fi

placement_args=()
if [[ -n "$SPLIT_MODE" ]]; then
    placement_args+=(-sm "$SPLIT_MODE")
fi
if [[ -n "$MAIN_GPU" ]]; then
    placement_args+=(-mg "$MAIN_GPU")
fi
if [[ -n "$TENSOR_SPLIT" ]]; then
    placement_args+=(-ts "$TENSOR_SPLIT")
fi
if [[ "$NO_HOST" == "1" ]]; then
    placement_args+=(--no-host)
fi
if [[ "$NO_OP_OFFLOAD" == "1" ]]; then
    placement_args+=(--no-op-offload)
fi

cd "$ROOT"

exec "$BIN" \
    -m "$MODEL" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CTX_SIZE" \
    --reasoning off \
    --reasoning-format none \
    --reasoning-budget -1 \
    --no-context-shift \
    -dev "$DEVICE" \
    -ngl "$N_GPU_LAYERS" \
    -fa "$FLASH_ATTN" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -t "$THREADS" \
    -tb "$THREADS_BATCH" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --temp 0 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 123 \
    --parallel "$PARALLEL" \
    "${perf_args[@]}" \
    "${fit_args[@]}" \
    "${placement_args[@]}" \
    "${mmproj_args[@]}" \
    "${chat_template_args[@]}" \
    --metrics \
    --no-webui \
    "${cache_args[@]}" \
    "${spec_args[@]}" \
    "$@"
