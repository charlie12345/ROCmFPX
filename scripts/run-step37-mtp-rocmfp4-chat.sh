#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp4-deepseek-clean-upstream}"
BIN="${BIN:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/Step-3.7-Flash-MTP-GGUF/Step-3.7-flash-BF16-MTP-to-ROCmFP4-STRIX_LEAN.gguf}"
BACKEND="${BACKEND:-Vulkan0}"
CTX_SIZE="${CTX_SIZE:-32768}"
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
CACHE_RAM="${CACHE_RAM:-0}"
USE_MEMORY_SCOPE="${USE_MEMORY_SCOPE:-1}"
MEMORY_MAX="${MEMORY_MAX:-112G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"

cd "$ROOT"

"$SCRIPT_DIR/check-step37-mtp-gguf-metadata.sh" "$MODEL"

scope_args=()
if [[ "$USE_MEMORY_SCOPE" == "1" ]]; then
    scope_args=(
        systemd-run --user --scope --quiet
        -p "MemoryMax=$MEMORY_MAX"
        -p "MemorySwapMax=$MEMORY_SWAP_MAX"
    )
fi

exec "${scope_args[@]}" env \
    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
    "$BIN" \
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
        --cache-ram "$CACHE_RAM" \
        --jinja \
        --reasoning off \
        -cnv \
        "$@"
