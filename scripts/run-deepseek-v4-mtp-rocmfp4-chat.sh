#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${ROOT_DIR}/models/templates/deepseek-ai-DeepSeek-V4.jinja}"
BACKEND="${BACKEND:-ROCm0}"
CONTEXT="${CONTEXT:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"
UBATCH_SIZE="${UBATCH_SIZE:-16}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-16}"
REASONING="${REASONING:-off}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Respond in English unless the user explicitly requests another language.}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
SPEC_DRAFT_N_GPU_LAYERS="${SPEC_DRAFT_N_GPU_LAYERS:-all}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-f16}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-f16}"
MEMORY_MAX="${MEMORY_MAX:-108G}"
MIN_AVAILABLE_MEMORY_GIB="${MIN_AVAILABLE_MEMORY_GIB:-110}"
MAX_TMP_USED_GIB="${MAX_TMP_USED_GIB:-2}"

if [[ ! -x "${BIN}" ]]; then
    printf 'missing CLI tool: %s\n' "${BIN}" >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 1
fi

if [[ ! -f "${CHAT_TEMPLATE_FILE}" ]]; then
    printf 'missing DeepSeek V4 chat template: %s\n' "${CHAT_TEMPLATE_FILE}" >&2
    exit 1
fi

MODEL_REALPATH="$(readlink -f -- "${MODEL}")"
if [[ "${MODEL_REALPATH}" == /mnt/ai-models/* ]]; then
    printf 'refusing external-drive runtime chat: %s\n' "${MODEL_REALPATH}" >&2
    printf 'use the internal NVMe model under /home/caf/strix-fp4/models\n' >&2
    exit 1
fi

available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
required_available_kib="$((MIN_AVAILABLE_MEMORY_GIB * 1024 * 1024))"
tmp_used_bytes="$(df -B1 --output=used /tmp | tail -n 1 | tr -d ' ')"
max_tmp_used_bytes="$((MAX_TMP_USED_GIB * 1024 * 1024 * 1024))"

if (( available_kib < required_available_kib )); then
    printf 'refusing full-model chat: only %.1f GiB RAM is available; require at least %s GiB\n' \
        "$(awk -v kib="${available_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }')" \
        "${MIN_AVAILABLE_MEMORY_GIB}" >&2
    exit 1
fi

if (( tmp_used_bytes > max_tmp_used_bytes )); then
    printf 'refusing full-model chat: /tmp uses %.1f GiB; require at most %s GiB\n' \
        "$(awk -v bytes="${tmp_used_bytes}" 'BEGIN { printf "%.1f", bytes / 1024 / 1024 / 1024 }')" \
        "${MAX_TMP_USED_GIB}" >&2
    exit 1
fi

printf 'DeepSeek V4 ROCmFP4 native-MTP chat: backend=%s context=%s reasoning=%s temp=%s kv=%s/%s draft_kv=%s/%s\n' \
    "${BACKEND}" "${CONTEXT}" "${REASONING}" "${TEMPERATURE}" "${CACHE_TYPE_K}" "${CACHE_TYPE_V}" \
    "${SPEC_DRAFT_TYPE_K}" "${SPEC_DRAFT_TYPE_V}"

exec systemd-run \
    --user \
    --pipe \
    --wait \
    --collect \
    -p "MemoryMax=${MEMORY_MAX}" \
    -p "MemorySwapMax=0" \
    /usr/bin/env \
        "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY=${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        "${BIN}" \
        -m "${MODEL}" \
        -dev "${BACKEND}" \
        -ngl "${N_GPU_LAYERS}" \
        --spec-draft-device "${BACKEND}" \
        --spec-draft-ngl "${SPEC_DRAFT_N_GPU_LAYERS}" \
        -c "${CONTEXT}" \
        -b "${BATCH_SIZE}" \
        -ub "${UBATCH_SIZE}" \
        -fa off \
        --fit off \
        --load-mode none \
        --no-repack \
        --offline \
        --jinja \
        --chat-template-file "${CHAT_TEMPLATE_FILE}" \
        --reasoning "${REASONING}" \
        --temp "${TEMPERATURE}" \
        -sys "${SYSTEM_PROMPT}" \
        --no-warmup \
        -t "${THREADS}" \
        -tb "${THREADS_BATCH}" \
        --spec-draft-threads "${THREADS}" \
        --spec-draft-threads-batch "${THREADS_BATCH}" \
        -ctk "${CACHE_TYPE_K}" \
        -ctv "${CACHE_TYPE_V}" \
        --spec-draft-type-k "${SPEC_DRAFT_TYPE_K}" \
        --spec-draft-type-v "${SPEC_DRAFT_TYPE_V}" \
        --spec-type draft-mtp \
        --spec-draft-n-min 0 \
        --spec-draft-n-max 1 \
        --spec-draft-p-min 0.0 \
        --spec-draft-p-split 0.10 \
        --no-spec-draft-backend-sampling \
        -cnv
