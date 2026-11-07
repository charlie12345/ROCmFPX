#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-3layer-rocmfp4-mxfp4.gguf}"
BACKEND="${2:-CPU}"
COMPLETION="${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-completion"
CONTEXT="${CONTEXT:-512}"
N_PREDICT="${N_PREDICT:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
TAIL_LINES="${TAIL_LINES:-160}"
NO_REPACK="${NO_REPACK:-0}"
NO_MMAP="${NO_MMAP:-1}"
LOG_FILE="${LOG_FILE:-}"
VERBOSITY="${VERBOSITY:-3}"
TRACE_FILE="${TRACE_FILE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FULL_MODEL_THRESHOLD_GIB="${FULL_MODEL_THRESHOLD_GIB:-50}"
MIN_AVAILABLE_MEMORY_GIB="${MIN_AVAILABLE_MEMORY_GIB:-110}"
MAX_TMP_USED_GIB="${MAX_TMP_USED_GIB:-2}"

if [[ ! -x "${COMPLETION}" ]]; then
    printf 'missing completion tool: %s\n' "${COMPLETION}" >&2
    printf 'build it with: cmake --build build-deepseek4-rocmfp4 --target llama-completion -j 12\n' >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 1
fi

MODEL_REALPATH="$(readlink -f -- "${MODEL}")"
if [[ "${MODEL_REALPATH}" == /mnt/ai-models/* ]]; then
    printf 'refusing external-drive runtime smoke: %s\n' "${MODEL_REALPATH}" >&2
    printf 'copy full runtime artifacts to the internal NVMe under /home/caf before testing\n' >&2
    exit 1
fi

model_bytes="$(stat -c '%s' -- "${MODEL}")"
full_model_threshold_bytes="$((FULL_MODEL_THRESHOLD_GIB * 1024 * 1024 * 1024))"
if (( model_bytes >= full_model_threshold_bytes )); then
    available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
    required_available_kib="$((MIN_AVAILABLE_MEMORY_GIB * 1024 * 1024))"
    tmp_used_bytes="$(df -B1 --output=used /tmp | tail -n 1 | tr -d ' ')"
    max_tmp_used_bytes="$((MAX_TMP_USED_GIB * 1024 * 1024 * 1024))"

    if (( available_kib < required_available_kib )); then
        printf 'refusing full-model smoke: only %.1f GiB RAM is available; require at least %s GiB\n' \
            "$(awk -v kib="${available_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }')" \
            "${MIN_AVAILABLE_MEMORY_GIB}" >&2
        exit 1
    fi

    if (( tmp_used_bytes > max_tmp_used_bytes )); then
        printf 'refusing full-model smoke: /tmp uses %.1f GiB; require at most %s GiB\n' \
            "$(awk -v bytes="${tmp_used_bytes}" 'BEGIN { printf "%.1f", bytes / 1024 / 1024 / 1024 }')" \
            "${MAX_TMP_USED_GIB}" >&2
        exit 1
    fi
fi

args=(
    -m "${MODEL}"
    -p "Write the numbers one through ten."
    -n "${N_PREDICT}"
    --ignore-eos
    --no-display-prompt
    -c "${CONTEXT}"
    -fit off
    --simple-io
    --no-warmup
    --log-colors off
)

if [[ "${BACKEND}" == "CPU" ]]; then
    args+=(-dev none -ngl 0 --no-op-offload)
else
    args+=(-dev "${BACKEND}" -ngl "${N_GPU_LAYERS}")
fi

if [[ "${NO_REPACK}" == "1" ]]; then
    args+=(--no-repack)
fi

if [[ "${NO_MMAP}" == "1" ]]; then
    args+=(--load-mode none)
fi

if [[ -n "${LOG_FILE}" ]]; then
    args+=(--log-file "${LOG_FILE}" --log-timestamps --log-prefix -lv "${VERBOSITY}")
fi

printf 'DeepSeek V4 ROCmFP4 smoke: backend=%s model=%s context=%s predict=%s ngl=%s repack=%s mmap=%s\n' \
    "${BACKEND}" "${MODEL}" "${CONTEXT}" "${N_PREDICT}" "${N_GPU_LAYERS}" \
    "$([[ "${NO_REPACK}" == "1" ]] && printf off || printf on)" \
    "$([[ "${NO_MMAP}" == "1" ]] && printf off || printf on)"
if [[ -n "${TRACE_FILE}" ]]; then
    timeout -k 2 "${TIMEOUT_SECONDS}" stdbuf -oL -eL "${COMPLETION}" "${args[@]}" 2>&1 | tee "${TRACE_FILE}"
else
    timeout -k 2 "${TIMEOUT_SECONDS}" "${COMPLETION}" "${args[@]}" 2>&1 | tail -n "${TAIL_LINES}"
fi
