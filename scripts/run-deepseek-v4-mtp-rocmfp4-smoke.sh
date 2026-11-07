#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-1layer-mtp-rocmfp4-mxfp4.gguf}"
BACKEND="${2:-ROCm0}"
BIN="${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-cli"
CONTEXT="${CONTEXT:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
UBATCH_SIZE="${UBATCH_SIZE:-16}"
N_PREDICT="${N_PREDICT:-4}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-16}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-1}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
FA="${FA:-off}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-480}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
SPEC_DRAFT_N_GPU_LAYERS="${SPEC_DRAFT_N_GPU_LAYERS:-all}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-f16}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-f16}"
FULL_MODEL_THRESHOLD_GIB="${FULL_MODEL_THRESHOLD_GIB:-50}"
MIN_AVAILABLE_MEMORY_GIB="${MIN_AVAILABLE_MEMORY_GIB:-110}"
MAX_TMP_USED_GIB="${MAX_TMP_USED_GIB:-2}"

if [[ ! -x "${BIN}" ]]; then
    printf 'missing CLI tool: %s\n' "${BIN}" >&2
    printf 'build it with: cmake --build build-deepseek4-rocmfp4 --target llama-cli -j 12\n' >&2
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

device_args=(-ngl 0 --spec-draft-ngl 0)
if [[ "${BACKEND}" != "CPU" ]]; then
    device_args=(-dev "${BACKEND}" --spec-draft-device "${BACKEND}" -ngl "${N_GPU_LAYERS}" --spec-draft-ngl "${SPEC_DRAFT_N_GPU_LAYERS}")
fi

printf 'DeepSeek V4 MTP ROCmFP4 smoke: backend=%s model=%s context=%s predict=%s ngl=%s spec_draft_ngl=%s kv=%s/%s draft_kv=%s/%s\n' \
    "${BACKEND}" "${MODEL}" "${CONTEXT}" "${N_PREDICT}" \
    "${N_GPU_LAYERS}" "${SPEC_DRAFT_N_GPU_LAYERS}" \
    "${CACHE_TYPE_K}" "${CACHE_TYPE_V}" "${SPEC_DRAFT_TYPE_K}" "${SPEC_DRAFT_TYPE_V}"

output="$(
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
    timeout --kill-after=10s "${TIMEOUT_SECONDS}" \
    "${BIN}" \
        -m "${MODEL}" \
        "${device_args[@]}" \
        -c "${CONTEXT}" \
        -b "${BATCH_SIZE}" \
        -ub "${UBATCH_SIZE}" \
        -fa "${FA}" \
        --fit off \
        --load-mode none \
        --no-repack \
        -t "${THREADS}" \
        -tb "${THREADS_BATCH}" \
        --spec-draft-threads "${THREADS}" \
        --spec-draft-threads-batch "${THREADS_BATCH}" \
        -ctk "${CACHE_TYPE_K}" \
        -ctv "${CACHE_TYPE_V}" \
        --spec-draft-type-k "${SPEC_DRAFT_TYPE_K}" \
        --spec-draft-type-v "${SPEC_DRAFT_TYPE_V}" \
        --spec-type draft-mtp \
        --spec-draft-n-max "${SPEC_DRAFT_N_MAX}" \
        --spec-draft-n-min "${SPEC_DRAFT_N_MIN}" \
        --spec-draft-p-min "${SPEC_DRAFT_P_MIN}" \
        --spec-draft-p-split "${SPEC_DRAFT_P_SPLIT}" \
        --no-spec-draft-backend-sampling \
        --temp 0 \
        --seed 123 \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -n "${N_PREDICT}" \
        -p '1 + 1 =' 2>&1
)"

printf '%s\n' "${output}"

if ! rg -q 'Prompt: .*Generation:' <<< "${output}"; then
    printf 'FAIL: DeepSeek V4 MTP smoke did not report generation speed\n' >&2
    exit 1
fi

printf 'PASS: DeepSeek V4 ROCmFP4 native-MTP smoke completed on %s\n' "${BACKEND}"
