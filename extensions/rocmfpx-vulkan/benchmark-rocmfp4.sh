#!/usr/bin/env bash
set -euo pipefail

mode=${1:-base}
bin_dir=${ROCMFPX_BIN_DIR:?set ROCMFPX_BIN_DIR to the build bin directory}
model=${ROCMFPX_MODEL:?set ROCMFPX_MODEL to the ROCmFP4 GGUF}
prompt='Write a correct iterative C++ function that reverses a singly linked list, then explain its time and space complexity.'

args=(
    -m "$model"
    -dev ROCmFPXVulkan0
    -ngl 999
    -c 4096
    -b 512
    -ub 256
    -fa on
    -t 16
    -tb 32
    --fit off
    -st
    --temp 0
    -n 128
    --color off
    --no-display-prompt
    -p "$prompt"
)

if [[ $mode == mtp4 ]]; then
    args+=(
        --spec-type draft-mtp
        --spec-draft-device ROCmFPXVulkan0
        --spec-draft-ngl all
        --spec-draft-type-k f16
        --spec-draft-type-v f16
        --spec-draft-n-max 4
        --spec-draft-n-min 0
        --spec-draft-p-min 0.0
        --spec-draft-backend-sampling
    )
elif [[ $mode != base ]]; then
    echo "usage: $0 [base|mtp4]" >&2
    exit 2
fi

export ROCMFPX_PLUGIN_PATH="$bin_dir/rocmfpx-vulkan-plugin.so"
exec "$bin_dir/llama-cli" "${args[@]}" </dev/null
