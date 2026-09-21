#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="${FRAMEWORK_REPO_PATH:-$(cd -- "$script_dir/../.." && pwd -P)}"
model="${MODEL_PATH:?MODEL_PATH is required}"
reference="${ROCMFPX_QUALITY_REFERENCE:?ROCMFPX_QUALITY_REFERENCE is required}"
hyperloom_root="${HYPERLOOM_ROOT:?HYPERLOOM_ROOT is required}"
hyperloom_python="${HYPERLOOM_PYTHON:-python3}"
max_hours="${HYPERLOOM_MAX_HOURS:-12}"
branch="$(git -C "$repo" branch --show-current)"
patch_path="$script_dir/patches/hyperloom-gfx1151.patch"

case "$branch" in
    main|master|integration/*|rocmfpx/upstream-*)
        echo "Refusing to run Hyperloom on protected branch: $branch" >&2
        exit 2
        ;;
esac

if [[ "$reference" == "$repo"/* ]]; then
    echo "ROCMFPX_QUALITY_REFERENCE must be outside the source worktree" >&2
    exit 2
fi

if [[ ! -f "$reference" ]]; then
    echo "Missing quality reference: $reference" >&2
    exit 2
fi

if ! PYTHONPATH="$hyperloom_root/src${PYTHONPATH:+:$PYTHONPATH}" "$hyperloom_python" -c 'from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES; assert AMD_GPU_DISPATCH_IDENTITIES.get("strixhalo") == ("gfx1151", 40)' >/dev/null 2>&1; then
    echo "Hyperloom does not have the required strixhalo/gfx1151 mapping." >&2
    echo "Apply $patch_path to the pinned Hyperloom checkout; see $script_dir/patches/README.md" >&2
    exit 2
fi

args=(
    -m hyperloom.inference_optimizer.cli -v optimize
    --framework custom
    --framework-path "$repo"
    --benchmark-scripts-dir "$script_dir"
    --model "$model"
    --gpu-type strixhalo
    --tp 1
    --max-hours "$max_hours"
    --extra-env "ROCMFPX_BACKEND=${ROCMFPX_BACKEND:-ROCm0}"
    --extra-env "ROCMFPX_BATCH=${ROCMFPX_BATCH:-2048}"
    --extra-env "ROCMFPX_BUILD_DIR=${ROCMFPX_BUILD_DIR:-$repo/build-hyperloom}"
    --extra-env "ROCMFPX_OBJECTIVE=${ROCMFPX_OBJECTIVE:-decode}"
    --extra-env "ROCMFPX_QUALITY_REFERENCE=$reference"
    --extra-env "ROCMFPX_REPETITIONS=${ROCMFPX_REPETITIONS:-3}"
    --extra-env "ROCMFPX_MTP_N_MAX=${ROCMFPX_MTP_N_MAX:-6}"
    --extra-env "ROCMFPX_MTP_P_MIN=${ROCMFPX_MTP_P_MIN:-0.6}"
    --extra-env "ROCMFPX_MTP_TOKENS=${ROCMFPX_MTP_TOKENS:-256}"
    --extra-env "ROCMFPX_UBATCH=${ROCMFPX_UBATCH:-512}"
)

if [[ -n "${ROCM_PATH:-}" ]]; then
    args+=(--extra-env "ROCM_PATH=$ROCM_PATH")
fi
if [[ "${ROCMFPX_SKIP_BUILD:-0}" == 1 ]]; then
    args+=(--extra-env "ROCMFPX_SKIP_BUILD=1")
fi

if [[ "${HYPERLOOM_EXPERIMENTAL_RDNA_KERNELS:-0}" != 1 ]]; then
    args+=(--no-kernel --no-enable-roofline)
fi

export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-$hyperloom_root/src/hyperloom/agents/kernel}"
export HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV="${HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV:-0}"
export INFERENCE_OPTIMIZER_RAY_EXEC="${INFERENCE_OPTIMIZER_RAY_EXEC:-0}"
export PYTHONPATH="$hyperloom_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$hyperloom_python" "${args[@]}"
