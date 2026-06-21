#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_OPTIONAL="${SKIP_OPTIONAL:-1}"

cd "$ROOT"

scripts/check-rocmfpx-qwen-all.sh
scripts/check-rocmfpx-model-capabilities.sh
scripts/check-rocmfpx-dynamic-draft.sh

if [[ "$SKIP_OPTIONAL" != "1" ]]; then
    scripts/build-rocmfpx-agent-fixtures.sh
    MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-long-context-smoke.sh
    scripts/check-rocmfpx-hermes-smoke.sh
    scripts/check-rocmfpx-openclaw-smoke.sh
    scripts/check-rocmfpx-moe-routing.sh
    scripts/check-rocmfpx-mtp-smoke.sh
    scripts/check-rocmfpx-eagle3-smoke.sh
    scripts/check-rocmfpx-speculative-smoke.sh
    scripts/check-rocmfpx-minimax-smoke.sh
    scripts/check-rocmfpx-agent-json-grammar.sh
fi

echo "ROCmFPX full validation passed (optional=${SKIP_OPTIONAL})"
