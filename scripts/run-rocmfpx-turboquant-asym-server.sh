#!/usr/bin/env bash
set -euo pipefail

echo "run-rocmfpx-turboquant-asym-server.sh: the turbo4 KV cache type is unsupported since 2026-09-15 (-ctk/-ctv take f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1)" >&2
exit 2

# Recommended TurboQuant asymmetric KV-cache profile for ROCmFPX serving.
#
# Keep K cache at q8_0 for attention/tool-call coherency and compress V cache
# with turbo4 for the memory win. This is a runtime cache policy; it does not
# change the ROCmFPX model-weight quant format.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALIAS="${ALIAS:-rocmfpx-turboquant-asym}"
export CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
export CACHE_TYPE_V="${CACHE_TYPE_V:-turbo4}"
export CACHE_TYPE_K_DRAFT="${CACHE_TYPE_K_DRAFT:-q8_0}"
export CACHE_TYPE_V_DRAFT="${CACHE_TYPE_V_DRAFT:-turbo4}"
export SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
export SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
export SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
export SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"

exec "$SCRIPT_DIR/run-rocmfpx-mtp-server.sh" "$@"
