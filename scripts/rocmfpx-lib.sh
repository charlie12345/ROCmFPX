#!/usr/bin/env bash
# Shared helpers for ROCmFPX check / sweep / build scripts.
#
# Source from a script with:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/rocmfpx-lib.sh"
#
# Provides:
#   - rocmfpx_setup_env          common backend env (HSA_OVERRIDE_GFX_VERSION, GGML_CUDA_ENABLE_UNIFIED_MEMORY)
#   - rocmfpx_require_binary P   exit 1 with a clear message if P is not executable
#   - rocmfpx_require_model P    skip-or-fail depending on SKIP_MISSING_MODEL
#   - rocmfpx_skip_json REASON   print a machine-readable skip JSON and exit 0
#   - rocmfpx_register_gate NAME RESULT NOTE  append to /tmp/rocmfpx-gate-summary.json

set -o pipefail

ROCMFPX_BACKEND_DEFAULT="${ROCMFPX_BACKEND_DEFAULT:-ROCm0}"
ROCMFPX_SKIP_MISSING_DEFAULT="${ROCMFPX_SKIP_MISSING_DEFAULT:-1}"
ROCMFPX_GATE_SUMMARY="${ROCMFPX_GATE_SUMMARY:-/tmp/rocmfpx-gate-summary.json}"

rocmfpx_setup_env() {
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
    export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"
    export BACKEND="${BACKEND:-$ROCMFPX_BACKEND_DEFAULT}"
    export SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-$ROCMFPX_SKIP_MISSING_DEFAULT}"
}

rocmfpx_require_binary() {
    local path="$1"
    if [[ ! -x "$path" ]]; then
        echo "missing required binary: $path" >&2
        return 1
    fi
}

rocmfpx_skip_json() {
    local reason="$1"
    local model="${2:-}"
    python3 - <<PY
import json
print(json.dumps({"status": "skip", "reason": "$reason", "model": "$model"}))
PY
}

rocmfpx_require_model() {
    local path="$1"
    if [[ -f "$path" ]]; then
        return 0
    fi
    if [[ "${SKIP_MISSING_MODEL:-1}" == "1" ]]; then
        rocmfpx_skip_json "missing model" "$path"
        exit 0
    fi
    echo "missing model: $path" >&2
    exit 1
}

rocmfpx_register_gate() {
    local name="$1"
    local status="$2"
    local note="${3:-}"
    local summary="$ROCMFPX_GATE_SUMMARY"
    local record
    record=$(BACKEND="${BACKEND:-}" MODEL="${MODEL:-}" python3 - "$name" "$status" "$note" <<'PY'
import json
import os
import sys
name, status, note = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({
    "name": name,
    "status": status,
    "note": note,
    "backend": os.environ.get("BACKEND", ""),
    "model": os.environ.get("MODEL", ""),
}))
PY
)
    if [[ -f "$summary" ]]; then
        python3 - "$summary" "$record" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
record = json.loads(sys.argv[2])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = {"gates": []}
if "gates" not in data:
    data = {"gates": []}
data["gates"].append(record)
path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
PY
    else
        echo "{\"gates\":[$record]}" >"$summary"
    fi
}
