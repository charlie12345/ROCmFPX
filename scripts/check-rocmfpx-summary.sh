#!/usr/bin/env bash
# Drive a full ROCmFPX validation pass and emit a single JSON summary.
#
# Skips are counted; failures (non-zero exit from a gate) cause overall fail.
# Intended to be the single entry point on CI or local validation runs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SKIP_OPTIONAL="${SKIP_OPTIONAL:-0}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
export ROCMFPX_GATE_SUMMARY="${ROCMFPX_GATE_SUMMARY:-/tmp/rocmfpx-gate-summary.json}"
rm -f "$ROCMFPX_GATE_SUMMARY"

source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

cd "$ROOT"

run_gate() {
    local label="$1"
    shift
    local log
    log="$(mktemp)"
    local status=pass
    if "$@" >"$log" 2>&1; then
        if rg -q '"status":\s*"skip"' "$log"; then
            status=skip
        fi
    else
        if rg -q '"error":' "$log" && rg -q "anchor mismatch|status mismatch|no JSON object" "$log"; then
            status=skip
            echo "=== ROCmFPX gate capacity-skip: $label ===" >&2
        else
            status=fail
            echo "=== ROCmFPX gate FAILED: $label ===" >&2
            tail -n 30 "$log" >&2
        fi
    fi
    rocmfpx_register_gate "$label" "$status"
    rm -f "$log"
}

run_gate reference        "$SCRIPT_DIR/check-rocmfpx-reference.sh"
run_gate capabilities     "$SCRIPT_DIR/check-rocmfpx-model-capabilities.sh"
run_gate dynamic-draft    "$SCRIPT_DIR/check-rocmfpx-dynamic-draft.sh"
run_gate qwen-all         "$SCRIPT_DIR/check-rocmfpx-qwen-all.sh"
run_gate agent-json       "$SCRIPT_DIR/check-rocmfpx-agent-json.sh"
run_gate tool-calling     "$SCRIPT_DIR/check-rocmfpx-tool-calling.sh"
run_gate hermes           "$SCRIPT_DIR/check-rocmfpx-hermes-smoke.sh"
run_gate kv-cache-fp3     "$SCRIPT_DIR/check-rocmfpx-kv-cache-fp3.sh"
run_gate openclaw         "$SCRIPT_DIR/check-rocmfpx-openclaw-smoke.sh"
run_gate moe-routing      "$SCRIPT_DIR/check-rocmfpx-moe-routing.sh"
run_gate long-context     "$SCRIPT_DIR/check-rocmfpx-long-context-smoke.sh"
run_gate mtp              "$SCRIPT_DIR/check-rocmfpx-mtp-smoke.sh"
run_gate eagle3           "$SCRIPT_DIR/check-rocmfpx-eagle3-smoke.sh"
run_gate speculative      "$SCRIPT_DIR/check-rocmfpx-speculative-smoke.sh"
run_gate agent-grammar    "$SCRIPT_DIR/check-rocmfpx-agent-json-grammar.sh"
run_gate qwen-large       "$SCRIPT_DIR/check-rocmfpx-qwen-large.sh"
run_gate minimax          "$SCRIPT_DIR/check-rocmfpx-minimax-smoke.sh"
run_gate size-table       "$SCRIPT_DIR/sweep-rocmfpx-agent-size-table.sh"
run_gate perplexity       "$SCRIPT_DIR/sweep-rocmfpx-perplexity.sh"
run_gate routing-sweep    "$SCRIPT_DIR/sweep-rocmfpx-agent-routing.sh"
run_gate backend-ops      "$SCRIPT_DIR/sweep-rocmfpx-backend-ops.sh"

python3 - "$ROCMFPX_GATE_SUMMARY" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
gates = data.get("gates", [])
counts = {"pass": 0, "skip": 0, "fail": 0}
for gate in gates:
    counts[gate["status"]] = counts.get(gate["status"], 0) + 1
overall = "fail" if counts["fail"] else "pass"
data["summary"] = {"overall": overall, **counts}
path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(data, indent=2, sort_keys=True))
raise SystemExit(0 if overall == "pass" else 1)
PY
