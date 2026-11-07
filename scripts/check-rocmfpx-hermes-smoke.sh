#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-$ROOT/../models/rocmfpx-hermes-tests/hermes-Q3_0_ROCMFPX_AGENT.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
export BACKEND MODEL

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-completion binary: $BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({"status": "skip", "reason": "missing Hermes ROCmFPX fixture", "model": "$MODEL"}))
PY
        exit 0
    fi
    echo "missing Hermes ROCmFPX fixture: $MODEL" >&2
    exit 1
fi

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

PROMPT='Hermes tool-call check. Return only JSON: {"tool_call":{"name":"browser.search","arguments":{"query":"ROCmFPX agent preset"}},"final":null}. Do not add markdown.'

extra_args=()
if [[ -n "${LLAMA_COMPLETION_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${LLAMA_COMPLETION_ARGS})
fi

timeout --kill-after=20s 3m "$BIN" \
    -m "$MODEL" -dev "$BACKEND" -ngl 999 -c "${CTX_SIZE:-4096}" \
    -t "${THREADS:-16}" -tb "${THREADS_BATCH:-32}" --load-mode none --simple-io \
    --no-display-prompt --no-warmup --no-perf -no-cnv -j '{}' \
    --temp 0 --seed 123 -n "${N_PREDICT:-128}" -p "$PROMPT" \
    "${extra_args[@]}" >"$tmp_out" 2>&1

python3 - "$tmp_out" <<'PY'
import json, os, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
start, end = text.find("{"), text.rfind("}")
obj = None
err = None
ok = False
try:
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    tc = obj.get("tool_call")
    if not isinstance(tc, dict):
        raise ValueError("missing tool_call object")
    if tc.get("name") != "browser.search":
        raise ValueError(f"wrong tool name: {tc.get('name')!r}")
    if tc.get("arguments", {}).get("query") != "ROCmFPX agent preset":
        raise ValueError("wrong query")
    if obj.get("final") is not None:
        raise ValueError("final should be null for tool call")
    ok = True
except Exception as exc:
    err = str(exc)
print(json.dumps({"status": "pass" if ok else "fail", "backend": os.environ.get("BACKEND"), "model": os.environ.get("MODEL"), "object": obj, "error": err}, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
