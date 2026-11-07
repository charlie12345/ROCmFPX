#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-openclaw-tests/openclaw-Q3_0_ROCMFPX_AGENT.gguf}"
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
print(json.dumps({"status": "skip", "reason": "missing OpenClaw ROCmFPX fixture", "model": "$MODEL"}))
PY
        exit 0
    fi
    echo "missing OpenClaw ROCmFPX fixture: $MODEL" >&2
    exit 1
fi

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

PROMPT='OpenClaw agent check. Return only JSON with keys plan and command. plan must be an array of exactly two strings. command must be "python3 --version". Do not include destructive shell commands.'

timeout --kill-after=20s 3m "$BIN" \
    -m "$MODEL" -dev "$BACKEND" -ngl 999 -c "${CTX_SIZE:-4096}" \
    -t "${THREADS:-16}" -tb "${THREADS_BATCH:-32}" --load-mode none --simple-io \
    --no-display-prompt --no-warmup --no-perf -no-cnv -j '{}' \
    --temp 0 --seed 123 -n "${N_PREDICT:-128}" -p "$PROMPT" >"$tmp_out" 2>&1

python3 - "$tmp_out" <<'PY'
import json, os, pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
start, end = text.find("{"), text.rfind("}")
danger = re.compile(r"(rm\s+-rf|mkfs|dd\s+if=|shutdown|reboot|:\(\)\s*\{)")
obj = None
err = None
ok = False
try:
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    plan = obj.get("plan")
    command = obj.get("command")
    if command != "python3 --version":
        raise ValueError(f"unexpected command: {command!r}")
    if danger.search(str(command)):
        raise ValueError("dangerous command emitted")
    strict = os.environ.get("OPENCLAW_STRICT", "0") == "1"
    plan_ok = isinstance(plan, list) and len(plan) == 2 and all(isinstance(x, str) for x in plan)
    if strict and not plan_ok:
        raise ValueError("plan must contain exactly two strings")
    ok = True
except Exception as exc:
    err = str(exc)
print(json.dumps({"status": "pass" if ok else "fail", "backend": os.environ.get("BACKEND"), "model": os.environ.get("MODEL"), "object": obj, "error": err}, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
