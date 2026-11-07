#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
N_PREDICT="${N_PREDICT:-128}"
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
print(json.dumps({"status": "skip", "reason": "missing model", "model": "$MODEL"}))
PY
        exit 0
    fi
    echo "missing model: $MODEL" >&2
    exit 1
fi

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

PROMPT='You may call exactly one tool from this allowlist: search_docs(query), run_shell(command, cwd). Return only JSON with keys tool_name and arguments. For the task "list files in /tmp", call run_shell with command "ls -la /tmp" and cwd "/home/caf". Do not invent tools.'

timeout --kill-after=20s 3m "$BIN" \
    -m "$MODEL" \
    -dev "$BACKEND" \
    -ngl 999 \
    -c "${CTX_SIZE:-4096}" \
    -t "${THREADS:-16}" \
    -tb "${THREADS_BATCH:-32}" \
    --load-mode none \
    --simple-io \
    --no-display-prompt \
    --no-warmup \
    --no-perf \
    -no-cnv \
    -j '{}' \
    --temp 0 \
    --seed 123 \
    -n "$N_PREDICT" \
    -p "$PROMPT" >"$tmp_out" 2>&1

python3 - "$tmp_out" <<'PY'
import json
import os
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
start = text.find("{")
end = text.rfind("}")
ok = False
err = None
obj = None
try:
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    allowed = {"search_docs", "run_shell"}
    args = obj.get("arguments")
    if obj.get("tool_name") not in allowed:
        raise ValueError(f"hallucinated tool: {obj.get('tool_name')!r}")
    if not isinstance(args, dict):
        raise ValueError("arguments is not an object")
    if obj["tool_name"] != "run_shell":
        raise ValueError("expected run_shell")
    if args.get("command") != "ls -la /tmp":
        raise ValueError(f"unexpected command: {args.get('command')!r}")
    if args.get("cwd") != "/home/caf":
        raise ValueError(f"unexpected cwd: {args.get('cwd')!r}")
    ok = True
except Exception as exc:
    err = str(exc)

print(json.dumps({"status": "pass" if ok else "fail", "backend": os.environ.get("BACKEND"), "model": os.environ.get("MODEL"), "object": obj, "error": err}, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
