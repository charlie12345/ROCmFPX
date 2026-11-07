#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
N_PREDICT="${N_PREDICT:-96}"
CTX_SIZE="${CTX_SIZE:-4096}"
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

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

run_case() {
    local name="$1"
    local prompt="$2"
    local out="$tmp_dir/$name.txt"

    timeout --kill-after=20s 3m "$BIN" \
        -m "$MODEL" \
        -dev "$BACKEND" \
        -ngl 999 \
        -c "$CTX_SIZE" \
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
        -p "$prompt" >"$out" 2>&1
}

run_case answer 'Return exactly this JSON object and nothing else: {"answer":"42","method":"addition"}.'
run_case repair 'The previous response was invalid JSON. Repair it and return exactly this JSON object: {"status":"fixed","valid":true,"items":[1,2,3]}.'
run_case state 'Return exactly this JSON object and nothing else: {"step":3,"memory":"alpha","next_action":"call_tool"}.'

python3 - "$tmp_dir" <<'PY'
import json
import os
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

expect = {
    "answer": lambda o: str(o.get("answer")) == "42" and o.get("method") == "addition",
    "repair": lambda o: o.get("status") == "fixed" and o.get("valid") is True and o.get("items") == [1, 2, 3],
    "state": lambda o: o.get("step") == 3 and "alpha" in str(o.get("memory", "")).lower() and o.get("next_action") == "call_tool",
}

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])

results = []
failed = False
for name, check in expect.items():
    text = (root / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
    repeated = bool(re.search(r"(.{12,}?)\1\1", text, re.S))
    try:
        obj = extract_json(text)
        ok = check(obj) and not repeated
        err = None
    except Exception as exc:
        obj = None
        ok = False
        err = str(exc)
    failed = failed or not ok
    results.append({"case": name, "ok": ok, "error": err, "object": obj})

print(json.dumps({"status": "fail" if failed else "pass", "backend": os.environ.get("BACKEND"), "model": os.environ.get("MODEL"), "results": results}, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
