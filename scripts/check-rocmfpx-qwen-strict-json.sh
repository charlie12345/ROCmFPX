#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
PROMPT="${PROMPT:-Return a JSON object with keys answer and method. answer must be 391 and method must be multiplication.}"
N_PREDICT="${N_PREDICT:-64}"

cd "$ROOT"

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

timeout --kill-after=20s 3m "$BIN" \
    -m "$MODEL" \
    -dev "$BACKEND" \
    -ngl 99 \
    -c 4096 \
    -t 16 \
    -tb 32 \
    --load-mode none \
    --simple-io \
    --no-display-prompt \
    --no-warmup \
    --no-perf \
    -no-cnv \
    -j '{}' \
    --temp 0 \
    -n "$N_PREDICT" \
    -p "$PROMPT" >"$tmp_out" 2>&1

python3 - "$tmp_out" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]

start = None
end = None
for i, line in enumerate(lines):
    if start is None and line.strip().startswith("{"):
        start = i
    if start is not None and line.strip() == "}":
        end = i
        break

if start is None or end is None or end <= start:
    raise SystemExit("failed to locate JSON object in llama-completion output")

payload = "\n".join(lines[start:end + 1])
obj = json.loads(payload)

if str(obj.get("answer")) != "391":
    raise SystemExit(f"unexpected answer: {obj.get('answer')!r}")
if obj.get("method") != "multiplication":
    raise SystemExit(f"unexpected method: {obj.get('method')!r}")

print(json.dumps(obj, indent=2, sort_keys=True))
PY
