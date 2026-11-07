#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
CTX_SIZE="${CTX_SIZE:-8192}"
FILLER_REPEAT="${FILLER_REPEAT:-180}"
export BACKEND MODEL CTX_SIZE

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

tmp_prompt="$(mktemp)"
tmp_out="$(mktemp)"
trap 'rm -f "$tmp_prompt" "$tmp_out"' EXIT

python3 - "$tmp_prompt" "$FILLER_REPEAT" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
filler = "\n".join(f"Filler line {i}: preserve the anchor but do not answer yet." for i in range(n))
path.write_text(
    "ANCHOR_CODE=ROCMFPX-AGENT-8192\n"
    + filler
    + "\nReturn exactly this JSON object and nothing else: {\"anchor\":\"ROCMFPX-AGENT-8192\",\"status\":\"ok\"}.\n",
    encoding="utf-8",
)
PY

timeout --kill-after=20s 4m "$BIN" \
    -m "$MODEL" -dev "$BACKEND" -ngl 999 -c "$CTX_SIZE" \
    -t "${THREADS:-16}" -tb "${THREADS_BATCH:-32}" --load-mode none --simple-io \
    --no-display-prompt --no-warmup --no-perf -no-cnv -j '{}' \
    --temp 0 --seed 123 -n "${N_PREDICT:-96}" -f "$tmp_prompt" >"$tmp_out" 2>&1

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
    if obj.get("anchor") != "ROCMFPX-AGENT-8192":
        raise ValueError(f"anchor mismatch: {obj.get('anchor')!r}")
    if obj.get("status") != "ok":
        raise ValueError(f"status mismatch: {obj.get('status')!r}")
    ok = True
except Exception as exc:
    err = str(exc)
print(json.dumps({"status": "pass" if ok else "fail", "backend": os.environ.get("BACKEND"), "ctx": int(os.environ.get("CTX_SIZE", "8192")), "model": os.environ.get("MODEL"), "object": obj, "error": err}, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
