#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
PORT="${PORT:-18123}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
HOST="${HOST:-127.0.0.1}"

export BACKEND MODEL

cd "$ROOT"

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "missing llama-server: $SERVER_BIN" >&2
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

log="$(mktemp)"
trap 'kill ${server_pid:-} 2>/dev/null || true; rm -f "$log"' EXIT

"$SERVER_BIN" \
    -m "$MODEL" -dev "$BACKEND" -ngl 999 -c 2048 \
    --host "$HOST" --port "$PORT" --load-mode none \
    >"$log" 2>&1 &
server_pid=$!

for _ in $(seq 1 60); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "llama-server failed to start" >&2
    tail -n 40 "$log" >&2 || true
    exit 1
fi

schema='{"type":"object","properties":{"answer":{"type":"string"},"method":{"type":"string"}},"required":["answer","method"],"additionalProperties":false}'
payload=$(python3 - <<PY
import json
schema = json.loads('$schema')
print(json.dumps({
    "prompt": "Return JSON only with answer 42 and method addition.",
    "n_predict": 96,
    "temperature": 0,
    "seed": 123,
    "json_schema": schema,
}))
PY
)

resp="$(curl -fsS "http://${HOST}:${PORT}/completion" -d "$payload")"

python3 - "$resp" <<'PY'
import json, os, sys
raw = sys.argv[1]
data = json.loads(raw)
text = data.get("content", "")
start, end = text.find("{"), text.rfind("}")
ok = False
err = None
obj = None
try:
    if start < 0 or end <= start:
        raise ValueError("no JSON object in completion content")
    obj = json.loads(text[start:end + 1])
    if str(obj.get("answer")) != "42" or obj.get("method") != "addition":
        raise ValueError(f"unexpected object: {obj!r}")
    ok = True
except Exception as exc:
    err = str(exc)
print(json.dumps({
    "status": "pass" if ok else "fail",
    "backend": os.environ.get("BACKEND"),
    "model": os.environ.get("MODEL"),
    "object": obj,
    "error": err,
}, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
