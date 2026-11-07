#!/usr/bin/env bash
# Agent-oriented OpenAI-compatible smokes for ROCmFPX GGUFs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
LLAMA_SERVER="${LLAMA_SERVER:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-}"
BACKEND="${BACKEND:-ROCm0}"
NGL="${NGL:-999}"
PORT="${PORT:-8138}"
ALIAS="${ALIAS:-rocmfpx-agentic-smoke}"
CTX="${CTX:-8192}"
OUT_DIR="${OUT_DIR:-$(mktemp -d /tmp/rocmfpx-agentic-smoke.XXXXXX)}"
REQUIRE_CLEAR_VRAM="${REQUIRE_CLEAR_VRAM:-1}"
VRAM_IDLE_SECONDS="${VRAM_IDLE_SECONDS:-10}"

if [[ -z "$MODEL" ]]; then
    echo "MODEL is required" >&2
    exit 2
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "missing llama-server: $LLAMA_SERVER" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$REQUIRE_CLEAR_VRAM" == "1" ]] && command -v rocm-smi >/dev/null 2>&1; then
    pids_out="$(rocm-smi --showpids 2>/dev/null || true)"
    if grep -qE '^[[:space:]]*[0-9]+[[:space:]]+' <<<"$pids_out"; then
        echo "ROCm KFD process is already active; clear VRAM before running this test." >&2
        echo "$pids_out" >&2
        exit 1
    fi
    sleep "$VRAM_IDLE_SECONDS"
fi

"$LLAMA_SERVER" \
    --log-disable \
    -m "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --alias "$ALIAS" \
    -np 1 \
    -dev "$BACKEND" \
    -ngl "$NGL" \
    -fa on \
    --load-mode mmap \
    -c "$CTX" \
    -b 512 \
    -ub 512 \
    -t 16 \
    -tb 32 \
    --jinja \
    --reasoning off \
    >"$OUT_DIR/server.log" 2>&1 &
server_pid="$!"

base_url="http://127.0.0.1:$PORT"
deadline=$((SECONDS + 180))
until curl -fsS "$base_url/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "server did not become healthy" >&2
        tail -80 "$OUT_DIR/server.log" >&2 || true
        exit 1
    fi
    sleep 1
done

chat_request() {
    local label="$1"
    local prompt="$2"
    local max_tokens="${3:-160}"
    local output="$OUT_DIR/$label.json"

    python3 - "$ALIAS" "$prompt" "$max_tokens" "$output" "$base_url" <<'PY'
import json
import sys
import urllib.request

model, prompt, max_tokens, output, base_url = sys.argv[1:6]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "stream": False,
}
req = urllib.request.Request(
    base_url + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    data = resp.read()
open(output, "wb").write(data)
PY

    python3 - "$label" "$output" <<'PY'
import json
import re
import sys

label, path = sys.argv[1:3]
data = json.load(open(path, encoding="utf-8"))
content = data["choices"][0]["message"].get("content") or ""

def extract_json(s):
    if "```" in s:
        for block in s.split("```"):
            b = block.strip()
            if b.startswith("json"):
                b = b[4:].strip()
            if "{" in b and "}" in b:
                s = b
                break
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(s[start:end + 1])

detail = ""
try:
    if label == "chat":
        if "42" not in content:
            raise ValueError(f"chat answer did not contain 42: {content!r}")
        detail = content[:240]
    elif label == "coding":
        if "def add" not in content or "return" not in content or "+" not in content:
            raise ValueError(f"coding answer missing expected add function: {content!r}")
        detail = content[:240]
    elif label == "json":
        obj = extract_json(content)
        if obj.get("answer") != 391 or obj.get("method") != "multiplication":
            raise ValueError(f"unexpected JSON object: {obj!r}")
        detail = json.dumps(obj, sort_keys=True)
    elif label == "tool":
        obj = extract_json(content)
        tc = obj.get("tool_call")
        if not isinstance(tc, dict):
            raise ValueError(f"missing tool_call object: {obj!r}")
        if tc.get("name") != "browser.search":
            raise ValueError(f"wrong tool name: {tc.get('name')!r}")
        args = tc.get("arguments") or {}
        if args.get("query") != "ROCmFPX agent preset":
            raise ValueError(f"wrong query: {args.get('query')!r}")
        detail = json.dumps(obj, sort_keys=True)
    elif label == "coherency":
        bullets = [line for line in content.splitlines() if re.match(r"^\s*(-|\*|\d+\.)\s+", line)]
        lower = content.lower()
        missing = [word for word in ("json", "tools", "chat") if word not in lower]
        if len(bullets) < 3:
            raise ValueError(f"expected at least three bullets: {content!r}")
        if missing:
            raise ValueError(f"missing coherency keywords {missing}: {content!r}")
        detail = content[:320]
    else:
        raise ValueError(f"unknown label: {label}")
except Exception as exc:
    print(json.dumps({"test": label, "status": "fail", "detail": str(exc), "output": path}, indent=2))
    raise SystemExit(1)

print(json.dumps({"test": label, "status": "pass", "detail": detail, "output": path}, indent=2))
PY
}

echo "model=$MODEL"
echo "backend=$BACKEND"
echo "base_url=$base_url"
echo "out_dir=$OUT_DIR"

curl -fsS "$base_url/v1/models" >/dev/null
echo "models: OK"

chat_request chat 'Answer with one short sentence: what is 19 plus 23?' 80
chat_request coding 'Output only Python code. Define a function add(a, b) that returns the sum of a and b.' 120
chat_request json 'Return only a JSON object with keys answer and method. answer must be 391 and method must be multiplication.' 80
chat_request tool 'Return only JSON with this shape: {"tool_call":{"name":"browser.search","arguments":{"query":"ROCmFPX agent preset"}}}. Do not add markdown.' 120
chat_request coherency 'In exactly three short bullet points, explain why agent model quantization should preserve JSON, tools, and chat coherency.' 180

stream_out="$OUT_DIR/stream.txt"
curl -fsS --max-time 180 -N "$base_url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"stream test\"}],\"stream\":true,\"max_tokens\":32}" \
    >"$stream_out"
grep -q 'chat.completion.chunk' "$stream_out"
grep -q '\[DONE\]' "$stream_out"
echo "streaming: OK"

echo "All ROCmFPX agentic smokes passed."
