#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-completion}"
MODEL="${MODEL:-$ROOT/../models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
N_PREDICT="${N_PREDICT:-256}"
TEMP="${TEMP:-0}"

cd "$ROOT"

rocmfpx_require_binary "$BIN"

global_extra_args=()
if [[ -n "${LLAMA_COMPLETION_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    global_extra_args=(${LLAMA_COMPLETION_ARGS})
fi

run_probe() {
    local name="$1"
    local prompt="$2"
    shift 2
    local extra_args=("${global_extra_args[@]}" "$@")
    local tmp_out
    tmp_out="$(mktemp)"
    if ! timeout --kill-after=30s 5m "$BIN" \
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
        --temp "$TEMP" \
        -n "$N_PREDICT" \
        -p "$prompt" \
        "${extra_args[@]}" >"$tmp_out" 2>&1; then
        echo "probe $name: llama-completion failed" >&2
        cat "$tmp_out" >&2
        rm -f "$tmp_out"
        exit 1
    fi

    python3 - "$name" "$tmp_out" <<'PY'
import json
import re
import sys

name, path = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Strip common llama-completion banners.
lines = [line for line in text.splitlines() if line.strip()]
body = "\n".join(lines)

print(body[-1200:])
print()

if name == "coding":
    if "def " not in body and "function" not in body.lower():
        raise SystemExit("coding probe: no function-like output")
    if body.lower().count("duplicate") < 1 and "def " not in body:
        raise SystemExit("coding probe: missing duplicate-finder logic")
elif name == "summary":
    bullets = [line for line in lines if re.match(r"^(\*|-|\d+\.)", line.strip())]
    if len(bullets) < 3:
        raise SystemExit(f"summary probe: expected >=3 bullets, got {len(bullets)}")
elif name == "json":
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise SystemExit("json probe: no JSON object found")
    obj = json.loads(body[start:end + 1])
    if str(obj.get("answer")) != "391":
        raise SystemExit(f"json probe: unexpected answer {obj.get('answer')!r}")
    if obj.get("method") != "multiplication":
        raise SystemExit(f"json probe: unexpected method {obj.get('method')!r}")

print(f"{name}: OK")
PY
    rm -f "$tmp_out"
}

probes="${COHERENCY_PROBES:-coding,summary,json}"
probes="${probes// /}"

IFS=',' read -r -a probe_list <<< "$probes"

for probe in "${probe_list[@]}"; do
    case "$probe" in
        coding)
            run_probe coding \
                "Write a Python function named find_duplicates that returns duplicate values from a list. Output only the function."
            ;;
        summary)
            run_probe summary \
                "Summarize why offsite backups matter in exactly three short bullet points."
            ;;
        json)
            run_probe json \
                "Return a JSON object with keys answer and method. answer must be 391 and method must be multiplication." \
                -j '{}'
            ;;
        *)
            echo "unknown coherency probe: $probe" >&2
            exit 1
            ;;
    esac
done

echo "All coherency probes passed for ${MODEL} (probes=${probes})"