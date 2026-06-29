#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-BF16.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-agent-routing-sweep}"
BACKEND="${BACKEND:-ROCm0}"
RUN_AGENT_JSON="${RUN_AGENT_JSON:-0}"
PRESETS="${PRESETS:-Q3_0_ROCMFPX Q3_0_ROCMFPX_AGENT Q6_0_ROCMFPX Q6_0_ROCMFPX_LEAN Q6_0_ROCMFPX_AGENT Q6_0_ROCMFPX_AGENT_LEAN Q8_0_ROCMFPX Q8_0_ROCMFPX_AGENT}"

cd "$ROOT"

if [[ ! -f "$MODEL_SRC" ]]; then
    fallback="/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q4_K_M.gguf"
    if [[ -f "$fallback" ]]; then
        MODEL_SRC="$fallback"
    fi
fi

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize binary: $QUANTIZE_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL_SRC" ]]; then
    echo "missing MODEL_SRC: $MODEL_SRC" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

for preset in $PRESETS; do
    log="$OUT_DIR/${preset}.dry-run.log"
    "$QUANTIZE_BIN" --dry-run "$MODEL_SRC" "$OUT_DIR/${preset}.gguf" "$preset" >"$log" 2>&1
done

if [[ "$RUN_AGENT_JSON" == "1" ]]; then
    for preset in $PRESETS; do
        model="$OUT_DIR/${preset}.gguf"
        log="$OUT_DIR/${preset}.quantize.log"
        MODEL="$model" BACKEND="$BACKEND" "$QUANTIZE_BIN" "$MODEL_SRC" "$model" "$preset" >"$log" 2>&1
        MODEL="$model" BACKEND="$BACKEND" "$ROOT/scripts/check-rocmfpx-agent-json.sh" >"$OUT_DIR/${preset}.agent-json.json" 2>&1 || true
    done
fi

python3 - "$OUT_DIR" "$MODEL_SRC" "$RUN_AGENT_JSON" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
src = sys.argv[2]
with_agent = sys.argv[3] == "1"
results = []
for dry in sorted(out.glob("*.dry-run.log")):
    preset = dry.name.removesuffix(".dry-run.log")
    text = dry.read_text(encoding="utf-8", errors="replace")
    size = re.search(r"quant size\s+=\s+([0-9.]+ MiB) \(([0-9.]+ BPW)\)", text)
    entry = {
        "preset": preset,
        "dry_run_ok": "error" not in text.lower() and "failed" not in text.lower(),
        "size": size.group(1) if size else None,
        "bpw": size.group(2) if size else None,
    }
    if with_agent:
        path = out / f"{preset}.agent-json.json"
        if path.exists():
            raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                payload = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                entry["agent_json_status"] = payload.get("status")
            except Exception as exc:
                entry["agent_json_status"] = "parse-fail"
                entry["agent_json_error"] = str(exc)
        else:
            entry["agent_json_status"] = "missing"
    results.append(entry)

failed = any(not item["dry_run_ok"] for item in results)
print(json.dumps({"status": "fail" if failed else "pass", "source": src, "run_agent_json": with_agent, "results": results}, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
