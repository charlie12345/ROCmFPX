#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-agent-size-table}"
PRESETS="${PRESETS:-Q3_K_M Q3_0_ROCMFPX Q3_0_ROCMFPX_AGENT Q6_K Q6_0_ROCMFPX Q6_0_ROCMFPX_LEAN Q6_0_ROCMFPX_AGENT Q6_0_ROCMFPX_AGENT_LEAN Q8_0 Q8_0_ROCMFPX Q8_0_ROCMFPX_AGENT}"

cd "$ROOT"

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
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

python3 - "$OUT_DIR" "$MODEL_SRC" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
src = sys.argv[2]
results = []
failed = False
for log in sorted(out.glob("*.dry-run.log")):
    preset = log.name.removesuffix(".dry-run.log")
    text = log.read_text(encoding="utf-8", errors="replace")
    size = re.search(r"quant size\s+=\s+([0-9.]+ MiB) \(([0-9.]+ BPW)\)", text)
    ok = size is not None and "error" not in text.lower()
    failed = failed or not ok
    results.append({
        "preset": preset,
        "size_mib": float(size.group(1).split()[0]) if size else None,
        "size": size.group(1) if size else None,
        "bpw": size.group(2) if size else None,
        "ok": ok,
    })

stock = {r["preset"]: r for r in results if r["preset"] in {"Q3_K_M", "Q6_K", "Q8_0"}}
for r in results:
    base = None
    if r["preset"].startswith("Q3_"):
        base = stock.get("Q3_K_M")
    elif r["preset"].startswith("Q6_"):
        base = stock.get("Q6_K")
    elif r["preset"].startswith("Q8_"):
        base = stock.get("Q8_0")
    if base and r.get("size_mib") is not None and base.get("size_mib") is not None:
        r["delta_mib_vs_stock"] = round(r["size_mib"] - base["size_mib"], 2)

print(json.dumps({
    "status": "fail" if failed else "pass",
    "source": src,
    "results": results,
}, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
