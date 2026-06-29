#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
PPL_BIN="${PPL_BIN:-$BUILD_DIR/bin/llama-perplexity}"
MODEL_SRC="${MODEL_SRC:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-perplexity-sweep}"
BACKEND="${BACKEND:-ROCm0}"
PRESETS="${PRESETS:-Q3_K_M Q3_0_ROCMFPX Q3_0_ROCMFPX_AGENT Q6_K Q6_0_ROCMFPX Q6_0_ROCMFPX_LEAN Q6_0_ROCMFPX_AGENT Q6_0_ROCMFPX_AGENT_LEAN Q8_0 Q8_0_ROCMFPX Q8_0_ROCMFPX_AGENT}"
RUN_QUANTIZE="${RUN_QUANTIZE:-1}"
CORPUS_FILE="${CORPUS_FILE:-}"
CTX_SIZE="${CTX_SIZE:-1024}"
CORPUS_REPEAT="${CORPUS_REPEAT:-32}"

cd "$ROOT"

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi
if [[ ! -x "$PPL_BIN" ]]; then
    echo "missing llama-perplexity: $PPL_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL_SRC" ]]; then
    echo "missing MODEL_SRC: $MODEL_SRC" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
tmp_corpus="$(mktemp)"
trap 'rm -f "$tmp_corpus"' EXIT
ALLOW_REQUANTIZE="${ALLOW_REQUANTIZE:-1}"

if [[ -n "$CORPUS_FILE" && -f "$CORPUS_FILE" ]]; then
    cp "$CORPUS_FILE" "$tmp_corpus"
else
    base_corpus="$(mktemp)"
    cat >"$base_corpus" <<'EOF'
The quick brown fox jumps over the lazy dog. Structured JSON answers require stable token logits across attention layers.
Return exactly one JSON object with answer, method, and status fields when asked for machine-readable output.
Agent presets spend extra bits on attention and FFN routing without changing ROCmFPx block layouts.
Hermes and OpenClaw harnesses validate tool-call JSON on ROCm backends with strict decoding when available.
EOF
    for _ in $(seq 1 "$CORPUS_REPEAT"); do
        cat "$base_corpus" >>"$tmp_corpus"
    done
    rm -f "$base_corpus"
fi

for preset in $PRESETS; do
    model="$OUT_DIR/$(basename "$MODEL_SRC" .gguf)-${preset}.gguf"
    if [[ "$RUN_QUANTIZE" == "1" || ! -f "$model" ]]; then
        quant_args=()
        if [[ "$ALLOW_REQUANTIZE" == "1" ]]; then
            quant_args+=(--allow-requantize)
        fi
        "$QUANTIZE_BIN" "${quant_args[@]}" "$MODEL_SRC" "$model" "$preset" >"$OUT_DIR/${preset}.quantize.log" 2>&1
    fi
    log="$OUT_DIR/${preset}.ppl.log"
    timeout --kill-after=20s 5m "$PPL_BIN" \
        -m "$model" -dev "$BACKEND" -ngl 99 -c "$CTX_SIZE" \
        -t "${THREADS:-8}" -tb "${THREADS_BATCH:-16}" \
        --no-mmap --no-warmup -f "$tmp_corpus" >"$log" 2>&1 || true
done

python3 - "$OUT_DIR" "$MODEL_SRC" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
src = sys.argv[2]
results = []
for log in sorted(out.glob("*.ppl.log")):
    preset = log.name.removesuffix(".ppl.log")
    text = log.read_text(encoding="utf-8", errors="replace")
    ppl = re.search(r"perplexity:\s*([0-9.]+)", text)
    results.append({
        "preset": preset,
        "perplexity": float(ppl.group(1)) if ppl else None,
        "ok": ppl is not None,
    })

failed = any(not item["ok"] for item in results)
print(json.dumps({"status": "fail" if failed else "pass", "source": src, "results": results}, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
