#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/llama-bench}"
MODEL_SRC="${MODEL_SRC:-}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfp6-competitive-sweep}"
MODEL_DIR="${MODEL_DIR:-$OUT_DIR/models}"
PRESETS="${PRESETS:-Q6_K Q6_0_ROCMFPX Q6_0_ROCMFPX_LEAN Q6_0_ROCMFPX_AGENT Q6_0_ROCMFPX_AGENT_LEAN}"
BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
RUN_QUANTIZE="${RUN_QUANTIZE:-1}"
RUN_BENCH="${RUN_BENCH:-1}"
ALLOW_REQUANTIZE="${ALLOW_REQUANTIZE:-1}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
REPEAT="${REPEAT:-3}"
THREADS="${THREADS:-16}"
NGL="${NGL:-999}"

usage() {
    cat <<EOF
Usage:
  MODEL_SRC=/path/source.gguf $0

Environment:
  OUT_DIR          output directory [$OUT_DIR]
  MODEL_DIR        directory for quantized/reused model files [$MODEL_DIR]
  PRESETS          presets to compare [$PRESETS]
  BACKENDS         devices to bench [$BACKENDS]
  RUN_QUANTIZE     1 quantizes missing/selected presets, 0 reuses outputs [$RUN_QUANTIZE]
  RUN_BENCH        1 runs llama-bench, 0 only sizes [$RUN_BENCH]
  PROMPT_TOKENS    llama-bench -p value [$PROMPT_TOKENS]
  GEN_TOKENS       llama-bench -n value [$GEN_TOKENS]
EOF
}

if [[ -z "$MODEL_SRC" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "$MODEL_SRC" ]]; then
    echo "missing MODEL_SRC: $MODEL_SRC" >&2
    exit 1
fi
if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi
if [[ "$RUN_BENCH" == "1" && ! -x "$BENCH_BIN" ]]; then
    echo "missing llama-bench: $BENCH_BIN" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$OUT_DIR/logs"
base="$(basename "$MODEL_SRC" .gguf)"

for preset in $PRESETS; do
    model="$MODEL_DIR/${base}-${preset}.gguf"
    dry="$OUT_DIR/logs/${preset}.dry-run.log"
    "$QUANTIZE_BIN" --dry-run "$MODEL_SRC" "$model" "$preset" >"$dry" 2>&1

    if [[ "$RUN_QUANTIZE" == "1" && ! -f "$model" ]]; then
        args=()
        if [[ "$ALLOW_REQUANTIZE" == "1" ]]; then
            args+=(--allow-requantize)
        fi
        "$QUANTIZE_BIN" "${args[@]}" "$MODEL_SRC" "$model" "$preset" >"$OUT_DIR/logs/${preset}.quantize.log" 2>&1
    fi

    if [[ "$RUN_BENCH" == "1" && -f "$model" ]]; then
        for backend in $BACKENDS; do
            log="$OUT_DIR/logs/${preset}.${backend}.bench.md"
            "$BENCH_BIN" \
                -m "$model" -dev "$backend" -ngl "$NGL" -fa on \
                -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPEAT" -o md \
                -t "$THREADS" -mmp 0 >"$log" 2>&1 || true
        done
    fi
done

python3 - "$OUT_DIR" "$MODEL_SRC" "$MODEL_DIR" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
src = pathlib.Path(sys.argv[2])
model_dir = pathlib.Path(sys.argv[3])
results = []

for dry in sorted((out / "logs").glob("*.dry-run.log")):
    preset = dry.name.removesuffix(".dry-run.log")
    text = dry.read_text(encoding="utf-8", errors="replace")
    size = re.search(r"quant size\s+=\s+([0-9.]+) MiB \(([0-9.]+) BPW\)", text)
    model = model_dir / f"{src.stem}-{preset}.gguf"
    item = {
        "preset": preset,
        "dry_run_ok": size is not None and "error" not in text.lower(),
        "dry_run_size_mib": float(size.group(1)) if size else None,
        "dry_run_bpw": float(size.group(2)) if size else None,
        "file_size_mib": round(model.stat().st_size / 1048576, 2) if model.exists() else None,
        "bench": {},
    }

    for bench in sorted((out / "logs").glob(f"{preset}.*.bench.md")):
        backend = bench.name.removesuffix(".bench.md").split(".", 1)[1]
        btxt = bench.read_text(encoding="utf-8", errors="replace")
        pp_vals = []
        tg_vals = []
        for line in btxt.splitlines():
            if not line.startswith("|") or " t/s " in line or "---" in line:
                continue
            cols = [col.strip() for col in line.strip().strip("|").split("|")]
            if len(cols) < 2:
                continue
            test = cols[-2]
            speed = cols[-1].split()[0]
            try:
                val = float(speed)
            except ValueError:
                continue
            if test.startswith("pp"):
                pp_vals.append(val)
            elif test.startswith("tg"):
                tg_vals.append(val)
        item["bench"][backend] = {
            "pp_tps": round(sum(pp_vals) / len(pp_vals), 2) if pp_vals else None,
            "tg_tps": round(sum(tg_vals) / len(tg_vals), 2) if tg_vals else None,
            "ok": bool(pp_vals or tg_vals),
        }
    results.append(item)

stock = next((r for r in results if r["preset"] == "Q6_K"), None)
if stock:
    for item in results:
        if item["dry_run_size_mib"] is not None and stock["dry_run_size_mib"] is not None:
            item["delta_mib_vs_q6_k"] = round(item["dry_run_size_mib"] - stock["dry_run_size_mib"], 2)
        if item["dry_run_bpw"] is not None and stock["dry_run_bpw"] is not None:
            item["delta_bpw_vs_q6_k"] = round(item["dry_run_bpw"] - stock["dry_run_bpw"], 4)

summary = {
    "source": str(src),
    "model_dir": str(model_dir),
    "out_dir": str(out),
    "results": results,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
