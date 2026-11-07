#!/usr/bin/env bash
# Rebuild and benchmark ROCmFP4 MMQ prefill tuning candidates on the local
# Strix ROCmFPX tree. This is intentionally full-model focused: backend-op
# wins are only useful if Qwen ROCmFP4 prompt throughput improves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-ROCmFP4-Agentic-GGUF/Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf}"
BASELINE_MODEL="${BASELINE_MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-Q4_K_M-GGUF/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfp4-prefill-mmq-sweep}"
HIP_ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1151}"
JOBS="${JOBS:-$(nproc)}"
DEVICE="${DEVICE:-ROCm0}"

PROMPT="${PROMPT:-512}"
GEN="${GEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
DEPTH="${DEPTH:-2}"
REPEAT="${REPEAT:-3}"
POLL="${POLL:-50}"

# Space-separated name:flags entries. Use commas inside flags when a variant
# needs more than one compile definition, for example:
#   VARIANTS='default: both:-DFOO=1,-DBAR=1'
VARIANTS="${VARIANTS:-default: fast-mmq-vdr4:-DGGML_ROCMFP4_FAST_Q8_1_MMQ_VDR=4 fast-mmq-vdr16:-DGGML_ROCMFP4_FAST_Q8_1_MMQ_VDR=16 dual-mmq-vdr16:-DGGML_ROCMFP4_Q8_1_MMQ_VDR=16 all-mmq-vdr4:-DGGML_ROCMFP4_Q8_1_MMQ_VDR=4,-DGGML_ROCMFP4_FAST_Q8_1_MMQ_VDR=4}"

cd "$ROOT"

mkdir -p "$OUT_DIR"

if [[ ! -s "$MODEL" ]]; then
    echo "missing ROCmFP4 model: $MODEL" >&2
    exit 1
fi

run_bench() {
    local bin="$1"
    local model="$2"
    local output="$3"

    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        "$bin" \
        -m "$model" \
        -dev "$DEVICE" \
        -ngl 999 \
        -fa on \
        -ctk q4_0 \
        -ctv q4_0 \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -lm none \
        -t "$THREADS" \
        --poll "$POLL" \
        -p "$PROMPT" \
        -n "$GEN" \
        -d "$DEPTH" \
        -r "$REPEAT" \
        --no-warmup \
        -o json >"$output"
}

write_summary() {
    python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.bench.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows.append({"file": path.name, "status": "invalid-json"})
        continue
    variant = path.name.removesuffix(".bench.json")
    for item in data:
        rows.append({
            "variant": variant,
            "devices": item.get("devices"),
            "model_type": item.get("model_type"),
            "prompt_tokens": item.get("n_prompt"),
            "gen_tokens": item.get("n_gen"),
            "tok_s": item.get("avg_ts"),
            "stddev_tok_s": item.get("stddev_ts"),
            "model_size": item.get("model_size"),
        })

summary = {"status": "pass", "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

echo "model: $MODEL"
echo "device: $DEVICE"
echo "out: $OUT_DIR"

if [[ -s "$BASELINE_MODEL" ]]; then
    echo "baseline model: $BASELINE_MODEL"
else
    echo "baseline model not found, skipping: $BASELINE_MODEL" >&2
    BASELINE_MODEL=""
fi

for spec in $VARIANTS; do
    name="${spec%%:*}"
    flags="${spec#*:}"
    if [[ "$flags" == "$name" ]]; then
        flags=""
    fi
    flags="${flags//,/ }"

    build_dir="$ROOT/build-rocmfp4-prefill-${name}"
    echo
    echo "== variant: $name =="
    echo "flags: ${flags:-<default>}"
    BUILD_DIR="$build_dir" \
        CMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
        CMAKE_HIP_FLAGS="${flags} ${CMAKE_HIP_FLAGS_EXTRA:-}" \
        JOBS="$JOBS" \
        "$SCRIPT_DIR/build-strix-rocmfp4-mtp.sh" llama-bench test-backend-ops

    echo "backend guard: $name"
    "$build_dir/bin/test-backend-ops" test -o MUL_MAT -b "$DEVICE" -p 'type_a=q4_0_rocmfp4.*type_b=f32' \
        >"$OUT_DIR/${name}.backend.log" 2>&1

    echo "bench rocmfp4: $name"
    run_bench "$build_dir/bin/llama-bench" "$MODEL" "$OUT_DIR/${name}.bench.json"

    if [[ -n "$BASELINE_MODEL" ]]; then
        echo "bench q4_k_m baseline with build: $name"
        run_bench "$build_dir/bin/llama-bench" "$BASELINE_MODEL" "$OUT_DIR/${name}.q4km.bench.json"
    fi

    write_summary >"$OUT_DIR/summary.log"
done

echo
write_summary
echo "summary: $OUT_DIR/summary.json"
