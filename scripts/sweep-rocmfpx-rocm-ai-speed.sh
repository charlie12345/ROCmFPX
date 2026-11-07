#!/usr/bin/env bash
# Build and benchmark ROCmFPX speed variants without changing GGUF formats.
#
# This harness is intentionally conservative: every variant must pass the
# ROCmFPX backend-op family gates before full-model llama-bench numbers are
# accepted into the summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-rocm-ai-speed}"
HIP_ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1151}"
JOBS="${JOBS:-$(nproc)}"

ROCMFP4_MODEL="${ROCMFP4_MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-ROCmFP4-Agentic-GGUF/Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf}"
Q4KM_MODEL="${Q4KM_MODEL:-/mnt/seconddrive/models/Qwen3.6-35B-A3B-MTP-Q4_K_M-GGUF/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf}"

DEVICES="${DEVICES:-ROCm0 Vulkan0}"
BACKENDS="${BACKENDS:-ROCm0 Vulkan0}"
PROMPT="${PROMPT:-512}"
GEN="${GEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
DEPTH="${DEPTH:-2}"
REPEAT="${REPEAT:-3}"
POLL="${POLL:-50}"

# Format: name:rocwmma_flag:extra_hip_flags
# Use commas inside extra flags when more than one compile definition is needed.
# rocwmma_flag is ignored: the GGML_HIP_ROCWMMA_FATTN CMake option was removed
# upstream (llama.cpp #26046) and is gone from this tree since 2026-09-15.
VARIANTS="${VARIANTS:-default:OFF:}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

if [[ ! -s "$ROCMFP4_MODEL" ]]; then
    echo "missing ROCmFP4 model: $ROCMFP4_MODEL" >&2
    exit 1
fi

if [[ ! -s "$Q4KM_MODEL" ]]; then
    echo "missing Q4_K_M baseline model: $Q4KM_MODEL" >&2
    exit 1
fi

run_bench() {
    local bin="$1"
    local model="$2"
    local device="$3"
    local output="$4"

    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}" \
        "$bin" \
        -m "$model" \
        -dev "$device" \
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
    stem = path.name.removesuffix(".bench.json")
    parts = stem.split(".")
    variant = parts[0]
    quant = parts[1] if len(parts) > 1 else "unknown"
    device = parts[2] if len(parts) > 2 else None
    for item in data:
        rows.append({
            "variant": variant,
            "quant": quant,
            "device": device or item.get("devices"),
            "model_type": item.get("model_type"),
            "model_size": item.get("model_size"),
            "prompt_tokens": item.get("n_prompt"),
            "gen_tokens": item.get("n_gen"),
            "tok_s": item.get("avg_ts"),
            "stddev_tok_s": item.get("stddev_ts"),
        })

summary = {"status": "pass", "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

echo "out: $OUT_DIR"
echo "rocmfp4: $ROCMFP4_MODEL"
echo "q4km: $Q4KM_MODEL"
echo "devices: $DEVICES"
echo "backends: $BACKENDS"

for spec in $VARIANTS; do
    IFS=: read -r name rocwmma extra_flags <<<"$spec"
    extra_flags="${extra_flags//,/ }"
    build_dir="$ROOT/build-rocmfpx-speed-${name}"

    echo
    echo "== variant: $name =="
    echo "extra HIP flags: ${extra_flags:-<none>}"

    BUILD_DIR="$build_dir" \
        CMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
        CMAKE_HIP_FLAGS="${extra_flags} ${CMAKE_HIP_FLAGS_EXTRA:-}" \
        JOBS="$JOBS" \
        "$SCRIPT_DIR/build-strix-rocmfp4-mtp.sh" llama-bench test-backend-ops

    echo "family backend gates: $name"
    BUILD_DIR="$build_dir" \
        BACKENDS="$BACKENDS" \
        OUT_DIR="$OUT_DIR/${name}.backend-ops" \
        "$SCRIPT_DIR/sweep-rocmfpx-backend-ops.sh" \
        >"$OUT_DIR/${name}.backend-ops.log"

    for device in $DEVICES; do
        echo "bench ROCmFP4 $name $device"
        run_bench "$build_dir/bin/llama-bench" "$ROCMFP4_MODEL" "$device" "$OUT_DIR/${name}.rocmfp4.${device}.bench.json"

        echo "bench Q4_K_M $name $device"
        run_bench "$build_dir/bin/llama-bench" "$Q4KM_MODEL" "$device" "$OUT_DIR/${name}.q4km.${device}.bench.json"

        write_summary >"$OUT_DIR/summary.log"
    done
done

echo
write_summary
echo "summary: $OUT_DIR/summary.json"
