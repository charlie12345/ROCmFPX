#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfp6-kernel-tune-$(date +%Y%m%d-%H%M%S)}"
BACKENDS="${BACKENDS:-ROCm0}"
PROFILES="${PROFILES:-baseline fast-signmag half-split fast-signmag-half-split}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
REPEAT="${REPEAT:-3}"
THREADS="${THREADS:-16}"
NGL="${NGL:-999}"
JOBS="${JOBS:-$(nproc)}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_BENCH="${RUN_BENCH:-1}"

usage() {
    cat <<EOF
Usage:
  MODEL=/path/model-Q6_0_ROCMFPX*.gguf $0

Environment:
  OUT_DIR        output directory [$OUT_DIR]
  PROFILES       baseline fast-signmag half-split fast-signmag-half-split
  BACKENDS       llama-bench devices [$BACKENDS]
  RUN_BUILD      build each profile before testing [$RUN_BUILD]
  RUN_BENCH      run llama-bench after build [$RUN_BENCH]
EOF
}

if [[ -z "$MODEL" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "missing MODEL: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/logs"

profile_flags() {
    case "$1" in
        baseline)
            printf '%s\n' ''
            ;;
        fast-signmag)
            printf '%s\n' '-DGGML_ROCMFP6_FAST_SIGNMAG_PACK=1'
            ;;
        half-split)
            printf '%s\n' '-DGGML_ROCMFP6_MMVQ_HALF_BLOCK_SPLIT=1'
            ;;
        fast-signmag-half-split)
            printf '%s\n' '-DGGML_ROCMFP6_FAST_SIGNMAG_PACK=1 -DGGML_ROCMFP6_MMVQ_HALF_BLOCK_SPLIT=1'
            ;;
        *)
            echo "unknown profile: $1" >&2
            return 2
            ;;
    esac
}

for profile in $PROFILES; do
    build_dir="$OUT_DIR/build-$profile"
    flags="$(profile_flags "$profile")"

    if [[ "$RUN_BUILD" == "1" ]]; then
        BUILD_DIR="$build_dir" JOBS="$JOBS" CMAKE_HIP_FLAGS="$flags" \
            "$ROOT/scripts/build-strix-rocmfp4-mtp.sh" llama-bench test-backend-ops \
            >"$OUT_DIR/logs/$profile.build.log" 2>&1
    fi

    bench_bin="$build_dir/bin/llama-bench"
    if [[ "$RUN_BENCH" == "1" ]]; then
        if [[ ! -x "$bench_bin" ]]; then
            echo "missing benchmark binary for $profile: $bench_bin" >&2
            continue
        fi
        for backend in $BACKENDS; do
            "$bench_bin" \
                -m "$MODEL" -dev "$backend" -ngl "$NGL" -fa on \
                -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPEAT" -o md \
                -t "$THREADS" -mmp 0 >"$OUT_DIR/logs/$profile.$backend.bench.md" 2>&1 || true
        done
    fi
done

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
results = []

for log in sorted((out / "logs").glob("*.bench.md")):
    profile, backend, _ = log.name.split(".", 2)
    text = log.read_text(encoding="utf-8", errors="replace")
    pp_vals = []
    tg_vals = []
    for line in text.splitlines():
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
    results.append({
        "profile": profile,
        "backend": backend,
        "pp_tps": round(sum(pp_vals) / len(pp_vals), 2) if pp_vals else None,
        "tg_tps": round(sum(tg_vals) / len(tg_vals), 2) if tg_vals else None,
        "ok": bool(pp_vals or tg_vals),
    })

baseline = {(r["backend"]): r for r in results if r["profile"] == "baseline"}
for r in results:
    base = baseline.get(r["backend"])
    if base:
        if r["pp_tps"] is not None and base["pp_tps"] is not None:
            r["pp_delta_vs_baseline"] = round(r["pp_tps"] - base["pp_tps"], 2)
        if r["tg_tps"] is not None and base["tg_tps"] is not None:
            r["tg_delta_vs_baseline"] = round(r["tg_tps"] - base["tg_tps"], 2)

summary = {"out_dir": str(out), "results": results}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "summary: $OUT_DIR/summary.json"
