#!/usr/bin/env bash
# Noise-free tg128 benchmark for Qwen3.5-4B-Q4_0_ROCMI4.
# Runs 5 separate llama-bench invocations, takes median, checks variance.
# Usage: ./qwen35-4b-benchmark.sh <path-to-llama-bench> [model-path]
set -euo pipefail

BENCH="${1:-$(dirname "$0")/../build-gfx1100/bin/llama-bench}"
MODEL="${2:-$HOME/models/GGUF/Qwen3.5-4B-Q4_0_ROCMI4.gguf}"
GPU_CTL="${GPU_CTL:-$HOME/gpu-coord/gpu-ctl}"
RUNS=5

# Reserve GPU for the benchmark window
"$GPU_CTL" reserve "qwen35-4b tg128 benchmark" 10 2>/dev/null || true

VALUES=()
echo "=== Running ${RUNS}x tg128 benchmark ==="
for i in $(seq 1 "$RUNS"); do
    # Each run acquires the GPU lock
    line=$("$GPU_CTL" run 120 "bench run $i" -- \
        "$BENCH" -m "$MODEL" -p 0 -n 128 -r 1 -o csv 2>/dev/null \
        | grep -v "ggml_rocm_init" | tail -1)

    # Extract avg_ts field (column 40 in CSV)
    ts=$(echo "$line" | awk -F',' '{gsub(/"/, "", $40); print $40}')
    VALUES+=("$ts")
    echo "  Run $i: ${ts} t/s"
done

# Compute median
SORTED=($(printf '%s\n' "${VALUES[@]}" | sort -n))
MID=$((RUNS / 2))
MEDIAN="${SORTED[$MID]}"

# Check variance: any run >15% from median is anomalous
ANOMALIES=0
for v in "${VALUES[@]}"; do
    # Compute percentage deviation using awk for float math
    dev=$(awk -v med="$MEDIAN" -v val="$v" 'BEGIN { d = (val - med) / med * 100; if (d < 0) d = -d; printf "%.1f", d }')
    if awk -v d="$dev" 'BEGIN { exit !(d > 15) }'; then
        echo "  WARNING: Run value $v is ${dev}% from median — anomalous"
        ANOMALIES=$((ANOMALIES + 1))
    fi
done

echo "=== Results ==="
echo "All values: ${VALUES[*]}"
echo "Median tg128: ${MEDIAN} t/s"

if [ "$ANOMALIES" -gt 0 ]; then
    echo "WARNING: $ANOMALIES anomalous run(s) detected — consider re-running"
fi

# Clean up reservation
"$GPU_CTL" done 2>/dev/null || true

echo "$MEDIAN"
