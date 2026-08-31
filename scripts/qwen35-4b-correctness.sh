#!/usr/bin/env bash
# Correctness gate for Qwen3.5-4B-Q4_0_ROCMI4 — 15 known-answer prompts.
# All must pass (expected substring present in output) for a run to be valid.
# Usage: ./qwen35-4b-correctness.sh [path-to-llama-completion] [model-path]
set -euo pipefail

CLI="${1:-$(dirname "$0")/../build-gfx1100/bin/llama-completion}"
MODEL="${2:-$HOME/models/GGUF/Qwen3.5-4B-Q4_0_ROCMI4.gguf}"
GPU_CTL="${GPU_CTL:-$HOME/gpu-coord/gpu-ctl}"

# Each prompt: "prompt|expected_substring" (case-insensitive substring match)
PROMPTS=(
    "The capital of France is|paris"
    "The capital of Japan is|tokyo"
    "The largest planet in our solar system is|jupiter"
    "The chemical symbol for gold is|au"
    "The author of Romeo and Juliet is|shakespeare"
    "The largest ocean on Earth is|pacific"
    "The square root of 144 is|12"
    "The currency of the United States is|dollar"
    "The tallest mountain on Earth is|everest"
    "The first president of the United States was|washington"
    "A group of lions is called|pride"
    "The primary language spoken in Brazil is|portuguese"
    "The human heart has|chamber"
    "The freezing point of water in Celsius is|0"
    "The Great Wall is located in|china"
)

PASS=0
FAIL=0
FAILED_PROMPTS=()

for entry in "${PROMPTS[@]}"; do
    prompt="${entry%%|*}"
    expected="${entry##*|}"

    # Run the model — 512 tokens for reasoning model thinking room
    output=$("$GPU_CTL" run 120 "correctness: ${prompt:0:30}" -- \
        "$CLI" -m "$MODEL" -p "$prompt" -n 512 --temp 0.0 --seed 42 \
        --no-display-prompt 2>/dev/null) || true

    # Case-insensitive substring check
    if echo "$output" | grep -qi "$expected"; then
        PASS=$((PASS + 1))
        echo "  PASS: $prompt -> found '$expected'"
    else
        FAIL=$((FAIL + 1))
        FAILED_PROMPTS+=("$prompt (expected: $expected)")
        echo "  FAIL: $prompt -> expected '$expected' not found"
    fi
done

echo "=== Correctness Results ==="
echo "Passed: $PASS / ${#PROMPTS[@]}"
echo "Failed: $FAIL / ${#PROMPTS[@]}"

if [ "$FAIL" -gt 0 ]; then
    echo "--- Failed prompts ---"
    for fp in "${FAILED_PROMPTS[@]}"; do
        echo "  $fp"
    done
    exit 1
fi

echo "ALL PROMPTS PASSED"
exit 0
