#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-bf16-to-ROCmFP4-STRIX_LEAN.gguf}"
MIN_DECODE_TPS="${MIN_DECODE_TPS:-12.7}"
RUN_ATTEMPTS="${RUN_ATTEMPTS:-3}"

PROMPT="Answer in one concise sentence: what is 17 plus 25?"

cd "$ROOT"

run_case() {
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    timeout --kill-after=20s 5m "$BIN" \
        -m "$MODEL" \
        -dev Vulkan0 \
        -ngl 999 \
        -fa on \
        --load-mode none \
        -t 16 \
        -tb 32 \
        -ctk f16 \
        -ctv f16 \
        -c 131072 \
        -b 2048 \
        -ub 512 \
        --temp 0 \
        --ignore-eos \
        --no-display-prompt \
        --simple-io \
        --no-warmup \
        -st \
        -cnv \
        --reasoning on \
        -n 96 \
        -p "$PROMPT" 2>&1
}

check_decode_floor() {
    local output="$1"

    local line decode_tps
    line="$(printf "%s\n" "$output" | rg "Prompt: .*Generation:" | tail -n 1 || true)"
    decode_tps="$(printf "%s\n" "$line" | sed -n 's/.*Generation: \([0-9.]*\) t\/s.*/\1/p')"

    if [[ -z "$decode_tps" ]]; then
        echo "FAIL: could not parse decode speed" >&2
        return 1
    fi

    awk -v got="$decode_tps" -v min="$MIN_DECODE_TPS" 'BEGIN {
        if (got + 0 < min + 0) {
            printf("FAIL: decode %.2f tok/s is below floor %.2f tok/s\n", got, min) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: decode %.2f tok/s meets floor %.2f tok/s\n", got, min);
    }'
}

for (( attempt = 1; attempt <= RUN_ATTEMPTS; ++attempt )); do
    output="$(run_case)"
    printf "%s\n" "$output"

    if check_decode_floor "$output"; then
        exit 0
    fi

    if (( attempt < RUN_ATTEMPTS )); then
        echo "WARN: DeepSeek decode guard attempt ${attempt}/${RUN_ATTEMPTS} missed the floor; retrying." >&2
    fi
done

exit 1
