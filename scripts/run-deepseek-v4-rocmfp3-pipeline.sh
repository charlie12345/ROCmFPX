#!/usr/bin/env bash
# Complete pipeline to copy, convert, and quantize DeepSeek-v4-Fable to ROCmFP3.
set -euo pipefail

# Configuration
SOURCE_DIR="/mnt/wd-easystore/hf-sources/DeepSeek-v4-Fable"
DEST_SRC_DIR="/mnt/wd-easystore/hf-sources/DeepSeek-v4-Fable"
OUTPUT_DIR="/mnt/ai-models/rocmfpx-quants/DeepSeek-v4-Fable"
CONVERSION_TMPDIR="/mnt/wd-easystore/.tmp"

ROOT_DIR="/home/caf/rocmfpxclone/ROCmFPX"
QUANTIZER="${ROOT_DIR}/build-strix-rocmfp4/bin/llama-quantize"

INTERMEDIATE="/mnt/wd-easystore/DeepSeek-v4-Fable-MTP-BF16.gguf"
QUANTIZED="${OUTPUT_DIR}/DeepSeek-v4-Fable-MTP-Q3_0_ROCMFPX.gguf"

INTERMEDIATE_PARTIAL="${INTERMEDIATE}.partial"
QUANTIZED_PARTIAL="${QUANTIZED}.partial"

echo "=== Pipeline Setup ==="
echo "Source/Dest:    $DEST_SRC_DIR"
echo "Output Dir:     $OUTPUT_DIR"
echo "Intermediate:   $INTERMEDIATE"
echo "Quantized:      $QUANTIZED"
echo "======================="

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONVERSION_TMPDIR"

# 1. Skip Copy step (reading directly from HDD to save SSD space)
echo "Step 1: Reading safetensors directly from HDD to bypass SSD space limit."

# 2. Convert to BF16 GGUF
cd "$ROOT_DIR"
if [ -f "$INTERMEDIATE" ]; then
    echo "Intermediate BF16 GGUF already exists. Skipping conversion."
else
    echo "Step 2: Converting HF checkpoint to GGUF BF16 with MTP..."
    TMPDIR="$CONVERSION_TMPDIR" python3 scripts/convert_deepseek_v4_modular.py \
        "$DEST_SRC_DIR" \
        --outtype bf16 \
        --deepseek4-include-mtp \
        --use-temp-file \
        --outfile "$INTERMEDIATE_PARTIAL"
    mv "$INTERMEDIATE_PARTIAL" "$INTERMEDIATE"
    echo "Conversion complete!"
fi

# 3. Quantize to Q3_0_ROCMFPX
if [ -f "$QUANTIZED" ]; then
    echo "Quantized GGUF already exists. Skipping quantization."
else
    echo "Step 3: Quantizing to Q3_0_ROCMFPX..."
    # Using the quantize-rocmfpx-agent.sh preset mapping
    # Preset: Q3_0_ROCMFPX
    # Run dry-run first
    echo "Running dry-run..."
    "$QUANTIZER" --dry-run "$INTERMEDIATE" Q3_0_ROCMFPX || true

    echo "Running actual quantization..."
    "$QUANTIZER" "$INTERMEDIATE" "$QUANTIZED_PARTIAL" Q3_0_ROCMFPX
    mv "$QUANTIZED_PARTIAL" "$QUANTIZED"
    echo "Quantization complete!"
fi

# 4. Clean up intermediate BF16 GGUF to free space
if [ -f "$INTERMEDIATE" ] && [ -f "$QUANTIZED" ]; then
    echo "Step 4: Cleaning up intermediate GGUF..."
    rm -f "$INTERMEDIATE"
    echo "Cleanup complete!"
fi

echo "=== Pipeline Finished Successfully! ==="
echo "Final model: $QUANTIZED"
