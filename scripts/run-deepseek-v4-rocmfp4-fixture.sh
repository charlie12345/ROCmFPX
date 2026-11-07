#!/usr/bin/env bash
set -euo pipefail

printf 'run-deepseek-v4-rocmfp4-fixture.sh: convert_hf_to_gguf.py no longer has --deepseek4-max-layers (or --deepseek4-include-mtp), so it cannot write an N-layer DeepSeek V4 fixture; see scripts/run-deepseek-v4-rocmfp4-full-conversion.sh for the full model\n' >&2
exit 2

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-/mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B}"
OUTPUT_DIR="${2:-/home/caf/strix-fp4/runtime-fixtures/deepseek-v4}"
MAX_LAYERS="${3:-1}"
INCLUDE_MTP="${4:-0}"
MTP_SUFFIX=""
MTP_ARGS=()
if [[ "${INCLUDE_MTP}" == "1" ]]; then
    MTP_SUFFIX="-mtp"
    MTP_ARGS+=(--deepseek4-include-mtp)
elif [[ "${INCLUDE_MTP}" != "0" ]]; then
    printf 'include MTP must be 0 or 1: %s\n' "${INCLUDE_MTP}" >&2
    exit 1
fi
INTERMEDIATE="${OUTPUT_DIR}/deepseek-v4-flash-${MAX_LAYERS}layer${MTP_SUFFIX}-mxfp4-f16.gguf"
QUANTIZED="${OUTPUT_DIR}/deepseek-v4-flash-${MAX_LAYERS}layer${MTP_SUFFIX}-rocmfp4-mxfp4.gguf"
QUANTIZER="${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-quantize"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

if [[ ! "${MAX_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'layer count must be a positive integer: %s\n' "${MAX_LAYERS}" >&2
    exit 1
fi

if [[ ! -x "${QUANTIZER}" ]]; then
    printf 'missing quantizer: %s\n' "${QUANTIZER}" >&2
    printf 'build it with: cmake --build build-deepseek4-rocmfp4 --target llama-quantize -j 12\n' >&2
    exit 1
fi

python3 convert_hf_to_gguf.py \
    "${SOURCE_DIR}" \
    --deepseek4-max-layers "${MAX_LAYERS}" \
    "${MTP_ARGS[@]}" \
    --outtype f16 \
    --outfile "${INTERMEDIATE}"

"${QUANTIZER}" --dry-run "${INTERMEDIATE}" Q4_0_ROCMFP4
"${QUANTIZER}" "${INTERMEDIATE}" "${QUANTIZED}" Q4_0_ROCMFP4

PYTHONPATH="${ROOT_DIR}/gguf-py" python3 - "${QUANTIZED}" "${MAX_LAYERS}" "${INCLUDE_MTP}" <<'PY'
from collections import Counter
from pathlib import Path
import sys

from gguf import GGUFReader

path = Path(sys.argv[1])
max_layers = int(sys.argv[2])
include_mtp = int(sys.argv[3])
reader = GGUFReader(path)
tensors = {tensor.name: tensor for tensor in reader.tensors}

expected = {
    "blk.0.ffn_down_exps.weight": "MXFP4",
    "blk.0.ffn_gate_exps.weight": "MXFP4",
    "blk.0.ffn_up_exps.weight": "MXFP4",
    "blk.0.ffn_gate_tid2eid.weight": "I32",
    "blk.0.attn_kv.weight": "Q4_0_ROCMFP4",
    "token_embd.weight": "Q4_0_ROCMFP4",
}

for name, tensor_type in expected.items():
    actual = tensors[name].tensor_type.name
    if actual != tensor_type:
        raise SystemExit(f"{name}: expected {tensor_type}, got {actual}")

if max_layers >= 3:
    compressed = [
        "blk.2.attn_compressor_ape.weight",
        "blk.2.attn_compressor_kv.weight",
        "blk.2.attn_compressor_gate.weight",
        "blk.2.attn_compressor_norm.weight",
        "blk.2.indexer_compressor_ape.weight",
        "blk.2.indexer_compressor_kv.weight",
        "blk.2.indexer_compressor_gate.weight",
        "blk.2.indexer_compressor_norm.weight",
    ]
    missing = [name for name in compressed if name not in tensors]
    if missing:
        raise SystemExit(f"missing compressed-layer tensors: {missing}")

if include_mtp:
    mtp_layer = max_layers
    expected_mtp = [
        f"blk.{mtp_layer}.nextn.e_proj.weight",
        f"blk.{mtp_layer}.nextn.h_proj.weight",
        f"blk.{mtp_layer}.nextn.enorm.weight",
        f"blk.{mtp_layer}.nextn.hnorm.weight",
        f"blk.{mtp_layer}.nextn.shared_head_norm.weight",
        f"blk.{mtp_layer}.nextn.hc_head_base.weight",
        f"blk.{mtp_layer}.nextn.hc_head_fn.weight",
        f"blk.{mtp_layer}.nextn.hc_head_scale.weight",
        f"blk.{mtp_layer}.ffn_gate_exps.weight",
        f"blk.{mtp_layer}.attn_kv.weight",
    ]
    missing = [name for name in expected_mtp if name not in tensors]
    if missing:
        raise SystemExit(f"missing MTP tensors: {missing}")

counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
print(f"validated {path}")
print(f"tensor count: {len(reader.tensors)}")
print(f"tensor types: {dict(sorted(counts.items()))}")
PY
