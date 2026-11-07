#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-/mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B}"
OUTPUT_DIR="${2:-/mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B}"
INCLUDE_MTP="${3:-0}"
ARTIFACT_PREFIX=""
CONVERT_ARGS=()
if [[ "${INCLUDE_MTP}" == "1" ]]; then
    # The converter no longer folds the MTP head into the target (--deepseek4-include-mtp is gone):
    # --mtp writes it as a separate draft GGUF, loaded with -md and --spec-type draft-mtp.
    # Run once with INCLUDE_MTP=0 for the target and once with 1 for the draft.
    ARTIFACT_PREFIX="mtp-"
    CONVERT_ARGS+=(--mtp --use-temp-file)
fi
CONVERSION_TMPDIR="${CONVERSION_TMPDIR:-${OUTPUT_DIR}/.tmp}"
INTERMEDIATE="${OUTPUT_DIR}/${ARTIFACT_PREFIX}DeepSeek-V4-Flash-180B-MXFP4-F16.gguf"
QUANTIZED="${OUTPUT_DIR}/${ARTIFACT_PREFIX}DeepSeek-V4-Flash-180B-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf"
INTERMEDIATE_PARTIAL="${INTERMEDIATE}.partial"
QUANTIZED_PARTIAL="${QUANTIZED}.partial"
QUANTIZER="${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-quantize"

mkdir -p "${OUTPUT_DIR}"
if [[ "${INCLUDE_MTP}" == "1" ]]; then
    mkdir -p "${CONVERSION_TMPDIR}"
fi
cd "${ROOT_DIR}"

if [[ ! -x "${QUANTIZER}" ]]; then
    printf 'missing quantizer: %s\n' "${QUANTIZER}" >&2
    printf 'build it with: cmake --build build-deepseek4-rocmfp4 --target llama-quantize -j 12\n' >&2
    exit 1
fi

for path in "${INTERMEDIATE}" "${QUANTIZED}" "${INTERMEDIATE_PARTIAL}" "${QUANTIZED_PARTIAL}"; do
    if [[ -e "${path}" ]]; then
        printf 'refusing to overwrite existing artifact: %s\n' "${path}" >&2
        exit 1
    fi
done

TMPDIR="${CONVERSION_TMPDIR}" python3 convert_hf_to_gguf.py \
    "${SOURCE_DIR}" \
    --outtype f16 \
    "${CONVERT_ARGS[@]}" \
    --outfile "${INTERMEDIATE_PARTIAL}"
mv "${INTERMEDIATE_PARTIAL}" "${INTERMEDIATE}"

"${QUANTIZER}" --dry-run "${INTERMEDIATE}" Q4_0_ROCMFP4
"${QUANTIZER}" "${INTERMEDIATE}" "${QUANTIZED_PARTIAL}" Q4_0_ROCMFP4

PYTHONPATH="${ROOT_DIR}/gguf-py" python3 - "${QUANTIZED_PARTIAL}" "${INCLUDE_MTP}" <<'PY'
from collections import Counter
from pathlib import Path
import sys

from gguf import GGUFReader

path = Path(sys.argv[1])
include_mtp = sys.argv[2] == "1"
reader = GGUFReader(path)
tensors = {tensor.name: tensor for tensor in reader.tensors}

expected = {
    "blk.0.ffn_down_exps.weight": "MXFP4",
    "blk.0.ffn_gate_exps.weight": "MXFP4",
    "blk.0.ffn_up_exps.weight": "MXFP4",
    "blk.0.ffn_gate_tid2eid.weight": "I32",
    "blk.0.attn_kv.weight": "Q4_0_ROCMFP4",
    "blk.2.attn_compressor_kv.weight": "Q4_0_ROCMFP4",
    "blk.2.indexer_compressor_kv.weight": "Q4_0_ROCMFP4",
    "blk.3.attn_compressor_kv.weight": "Q4_0_ROCMFP4",
    "token_embd.weight": "Q4_0_ROCMFP4",
}
if include_mtp:
    # The --mtp draft holds only the MTP layer (blk.43, e/h projections merged into eh_proj),
    # its hyper-connection head (output_hc_*) and the embeddings. None: the tensor must exist;
    # its type has not been checked on this two-file layout yet.
    expected = {
        "blk.43.nextn.eh_proj.weight": None,
        "blk.43.nextn.enorm.weight": "F32",
        "blk.43.nextn.hnorm.weight": "F32",
        "blk.43.nextn.shared_head_norm.weight": "F32",
        "output_hc_base.weight": None,
        "output_hc_fn.weight": None,
        "output_hc_scale.weight": None,
        "blk.43.ffn_down_exps.weight": "MXFP4",
        "blk.43.attn_kv.weight": None,
        "token_embd.weight": None,
    }

for name, tensor_type in expected.items():
    actual = tensors[name].tensor_type.name
    if tensor_type is not None and actual != tensor_type:
        raise SystemExit(f"{name}: expected {tensor_type}, got {actual}")

counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
print(f"validated {path}")
print(f"tensor count: {len(reader.tensors)}")
print(f"tensor types: {dict(sorted(counts.items()))}")
PY

mv "${QUANTIZED_PARTIAL}" "${QUANTIZED}"
printf 'full DeepSeek V4 ROCmFP4 artifact: %s\n' "${QUANTIZED}"
