# Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW Recipe

This recipe reproduces the released
`Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW` model. The Hugging Face card
and GGUF filename are the model identity source of truth. UltraQuality is the
recipe label for the ranked leave-32 ROCmFPX/Q6_K splice candidate:

```text
Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW.gguf
```

## Recipe Summary

| Field | Value |
| --- | --- |
| Implementation/topology | `qwen35` dense hybrid |
| Public model name | `Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW` |
| Internal recipe ID | `rocmfpx.qwen35.dense.ultraquality.v1` |
| Quant policy | ROCmFPX Q6 base with ranked leave-32 attention tensors and Q6_K splice |
| Output BPW | 7.6146 |
| Output bytes | 26,004,616,416 |
| SHA256 | `14cb3fb0670163a1b0f73c5df521ce0513cfddd7609d75d0640d00a07537073e` |
| Serving profile | Strix Halo ROCm, q8 target KV, f16 MTP draft KV |

The policy starts from `Q6_0_ROCMFPX` and uses a generated
`--tensor-type-file` to restore selected dense FFN and attention tensors to
`Q6_K`, while leaving the best 32 attention tensors in ROCmFPX according to
`qwable-ultraquality-7p61bpw-attention-rank.csv`.

## Files

- `docs/recipes/qwable-ultraquality-7p61bpw-rankleave32.md`: this recipe.
- `docs/recipes/qwable-ultraquality-7p61bpw-attention-rank.csv`: attention
  ranking input for the leave-32 policy.
- `scripts/rocmfpx-ranked-policy.py`: converts the rank CSV into a
  `llama-quantize --tensor-type-file`.
- `scripts/quantize-rocmfpx-agent.sh`: wrapper around `llama-quantize` with
  ROCmFPX preset selection and tensor-type-file pass-through.
- `scripts/run-rocmfpx-mtp-server.sh`: reproducible served-MTP launch wrapper.

## Build ROCmFPX Tools

```bash
cmake -S . -B build-ultraquality-rocm \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=ON

cmake --build build-ultraquality-rocm -j --target llama-quantize llama-server
```

Use the exact ROCm target for your Strix Halo system if it differs from
`gfx1151`.

## Generate the Tensor Policy

```bash
python3 scripts/rocmfpx-ranked-policy.py \
  --rank-csv docs/recipes/qwable-ultraquality-7p61bpw-attention-rank.csv \
  --leave-count 32 \
  --base-type q6_0_rocmfpx \
  --restore-type q6_k \
  --output build-ultraquality-rocm/ultraquality-leave32.tensor-type
```

The generated tensor-type file is the policy splice. With the included rank
CSV and `--leave-count 32`, it reproduces the UltraQuality rankleave32 lane.

## Quantize

Set `SRC` to the BF16/F16 GGUF for the Qwen3.6 27B MTP base and `OUT` to the
desired UltraQuality GGUF path:

```bash
SRC=/models/Qwen3.6-27B-MTP-BF16.gguf \
OUT=/models/Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW.gguf \
FORMAT=rocmfp6 \
PROFILE=straight \
BUILD_DIR=$PWD/build-ultraquality-rocm \
TENSOR_TYPE_FILE=$PWD/build-ultraquality-rocm/ultraquality-leave32.tensor-type \
scripts/quantize-rocmfpx-agent.sh
```

Expected output characteristics:

```text
BPW: 7.6146
bytes: 26004616416
sha256: 14cb3fb0670163a1b0f73c5df521ce0513cfddd7609d75d0640d00a07537073e
```

If the base GGUF differs, the byte count and hash will differ. The policy is
the reproducible part of the recipe; bit-identical output also requires the
same source GGUF and the same ROCmFPX quantizer revision.

## Serve With MTP

```bash
MODEL=/models/Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW.gguf \
BIN=$PWD/build-ultraquality-rocm/bin/llama-server \
DEVICE=ROCm0 \
SPEC_DRAFT_DEVICE=ROCm0 \
ALIAS=qwable-27b-chadrock-rocmfpx-ultraquality-7p61bpw \
CTX_SIZE=65536 \
CACHE_TYPE_K=q8_0 \
CACHE_TYPE_V=q8_0 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
POLL=100 \
POLL_BATCH=1 \
PERF_PRESET=latency \
scripts/run-rocmfpx-mtp-server.sh
```

The served profile uses:

- one slot: `--parallel 1`
- target KV: `q8_0/q8_0`
- draft KV: `f16/f16`
- draft MTP: `--spec-type draft-mtp`
- draft cap: `--spec-draft-n-max 6`
- backend sampling disabled for the draft path
- prompt cache disabled by the wrapper when `STRICT_BENCH=1`

## Reference Result

The card-refresh served row for this lane used a 20 KB prompt shape with 3946
prompt tokens and 512 generated tokens:

| Metric | Value |
| --- | ---: |
| Prompt tokens | 3946 |
| Generated tokens | 512 |
| Prompt throughput | 209.835 tok/s |
| Decode throughput | 25.921 tok/s |
| Wall time | 38.564 s |
| Time to first token | 18.811 s |
| Draft accepted | 437 / 439 |

Quality evidence for the promoted lane:

| Metric | Value |
| --- | ---: |
| PPL | 6.5212 +/- 0.09323 |
| Mean KLD | 0.002420 +/- 0.000481 |
| P99 KLD | 0.019161 |
| P99.9 KLD | 0.150872 |
| Same-top rate | 97.843% +/- 0.227 |
| HermesAgent-20 | 82 |
| HumanEval base | 160 / 164 |
| HumanEval+ | 154 / 164 |

## Notes

- The older STRIX QUALITY recipe used broader Q6/Q8 promotion. UltraQuality is
  the ranked leave-32/Q6_K-splice replacement.
- A stricter quality-tail fallback exists at ranked leave-16, but this
  UltraQuality recipe is the promoted 7.61 BPW leave-32 lane.
- The public report for the lane is published at
  `https://llm.ciru.ai/research/qwable-ultra-quality-761bpw-report-20260701.html`.
