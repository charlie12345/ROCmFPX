# ThinkingCap-Qwen3.8-27B ROCMFPX-MQ-Q4S

A 4-bit ROCmFP4 quantization of
[ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)
(BottleCap's Qwen3.8-27B fine-tune) that promotes the 61 most sensitive tensors
to `Q6_0_ROCMFPX`. Built for the 20 GB RX 7900 XT: the promotions cost no decode
speed, prefill speed or VRAM.

Why this model: at the `xhigh` reasoning effort it solves the same agentic and
reasoning tasks as the base-model fine-tunes we ran before while emitting about
a third fewer output tokens (see [Evidence](#evidence)). Since output is ~99%
thinking, that is a third less wall time on reasoning-heavy work at identical
decode speed.

## Lane

- Public name: `ThinkingCap-Qwen3.8-27B-ROCMFPX-MQ-Q4S`
- Policy: `Q4_0_ROCMFP4` bulk, 61 whole-tensor promotions to `Q6_0_ROCMFPX`
- Tensor policy file: [`qwen3.8-27b-mq-q4s.tensor-types.txt`](qwen3.8-27b-mq-q4s.tensor-types.txt)
- Tensor inventory: 360 F32, 445 `Q4_0_ROCMFP4`, 61 `Q6_0_ROCMFPX` (866 total)
- Size: 16,075,544,608 bytes (14.97 GiB), 4.70 BPW over the quantized tensors
- Loads only in ROCmFPX builds: the two tensor types do not exist in stock
  `llama.cpp`.

Promoted tensors:

| tensors | count | where |
| --- | ---: | --- |
| `output.weight` | 1 | |
| `blk.64.nextn.eh_proj.weight` | 1 | MTP head projection |
| `attn_k`, `attn_v`, `attn_output` | 3 x 17 | the 16 full-attention layers (3, 7, 11, ... 63) and the MTP block 64 |
| `ffn_down` | 8 | boundary layers 0, 1, 2, 3, 61, 62, 63, 64 |

Why this shape: on RDNA3 the K-quants decode about 60% slower than the ROCmFP4
kernel path, so a mixture that promotes to `Q5_K`/`Q6_K` (what the automatic
planner picks) pays for its quality in speed. `Q6_0_ROCMFPX` stays on the fast
path. The promoted set is the planner's own sensitivity ranking, re-expressed
in that type. The attention promotions are nearly free (GQA, 4 KV heads); the
eight `ffn_down` tensors carry most of the added size (0.33 GiB in total).

The policy was derived on a different Qwen3.8-27B fine-tune and transfers
unchanged: the planner's no-pin picks on the ThinkingCap f16 and the top-61
importance-matrix ranking are identical, as are all tensor names and shapes.

## Reproduction

Source: the F16 GGUF from
[`bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF`](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF)
(gated, instant approval; 50.9 GiB, one file) and an importance matrix for
ThinkingCap (the one used here: `thinkingcap-imatrix.gguf` from
[`vmarcelo/ThinkingCap-Qwen3.8-27B-MIX_GGUF`](https://huggingface.co/vmarcelo/ThinkingCap-Qwen3.8-27B-MIX_GGUF),
496 entries on the calibration-v6 corpus). Never requantize from a low-bit file.

```bash
llama-quantize \
  --imatrix thinkingcap-imatrix.gguf \
  --tensor-type-file docs/rocmfpx/qwen3.8-27b-mq-q4s.tensor-types.txt \
  ThinkingCap-Qwen3.8-27B-f16.gguf \
  ThinkingCap-Qwen3.8-27B-ROCMFPX-MQ-Q4S.gguf \
  Q4_0_ROCMFP4 24
```

About 4 minutes on a 12-core CPU; the GPU is not used. Add `--dry-run` first:
it must report `quant size = 15320.35 MiB (4.70 BPW)` and convert nothing to
`q5_K`.

The policy file pins **every** quantized tensor, not only the promoted ones.
That is deliberate: left to itself the quantizer escalates a varying set of
`attn_qkv` and `ffn_gate` tensors to `Q5_K`, which silently puts them on the
slow kernel path.

The tensor names are those of the `qwen35` 27B architecture, so the same file
applies to Qwen3.8-27B itself and to other fine-tunes of it. Use an importance
matrix computed for the model you quantize.

For vision, pair the file with the f16 `mmproj` of the base Qwen3.8-27B model.
BottleCap ships its own projector; it is numerically identical to the base one
(334/334 tensors within f16 rounding), so either works.

## Evidence

RX 7900 XT 20 GB, ROCm 7.2 HIP, Windows 11.

Perplexity, wikitext-2 test, 580 chunks, `-c 512 -b 512 -fa on`, f16 KV:

| file | PPL | size GiB |
| --- | ---: | ---: |
| `MQ-Q4S` (61 tensors) | 7.0399 ± 0.046 | 14.97 |

Speed and memory, served with `-c 81920`, q4_0 KV, MTP draft `n-max 4`,
`-ub 256`, with the vision `mmproj` loaded:

| measure | value |
| --- | ---: |
| prose decode, MTP, tok/s | 40.9 |
| prefill at 20k / 50k, tok/s | 569 / 467 |
| VRAM after a 50k prefill, GB | 18.11 dedicated + 0.89 shared |
| MTP draft acceptance | 0.79 |

Token economy, 50 runs at `reasoning_effort: xhigh` (4 tool-calling agentic
tasks x 3, 10 reasoning tasks x 2, 9 harder python-verified tasks x 2,
temperature 1, exact-match grading), against a Qwen3.8-27B fine-tune quantized
with the same recipe:

| family | output tokens, other | output tokens, ThinkingCap | correct, other | correct, ThinkingCap |
| --- | ---: | ---: | ---: | ---: |
| agentic A-D | 3,651 | 4,038 | 12/12 | 12/12 |
| reasoning R1-R10 | 10,965 | 3,754 | 19/20 | 18/20 |
| hard H1-H9 | 18,483 | 13,529 | 17/18 | 18/18 |
| all | 33,099 | 21,321 | 48/50 | 48/50 |

Decode speed was identical in the same runs (62.7 vs 63.7 tok/s median), so
the token saving is a wall-time saving. At `low` effort the two are at parity:
the gain is a shorter high-effort trace, not a shorter floor.

## Notes

- If a variant of this recipe spills VRAM on your card, drop the eight
  `ffn_down` promotions first.
- A 24 GB RX 7900 XTX has room for more promotions or more context. Not
  measured.
- ThinkingCap-Qwen3.8-27B is distributed under the PolyForm Small Business
  License 1.0.0 (the Qwen part under Apache-2.0); check its terms before
  redistributing a quantized file.
