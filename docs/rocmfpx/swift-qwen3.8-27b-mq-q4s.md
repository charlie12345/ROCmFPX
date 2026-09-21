# Swift-Qwen3.8-27B ROCMFPX-MQ-Q4S

A 4-bit ROCmFP4 quantization of Swift-Qwen3.8-27B (a Qwen3.8-27B fine-tune) that
promotes the 61 most sensitive tensors to `Q6_0_ROCMFPX`. Built for the 20 GB
RX 7900 XT: the promotions cost no decode speed, prefill speed or VRAM.

## Lane

- Public name: `Swift-Qwen3.8-27B-ROCMFPX-MQ-Q4S`
- Policy: `Q4_0_ROCMFP4` bulk, 61 whole-tensor promotions to `Q6_0_ROCMFPX`
- Tensor policy file: [`swift-qwen3.8-27b-mq-q4s.tensor-types.txt`](swift-qwen3.8-27b-mq-q4s.tensor-types.txt)
- Tensor inventory: 360 F32, 445 `Q4_0_ROCMFP4`, 61 `Q6_0_ROCMFPX` (866 total)
- Size: 16,075,544,320 bytes (14.97 GiB), 4.70 BPW over the quantized tensors
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

## Reproduction

Source: the F16 GGUF shards and the importance matrix from
[`ukisai/Swift-Qwen3.8-27B-GGUF`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF).
Never requantize from a low-bit file.

```bash
llama-quantize \
  --imatrix ukisai_Swift-Qwen3.8-27b-imatrix.gguf \
  --tensor-type-file docs/rocmfpx/swift-qwen3.8-27b-mq-q4s.tensor-types.txt \
  Swift-Qwen3.8-27B-F16-00001-of-00003.gguf \
  Swift-Qwen3.8-27B-ROCMFPX-MQ-Q4S.gguf \
  Q4_0_ROCMFP4 24
```

About 7 minutes on a 12-core CPU; the GPU is not used. Add `--dry-run` first:
it must report `quant size = 15320.35 MiB (4.70 BPW)` and convert nothing to
`q5_K`.

The policy file pins **every** quantized tensor, not only the promoted ones.
That is deliberate: left to itself the quantizer escalates a varying set of
`ffn_gate` tensors to `Q5_K`, which silently puts them on the slow kernel path.

The tensor names are those of the `qwen35` 27B architecture, so the same file
applies to Qwen3.8-27B itself and to other fine-tunes of it. Only the Swift
build has been measured. Use an importance matrix computed for the model you
quantize.

For vision, pair the file with the f16 `mmproj` of the base model.

## Evidence

RX 7900 XT 20 GB, ROCm 7.2 HIP, Windows 11.

Perplexity, wikitext-2 test, 580 chunks, `-c 512 -b 512 -fa on`, f16 KV:

| file | PPL | size GiB |
| --- | ---: | ---: |
| `MQ-Q4S` (61 tensors) | 7.1072 | 14.97 |

Speed and memory, served with `-c 81920`, q4_0 KV, MTP draft `n-max 4`,
`-ub 256`:

| measure | value |
| --- | ---: |
| prose decode, MTP, tok/s | 44.2 |
| prefill at 20k / 50k, tok/s | 460 / 358 |
| 12-task agentic suite, wall s | 72.9 |
| agentic suite, tasks passed | 12/12 |
| peak VRAM, GB | 18.19 |

Peak VRAM is set by the HIP memory pool's high-water mark on this card, not by
the size of the weights.

## Notes

- If a variant of this recipe spills VRAM on your card, drop the eight
  `ffn_down` promotions first.
- A 24 GB RX 7900 XTX has room for more promotions or more context. Not
  measured.
- Swift-Qwen3.8-27B is distributed under the Swift Open License; check its
  terms before redistributing a quantized file.
