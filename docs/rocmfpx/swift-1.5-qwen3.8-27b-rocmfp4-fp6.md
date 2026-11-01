# Swift-1.5-Qwen3.8-27B ROCmFP4-FP6

A 4-bit ROCmFP4 quantization of
[Swift 1.5 Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
(UkisAI's Qwen3.8-27B fine-tune) that promotes the 61 most sensitive tensors
to `Q6_0_ROCMFPX`. Built for the 20 GB RX 7900 XT: the promotions cost no decode
speed, prefill speed or VRAM.

Why this model, and why at `reasoning_effort: low`: on our 50-task set it
solved everything at `low`, in the least wall time of any setting we measured
(see [Evidence](#evidence)). Lower-effort thinking text is also more
predictable, so the MTP draft is accepted more often and decode runs about 20%
faster than at `xhigh`. On replayed turns of our own agent chats, `low` was also
the quickest and most often right of the three settings (see
[Real agent turns](#real-agent-turns)).

## Lane

- Public name: `Swift-1.5-Qwen3.8-27B-ROCmFP4-FP6`
- Policy: `Q4_0_ROCMFP4` bulk, 61 whole-tensor promotions to `Q6_0_ROCMFPX`
- Tensor policy file: [`qwen3.8-27b-rocmfp4-fp6.tensor-types.txt`](qwen3.8-27b-rocmfp4-fp6.tensor-types.txt)
- Tensor inventory: 360 F32, 445 `Q4_0_ROCMFP4`, 61 `Q6_0_ROCMFPX` (866 total)
- Size: 16,075,544,864 bytes (14.97 GiB), 4.70 BPW over the quantized tensors
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
unchanged: the planner's no-pin picks on the Swift 1.5 BF16 are identical
(182 of 182), as are all tensor names and shapes.

## Reproduction

Source: the BF16 GGUF (two shards, 50.9 GiB) and the importance matrix
(`ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf`, calibration-v6 corpus) from
[`bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF`](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF).
Pass the first shard; the second is found automatically. Never requantize from
a low-bit file.

```bash
llama-quantize \
  --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf \
  --tensor-type-file docs/rocmfpx/qwen3.8-27b-rocmfp4-fp6.tensor-types.txt \
  ukisai_Swift-1.5-Qwen3.8-27b-bf16-00001-of-00002.gguf \
  Swift-1.5-Qwen3.8-27B-ROCmFP4-FP6.gguf \
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

For vision, pair the file with the
[BF16 `mmproj`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/main/mmproj-Qwen3.8-27B-BF16.gguf)
of the base Qwen3.8-27B model. On a Ryzen 9 7900 it encodes an image 8% faster
than the f16 file on the CPU (9.18 against 9.95 s at 1,443 tokens), with the same
values within rounding. Swift 1.5 ships an f16 projector that is bit-identical
to the base f16 one (334/334 tensors), so that works too.

## Evidence

RX 7900 XT 20 GB, ROCm 7.2 HIP, Windows 11.

Perplexity, wikitext-2 test, 580 chunks, `-c 512 -b 512 -fa on`, f16 KV:

| file | PPL | size GiB |
| --- | ---: | ---: |
| `ROCmFP4-FP6` (61 tensors) | 7.1516 ± 0.048 | 14.97 |

Speed and memory on 9 and 10 October 2026, served with `-c 106496` on two
slots, q4_0 KV, MTP draft `n-max 4` over 16,384 candidate tokens
(`ROCMFPX_DRAFT_VOCAB`), six query heads per attention block
(`GGML_HIP_FA_GQA6`), `-ub 1024`, vision `mmproj` on the CPU:

| measure | value |
| --- | ---: |
| prose / code decode, MTP, tok/s, median (peak) | 50.1 (53.2) / 73.4 (81.8) |
| decode at 41k depth, MTP, tok/s | 55.9 |
| decode without MTP, tok/s | 34.6 |
| prefill of a cold 17k / 41k prompt, tok/s | 679 / 621 |
| the same without the MTP draft, tok/s | 701 / 640 |
| VRAM at load, GB | 18.89 dedicated + 0.79 shared |

Reasoning effort, 50 runs per setting (4 tool-calling agentic tasks x 3,
10 reasoning tasks x 2, 9 harder python-verified tasks x 2, temperature 1,
exact-match grading), same server, effort set per request:

| effort | output tokens | wall s | correct | decode tok/s, median | MTP draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low` | 24,760 | 341 | 50/50 | 78.1 | 0.90 |
| `medium` | 25,793 | 359 | 48/50 | 76.3 | 0.88 |
| `xhigh` | 28,892 | 474 | 48/50 | 65.3 | 0.80 |

By task family at `low`: agentic A-D 4,509 tokens, 12/12; reasoning R1-R10
5,508 tokens, 20/20; hard H1-H9 14,743 tokens, 18/18.

Repeating a setting moves the token total by a few percent and the score by
one or two answers, so `low` and `medium` are not separable on this set; the
step to `xhigh` (39% more wall time for no more correct answers) is. Thinking
is never switched off: `low` is the floor we run.

### Real agent turns

26 turns of our own chief-of-staff agent: greetings, task notifications to relay,
look-ups, actions, delegations and web questions. Each turn is replayed with the
agent's real system prompt and 17 tools, the conversation before it, and the
tool results recorded at the time. One run per turn, 21 October 2026, with the
production flags above and `--reasoning-budget 4096`. The time is model time to
the final reply, since the tools answer instantly. A check per turn tests the
outcome: the fact relayed, the card created for the right agent, the command run.

| model, effort | passed | median s | p90 s | model calls per turn |
| --- | ---: | ---: | ---: | ---: |
| Swift 1.5, `low` | 25/26 | 13.5 | 38 | 2.9 |
| Swift 1.5, `medium` | 23/26 | 17.1 | 41 | 3.0 |
| Swift 1.5, `xhigh` | 24/26 | 20.7 | 36 | 3.6 |
| Thinklite-27B High, `low` | 26/26 | 20.5 | 38 | 3.1 |
| Thinklite-27B High, `medium` | 22/26 | 20.7 | 57 | 3.7 |

Every Swift 1.5 setting missed the same turn: asked to have rumours checked on
X, it searched itself instead of delegating to the research agent. The other
misses at `medium` and `xhigh` were extra look-ups before an otherwise correct
answer.

## Reasoning cap

In production this model runs with `--reasoning-budget 4096`. Over its first
three weeks in production, 27 September to 19 October 2026 (7,231 model calls
from nine agents), it thought 104 tokens per call at the median. 95% of the
calls stayed under 1,479 tokens and 99% under 3,863; the longest took 17,160.
A cap of 4,096 reaches about 0.9% of the calls and 6% of all thinking tokens.
The longest were workers that drafted a whole file or script in their thinking
before writing it in one tool call, not loops.

When the budget runs out, the server forces the budget message and the end of
thinking, and the model answers.

## Notes

- If a variant of this recipe spills VRAM on your card, drop the eight
  `ffn_down` promotions first.
- A 24 GB RX 7900 XTX has room for more promotions or more context. Not
  measured.
- Swift 1.5 Qwen3.8-27B is distributed under the Swift Open License v1.0
  (free for individuals and organisations under US$1M gross annual revenue);
  check its terms before redistributing a quantized file.
