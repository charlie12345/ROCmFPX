# ROCmFPX Q3 QualityPlus Recipe for Step35 MoE

Internal recipe ID: `rocmfpx-q3.step35.moe.qualityplus.v1`

This is the recipe used by the released
`Step-3.7-Flash-ROCmFPX-Q3-QualityPlus-*.gguf` artifact set in
`jcbtc/Step-3.7-Flash-ROCmFPX-Q3-QualityPlus`.

`QualityPlus` is part of this released recipe label. Do not shorten the recipe
to generic `Q3`, and do not rename the released model.

## Contract

| Field | Value |
| --- | --- |
| Implementation | `step35` |
| Topology | MoE, 288 experts with 8 active experts |
| Public recipe label | `ROCmFPX Q3 QualityPlus` |
| Stable recipe ID | `rocmfpx-q3.step35.moe.qualityplus.v1` |
| Released build size | 3.57 BPW, 9 GGUF shards |

## Released Tensor Policy

- Large routed `ffn_*_exps` tensors: `Q3_0_ROCMFPX`
- Attention query and output tensors: `Q5_K`
- Attention key and value tensors: `Q4_K`
- Shared and dense FFN tensors: `Q5_K`
- Output and token embeddings: `Q4_0_ROCMFP4_FAST`

This routing is tied to Step35's expert and attention structure. It is not a
generic Q3 recipe for Qwen35 or Gemma4.
