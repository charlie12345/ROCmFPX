# Released Model Recipe Catalog

This catalog keeps two identifiers separate:

- The Hugging Face card title and GGUF filename define the released model
  identity. They must not be rewritten to make recipe names uniform.
- A recipe ID identifies the quantization policy that produced an artifact.
  Recipe IDs may include implementation family and topology even when the
  released model name does not.

Map recipes to individual GGUF artifacts, not only to repositories. The Ace
Saber and Coder repositories each contain two different model artifacts.

## Recipe ID Contract

Use this internal form for new recipe metadata:

```text
<format>.<implementation-family>.<topology>.<tier>.v<policy-version>
```

The implementation family is the llama.cpp architecture contract, such as
`qwen35`, `qwen35moe`, `step35`, or `gemma4`. The topology is at least `dense`
or `moe`. Actual BPW belongs to the built artifact, not the stable recipe ID.

## Published Artifact Map

| Published GGUF artifact | Hugging Face repository | Internal recipe ID |
| --- | --- | --- |
| `Step-3.7-Flash-ROCmFPX-Q3-QualityPlus-*.gguf` | `jcbtc/Step-3.7-Flash-ROCmFPX-Q3-QualityPlus` | `rocmfpx-q3.step35.moe.qualityplus.v1` |
| `Qwen3.6-35B-A3B-NSC-ACE-SABER-MTP-F16-to-ROCmFP4-STRIX_LEAN.gguf` | `jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp` | `rocmfp4.qwen35.moe.strix-lean.v1` |
| `CHADROCK-35B-Ace-Saber-MTP-ROCmFPX-MoEQuality-7.07BPW.gguf` | `jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp` | `rocmfpx.qwen35.moe.moequality.v1` |
| `Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW.gguf` | `jcbtc/Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW` | `rocmfpx.qwen35.dense.ultraquality.v1` |
| `Qwen3.6-35B-A3B-HaloStrix-Dyn-MTP-v7.gguf` | `jcbtc/qwen3.6-35b-a3b-crown-halo-mtp-dynamic` | `crown.qwen35.moe.halo-dynamic.v7` |
| `Qwopus3.6-27B-v2-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf` | `jcbtc/qwopus3.6-27b-v2-chadrock-rocmfp4-mtp` | `rocmfp4.qwen35.dense.strix-lean.v1` |
| `CHADROCK3.6-35B-UNCENSORED-MTP-STRIX-LEAN.gguf` | `jcbtc/CHADROCK3.6-35B-UNCENSORED-MTP-STRIX-LEAN` | `rocmfp4.qwen35.moe.strix-lean.v1` |
| `CHADROCK3.6-27B-Coder-MTP-ROCmFP4-STRIX_LEAN.gguf` | `jcbtc/chadrock3.6-27b-coder-rocmfp4-mtp` | `rocmfp4.qwen35.dense.strix-lean.v1` |
| `CHADROCK3.6-35B-A3B-Coder-MTP-ROCmFPX-MoEQuality-7.08BPW.gguf` | `jcbtc/chadrock3.6-27b-coder-rocmfp4-mtp` | `rocmfpx.qwen35.moe.moequality.v1` |
| `CHADROCK3.6-40B-Opus-Deckard-Uncensored-Thinking-NEO-CODE-Di-IMatrix-ROCmFP4.gguf` | `jcbtc/chadrock3.6-40b-opus-deckard-uncensored-thinking-neo-code-di-imatrix-rocmfp4` | `rocmfp4.qwen35.dense.strix-lean.v1` |
| `CHADROCK3.6-27B-Pi-Agent-MTP-ROCmFP4-STRIX_LEAN.gguf` | `jcbtc/chadrock3.6-27b-pi-agent-rocmfp4-mtp` | `rocmfp4.qwen35.dense.strix-lean.v1` |
| `Qwable-5-27B-Chadrock-v2-ROCmFP4.gguf` | `jcbtc/qwable-5-27b-chadrock-v2-rocmfp4` | `rocmfp4.qwen35.dense.strix-lean.v1` |
| `Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY.gguf` | `jcbtc/Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY` | `rocmfpx-q6.qwen35.dense.strix-quality.v1` |
| `Qwable-5-27B-Chadrock-v2-ROCmFP6-QUALITY.gguf` | `jcbtc/Qwable-5-27B-Chadrock-v2-ROCmFP6-QUALITY` | `rocmfpx-q6.qwen35.dense.strix-quality.v1` |

The lowercase UltraQuality repository is a partial duplicate of the same
released artifact, not another model. The public Qwable ROCmFP6 model remains
named `ROCmFP6-QUALITY`; the internal recipe metadata may identify its Strix
Quality policy without changing that released name.

The same MoEQuality policy produced the 7.07 BPW Ace Saber and 7.08 BPW Coder
artifacts. Conversely, the ROCmFP4 artifacts in those repositories are Strix
Lean models and are not MoEQuality models.

The machine-readable form of this table is
[`release-recipe-map.tsv`](release-recipe-map.tsv).

## Architecture Boundaries

- Dense and MoE recipes are separate contracts, even when they currently
  resolve to the same llama-quantize preset.
- Qwen35/Qwen35MoE, Step35, and Gemma4 are separate implementation contracts.
  Their attention, SSM, expert, shared-expert, embedding, and MTP structures
  differ.
- Fine-tune names such as Ace Saber, Qwopus, Qwable, Pi Agent, and Deckard do
  not create a recipe family by themselves. Reuse is allowed only within the
  same implementation/topology contract and after validation.
- `Chadrock` is model/release/runtime branding, not a quantization recipe.

Future Gemma4 work should begin with separate provisional contracts such as
`rocmfp4.gemma4.dense-qat.strix-lean.v1` and
`rocmfp4.gemma4.moe-qat.strix-lean.v1`. Do not present them as released or
validated recipes until their quality gates pass.

## Validation Gate

A recipe may be reused for another artifact only after all of these pass:

1. Architecture and topology match the recipe contract.
2. Tensor selection is reviewed against the target architecture.
3. PPL or KLD is compared against the source model.
4. A family-relevant behavior benchmark passes.
5. Served MTP acceptance, memory, and end-to-end speed are measured when MTP
   is part of the release.

Recipe-specific notes:

- [Qwen35 dense UltraQuality](qwable-ultraquality-7p61bpw-rankleave32.md)
- [Qwen35 MoE MoEQuality](qwen35-moe-rocmfpx-moequality.md)
- [Step35 MoE Q3 QualityPlus](step35-rocmfpx-q3-qualityplus.md)
