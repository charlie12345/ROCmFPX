# ROCmFPX MoEQuality Recipe for Qwen35 MoE

Internal recipe ID: `rocmfpx.qwen35.moe.moequality.v1`

This is the Qwen35MoE quality policy used by two released artifacts:

- `CHADROCK-35B-Ace-Saber-MTP-ROCmFPX-MoEQuality-7.07BPW.gguf`
- `CHADROCK3.6-35B-A3B-Coder-MTP-ROCmFPX-MoEQuality-7.08BPW.gguf`

The artifact names above come from their Hugging Face cards and remain the
release source of truth. The 7.07 and 7.08 BPW values are outputs from two
build instances; they do not identify separate recipe policies.

## Contract

| Field | Value |
| --- | --- |
| Implementation | `qwen35moe` |
| Topology | MoE |
| Public recipe label | `ROCmFPX MoEQuality` |
| Policy lineage | H29-B-derived MoE quality tensor policy |
| Stable recipe ID | `rocmfpx.qwen35.moe.moequality.v1` |

MoEQuality is not the dense Qwen35 UltraQuality recipe. It is also not the
ROCmFP4 Strix Lean recipe used by the separate ROCmFP4 artifacts in the Ace
Saber and Coder repositories.

Do not reuse this recipe for dense Qwen, Step35, or Gemma4 from the label
alone. A new architecture/topology contract and validation pass are required.
