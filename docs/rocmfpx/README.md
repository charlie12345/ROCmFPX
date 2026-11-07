# ROCmFPX

ROCmFPX is a family of 32-weight GGML quantization blocks developed for
portable inference, with AMD RDNA as the first optimization target. Carlo
Pasquale (Charlie12345) is the creator and founder of the ROCmFPX format family
and of ROCmFP3, ROCmFP4, ROCmFP6, and ROCmFP8.

## Format status

| Format | Encoded size | Disk contract | CPU | Vulkan | HIP |
| --- | ---: | --- | --- | --- | --- |
| ROCmFP2 | 2.50 bpw | frozen | yes | yes | yes |
| ROCmFP3 | 3.50 bpw | frozen | yes | yes | yes |
| ROCmFP4 | 4.25/4.50 bpw variants | frozen | yes | yes | yes |
| ROCmFP5 | 5.50 bpw | new, version 1 | yes | yes | yes |
| ROCmFP6 | 6.50 bpw | frozen | yes | yes, packed-layout verified | yes |
| ROCmFP7 | 7.50 bpw | new, version 1 | yes | yes | yes |
| ROCmFP8 | 8.25 bpw | frozen | yes | yes | yes |
| ROCmI4 | 4.25 bpw | not frozen | yes | no | yes |
| NVFP4 | 4.50 bpw | upstream GGML | yes | yes | yes |

“Yes” means a registered execution path exists; it is not a performance claim
for every GPU. On the CPU, every ROCmFPX type, ROCmFP4 and ROCmI4 has an AVX2
integer dot against Q8_0 since 2026-11-06 (`[TAG_ROCMFPX_CPU_DOT]`; the
[expert handoff](../expert-handoff.md) computes MoE experts with it). Builds
without AVX2 keep the float fallback. Vulkan has no ROCmI4 matmul, so those ops
run on the CPU. The ROCmFPX types match the reference implementation on
`gfx1100` (RX 7900 XT) with HIP; the FP5 and FP7 CPU, Vulkan and HIP paths were
also verified on `gfx1151` by the ROCmFPX project.

The base presets minimize size at the named format. `*_AGENT` presets for
ROCmFP2/3/5/6/7/8 keep the same base format but promote sensitive embeddings,
output, attention, and selected FFN tensors. This is tensor mixing, so an agent
preset can be slightly larger than the straight preset without changing any
individual block layout.

## Entry points

- [Formats and compatibility](FORMATS.md)
- [Quant mixing](QUANT-MIXING.md)
- [Swift-1.5-Qwen3.8-27B ROCmFP4-FP6 recipe for the RX 7900 XT](swift-1.5-qwen3.8-27b-rocmfp4-fp6.md)
- [Nanbeige4.3-33B IQ3-FP6 recipe for the RX 7900 XT](nanbeige4.3-33b-iq3-fp6.md)
- [Plugin and sidecar ABI](PLUGINS.md)
- [Native draft vocabulary for MTP drafting](native-draft-vocabulary.md)
- [Upstream maintenance](UPSTREAM-MAINTENANCE.md)
- [Windows AMD multi-GPU bridge](../../README.md#windows-amd-multi-gpu-bridge)
- [Hyperloom adapter](../../tools/hyperloom/README.md)
- [CI policy](CI.md)
- [Other AMD cards](../../README.md#other-cards)
- [Optional Vulkan ROCmFP4 CM1 acceleration and Qwen example](VULKAN-CM1.md)
- [Release process](RELEASES.md)
- The ROCmFPX project's benchmark and handoff notes from before the fork (June-July 2026) in `docs/ROCmFP*.md`

ROCmFPX uses the repository's MIT license. See `NOTICE` for attribution and
`LICENSE` for the terms that apply to copies and substantial portions.
