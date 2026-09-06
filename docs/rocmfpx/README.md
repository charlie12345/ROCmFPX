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
| NVFP4 | 4.50 bpw | upstream GGML | yes | backend-dependent | backend-dependent |

“Yes” means a registered execution path exists; it is not a performance claim
for every GPU. FP5 and FP7 CPU, Vulkan, and HIP paths match the reference
implementation on gfx1151. Other RDNA generations still require physical-device
qualification.

The base presets minimize size at the named format. `*_AGENT` presets for
ROCmFP2/3/5/6/7/8 keep the same base format but promote sensitive embeddings,
output, attention, and selected FFN tensors. This is tensor mixing, so an agent
preset can be slightly larger than the straight preset without changing any
individual block layout.

## Entry points

- [Formats and compatibility](FORMATS.md)
- [Quant mixing](QUANT-MIXING.md)
- [Plugin and sidecar ABI](PLUGINS.md)
- [Windows AMD multi-GPU bridge](../../README.md#windows-amd-multi-gpu-bridge)
- [Hyperloom adapter](../../tools/hyperloom/README.md)
- [CI policy](CI.md)
- [AMD support tiers](SUPPORT.md)
- [Optional Vulkan ROCmFP4 CM1 acceleration and Qwen example](VULKAN-CM1.md)
- [Release process](RELEASES.md)
- Existing detailed benchmark and handoff documents in `docs/ROCmFP*.md`

ROCmFPX uses the repository's MIT license. See `NOTICE` for attribution and
`LICENSE` for the terms that apply to copies and substantial portions.
