# ROCmI4 W4A4 / native IU4 experiment

ROCmI4 normally uses the exact int8 MMQ path. On gfx1151, an optional W4A4
path can instead use native IU4 WMMA for prompt processing and MTP target
verification:

```sh
cmake -S . -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_HIP_ROCMI4_W4A4=ON
```

The option defaults to `OFF`. When enabled, it is compiled only for RDNA3.5
device code and selected only when the runtime device reports RDNA3.5. All
other architectures retain the exact int8 MMQ path.

## Accuracy and scope

This is a deliberately lossy activation-quantization experiment, not a new
exact ROCmI4 representation. ROCmI4 weights are unchanged. During MMQ, float
activations are quantized directly to a signed 4-bit grid and multiplied by
the packed ROCmI4 weights with gfx1151 IU4 WMMA instructions.

Ordinary non-speculative token generation uses MMVQ and is unchanged. Batched
prompt processing and batched MTP target verification use MMQ, so the path can
improve end-to-end MTP decode when enough proposed tokens are accepted. It does
not affect ROCmFP2, ROCmFP3, or other tensor types.

Do not enable this option in builds that require exact backend-test agreement.
No tolerances are relaxed when it is enabled; expected W4A4 numerical
differences remain visible to tests and users.

## Measured tradeoff

On the development gfx1151 host with Qwen3.8-27B Q4_0_ROCMI4, a fixed-shape
pp512 test measured 566.46 prompt tokens/s versus 465.50 tokens/s for exact
int8 MMQ. Plain non-speculative tg128 stayed near 13.8 tokens/s.

The practical decode gain appears when strict MTP batches target verification.
A matched 10-task HumanEval pilot measured 49.40 tokens/s mean with W4A4 versus
41.63 tokens/s with exact int8 MMQ, an 18.66 percent gain. A full W4A4
HumanEval run measured 44.39 tokens/s mean and 45.23 tokens/s median. A
25-chunk perplexity sample increased by about 5.4 percent, so the speed path
remains opt-in.

Treat these values as host-specific qualification data, not a universal
performance guarantee. Re-run exact-versus-W4A4 quality and throughput checks
on the intended model before deployment.

## Rollback

Disable `GGML_HIP_ROCMI4_W4A4` and rebuild. Because the exact implementation
remains in the same source tree and is the default branch of every dispatch,
rollback requires no model conversion or data changes.
