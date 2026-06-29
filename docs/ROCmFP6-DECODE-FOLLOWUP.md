# ROCmFP6 Decode Follow-Up

Date: 2026-06-27

Model:

`/mnt/seconddrive/models/Qwythos-9B-Claude-Mythos-5-1M-GGUF`

Commands used `llama-bench -p 512 -n 128 -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16`, with `-r 3` for final-state checks and `-r 5` for probes.

## Current Comparison

| Quant | Backend | Size | pp512 tok/s | tg128 tok/s |
|---|---:|---:|---:|---:|
| Q6_0_ROCMFPX_AGENT, packed default | ROCm0 | 8.06 GiB | 730.16 | 25.59 |
| Q6_0_ROCMFPX_AGENT, expanded-device experiment | ROCm0 | 8.06 GiB | 737.98 | 23.76 |
| Q6_K | ROCm0 | 7.08 GiB | 794.78 | 28.97 |
| Q6_0_ROCMFPX_AGENT | Vulkan0 | 8.06 GiB | 1055.21 | 22.78 |
| Q6_K | Vulkan0 | 7.08 GiB | 954.70 | 31.04 |

## Vulkan MMVQ Shape Probe

The FP6 Vulkan `mul_mat_vecq` path was tested with larger `K_PER_ITER` values:

| FP6 Vulkan shape | pp512 tok/s | tg128 tok/s | Decision |
|---|---:|---:|---|
| 8-wide default | 1055.21 | 22.78 | keep |
| 16-wide | 1062.88 | 22.80 | reject; decode tie |
| 32-wide | 1055.77 | 22.79 | reject; decode tie |

The wider shapes pass Vulkan `MUL_MAT` correctness, but they do not improve decode.
The source is restored to the 8-wide default.

## Vulkan FP6 Load/Index Probe

The FP6 Vulkan MMVQ path was also tested with a 16-bit alias over the expanded
34-byte Vulkan block:

- `block_rocmfpx_fp6_packed16` reads two halfwords instead of four scalar bytes
  for each `dotPacked4x8EXT` operand.
- The FP6 A-side MMVQ base index is computed as `iqs * K_PER_ITER + 4 * lane`,
  so the 8-wide path addresses the same 8 quant values as the matching Q8_1 B-side
  cache lane.

Result on Qwythos-9B ROCmFP6 Agent, Vulkan0, `-r 5`:

| FP6 Vulkan path | pp512 tok/s | tg128 tok/s | Decision |
|---|---:|---:|---|
| 8-wide default baseline | 1055.21 | 22.78 | baseline |
| packed16 load + corrected base index | 1061.94 | 22.77 | keep only as cleanup; no decode win |
| final reverted 8-wide + packed16 cleanup | 1068.57 | 22.78 | keep; prefill only |

The path passes Vulkan `MUL_MAT` correctness for `q6_0_rocmfpx`. FP4 and FP4_FAST
Vulkan `MUL_MAT` guardrails also pass.

## Rejected Follow-Up Probes

Additional Vulkan decode probes after the packed16/index cleanup:

| Probe | pp512 tok/s | tg128 tok/s | Decision |
|---|---:|---:|---|
| FP6 q8_1 one-row dispatch | 1066.64 | 22.79 | reject; decode tie |
| FP6 16-wide MMVQ with corrected base index | 1063.49 | 22.80 | reject; decode tie |
| FP6 direct UE4M3 scale math, no LUT init | 755.42 | 19.68 | reject; clear regression |
| FP6 fixed 8-wide inner loop, no per-lane branch | 1059.28 | 22.78 | cleanup only; decode tie |
| FP6 full-block 32-value MMVQ, two-scale dot sum | 1061.49 | 22.81 | reject; decode tie |

The direct-scale path was worse because the arithmetic cost outweighed the removed
shared LUT initialization. Keep the UE4M3 LUT path for FP6.

The fixed 8-wide inner loop follows the Vulkan/RDNA direction of minimizing
branch/register overhead around packed int8 dot operations, but end-to-end decode
does not move. That suggests the remaining gap to Q6_K is not in the tiny FP6
operand packing sequence alone.

The full-block 32-value MMVQ path follows the same integer dot-product direction,
but it only changes dispatch granularity inside the generic matvec pipeline. It
also ties decode, so the source is restored to the 8-wide FP6 path.

## Decision

The expanded-device FP6 experiment is not a default promotion candidate on ROCm.

It keeps packed GGUF compatibility and can help Vulkan prefill, but it does not beat
Q6_K decode. On ROCm0 it loses more than the 3 tok/s tolerance, so it should stay
experimental until the decode path is improved.

The HIP/ROCm build keeps packed FP6 device storage by default, with the expanded
layout still gated by `GGML_ROCMFP6_EXPANDED_DEVICE=1`. The Vulkan path in this
branch uses expanded signed-byte device storage for FP6 to avoid 6-bit unpacking
inside shaders; the packed GGUF file format remains compatible.

## Best Next Decode Targets

- ROCm0: keep the packed FP6 device path as the default and tune MMVQ decode directly.
  Expanding FP6 blocks from 26 bytes to 34 bytes increases memory traffic too much for
  single-token decode.
- Vulkan0: target `mul_mat_vecq` for FP6. The current shader does signed-byte loads and
  integer dot product, but decode remains memory/kernel-shape limited.
- Larger Vulkan FP6 `K_PER_ITER` shapes were tested and rejected. The next Vulkan work
  should be a different kernel strategy, not another simple width change.
- A useful next kernel experiment is a separate FP6 matvec shader that maps one
  subgroup to a full 32-value FP6 block and reduces across rows directly, instead of
  sharing the generic q8_1 `mul_mat_vecq` row/column pipeline.
- Keep FP4/FAST unchanged; FP6 tuning should stay isolated to `Q6_0_ROCMFPX`.
