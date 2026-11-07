# ROCmFP4 Decode Speed Experiments

Controlled decode-speed work on top of the stable ROCmFP4 tree. The default
build remains unchanged; every tuning path below is opt-in.

## Branch

This work started on the ROCmFPX branch `experiment-ROCmFP4-decode-speed`,
created from `4795079b0`. The profiles and knobs below are on `main` now; that
branch is not among this repository's branches.

## Rules

- Stability wins over decode speed.
- Do not change the ROCmFP4 GGUF layout or quantized value semantics for a speed
  experiment.
- The old rule not to revert the `SPLIT_MODE_TENSOR` graph-reuse safety guard
  no longer applies: that guard left `src/llama-context.cpp` in the 2026-09-15
  ROCmFPX upstream merge (`40c5e4965`). A graph is now reused only right after
  one of the same size (`[TAG_GRAPH_ARENAS]`).
- Promote only changes that pass focused ROCmFP4 kernel guards and the real
  slowed-model decode guard.
- Keep rejected experiments documented so they are not repeated.

## Build Profiles

Use a separate build directory for every profile:

```bash
ROCMFP4_DECODE_TUNE=strix-moe-rpb1 \
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/build-strix-rocmfp4-mtp.sh
```

The same `ROCMFP4_DECODE_TUNE` profiles are also accepted by
`scripts/build-rocmfp4.sh`, so the RDNA2, RDNA3, and RDNA4 wrappers inherit
them.

Available profiles:

| Profile | Effect | Intended use |
|---|---|---|
| `stable` | Current default flags | Baseline |
| `strix-moe-rpb1` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=1` | MoE decode launch-shape check |
| `strix-moe-rpb2` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=2` | Explicit current default |
| `strix-moe-rpb3` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=3` | MoE decode launch-shape check |
| `strix-moe-rpb4` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=4` | MoE decode launch-shape check |
| `strix-nwarps1` | `-DGGML_ROCMFP4_RDNA35_NWARPS=1` | RDNA3.5 MMVQ launch check |
| `strix-nwarps2` | `-DGGML_ROCMFP4_RDNA35_NWARPS=2` | Explicit current default |
| `strix-nwarps4` | `-DGGML_ROCMFP4_RDNA35_NWARPS=4` | RDNA3.5 MMVQ launch check |
| `strix-mmid3` | `-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3` | MMVQ/MMQ routing threshold check |
| `strix-mmid4` | `-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4` | MMVQ/MMQ routing threshold check |

The full list lives in `scripts/rocmfp4-decode-tune-flags.sh`. It also has
`rocmfpx-strix-*` profiles for the ROCmFPX types (`GGML_ROCMFPX_*` and
`GGML_ROCMFP6_Q8_1_MMVQ_VDR`), and `ROCMFPX_DECODE_TUNE` takes precedence over
`ROCMFP4_DECODE_TUNE`. The `RDNA35_*` knobs only change RDNA3.5 (gfx1151)
launches; `MOE_MMVQ_ROWS_PER_BLOCK` applies on every HIP GPU.

You can still pass direct HIP flags when a profile is too narrow:

```bash
CMAKE_HIP_FLAGS="-DGGML_ROCMFP4_FATTN_V_NTHREADS=4" \
BUILD_DIR=build-strix-rocmfp4-fa-v4 \
scripts/build-strix-rocmfp4-mtp.sh
```

## Graph Build Timing

If a model slowed down, first measure graph rebuild overhead before changing
kernels. `LLAMA_GRAPH_BUILD_TIMING` and its perf footer were removed in the
2026-09-15 upstream merge; use `LLAMA_DECODE_PROFILE=N` instead:

```bash
LLAMA_DECODE_PROFILE=32 \
BUILD_DIR=build-strix-rocmfp4 \
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
```

Every N ubatches it logs a `decode profile [main]` (or `[draft]`) line with the
number of graphs built and the host milliseconds per ubatch for memory, graph,
inputs and submit (`N=1` means every 256). `LLAMA_GRAPH_REUSE_DISABLE=1` turns
graph reuse off for an A/B.

If graph time is a small fraction of total eval time, focus on kernels and
speculative profile settings. If it is large, work on graph reuse first.

## Code Tweak Map

| Area | File | Safe experiment |
|---|---|---|
| ROCmFP4 MoE MMVQ launch shape | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK` |
| RDNA3.5 ROCmFP4 MMVQ warps | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_RDNA35_NWARPS` and `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS` |
| MMVQ vs MMQ routing | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH` |
| ROCmFP4 FlashAttention shape | `ggml/src/ggml-cuda/fattn-vec.cuh` | `GGML_ROCMFP4_FATTN_*` compile flags |
| Build profiles | `scripts/rocmfp4-decode-tune-flags.sh` | Add opt-in profiles only |
| Graph rebuild timing | `src/llama-context.cpp` | `LLAMA_DECODE_PROFILE=N` (`[TAG_DECODE_PROFILE]`) |
| Runtime queue behavior | HIP runtime (environment only) | `GPU_MAX_HW_QUEUES` A/B; `common/common.cpp` no longer sets a default since the 2026-09-15 merge |

## Validation

At minimum, run the relevant focused model guard plus the ROCmFP4 runtime guards:

```bash
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh

BIN=build-strix-rocmfp4-rpb1/bin/test-backend-ops \
scripts/check-rocmfp4-rocm-runtime-regression.sh
```

The backend-op guards take their binary from `BIN` and ignore `BUILD_DIR`;
`check-rocmfp4-all-regression.sh` derives both binaries from `BUILD_DIR`
itself. The Qwen MTP guards print `SKIP` and exit 0 when their model file is
missing, so set `MODEL=...` (or `SKIP_MISSING_MODEL=0`) for a real result.

For a candidate that looks faster, also run:

```bash
INCLUDE_QWEN35_A3B_GUARD=1 \
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/check-rocmfp4-all-regression.sh
```

## Larger Quantization Research

Update: these formats now exist as the experimental ROCmFPX types
`Q3_0_ROCMFPX`, `Q6_0_ROCMFPX` and `Q8_0_ROCMFPX`, plus `Q2_0_ROCMFPX`,
`Q5_0_ROCMFPX` and `Q7_0_ROCMFPX`; see
[ROCmFPX-EXPERIMENT.md](ROCmFPX-EXPERIMENT.md). The plan below is kept as
written.

ROCmFP8, ROCmFP6, and ROCmFP3 should be treated as new GGUF formats, not as
kernel-only speed patches. A real format needs all of the following:

- A precise block layout and scale policy.
- CPU quantize/dequantize reference code.
- GGUF type IDs and quantization presets.
- ROCm/HIP copy, dequant, MMVQ, MMQ, and FlashAttention support.
- Vulkan decode and copy support if the format is user-visible.
- Focused numerical tests and end-to-end model regression guards.

Recommended order:

1. `ROCmFP8`: easiest to prototype because values are byte-aligned. It can be a
   quality-oriented intermediate format or a faster fallback when FP4 quality is
   too tight.
2. `ROCmFP6`: more complex because packing is awkward, but it may be useful as a
   quality/speed middle ground if memory bandwidth still improves.
3. `ROCmFP3`: highest risk. It needs stronger tensor-aware protection because
   the value grid is very small and quality regressions are likely.

Do not call a format `ROCmFP8`, `ROCmFP6`, or `ROCmFP3` in release notes unless
the README clearly states that it is this repository's experimental GGUF format,
not an AMD hardware-native data type.
