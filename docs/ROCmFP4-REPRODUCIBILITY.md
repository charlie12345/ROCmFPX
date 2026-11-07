# ROCmFP4 Reproducibility Standard

This tree should only claim ROCmFP4 gains that are reproducible from a clean
GitHub checkout, a named build, named model files, exact flags, and a recorded
hardware target.

> Scope: this is the ROCmFPX standard for ROCmFP4 on Strix Halo (gfx1151)
> under Linux, written 2026-05. The scripts default to the gfx1151
> `build-strix-rocmfp4` build, `HSA_OVERRIDE_GFX_VERSION=11.5.1` and model
> paths on the original author's machine. RDNA-FAST's own numbers are measured
> on an RX 7900 XT (gfx1100) under Windows, built with ROCm 10.1.

## Promotion Rules

A ROCmFP4 optimization is promoted only when all of these are true:

- The changed code is specific to ROCmFP4, MTP scheduling, or a documented
  Strix Halo backend path.
- The exact benchmark command, context window, cache type, MTP settings,
  reasoning state, and model paths are recorded.
- The comparison uses the same binary for ROCmFP4 and the non-ROCmFP4 baseline
  whenever the model formats allow it.
- End-to-end decode speed improves or ties while the sustained decode guard
  does not regress.
- `scripts/check-rocmfp4-all-regression.sh` passes after the change.
- Rejected experiments are documented with the reason they were not promoted.

Microbenchmarks are useful for finding bottlenecks, but they are not enough to
claim a user-visible win. A microbench-only gain stays behind a compile-time or
runtime knob until an end-to-end decode guard also benefits.

## Current Proof Commands

Run these from the root of the checkout.

Build the reproducible Strix Halo ROCmFP4 + MTP binary:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
```

Run the full promoted regression gate:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.5.1 scripts/check-rocmfp4-all-regression.sh
```

Run the controlled Qwen3.6 27B MTP comparison and write a markdown report:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.5.1 scripts/reproduce-rocmfp4-qwen-mtp-comparison.sh
```

The regression and reproduction scripts resolve the checkout path from their
own location, so they can run from a different GitHub clone directory. Model
paths are intentionally explicit and can be overridden with `ROCMFP4_MODEL=...`,
`BASELINE_MODEL=...`, or `MODEL=...` depending on the script.

By default this uses the promoted sustained profile:

```bash
--spec-draft-n-max 4 --spec-draft-n-min 0 \
--spec-draft-p-min 0.0 --spec-draft-p-split 0.10 \
-c 262144 -b 512 -ub 512 -ctk q4_0 -ctv q4_0 -n 160
```

The comparison script records:

- date and host,
- git commit,
- binary version,
- model paths and sizes,
- backend,
- context window,
- MTP flags,
- reasoning/tools state,
- prompt tokens per second,
- decode tokens per second,
- full command flags.

The Qwen MTP guards in the regression gate print `SKIP` and exit 0 when their
model file is missing (`SKIP_MISSING_MODEL=1` is the default), so the full gate
passes without them. Set `MODEL=...` or `SKIP_MISSING_MODEL=0` when the gate
has to prove a decode result. They also fail when `llama-cli` is older than
`common/speculative.cpp`, `common/arg.cpp`, `src/llama-context.cpp` or
`src/llama-model.cpp` (`STALE_BINARY_CHECK=1`). The dense 27B guard floors are
`12.0` / `11.5 tok/s` (short / sustained) since 2026-06-24.

## Current Proven Gain

The strongest replicated end-to-end result so far is Qwen3.6 27B MTP at 262k
context on Framework AMD Strix Halo 395+ with 128 GB unified RAM:

| Model | Backend | Context | MTP | Reasoning | Tools | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | on | off | off | 27.6 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | ROCm0 | 262144 | on | off | off | 15.7 |
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | on | off | off | 24.9 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | Vulkan0 | 262144 | on | off | off | 18.4 |

Those numbers are hardware- and driver-sensitive. They should be presented as
Strix Halo measurements, not as a universal llama.cpp result.

Latest local reproduction report:
`bench-reports/rocmfp4-qwen-mtp-comparison-20260524-124911.md`.
That report is intentionally ignored by git because it is generated output; the
script above recreates the same artifact.

## What Makes This Build Distinct

ROCmFP4 is not a renamed Q4 path. The promoted path includes:

- Codebook10 4-bit weights with finite unsigned E4M3 half-scales.
- Dual-scale and FAST layouts, selected by tensor role instead of one global
  quant choice.
- ROCm/HIP vector-dot code that expands ROCmFP4 nibbles directly into DP4A
  operands.
- ROCmFP4-specific ROCm FlashAttention thread grouping.
- ROCmFP4-specific Vulkan scale lookup logic.
- ROCmFP4 source/dequant CPY kernels so fallback conversion paths do not leave
  the format-specific implementation.
- MTP host-loop cleanup for long-running target/draft decode (that host code
  was replaced by the ROCmFPX upstream MTP code in the 2026-09-15 merge,
  `40c5e4965`).

The next high-upside work must keep this same proof standard: fused ROCmFP4
matvec changes, long-context attention changes, and speculative scheduler
changes should ship only after the benchmark report and full guard agree.
