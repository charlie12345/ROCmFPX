# ROCmFP4 UE4M4 gfx1151 Results (standalone experiment)

This document records a third-party standalone kernel experiment on native `gfx1151`. It compares a 4.25-BPW Codebook10 format with a conventional unsigned UE4M4 scale against the current ROCmFP4 fast and dual-scale mechanisms, Q4_0, and Q4_K.

The result is evidence for a possible format direction. It is **not** a proposed GGML type implementation and does not establish model, server, or deployment performance.

## Identity boundary

The tested candidate, `rocmfp4-ue4m4-fast`, stores 32 weights as 16 Codebook10 bytes plus one conventional unsigned UE4M4 scale byte.

## Summary

Across eight exact DeepSeek routed-expert matrix shapes, `rocmfp4-ue4m4-fast` ranked first by median-of-round-medians in six shapes and second in two.

The negative and uncertain results are material:

- the incumbent fast kernel was 1.35% faster at MMQ down `N=9` by aggregate median; the paired-round result was −1.27% with a deterministic 95% bootstrap interval of −1.77% to −0.91%;
- incumbent dual was 0.88% faster at MMQ gate/up `N=2048` by aggregate median, while the paired interval crossed zero;
- UE4M4 was only 2.08% faster than Q4_0 at MMQ gate/up `N=2048`, and that paired interval also crossed zero;
- MMVQ gate/up had large timing spikes, so these data do not support a tail-latency superiority claim.

UE4M4 beat Q4_K by aggregate median at all eight shapes. This is a kernel result for fixed work, not a full-model throughput result.

## Format identities

| Candidate | Geometry | BPW | Description |
|---|---|---:|---|
| `rocmfp4-ue4m4-fast` | flat32 | 4.25 | 16 Codebook10 bytes plus one unsigned UE4M4 scale byte |
| `rocmfp4-incumbent-fast` | flat32 | 4.25 | current single-scale ROCmFP4 fast control |
| `rocmfp4-incumbent-dual` | flat32 | 4.50 | current dual-scale ROCmFP4 control |
| `q4_0-raw` | flat32 | 4.50 | stock Q4_0 control |
| `q4_K-raw` | K256 | 4.50 | stock Q4_K ecosystem control |

Changing the scale domain changes reconstruction semantics and legal byte states. A future integrated implementation must use a versioned type identity rather than silently reusing the current fast type.

## Hardware and method

- GPU: Radeon 8060S, 20 compute units, wave32;
- target: native `gfx1151`; `HSA_OVERRIDE_GFX_VERSION` unset;
- compiler: HIP 7.14.60850 / AMD clang 23.0.0git;
- candidate build flags: `--offload-arch=gfx1151 -O2`; no custom tuning flags;
- `HSA_ENABLE_SDMA=0`;
- MMVQ shapes: gate/up `K=4096, N=1`; down `K=2048, N=1`;
- MMQ shapes: the same gate/up and down matrices at `N=9`, `512`, and `2048`;
- five exact frozen binaries;
- three warmup rounds and 12 measured rounds per board;
- five timing samples per candidate/shape/round, or 60 measured samples per candidate/shape;
- first 10 measured rounds formed five reverse pairs; two seeded Latin shifts completed position balancing for the odd five-candidate roster;
- each table latency is the median of the 12 within-round medians.

There were 150 independently receipted cells: 30 warmup and 120 measured. Independent validation passed all 150 receipts.

## Repeated performance results

Positive percentages mean UE4M4 completed the same fixed kernel work faster than the named control. They are calculated as `control_ms / ue4m4_ms - 1` from the median-of-round-medians. The p95 column is across the 12 UE4M4 round medians.

| Board / shape | UE4M4 median ms | UE4M4 p95 ms | vs fast | vs dual | vs Q4_0 | vs Q4_K | rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMVQ gate/up `N=1` | 0.038320 | 0.162881 | +13.05% | +26.57% | +21.56% | +19.78% | 1/5 |
| MMVQ down `N=1` | 0.023100 | 0.024080 | +21.13% | +26.23% | +17.23% | +29.44% | 1/5 |
| MMQ gate/up `N=9` | 0.083801 | 0.118960 | +0.64% | +32.39% | +51.19% | +83.15% | 1/5 |
| MMQ gate/up `N=512` | 20.363502 | 21.561838 | +6.52% | +23.41% | +26.71% | +39.45% | 1/5 |
| MMQ gate/up `N=2048` | 95.269955 | 98.187233 | +9.73% | **−0.88%** | +2.08% | +34.95% | 2/5 |
| MMQ down `N=9` | 0.109600 | 0.111721 | **−1.35%** | +64.91% | +73.72% | +97.28% | 2/5 |
| MMQ down `N=512` | 21.093603 | 22.610012 | +8.05% | +2.22% | +20.05% | +46.78% | 1/5 |
| MMQ down `N=2048` | 67.171089 | 78.673355 | +64.98% | +16.92% | +33.86% | +32.73% | 1/5 |

The paired-round analysis in the raw JSON uses the same round number for candidate and control, computes `(control_ms / ue4m4_ms - 1) × 100`, and reports the median of 12 paired values. Its deterministic 20,000-resample bootstrap is useful for identifying robust direction, but 12 rounds do not justify architecture-wide claims.

Selected paired results:

| Shape / control | Paired median | Deterministic 95% interval |
|---|---:|---:|
| MMVQ down vs fast | +20.29% | +18.53% to +24.60% |
| MMVQ down vs Q4_0 | +18.24% | +16.00% to +21.89% |
| MMVQ down vs Q4_K | +31.11% | +27.90% to +32.71% |
| MMQ down `N=9` vs fast | **−1.27%** | **−1.77% to −0.91%** |
| MMQ gate/up `N=2048` vs dual | −2.29% | −6.50% to +5.57% |
| MMQ gate/up `N=2048` vs Q4_0 | +1.69% | −5.39% to +7.02% |
| MMQ down `N=2048` vs fast | +61.13% | +53.48% to +69.20% |

## Offline quality context

These values come from a corrected CPU-only sampled oracle, not model perplexity or KL divergence. Flat32 candidates share one sampled population. Q4_K uses K256 geometry and is shown only as an ecosystem control; it is not a paired flat32 quality comparison.

| Candidate | Geometry | BPW | held-out weighted MSE | sampled-output relative MSE |
|---|---|---:|---:|---:|
| `rocmfp4-ue4m4-fast` | flat32 | 4.25 | 1.72626231e-07 | 1.68693537e-05 |
| `rocmfp4-incumbent-fast` | flat32 | 4.25 | 1.93869918e-07 | 7.73350830e-04 |
| `rocmfp4-incumbent-dual` | flat32 | 4.50 | 1.23233464e-07 | 9.49578178e-05 |
| `q4_0-raw` | flat32 | 4.50 | 1.86185552e-06 | 1.53136801e-03 |
| `q4_K-raw` | K256 | 4.50 | 1.71005460e-06 | 1.56274494e-04 |

At equal 4.25 BPW, UE4M4 had 10.96% lower held-out weighted MSE than incumbent fast. Against incumbent dual it used 5.56% fewer bits and had lower sampled-output error, but its weighted MSE was 40.08% worse. That is a Pareto trade, not domination.

No perplexity, KL-divergence, converted-model, or serving result exists for this candidate.

## Correctness and generated ISA

The standalone candidate passed 64 source-matched scalar/device reconstruction cases with zero fixture mismatches. Dense MMVQ, dense MMQ, and a routed-semantics harness reported zero generic fallbacks. The routed harness covered duplicate, empty, inactive, and malformed route cases.

The exact executed binary was SHA-256 `75d567bfe3dd5562de385c3084cd041060f1c05009b289543d3136cedf83aa5b`. Symbol-correlated disassembly contained eight generated `v_dot4_i32_iu8` instructions in each dense kernel:

| Kernel | instructions | VGPR | SGPR | spills | symbol size |
|---|---:|---:|---:|---:|---:|
| MMVQ | 253 | 48 | 22 | 0 | 1,392 bytes |
| MMQ | 251 | 48 | 24 | 0 | 1,504 bytes |

This is compiler-generated DP4A after fused Codebook10 reconstruction. It is not a native FP4 operand claim.

## Raw data and bindings

All 2,400 measured timing samples, round medians, rankings, exact candidate hashes, schedules, quality context, and paired intervals are in [`rocmfp4-ue4m4-gfx1151-fiveway.json`](rocmfp4-ue4m4-gfx1151-fiveway.json).

The exact reviewed source and fixture are published in commit [`55f5a75cc4a32683aad2a893e993b49085791d6f`](https://github.com/hstolte11-collab/ROCmFPX/tree/55f5a75cc4a32683aad2a893e993b49085791d6f/experiments/gfx1151/ue4m4-fast).

Public JSON SHA-256: `723b243ef73f3e322faf7a4f8b279d4cf8efa0e8ad233f99cbbee74ce0d2fd1d`

The JSON also binds:

- original result: `fa47524ca96681942b0460f85fa7336bcfef5e02b9d85e9da50add2681a6095d`;
- independent validation: `47db0946ace26108b785c3bd4bb410b9f05e2afa4cf412e105a326af1d831251`;
- immutable five-way lock: `e4ea15b95d52ccc0db8942176bcd2601e62f9802cb540404f76af6522562b55a`;
- candidate source: `b88c59cce0a2950e99bd9849dc3a3bd0e1de81dfc6d31016569bce45bd79ec53`;
- source-matched fixture: `66824df16ca1c07950cdfe42153b941ffe8a4fee0d0ec6eb980fb369220fedb9`.

## Claim boundary and next gates

This evidence supports discussion of an unsigned UE4M4 scale as a possible versioned ROCmFP4 direction on `gfx1151`. It does not support merging a new type yet.

A code contribution would still require, at minimum:

1. an agreed versioned wire/type identity;
2. CPU quantize/dequantize and vector-dot support;
3. GGUF mappings and bounded converted artifacts;
4. integrated `GET_ROWS`, MMVQ, MMQ, and routed operation tests;
5. pure-CPU comparisons;
6. perplexity and KL divergence against native precision and similar-size controls;
7. full-model and serving evidence before any deployment claim.
