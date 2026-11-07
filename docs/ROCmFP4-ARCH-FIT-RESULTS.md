# ROCmFP4 Arch-Fit Results (third-party)

Independent measurements of `Q4_0_ROCMFP4_STRIX_LEAN` across four model architectures on a Ryzen AI Max+ 395 (gfx1151 / Radeon 8060S), 128 GB unified memory, ROCm 7.2.4.

> Historical (measured 2026-07, ROCm 7.2.4, not re-measured since). Later changes that can move these numbers: prompt matmuls on ROCmFP4 weights can take a WMMA GEMM path on gfx11 GPUs, gfx1151 included (`[TAG_WMMA_GEMM]`, 2026-10-22), and RDNA-FAST builds use ROCm 10.1 since 2026-11-05.

Contributed to answer a question I had before starting and could not find answered anywhere: **which architectures actually get faster under ROCmFP4, and by how much?**

All four builds used this fork. Failures are included.

## Summary

| Model | Arch | Active params | Decode @ short | Decode @ long | Size | Baseline |
|---|---|---:|---:|---:|---:|---|
| Laguna-S-2.1 118B-A8B | `laguna` | ~8B | **+62.6%** | **+43.6%** | −18% | Q4_K_M |
| Step-3.7-Flash 198B MoE | `step35` | ~11B | **+18%** | **+20%** | +10% | UD-IQ4_XS |
| Leanstral-1.5 119B-A6B | `deepseek2` (MLA) | ~6.5B | +1.5% | **−8.2%** | −13% | Q4_K_M |
| KAT-Coder-V2.5-Dev 35B-A3B | `qwen35moe` (hybrid linear) | ~3B | +12% | **−37%** | −2.4 GiB | Q4_K_M |

Baselines are not uniform. Step-3.7-Flash was compared against `UD-IQ4_XS` (~3.9 BPW), which is why its size went *up* — `STRIX_LEAN` is ~4.26 BPW. Against `Q4_K_M`, ROCmFP4 came out 13–18% smaller in every case.

## Detail

**Laguna-S-2.1** (`laguna`, ~8B active) — the strongest result.

| Quant | prompt_n | prefill tok/s | decode tok/s | size |
|---|---:|---:|---:|---:|
| Q4_K_M | 8116 | 445.6 | 17.36 | 71 GiB |
| ROCmFP4 STRIX_LEAN | 8116 | 381.8 | **28.23** | 58.3 GiB |
| Q4_K_M | 32223 | 372.4 | 14.15 | 71 GiB |
| ROCmFP4 STRIX_LEAN | 32223 | 327.3 | **20.32** | 58.3 GiB |

Prefill cost −14% / −12%. Quality parity on reasoning, code and tool-call probes.

**Step-3.7-Flash** (`step35`, ~11B active) — 331 vs 296 prefill and 18.8 vs 15.9 decode at 8K; 283 vs 261 and 14.9 vs 12.4 at 32K. Notably this is the only build here where **prefill also improved**; every other one paid the usual 12–14% prefill penalty. Quantized from the official `Q8_0` GGUF with `--allow-requantize` (BF16 is 394 GB and would not fit alongside the output).

**Leanstral-1.5-119B-A6B** (`deepseek2`/MLA, 128 experts / 4 active) — 37.41 vs 36.85 decode at 7616 tokens, 23.70 vs 25.83 at 23081 tokens, ~59 vs ~68 GiB. The format loads and runs `deepseek2` correctly and output was coherent; it simply does not move decode here.

**KAT-Coder-V2.5-Dev** (`qwen35moe`, hybrid linear + full attention, 262K ctx) — +12% at ~9K, **−37% at ~37K** (25 vs 40 tok/s). Discarded rather than published. For a long-context coding model, `Q4_K_M` was the better quant.

## Observation

Across these four points, **active parameter count tracks the decode result** better than total size or expert count. ~8B and ~11B active gained substantially; ~6.5B active with MLA was flat; ~3B active with hybrid-linear attention regressed at long context.

Active-param count and attention type are confounded in this sample — MLA and linear attention both move work away from the FFN path where the FP4 kernels operate, which is a plausible mechanism for the same observation. Four data points is a hypothesis, not a proof, and I'd rather it were tested than believed.

An earlier hypothesis of mine — that a high *expert count* (128 in Leanstral) would predict a strong FP4 win — was wrong. Expert count did not predict anything; active parameters did.

## Secondary finding: measure at more than one context length

KAT-Coder measured at ~9K alone looks like a +12% win. At ~37K it is 37% slower. Any single-short-context benchmark would have shipped that as a success.

Two methodology notes that changed my own numbers materially:

- **Force equal generation length.** Decode tok/s over 4 tokens and over 256 tokens are different quantities. One early comparison of mine produced soft numbers because a baseline stopped at 4 tokens.
- **Defeat the prefix cache.** Put a fresh nonce at the *start* of each prompt and assert `cached_tokens == 0`. A contaminated run inflated prefill in one of my campaigns until it was caught.

Harness and full per-model write-ups: https://github.com/kingjones30/strix-halo-quant-lab

Published quants: https://huggingface.co/kingjones777
