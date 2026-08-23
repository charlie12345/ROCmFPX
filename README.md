# ROCmFPX for llama.cpp

ROCmFPX adds experimental AMD-focused 2-, 3-, 4-, 6-, and 8-bit GGUF model-weight
formats to `llama.cpp`, with CPU reference paths and accelerated HIP/ROCm and
Vulkan kernels.

> **Status:** ROCmFPX is an experimental feature family on the canonical `main`
> branch. APIs, tuning choices, and performance can change. Results depend on
> hardware, drivers, model, prompt, and quantization recipe; use BF16/F16 sources
> for quality comparisons.

## Disclosure

**#AD #AMDAI**

A huge thank you to AMD for providing the hardware that powers the ongoing
development of ROCmFPX.

The development system provided by AMD includes:

- AMD Ryzen™ Threadripper™ 9980X
- AMD Radeon™ AI PRO R9700 GPU
- 128 GB ECC DDR5 Memory
- 2 TB NVMe SSD

This hardware enables me to continue developing, optimizing, benchmarking, and
testing ROCmFPX for the community. AMD provided this hardware as part of a
creator partnership. All development, benchmarks, code, testing, documentation,
and opinions shared in this repository are my own.

Thank you, AMD, for supporting the continued advancement of open AI development
on Radeon hardware.

**#AD #AMDAI**

## Developer Update — ROCmFP2 Lands On Main (July 2026)

ROCmFP2 is now available on the canonical `main` branch as
`Q2_0_ROCMFPX`. It uses a 2.50-bpw block layout with an S40
`{-4, -1, +1, +4}` codebook and dual UE4M3 scales. The smaller blocks reduce
model storage and memory traffic compared with ROCmFP4. The 2.50-bpw figure
describes native ROCmFP2 weight blocks; complete GGUF BPW can be higher because
files also contain metadata and tensors stored in other types. The latest
Vulkan dense and routed/MoE Q8_1 decode kernels landed through
[PR #42](https://github.com/charlie12345/ROCmFPX/pull/42).

[Download the current `main` source (ZIP)](https://github.com/charlie12345/ROCmFPX/archive/refs/heads/main.zip)
or use the clone command in [Quick Start](#quick-start-strix-halo--gfx1151).
Older tags and previously built binaries do not receive these updates
automatically.

| Current status | Formats | Notes |
|---|---|---|
| **Optimized and validated** | ROCmFP2, ROCmFP4 | Performance-tuned and benchmarked on the tested Strix Halo Vulkan and HIP/ROCm paths |
| **Development preview** | ROCmFP3, ROCmFP6, ROCmFP8 | Included in the tree, but kernel optimization, routing, and model coverage are still being improved |

Fresh matched Qwen3.6-35B-A3B tests used the same prompt and 256-token,
non-speculative decode workload from an internal NVMe:

| Backend | ROCmFP2 | ROCmFP4 STRIX_LEAN | ROCmFP2 advantage |
|---|---:|---:|---:|
| Vulkan0 | 90.30 tok/s | 76.20 tok/s | 18.50% |
| ROCm0 (ROCm 7.14) | 75.90 tok/s | 67.50 tok/s | 12.44% |

ROCmFP2 also passed exhaustive packed-codebook validation and dense/routed
Vulkan kernel gates. These measurements demonstrate the benefit on the tested
system, not a universal speed or quality guarantee. ROCmFP2 is more lossy than
ROCmFP4, so compare important workloads against the BF16/F16 source.

## Why ROCmFPX?

- **AMD-first weight formats:** ROCmFP2, ROCmFP3, ROCmFP4, ROCmFP6, and
  ROCmFP8 are real GGUF model-weight quants, not just K/V-cache compression.
- **Native accelerated paths:** HIP/ROCm and Vulkan kernels are backed by CPU
  reference implementations for correctness testing.
- **Speed and size choices:** ROCmFP2 is the smallest optimized family, while
  ROCmFP4 is the speed-first 4-bit family; existing Qwen comparisons put
  ROCmFP4 files about 12% below the matched Q4_K_M size.
- **Agent-aware presets:** coherent/agent recipes protect tensors that matter for
  code, JSON, tool calling, and structured output.
- **Built-in MTP acceleration:** models with an MTP/NextN head—including
  M-RoPE Qwen models—can use target-verified self-speculative decoding without
  loading a separate draft model.
- **Broad validation:** the promoted source was exercised through local
  CPU/Vulkan/ROCm tests and cross-platform CI covering Windows, macOS/Metal,
  WebUI provisioning, and Apple packaging.

Start with [Quick Start](#quick-start-strix-halo--gfx1151), choose a format in
[Which Format Should I Pick?](#which-format-should-i-pick), or jump directly to
[MTP Speculative Decoding](#faster-decode-mtp-speculative-decoding).

## Verified MTP Results — Strix Halo, 2026-07-12

These are local command-line decode results from the promoted `main` source on
Strix Halo (`gfx1151`). Throughput is the final `Generation:` rate from
`llama-cli`; runs used full GPU offload, FlashAttention, `-c 4096`, greedy
sampling (`--temp 0`), `-b 512 -ub 512`, the same prompt within each row, and
one model at a time.

| Model and backend | Tokens | MTP profile | No MTP | MTP result | Speedup |
|---|---:|---|---:|---:|---:|
| Qwable-5-27B-Coder ROCmFP4 COHERENT_AGENT, Vulkan0 | 64 | `n6 / p0.60` | 14.0 t/s | 33.2 t/s | 2.37x |
| Qwen3.6-35B-A3B ROCmFP4 STRIX_LEAN-FRESH, Vulkan0 | 256 | `n4 / p0.55` | 76.5 t/s | 116.1 t/s median, 118.3 peak | 1.52x median |
| Qwen3.6-35B-A3B ROCmFP4 STRIX_LEAN-FRESH, ROCm0 | 256 | `n4 / p0.55` | — | 106.2 t/s median | — |

Qwable is a matched single 64-token pair. Qwen no-MTP is one 256-token run;
the MTP values are the median and peak of three matched 256-token runs.

The promoted source and the pre-promotion experimental build were effectively
tied on Qwen3.6: median differences were `-0.7%` on Vulkan and `-0.3%` on ROCm.
On a longer 512-token Vulkan run, the promoted source reached `110.7 t/s` versus
`107.2 t/s` for the experimental build. Qwable's 256-token branch comparison
was also tied: promoted/experimental measured `32.9/33.0 t/s` on Vulkan and
`32.3/32.3 t/s` on ROCm.

MTP gains are content-dependent: predictable code, JSON, and lists usually
accept more draft tokens than creative prose. Treat the profiles above as tested
starting points, not universal defaults.

## Experimental ROCmI4 IU4 Acceleration (`gfx1151`)

ROCmFPX includes an opt-in HIP/ROCm path for `Q4_0_ROCMI4` models on AMD
Strix Halo. It keeps ROCmI4 weights packed as signed four-bit values and uses
the native `v_wmma_i32_16x16x16_iu4` instruction for batched matrix
multiplication. The feature is intended for prompt processing and MTP target
verification; ordinary one-token-at-a-time generation continues to use MMVQ.

The path is deliberately conservative:

- `GGML_HIP_ROCMI4_W4A4` defaults to `OFF`.
- Device code is compiled only under the exact `__gfx1151__` target.
- Runtime dispatch also requires the exact `gfx1151` device identifier.
- Other GPUs and default builds use the existing exact int8 MMQ path.
- No backend-test tolerance is weakened when W4A4 is enabled.

### Architecture and terminology

These names describe different layers of the system:

| Term | What it describes | Representation and role |
|---|---|---|
| **ROCmI4** | GGUF model-weight format | Signed four-bit weight codes, packed two per byte with block scales. This is the model stored on disk. |
| **INT4** | Generic integer width | Four-bit integers: usually signed `-8..7` or unsigned `0..15`. INT4 does not imply a particular file format or GPU instruction. |
| **IU4** | AMD WMMA instruction spelling | The gfx1151 integer matrix instruction consumes packed four-bit operands and accumulates into int32. It is a compute path, not a GGUF quant type. |
| **INT8** | Eight-bit integer arithmetic | Wider integer range. The exact ROCmI4 MMQ fallback expands packed weight codes and quantizes activations for int8 computation. |
| **FP8** | Eight-bit floating point | Uses sign, exponent, and mantissa, commonly E4M3 or E5M2. It has different range, precision, scaling, and kernels from INT4/IU4. |
| **FP4** | Four-bit floating point | A floating-point encoding with exponent/mantissa behavior. It is not the same representation as ROCmI4 or signed INT4. |

`W4A4` means four-bit weights and four-bit activations during the accelerated
matrix multiply. The ROCmI4 weights themselves are unchanged; the additional
loss comes from quantizing float activations directly to a signed four-bit grid.

### Why MTP decode becomes faster

Non-speculative decode evaluates one new token at a time and is mostly
memory-bandwidth bound, so the W4A4 MMQ path does not materially change a plain
`tg128` result. MTP changes the workload:

1. The model's embedded NextN/MTP head proposes up to several future tokens.
2. The target model verifies those proposed tokens together in a batch.
3. Batched verification uses MMQ and therefore reaches the native IU4 path.
4. Accepted draft tokens become output tokens without separate serial target
   evaluations.

On the development Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) system, a
matched 10-task Qwen3.8-27B HumanEval pilot with strict MTP-16 measured:

| Build | Mean decode | Aggregate decode | Prompt processing |
|---|---:|---:|---:|
| Exact ROCmI4 int8 MMQ | 41.63 tok/s | 40.90 tok/s | 283.83 tok/s |
| ROCmI4 W4A4 IU4 MMQ | **49.40 tok/s** | **48.29 tok/s** | **318.83 tok/s** |

The W4A4 run improved mean end-to-end decode by 18.66%. A separate full
164-task run measured 44.39 tok/s mean and 45.23 tok/s median, with HumanEval
pass@1 94.5% and HumanEval+ pass@1 91.5%. A 25-chunk perplexity sample was
about 5.4% higher than the exact path, so W4A4 remains an explicit quality/
speed tradeoff rather than a default.

### Build the opt-in path

Use a separate build directory so the exact build remains available:

```bash
cmake -S . -B build-rocmI4-w4a4 \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_HIP_ROCMI4_W4A4=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build-rocmI4-w4a4 --target llama-cli llama-server llama-bench -j 16
```

ROCm installations that do not identify Strix Halo natively may also need:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

The server prints `ROCmI4 W4A4: enabled` when the accelerated path is active.
If the compiled binary runs on an unsupported architecture, it reports the
fallback and uses exact int8 MMQ.

### Fast strict-MTP server profile

For a Qwen model that contains its embedded MTP/NextN tensors:

```bash
build-rocmI4-w4a4/bin/llama-server \
  -m model-Q4_0_ROCMI4.gguf \
  --host 127.0.0.1 --port 8116 \
  -dev ROCm0 -ngl 999 -np 1 -c 262144 \
  -b 512 -ub 256 -t 16 -tb 32 -fa on \
  -ctk f16 -ctv f16 --jinja \
  --spec-type draft-mtp --spec-mtp-strict-qwen \
  --spec-draft-device ROCm0 --spec-draft-ngl all \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-n-max 16 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.60 --spec-draft-backend-sampling
```

MTP acceptance and speed depend on content. Start with `n16 / p0.60` only for
the qualified Qwen3.8 profile; reduce `--spec-draft-n-max` if another model or
long-running workload shows lower acceptance or stability. Strict MTP preserves
the selected target path's greedy decisions, but it does not remove the
approximation introduced by W4A4 activation quantization.

For implementation details, validation commands, and rollback instructions,
see [`ggml/rocmfpx/ROCMI4.md`](ggml/rocmfpx/ROCMI4.md).

## Quick Start (Strix Halo / `gfx1151`)

Four commands from clone to a running model. For other AMD GPUs, swap the build
script using the [Clone And Build](#clone-and-build) table.

```bash
# 1. Get the code (canonical main branch)
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout main

# 2. Build for Strix Halo
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh          # -> build-strix-rocmfp4/

# 3. Quantize a BF16/F16 GGUF to ROCmFP4 (4.25 bpw, speed-first layout)
build-strix-rocmfp4/bin/llama-quantize model-BF16.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST

# 4. Run it with the fastest backend measured on this Strix Halo system
build-strix-rocmfp4/bin/llama-cli \
  -m model-ROCMFP4_FAST.gguf -dev Vulkan0 -ngl 999 -fa on --jinja
```

That is the whole loop: **build → quantize → run.** The sections below explain
each format, how to convert an existing NVFP4 model, and how to squeeze more
decode speed with speculative decoding.

For the smallest optimized format, replace step 3 with:

```bash
build-strix-rocmfp4/bin/llama-quantize \
  model-BF16.gguf model-ROCMFP2.gguf Q2_0_ROCMFPX
```

The same `llama-cli` command can then load `model-ROCMFP2.gguf`.

For a model that contains an MTP/NextN head, add a tested starting profile:

```bash
build-strix-rocmfp4/bin/llama-cli \
  -m model-with-MTP.gguf -dev Vulkan0 -ngl 999 -fa on --jinja \
  --temp 0 --spec-type draft-mtp \
  --spec-draft-n-max 6 --spec-draft-p-min 0.6
```

For Qwen3.6-35B-A3B on the tested Strix Halo system, `n4 / p0.55` was faster:

```text
--spec-draft-n-max 4 --spec-draft-p-min 0.55
```

To use HIP/ROCm instead of Vulkan:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
build-strix-rocmfp4/bin/llama-cli \
  -m model-ROCMFP4_FAST.gguf -dev ROCm0 -ngl 999 -fa on --jinja
```

## Tested Support

| Target | Status |
|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | locally built and benchmarked with Vulkan and HIP/ROCm; Vulkan was fastest for tested decode workloads |
| RDNA2 (`gfx1030`), RDNA3 (`gfx1100`), RDNA4 (`gfx1200`) | dedicated build scripts are provided; results vary by GPU and ROCm version |
| CPU | reference and correctness paths; not the recommended performance backend |
| Vulkan | accelerated and the recommended decode starting point on tested Strix Halo hardware |
| HIP/ROCm | accelerated and validated on the tested Strix Halo system |

## Which Format Should I Pick?

| Goal | Use | Why |
|---|---|---|
| **Smallest optimized** | `Q2_0_ROCMFPX` | 2.50-bpw blocks with lower storage and memory traffic; validate quality for each model |
| **Speed-first 4-bit** | `Q4_0_ROCMFP4_FAST` | 4.25 bpw, single scale/block — the speed-oriented 4-bit default |
| **Balanced 4-bit** | `Q4_0_ROCMFP4` | 4.50 bpw, dual per-16 scale — a touch more precision |
| **Agents / tools / JSON / code** | `Q4_0_ROCMFP4_COHERENT` (or any `*_AGENT`) | protects the tensors that keep structured output correct |
| **Strix Halo tuned recipe** | `Q4_0_ROCMFP4_STRIX_LEAN` | attn-K/V quality recipe tuned on `gfx1151` |
| **Experimental gfx1151 IU4 + MTP** | `Q4_0_ROCMI4` with W4A4 enabled | native IU4 batched verification for higher speculative decode throughput; opt-in and lossy |
| **Preview higher-bit references** | `Q6_0_ROCMFPX` / `Q8_0_ROCMFPX` | 6.5 / 8.25 bpw; optimization and model coverage are still in progress |
| **Preview 3-bit** | `Q3_0_ROCMFPX` | 3.5 bpw; optimization and model coverage are still in progress |

Rule of thumb: choose **`Q2_0_ROCMFPX`** when capacity and memory traffic matter
most, or start with **`Q4_0_ROCMFP4_FAST`** for a less aggressive speed-oriented
quant. Use a **`*_COHERENT` / `*_AGENT`** preset if the model does tool-calling,
JSON, or coding. Always compare against your BF16/F16 source for real quality
checks.

## What Is ROCmFPX?

ROCmFPX is a family of GGUF model-weight quants:

| Family name | GGUF preset | Role | Optimization status |
|---|---|---|---|
| ROCmFP2 | `Q2_0_ROCMFPX` | smallest ROCmFPX weight format; 2.50-bpw block layout | optimized and validated on tested Strix Halo paths |
| ROCmFP3 | `Q3_0_ROCMFPX` | low-bit ROCmFPX weight format | development preview |
| ROCmFP4 | `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST` | promoted 4-bit ROCm family baseline | optimized and validated on tested Strix Halo paths |
| ROCmI4 | `Q4_0_ROCMI4` | signed integer four-bit weights; optional native gfx1151 IU4/W4A4 MMQ | exact path by default; W4A4 is experimental and opt-in |
| ROCmFP6 | `Q6_0_ROCMFPX` | middle quality/size ROCmFPX weight format | development preview |
| ROCmFP8 | `Q8_0_ROCMFPX` | high-quality ROCmFPX reference format | development preview |

Agent-specific versions are also available:

| Family name | Agent preset | Role |
|---|---|---|
| ROCmFP3 Agent | `Q3_0_ROCMFPX_AGENT` | low-bit ROCmFPX with protected agent tensors |
| ROCmFP6 Agent | `Q6_0_ROCMFPX_AGENT` | middle ROCmFPX with protected agent tensors |
| ROCmFP8 Agent | `Q8_0_ROCMFPX_AGENT` | high-quality ROCmFPX with protected agent tensors |
| ROCmFP4 Agent | `Q4_0_ROCMFP4_COHERENT` | ROCmFP4 coherent agent-oriented preset |

ROCmFPX is not a K/V-cache-only compression trick. It is a set of actual GGUF
model-weight tensor formats with CPU reference paths plus ROCm/HIP and Vulkan
kernel coverage.

## Community Builds

Models published by the community using ROCmFPX / ROCmFP4. Listed to help people
find a build already quantised for their hardware, and to show the range of
architectures the formats have been exercised on.

Sizes are the total of all GGUF files in each repository (some ship several quant
variants, a speculative drafter, or a vision projector, so this is not the download
size of a single model file). Compiled from Hugging Face repository metadata; no
build here has been benchmarked or endorsed by this project. Capped at three entries
per publisher to keep the list representative.

| Repository | Total GGUF size |
|---|---|
| [`christopher-kapic/AEON-Ultimate-ROCmFP4-Strix-Halo-GGUF`](https://huggingface.co/christopher-kapic/AEON-Ultimate-ROCmFP4-Strix-Halo-GGUF) | 72.3 GiB |
| [`jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp`](https://huggingface.co/jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp) | 47.0 GiB |
| [`Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3`](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | 186.8 GiB |
| [`Geometric-AI/DeepSeek-V4-Flash-0731-ROCmFP3-MIX`](https://huggingface.co/Geometric-AI/DeepSeek-V4-Flash-0731-ROCmFP3-MIX) | 186.8 GiB |
| [`cafonez/Escha-W2-35B-A3B-ROCmFP2`](https://huggingface.co/cafonez/Escha-W2-35B-A3B-ROCmFP2) | 12.2 GiB |
| [`raulvidis/KAT-Coder-V2.5-Dev-ROCmFP4-STRIX-MTP-GGUF`](https://huggingface.co/raulvidis/KAT-Coder-V2.5-Dev-ROCmFP4-STRIX-MTP-GGUF) | 18.2 GiB |
| [`jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF`](https://huggingface.co/jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF) | 60.9 GiB |
| [`kingjones777/Laguna-S-2.1-ROCmFP4-STRIX_LEAN-GGUF`](https://huggingface.co/kingjones777/Laguna-S-2.1-ROCmFP4-STRIX_LEAN-GGUF) | 58.3 GiB |
| [`kingjones777/Leanstral-1.5-119B-A6B-ROCmFP4-STRIX_LEAN-GGUF`](https://huggingface.co/kingjones777/Leanstral-1.5-119B-A6B-ROCmFP4-STRIX_LEAN-GGUF) | 59.0 GiB |
| [`raulvidis/Ling-3.0-flash-ROCmFP4-STRIX-MTP-GGUF`](https://huggingface.co/raulvidis/Ling-3.0-flash-ROCmFP4-STRIX-MTP-GGUF) | 64.9 GiB |
| [`christopher-kapic/MiMo-V2.5-ROCmFP4-GGUF`](https://huggingface.co/christopher-kapic/MiMo-V2.5-ROCmFP4-GGUF) | 433.4 GiB |
| [`gsrunion/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN-DFLASH-GGUF`](https://huggingface.co/gsrunion/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN-DFLASH-GGUF) | 18.0 GiB |
| [`gsrunion/Ornith-1.0-35B-ROCmFPX-GGUF`](https://huggingface.co/gsrunion/Ornith-1.0-35B-ROCmFPX-GGUF) | 98.6 GiB |
| [`julianmb/Ornith-1.0-35B-ROCmFPX-StrixHalo`](https://huggingface.co/julianmb/Ornith-1.0-35B-ROCmFPX-StrixHalo) | 48.4 GiB |
| [`vmlinux/Qwen3.5-122B-A10B-Heretic-ROCmFP4-iMatrix-GGUF`](https://huggingface.co/vmlinux/Qwen3.5-122B-A10B-Heretic-ROCmFP4-iMatrix-GGUF) | 62.8 GiB |
| [`vmlinux/Qwen3.5-122B-A10B-ROCmFP4-iMatrix-GGUF`](https://huggingface.co/vmlinux/Qwen3.5-122B-A10B-ROCmFP4-iMatrix-GGUF) | 62.8 GiB |
| [`lmcoleman/Qwen3.6-27B-Fable-Fusion-711-MTP-ROCmFPX-GGUF`](https://huggingface.co/lmcoleman/Qwen3.6-27B-Fable-Fusion-711-MTP-ROCmFPX-GGUF) | 69.2 GiB |
| [`plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF) | 17.4 GiB |
| [`philtheriver/Qwen3.6-27B-ROCmFPX`](https://huggingface.co/philtheriver/Qwen3.6-27B-ROCmFPX) | 85.5 GiB |
| [`plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF) | 20.2 GiB |
| [`gsrunion/Qwen3.6-35B-A3B-ROCmFP4-STRIX_LEAN-GGUF`](https://huggingface.co/gsrunion/Qwen3.6-35B-A3B-ROCmFP4-STRIX_LEAN-GGUF) | 18.6 GiB |
| [`raulvidis/Qwen3.6-35B-A3B-ROCmFP4_FAST-GGUF`](https://huggingface.co/raulvidis/Qwen3.6-35B-A3B-ROCmFP4_FAST-GGUF) | 17.7 GiB |
| [`plunderstruck/Qwen3.6-40B-Deckard-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-40B-Deckard-MTP-ROCmFP4-GGUF) | 22.4 GiB |
| [`philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX`](https://huggingface.co/philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX) | 156.5 GiB |
| [`jcbtc/qwopus3.6-27b-v2-chadrock-rocmfp4-mtp`](https://huggingface.co/jcbtc/qwopus3.6-27b-v2-chadrock-rocmfp4-mtp) | 13.8 GiB |
| [`philtheriver/Qwopus3.6-27B-v2-MTP-ROCmFPX`](https://huggingface.co/philtheriver/Qwopus3.6-27B-v2-MTP-ROCmFPX) | 156.5 GiB |
| [`lmcoleman/Tess-4-27B-ROCmFPX-GGUF`](https://huggingface.co/lmcoleman/Tess-4-27B-ROCmFPX-GGUF) | 160.9 GiB |
| [`lmcoleman/ThinkingCap-Qwen3.6-27B-ROCmFPX-GGUF`](https://huggingface.co/lmcoleman/ThinkingCap-Qwen3.6-27B-ROCmFPX-GGUF) | 36.2 GiB |

To add yours, open a PR editing this table. Please keep it to actual ROCmFPX /
ROCmFP4 builds and use the `rocmfp4` or `strix-halo` tag on the model so it is
discoverable on the Hub.

## Contributors And Credit

This work builds on `llama.cpp`; upstream authors and contributors retain credit
under the MIT license. See `AUTHORS`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.

ROCmFP4 and ROCmFPX experiment work in this repository is maintained by
`charlie12345` / `caf`.

Additional ROCmFPX contributors:

- `ciru-ai`: ROCmFP2 core format/runtime and frozen codebook; ROCmFP3 Vulkan
  matvec/dequant speed path.
- Tom Turney / `PlunderStruck` / Aydan S.: TurboQuant `turbo3`/`turbo4`
  K/V-cache quantization paths for ROCm/HIP and Vulkan.

## Why It Is Different From Regular Quants

Most regular GGUF quants target broad size/quality tradeoffs. ROCmFPX is
AMD-oriented and keeps the ROCmFP4 discipline:

- 32-weight blocks for CPU, HIP, and Vulkan kernel compatibility
- finite unsigned UE4M3 scale bytes
- explicit integer-code-times-decoded-scale dequant math
- reconstruction-MSE scale selection where low-bit coherency needs it
- tensor-aware routing for low-bit coherency instead of applying one blunt type
  everywhere
- optional agent presets for JSON, tool calling, coding, and chat coherency

The agent presets do not invent a separate dequant kernel. They use the same
ROCmFPX math but protect the tensors that tend to break structured output:
token/output embeddings, attention Q/K/V/O, selected FFN-down, and selected
FFN-gate tensors.

## Detailed and Historical Benchmarks

These are additional pre-promotion comparisons from a Strix Halo / `gfx1151`
system. Treat them as historical local data, not a universal benchmark. All
rows within each table used the same model pair, backend, batch shape, K/V
cache, FlashAttention setting, and one test at a time.

### Qwen3.6 27B, Vanilla No-MTP

Model pair:

- Baseline: `Qwen3.6-27B-Q4_K_M.gguf`, `16.55 GB`
- ROCmFPX: `Qwen3.6-27B-VANILLA-NO-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf`, `14.59 GB`
- Size delta: ROCmFP4 is `11.82%` smaller
- Test: `llama-bench`, `pp512 + tg128`, MTP/speculative decoding disabled

| Backend | Quant | Prompt fill tok/s | Decode tg128 tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 336.97 | 11.74 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 328.03 | 13.53 |
| Vulkan0 | Q4_K_M | 352.04 | 12.89 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 376.98 | 14.27 |

On this 27B vanilla run, ROCmFP4 was slightly behind Q4_K_M for ROCm prompt
fill, but faster for decode on both ROCm and Vulkan. Vulkan ROCmFP4 also led
prompt fill.

### Qwen3.6 35B A3B Weight-Quant Comparison

Model pair:

- Baseline: `Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf`, `21.71 GB`
- ROCmFPX: `Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf`, `19.05 GB`
- Size delta: ROCmFP4 is `12.28%` smaller
- Test: `llama-bench`, `pp512 + tg128`; this table measures the weight quants,
  not speculative MTP acceleration

| Backend | Quant | Prompt fill tok/s | Decode tg128 tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 1353.50 | 59.00 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 1301.21 | 66.42 |
| Vulkan0 | Q4_K_M | 1065.83 | 70.57 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 1200.81 | 76.71 |

The same 35B A3B pair was also run through a 20-prompt Hermes-style agent
smoke:

| Backend | Quant | Prompt tok/s | Generation tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 699.7 | 31.9 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 731.4 | 47.1 |
| Vulkan0 | Q4_K_M | 654.0 | 40.2 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 730.9 | 57.5 |

On this 35B A3B comparison, ROCmFP4 was smaller and faster on decode/generation
across ROCm and Vulkan. ROCm prompt fill was still slightly behind Q4_K_M in
`llama-bench`, while Vulkan prompt fill and Hermes-style prompts favored
ROCmFP4.

## Clone And Build

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
```

Most users should stay on `main`. The preserved
`experimental-rocmfpx-branch` exists for history and rollback comparisons.

Pick the build script for your machine:

| Hardware | Build command | Output folder |
|---|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | `env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh` | `build-strix-rocmfp4/` |
| RDNA2 / RX 6000 (`gfx1030` class) | `env JOBS=16 scripts/build-rdna2.sh` | `build-rdna2/` |
| RDNA3 / RX 7000 (`gfx1100` class) | `env JOBS=16 scripts/build-rdna3.sh` | `build-rdna3/` |
| RDNA4 / RX 9000 (`gfx1200` class) | `env JOBS=16 scripts/build-rdna4.sh` | `build-rdna4/` |
| RDNA4 / RX 9000 — self-contained (no system ROCm) | `env JOBS=16 scripts/build-rocmfp4-rocm714-local.sh` | `build-rdna4-rocm714/` |
| Vulkan-only / manual | use the Vulkan CMake path in `docs/BUILD-AMD-ARCHITECTURES.md` | custom |

For Strix Halo, the common runtime environment is:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
```

Key binaries after build:

```text
build-strix-rocmfp4/bin/llama-quantize
build-strix-rocmfp4/bin/llama-cli
build-strix-rocmfp4/bin/llama-server
build-strix-rocmfp4/bin/llama-bench
build-strix-rocmfp4/bin/test-backend-ops
```

For RDNA2/RDNA3/RDNA4 builds, use the same binary names under that build
folder, for example `build-rdna3/bin/llama-quantize`.

The `build-rocmfp4-rocm714-local.sh` script (RDNA4 / RX 9000) downloads the
ROCm 7.14.0a20260624 toolchain automatically and bundles the required
runtime libraries alongside the binaries. The resulting build is
self-contained — no system-wide ROCm install or `LD_LIBRARY_PATH` needed
at runtime.

## Quantize Straight ROCmFPX Models

Use BF16 or F16 GGUF sources. The wrapper keeps split GGUFs split by default.

ROCmFP2:

```bash
build-strix-rocmfp4/bin/llama-quantize \
  /path/to/model-BF16.gguf /path/to/model-Q2_0_ROCMFPX.gguf Q2_0_ROCMFPX
```

ROCmFP3:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP4:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4.gguf \
  FORMAT=rocmfp4 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP6:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX.gguf \
  FORMAT=rocmfp6 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP8:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX.gguf \
  FORMAT=rocmfp8 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

You can also call `llama-quantize` directly:

```bash
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q2.gguf Q2_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q3.gguf Q3_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q4.gguf Q4_0_ROCMFP4
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q6.gguf Q6_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q8.gguf Q8_0_ROCMFPX
```

For low-bit ROCmFPX quants, pass an imatrix when you have one:

```bash
IMATRIX=/path/to/imatrix.gguf \
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

The wrapper forwards `IMATRIX` to `llama-quantize --imatrix`. ROCmFP3,
ROCmFP6, and ROCmFP8 use imatrix-weighted scale search; ROCmFP4 has its own
imatrix path.

## Quantize Agent ROCmFPX Models

Use agent mode when the model will be used for Hermes/OpenClaw-style workflows,
tool calling, JSON output, coding, or chat agents.

ROCmFP3 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp3 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP6 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp6 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP8 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp8 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP4 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4_COHERENT_AGENT.gguf \
  FORMAT=rocmfp4 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

The wrapper maps `FORMAT` and `PROFILE` like this:

| FORMAT | PROFILE | Preset |
|---|---|---|
| `rocmfp3` | `straight` | `Q3_0_ROCMFPX` |
| `rocmfp3` | `agent` | `Q3_0_ROCMFPX_AGENT` |
| `rocmfp4` | `straight` | `Q4_0_ROCMFP4` |
| `rocmfp4` | `agent` | `Q4_0_ROCMFP4_COHERENT` |
| `rocmfp6` | `straight` | `Q6_0_ROCMFPX` |
| `rocmfp6` | `agent` | `Q6_0_ROCMFPX_AGENT` |
| `rocmfp8` | `straight` | `Q8_0_ROCMFPX` |
| `rocmfp8` | `agent` | `Q8_0_ROCMFPX_AGENT` |

## Convert An Existing NVFP4 Model To ROCmFP4

If you already have an **NVFP4** GGUF, you can re-map it onto the ROCmFP4 kernel
path without re-quantizing from BF16. This is the closest-matching conversion
ROCmFPX supports: NVFP4 and ROCmFP4 use the **same UE4M3 scale** and share **7 of
8 codebook levels** — only the top magnitude level differs (NVFP4 `12` vs ROCmFP4
`10`), so almost every weight maps over cleanly.

```bash
# Same 4.50 bpw as NVFP4 (closest quality match):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4.gguf Q4_0_ROCMFP4

# Smaller 4.25 bpw (speed-first layout, a little more loss):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST
```

- `--allow-requantize` is **required**: NVFP4 GGUFs usually keep `output.weight` at
  a higher-precision type (e.g. `q6_K`), so the file has mixed source types.
- Example measured on a 9B NVFP4 model (wikitext-2, `gfx1151`): the 4.50 bpw target
  landed within noise of the NVFP4 source perplexity; the 4.25 bpw `FAST` target was
  ~5% higher perplexity for a ~10% smaller file. Numbers are model-dependent — always
  A/B against the NVFP4 source on your own prompts.
- To make *every* tensor ROCmFP4 (a uniform "even" file), use the `Q4_0_ROCMFP4_EVEN`
  / `Q4_0_ROCMFP4_FAST_EVEN` presets, which imply `--pure`.

## Faster Decode: MTP Speculative Decoding

If your model ships with an **MTP / NextN** draft head (many recent models do),
you can turn on self-speculative decoding for a real decode speedup — no separate
draft model needed. This is the most effective way to push decode throughput past
what the weight format alone can do, because accepted draft tokens produce several
tokens per weight read.

MTP helps **both dense and MoE** models here. On `gfx1151`, `-dev Vulkan0`
was the fastest backend in the validated Qwen3.6 and Qwable comparisons.

```bash
# General starting profile for a model with an embedded MTP/NextN head
build-strix-rocmfp4/bin/llama-cli \
  -m model-with-MTP.gguf -dev Vulkan0 -ngl 999 -fa on --jinja \
  --temp 0 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.6
```

- **Tune per model and workload.** `n6 / p0.60` is a useful starting point, but
  the validated Qwen3.6-35B-A3B profile was faster at `n4 / p0.55`. Very low
  `p_min` can waste work on rejected drafts; an overly high value can miss useful
  draft tokens.
- The speedup is **content-dependent**: structured / predictable output (code, lists,
  JSON) accepts more drafts and gains most; free-form creative text gains less.
- It is **lossless**: at greedy (`--temp 0`) the output matches non-speculative
  decoding token-for-token (the target model verifies every drafted token).
- See [Verified MTP Results](#verified-mtp-results--strix-halo-2026-07-12) for
  current Qwable and Qwen3.6 measurements, profiles, and branch-parity context.

**M-RoPE models (`qwen35` / `qwen35moe`, and any IMROPE/MROPE arch):** MTP now
works on these. They use 4-D M-RoPE positions, and the batch position check
previously rejected the MTP draft/verify batch every step (`for M-RoPE, it is
required that the position satisfies: X < Y`), so MTP silently fell back to
plain decode. The MTP hook batch is a hybrid (token id **plus** an injected
hidden-state row) and is allowed to reuse positions like an embedding batch, so
the strict check is now gated on `batch.token && !batch.embd`
(`src/llama-batch.cpp`). If you are on an older build and see that `X < Y` error
spamming during MTP, this is the fix. NEOX-RoPE MTP (e.g. Gemma4 assistants) was
never affected.

## What The Agent Preset Protects

The agent profile is a tensor-routing choice. It keeps the ROCmFPX block
formats but spends more bits on tensors that affect structured behavior:

- token and output embeddings
- attention Q/K/V/O tensors
- selected FFN-down tensors
- selective FFN-gate tensors
- bulk FFN-up tensors stay on the family quant where possible

This is why agent quants are slightly larger than straight quants. The goal is
to preserve JSON shape, tool-call shape, coding behavior, and chat coherency
without forcing the whole model to a generic high-bit quant.

## Run A Quantized Model

Simple ROCm run:

```bash
build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  -b 512 \
  -ub 512 \
  --jinja
```

OpenAI-compatible server:

```bash
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  --host 127.0.0.1 \
  --port 8138 \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  -b 512 \
  -ub 512 \
  --jinja \
  --reasoning off
```

## K/V Cache Rule

ROCmFPX model quants and K/V cache types are separate runtime controls.

The current guard promotes `-ctk q3_0_rocmfpx` to `q6_0_rocmfpx` because fp3 K
cache was below the observed tool-call and agent coherency floor. `q3_0_rocmfpx`
can still be used for V cache.

TurboQuant K/V cache support is already built into this tree as the `turbo3`
and `turbo4` runtime cache types, including CPU reference tests plus ROCm/HIP
and Vulkan paths. TurboQuant is not a ROCmFPX model-weight quant; use it with
`-ctk` and `-ctv` at runtime.

The recommended safe TurboQuant+ style policy is asymmetric K/V:

```bash
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  -dev Vulkan0 \
  -ngl 999 \
  -fa on \
  -ctk q8_0 \
  -ctv turbo4 \
  --jinja
```

For the ROCmFPX MTP server wrapper, use the preset script:

```bash
MODEL=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
DEVICE=Vulkan0 \
scripts/run-rocmfpx-turboquant-asym-server.sh
```

This keeps K cache at `q8_0`, where attention quality and tool calls are more
sensitive, and uses `turbo4` for V cache, where compression is usually cheaper.
You can still run symmetric TurboQuant for sweeps with `-ctk turbo3 -ctv turbo3`
or `-ctk turbo4 -ctv turbo4`, but do not treat those as the default agentic
serving profile.

For symmetric TurboQuant experiments, first/last-layer K protection is available
as an opt-in compatibility knob:

```bash
LLAMA_KV_TURBO_BOUNDARY_LAYERS=2 \
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model.gguf \
  -ctk turbo4 \
  -ctv turbo4
```

With that flag, the first and last two model layers use `q8_0` for K cache
while the middle layers use the requested TurboQuant type. V boundary protection
is off by default; enable it only for experiments with
`LLAMA_KV_TURBO_BOUNDARY_V=1`.

Do not import the Python `turboquant_plus` research package into this C/C++ tree
as-is. The low-risk production findings are the asymmetric K/V policy and
documentation. QJL and turbo2 are intentionally not enabled here, and block-size
128 would require a GGML block-layout change and compatibility work.

## Test Agent Behavior

The agentic smoke harness checks chat, coding, JSON, tool-call JSON, coherency,
and streaming. It also refuses to start when ROCm reports an active KFD process,
so each run starts after VRAM/process cleanup.

```bash
MODEL=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
BACKEND=ROCm0 \
ALIAS=rocmfpx-agent \
OUT_DIR=/tmp/rocmfpx-agentic-smoke \
scripts/check-rocmfpx-agentic-smoke.sh
```

## Code Layout

- `ggml/rocmfpx/` - ROCmFP2/ROCmFP3/ROCmFP6/ROCmFP8 reference formats
- `ggml/rocmfp4/` - ROCmFP4 reference path this family inherits from
- `scripts/quantize-rocmfpx-agent.sh` - simple straight-vs-agent quant wrapper
- `scripts/check-rocmfpx-agentic-smoke.sh` - OpenAI-compatible agent smoke test
- `docs/ROCmFPX-HANDOFF.md` - detailed handoff for reviewers and other agents
- `docs/ROCmFPX-EXPERIMENT.md` - experiment history, routing notes, and gates
- `docs/BUILD-AMD-ARCHITECTURES.md` - RDNA2/RDNA3/RDNA4/Strix build details

## License

This repository is based on `llama.cpp` and keeps the upstream MIT license. See
`LICENSE` for details. Bundled third-party notices are listed in
`THIRD_PARTY_NOTICES.md`.
