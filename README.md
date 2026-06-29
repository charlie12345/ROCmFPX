# ROCmFPX for llama.cpp

Experimental AMD-focused ROCmFP3, ROCmFP4, ROCmFP6, and ROCmFP8 quantization
formats for `llama.cpp`.

This repository is for people who want to download, compile, quantize, and test
the ROCmFPX family directly from:

```text
https://github.com/charlie12345/ROCmFPX/tree/experimental-rocmfpx-branch
```

The same source is intended to live on `main` so GitHub shows the ROCmFPX
instructions by default.

> Status: experimental branch. This branch is for community testing before
> changes are promoted to `main`. APIs, serving defaults, benchmark thresholds,
> and ROCmFPX tuning choices may change quickly. Results are hardware-, driver-,
> model-, and prompt-sensitive, so use BF16/F16 sources for real quality tests.

## What Is ROCmFPX?

ROCmFPX is a family of GGUF model-weight quants:

| Family name | GGUF preset | Role |
|---|---|---|
| ROCmFP3 | `Q3_0_ROCMFPX` | smallest experimental ROCmFPX weight format |
| ROCmFP4 | `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST` | promoted 4-bit ROCm family baseline |
| ROCmFP6 | `Q6_0_ROCMFPX` | middle quality/size ROCmFPX weight format |
| ROCmFP8 | `Q8_0_ROCMFPX` | high-quality ROCmFPX reference format |

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

## Contributors And Credit

This work builds on `llama.cpp`; upstream authors and contributors retain credit
under the MIT license. See `AUTHORS`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.

ROCmFP4 and ROCmFPX experiment work in this branch is maintained by
`charlie12345` / `caf`.

Additional ROCmFPX contributors:

- `ciru-ai`: ROCmFPX FP3 Vulkan matvec/dequant speed path.
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

## Preliminary Benchmarks

These are local experimental-branch results from a Strix Halo / `gfx1151`
system. Treat them as early comparison data, not a final benchmark site. All
rows used the same model pair, backend, batch shape, q4 K/V cache, Flash
Attention enabled, and one test at a time.

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

### Qwen3.6 35B A3B MTP

Model pair:

- Baseline: `Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf`, `21.71 GB`
- ROCmFPX: `Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf`, `19.05 GB`
- Size delta: ROCmFP4 is `12.28%` smaller
- Test: `llama-bench`, `pp512 + tg128`

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

On this 35B A3B MTP run, ROCmFP4 was smaller and faster on decode/generation
across ROCm and Vulkan. ROCm prompt fill was still slightly behind Q4_K_M in
`llama-bench`, while Vulkan prompt fill and Hermes-style prompts favored
ROCmFP4.

## Clone And Build

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
```

If you specifically want the experimental branch name:

```bash
git checkout experimental-rocmfpx-branch
```

Pick the build script for your machine:

| Hardware | Build command | Output folder |
|---|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | `env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh` | `build-strix-rocmfp4/` |
| RDNA2 / RX 6000 (`gfx1030` class) | `env JOBS=16 scripts/build-rdna2.sh` | `build-rdna2/` |
| RDNA3 / RX 7000 (`gfx1100` class) | `env JOBS=16 scripts/build-rdna3.sh` | `build-rdna3/` |
| RDNA4 / RX 9000 (`gfx1200` class) | `env JOBS=16 scripts/build-rdna4.sh` | `build-rdna4/` |
| Vulkan fallback | use the Vulkan CMake path in `docs/BUILD-AMD-ARCHITECTURES.md` | custom |

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

## Quantize Straight ROCmFPX Models

Use BF16 or F16 GGUF sources. The wrapper keeps split GGUFs split by default.

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
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q3.gguf Q3_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q4.gguf Q4_0_ROCMFP4
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q6.gguf Q6_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q8.gguf Q8_0_ROCMFPX
```

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

## Local Reference Results

Current Strix Halo local reference points:

| Model | Size / BPW | Result |
|---|---:|---|
| ROCmFP8 Agent from BF16 | `31,568.94 MiB / 8.39 BPW` | agentic smoke pass |
| ROCmFP4 Agent from BF16 | `17,136.79 MiB / 4.55 BPW` | agentic smoke pass |
| BF16 baseline | source | agentic smoke pass |

ROCmFP4 Agent benchmark on ROCm0:

```text
pp512: 650.63 t/s
tg128: 76.55 t/s
```

## Code Layout

- `ggml/rocmfpx/` - ROCmFP3/ROCmFP6/ROCmFP8 reference formats
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
