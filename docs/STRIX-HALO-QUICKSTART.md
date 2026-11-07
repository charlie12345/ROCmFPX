# Strix Halo Quickstart

Strix Halo / RDNA3.5 (`gfx1151`) install guide. For other AMD GPUs (RDNA2,
RDNA3, RDNA4), see [`BUILD-AMD-ARCHITECTURES.md`](BUILD-AMD-ARCHITECTURES.md).

Use this repository when you want the ready-to-build llama.cpp fork with
ROCmFP4, MTP, ROCm/HIP, and Vulkan integration already applied.

```bash
git clone https://github.com/serrynaimo/RDNA-FAST.git
cd RDNA-FAST
```

This guide comes from ROCmFPX, whose proof target was Strix Halo. RDNA-FAST is
tuned and measured on an RX 7900 XT (`gfx1100`); on `gfx1151` it is checked to
build (see [Other cards](../README.md#other-cards) in the README). On Windows,
build with ROCm 10.1 as in the [README quick start](../README.md#quick-start),
using the `gfx1151` tarball and target.

## Target Hardware

ROCmFPX tuned and validated this build on:

```text
Framework AMD Strix Halo 395+, 128 GB unified RAM, gfx1151
```

Other AMD systems may work, but they are not the proof target for ROCmFPX's
published numbers.

## Prerequisites

Install the normal llama.cpp Linux build tools plus ROCm/HIP and Vulkan support:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential clang pkg-config \
  glslc vulkan-tools
```

ROCm must see the Strix Halo GPU:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 rocminfo
```

Vulkan should list a GPU device:

```bash
vulkaninfo --summary
```

## Build

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
```

For RDNA2, RDNA3, or RDNA4 desktop GPUs, see
[`BUILD-AMD-ARCHITECTURES.md`](BUILD-AMD-ARCHITECTURES.md) and run the matching
wrapper script (`build-rdna2.sh`, `build-rdna3.sh`, or `build-rdna4.sh`).

The build script enables ROCm/HIP and Vulkan, disables NVIDIA CUDA, targets
`gfx1151`, and writes binaries under:

```text
build-strix-rocmfp4/bin/
```

Key binaries:

```text
llama-cli
llama-server
llama-quantize
llama-bench
test-backend-ops
test-quantize-fns
test-quantize-perf
```

## Quantize A Model

For real quality testing, start from an F16 or BF16 GGUF source. Requantizing an
already heavily quantized file is only useful for smoke tests.

Compact Strix profile:

```bash
./build-strix-rocmfp4/bin/llama-quantize \
  /path/to/source-bf16.gguf \
  /path/to/model-ROCmFP4-STRIX_LEAN.gguf \
  Q4_0_ROCMFP4_STRIX_LEAN
```

Quality-biased Strix profile:

```bash
./build-strix-rocmfp4/bin/llama-quantize \
  /path/to/source-bf16.gguf \
  /path/to/model-ROCmFP4-STRIX.gguf \
  Q4_0_ROCMFP4_STRIX
```

Pure experimental formats:

```bash
./build-strix-rocmfp4/bin/llama-quantize source.gguf out-dual.gguf Q4_0_ROCMFP4
./build-strix-rocmfp4/bin/llama-quantize source.gguf out-fast.gguf Q4_0_ROCMFP4_FAST
```

## Run Interactive ROCm

Use `--jinja` when a model has a modern chat template. Use `--reasoning on` only
for models whose template supports reasoning.

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
./build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-ROCmFP4-STRIX_LEAN.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -c 262144 \
  -b 512 \
  -ub 512 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --jinja \
  -if
```

## Run Interactive MTP

For a model with native MTP draft heads, add speculative draft flags:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
./build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-ROCmFP4-STRIX_LEAN.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -c 262144 \
  -b 512 \
  -ub 512 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 \
  --spec-draft-p-split 0.10 \
  --spec-draft-type-k q4_0 \
  --spec-draft-type-v q4_0 \
  --jinja \
  -if
```

Remove the `--spec-*` flags for models that do not support MTP.

## Validate

After building, run the full promoted gate:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.5.1 scripts/check-rocmfp4-all-regression.sh
```

Focused checks while iterating:

```bash
scripts/check-rocmfp4-quant-regression.sh
scripts/check-rocmfp4-rocm-runtime-regression.sh
scripts/check-rocmfp4-rocm-fattn-regression.sh
scripts/check-rocmfp4-vulkan-runtime-regression.sh
scripts/check-rocmfp4-qwen-mtp-regression.sh
```

Some validation scripts use local default model paths. Override `MODEL`,
`ROCMFP4_MODEL`, or `BASELINE_MODEL` if your models live elsewhere.

## Troubleshooting

- `No HIP GPUs are available`: verify ROCm sees the GPU with
  `HSA_OVERRIDE_GFX_VERSION=11.5.1 rocminfo`.
- Chat template runtime error: add `--jinja`.
- Out of memory or very slow context setup: lower `-c`, `-b`, or `-ub`, then
  retest.
- No model weights are included in this repository. Download model files
  separately and follow their licenses.
- llama.cpp stores the HIP backend in paths named `ggml-cuda`; this build sets
  `-DGGML_CUDA=OFF` and uses those files for AMD HIP/ROCm.
