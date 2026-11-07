# Building ROCmFP4 for AMD GPUs

ROCmFP4 runs on **CPU**, **Vulkan**, and **ROCm/HIP**. Most users want the HIP
backend for best performance. To build HIP support, you must compile for your
GPU's `gfx` target.

Vulkan does not need a `gfx` target at compile time and works across RDNA2+ AMD
GPUs. RDNA-FAST tunes and measures the HIP backend on an RX 7900 XT (`gfx1100`);
the regression scripts below keep ROCmFPX's Strix Halo (`gfx1151`) defaults.

The scripts in this guide are Linux (bash) builds inherited from ROCmFPX.
RDNA-FAST itself builds with ROCm 10.1; the Windows build is the
[README quick start](../README.md#quick-start).

---

## Choose Your Build Path

| Your GPU | Example cards | Easiest build command | Build output folder |
|---|---|---|---|
| **Strix Halo / RDNA3.5** | Ryzen AI MAX+ 395, Framework Desktop | `scripts/build-strix-rocmfp4-mtp.sh` | `build-strix-rocmfp4/` |
| **RDNA2** | RX 6700 XT, RX 6800 | `scripts/build-rdna2.sh` | `build-rdna2/` |
| **RDNA3** | RX 7600, RX 7900 XTX, RX 7800 XT | `scripts/build-rdna3.sh` | `build-rdna3/` |
| **RDNA4** | RX 9070 XT | `scripts/build-rdna4.sh` | `build-rdna4/` |
| **Vega 20 / gfx906 experimental** | Radeon Instinct MI50 / MI60 | `scripts/build-gfx906.sh` | `build-gfx906/` |
| **Windows** | RX 7000 / 9000, Strix Halo | ROCm 10.1 `cmake` command in the [README quick start](../README.md#quick-start) | `build-hip/` |
| **Vulkan only** | Any AMD GPU with Vulkan drivers | see [Vulkan-only build](#vulkan-only-no-hip-arch-needed) | `build-vulkan/` |

All Linux scripts accept `JOBS=16` to control parallel compile jobs:

```bash
env JOBS=16 scripts/build-rdna3.sh
```

---

## Find Your `gfx` Target

On Linux, check what ROCm reports:

```bash
rocminfo | grep -m1 "Name:"
rocminfo | grep -m1 "gfx"
```

Then match your GPU to this table:

| AMD generation | Example hardware | Typical `gfx` IDs | Build target | Linux runtime fallback |
|---|---|---|---|---|
| Vega 20 / GCN5 | Radeon Instinct MI50 / MI60 | `gfx906` | `gfx906` | use native `gfx906` when ROCm supports it |
| RDNA1 | RX 5700 XT, RX 5600 | `gfx1010`, `gfx1012` | `gfx1010` | `HSA_OVERRIDE_GFX_VERSION=10.1.0` |
| RDNA2 | RX 6700/6800/6900 | `gfx1030`–`gfx1037` | `gfx1030` | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| RDNA3 | RX 7600, RX 7900 XTX/XT/GRE, RX 7800 XT, Radeon 780M / 760M | `gfx1100`–`gfx1103` | `gfx1100` | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| RDNA3.5 | Strix Halo, Ryzen AI MAX+, Radeon 890M / 880M / 860M | `gfx1150`–`gfx1152` | `gfx1151` | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |
| RDNA4 | RX 9060 XT (Navi 44) | `gfx1200` | `gfx1200` | use native `gfx` when ROCm supports it |
| RDNA4 | RX 9070 / 9070 XT / AI PRO R9700 (Navi 48) | `gfx1201` | `gfx1201` | must be built as `gfx1201`, see below |

**Tips**

- Sub-variants usually map to the nearest base target (`gfx1035` → `gfx1030`,
  `gfx1102` → `gfx1100`).
- **RDNA4 is the exception: `gfx1201` is not interchangeable with `gfx1200`.** A
  `gfx1200` build on a `gfx1201` card links and loads a model, then segfaults or
  hangs with no error message (issues #18 and #37). The build scripts detect the
  installed GPU and pick the matching target, and they check the emitted code
  objects afterwards, so this now fails loudly instead of silently.
- ROCmFPX's published benchmark numbers and the regression guards assume
  **Strix Halo / `gfx1151`**. RDNA-FAST's README numbers are from an RX 7900 XT
  (`gfx1100`).
- Vega 20 / `gfx906` is an experimental community target. It is not RDNA/CDNA,
  and should be validated on real MI50/MI60 hardware before claiming support.
- `HSA_OVERRIDE_GFX_VERSION` works on **Linux only** — not on Windows.

---

## Build Scripts

This repository provides one generic builder plus thin wrappers per generation.
You do not need separate full build scripts for each architecture.

| Script | Target | Notes |
|---|---|---|
| `scripts/build-strix-rocmfp4-mtp.sh` | `gfx1151` | Validated default; includes regression-test binaries |
| `scripts/build-rdna2.sh` | `gfx1030` default | RX 6000 class; uses the installed RDNA2 target if it differs |
| `scripts/build-rdna3.sh` | `gfx1100` default | RX 7000 class; uses the installed RDNA3/3.5 target if it differs |
| `scripts/build-rdna4.sh` | `gfx1200` default | RX 9000 class; uses `gfx1201` automatically on Navi 48 cards |
| `scripts/build-gfx906.sh` | `gfx906` | Experimental Vega 20 / MI50 / MI60 community target |
| `scripts/build-rocmfp4.sh` | any `gfx` | Generic — set `CMAKE_HIP_ARCHITECTURES` yourself |

There is no Windows build script; see [Windows](#windows).

Every wrapper honours `CMAKE_HIP_ARCHITECTURES`, so an explicit target always wins:

```bash
CMAKE_HIP_ARCHITECTURES=gfx1201 scripts/build-rdna4.sh
```

Two safeguards apply to `build-rocmfp4.sh` and the wrappers that call it
(`build-strix-rocmfp4-mtp.sh` runs CMake directly and has neither):

- Changing target reuses nothing: if the build directory was configured for a
  different `gfx`, it is removed first, because CMake updates the cache while
  object files for the old target can survive. Set `ROCMFPX_KEEP_STALE_BUILD=1`
  to keep the directory instead.
- After the build, the emitted code objects are compared against the requested
  target and the build fails if they disagree.

Generic example (any single target):

```bash
env CMAKE_HIP_ARCHITECTURES=gfx1100 BUILD_DIR=build-rdna3 scripts/build-rocmfp4.sh
```

---

## Common CMake Flags

The Linux build scripts pass:

| Flag | Value | Why |
|---|---|---|
| `GGML_HIP` | `ON` | Enable ROCm/HIP backend |
| `GGML_VULKAN` | `ON` | Enable Vulkan (recommended fallback) |
| `GGML_CUDA` | `OFF` | Disable NVIDIA CUDA |
| `GGML_HIP_FORCE_MMQ` | `ON` | No effect at present: the option is still declared, but nothing has read it since the September 2026 ROCmFPX merge. `GGML_CUDA_FORCE_MMQ` is the switch that forces MMQ |
| `CMAKE_BUILD_TYPE` | `Release` | Release build |
| `CMAKE_HIP_ARCHITECTURES` or `GPU_TARGETS` | your `gfx` | GPU ISA to compile for |

`GPU_TARGETS` and `CMAKE_HIP_ARCHITECTURES` are equivalent here — use either one.

---

## Per-Architecture Commands

### Strix Halo / RDNA3.5 (validated default)

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
```

Run with:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
./build-strix-rocmfp4/bin/llama-cli -m model.gguf -dev ROCm0 -ngl 999 ...
```

Full Strix install guide: [`docs/STRIX-HALO-QUICKSTART.md`](STRIX-HALO-QUICKSTART.md)

### RDNA2 — Linux

```bash
env JOBS=16 scripts/build-rdna2.sh
```

RDNA-FAST has not checked RDNA2 builds again (see
[Other cards](../README.md#other-cards) in the README).

If ROCm does not recognize your exact `gfx` ID:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ./build-rdna2/bin/llama-cli -m model.gguf -dev ROCm0 ...
```

### RDNA3 — Linux

```bash
env JOBS=16 scripts/build-rdna3.sh
```

Runtime fallback:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 ./build-rdna3/bin/llama-cli -m model.gguf -dev ROCm0 ...
```

### RDNA4 — Linux

```bash
env JOBS=16 scripts/build-rdna4.sh
```

On a Navi 48 card (RX 9070, 9070 XT, AI PRO R9700) this builds `gfx1201`
automatically; on Navi 44 (RX 9060 XT) it builds `gfx1200`. Force one with:

```bash
CMAKE_HIP_ARCHITECTURES=gfx1201 env JOBS=16 scripts/build-rdna4.sh
```

Requires a ROCm version with device libraries for the target you build. If HIP is
not ready yet, use the [Vulkan-only path](#vulkan-only-no-hip-arch-needed).

### Vega 20 / gfx906 — Linux Experimental

```bash
env JOBS=16 scripts/build-gfx906.sh
```

This target is intended for community testing on Radeon Instinct MI50 / MI60
hardware. It is additive and does not change the RDNA2/RDNA3/Strix/RDNA4 build
defaults.

Minimum validation before reporting it as working:

```bash
./build-gfx906/bin/test-backend-ops -b ROCm0
./build-gfx906/bin/test-quantize-fns
./build-gfx906/bin/llama-bench -m model.gguf -dev ROCm0 -ngl 999
```

If HIP support is unreliable on a specific ROCm version, try the
[Vulkan-only path](#vulkan-only-no-hip-arch-needed) first.

### Windows

Build with ROCm 10.1 as in the [README quick start](../README.md#quick-start):
AMD's TheRock tarball for your GPU family (`gfx110X-all` for RDNA3, `gfx120X-all`
for RDNA4, `gfx1151` for Strix Halo), then the `cmake` command shown there with
your target in `-DGPU_TARGETS` and `-DCMAKE_HIP_ARCHITECTURES`. Binaries land in
`build-hip\bin\`, next to the HIP runtime DLLs the build copies there.

`HSA_OVERRIDE_GFX_VERSION` does not work on Windows. The binary must match a
ROCm-supported `gfx` target.

---

## Multi-GPU / Distribution Build

Build one binary for several AMD GPUs by listing targets separated by semicolons.
Compile time and binary size increase significantly.

```bash
cmake -S . -B build-multi \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_FORCE_MMQ=ON \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  -DGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201"

cmake --build build-multi -j "$(nproc)"
```

| Use case | `GPU_TARGETS` |
|---|---|
| RDNA2 only | `gfx1030` |
| RDNA3 only | `gfx1100` |
| RDNA3 + Strix Halo | `gfx1100;gfx1150;gfx1151` |
| All current consumer AMD | `gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201` |
| Experimental MI50/MI60 add-on | append `gfx906` only if you intend to test Vega 20 |

On Windows, `ggml-hip.dll` must stay below 2 GB or Windows will not load it. A
build for every target is larger than that, so the RDNA-FAST release
(`.github/workflows/rdna-fast-release.yml`) builds one GPU family per archive:
`gfx1100;gfx1101;gfx1102;gfx1103`, `gfx1150;gfx1151;gfx1152` and
`gfx1200;gfx1201`.

---

## Runtime Environment Variables

| Variable | When to use | Example |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | Linux; GPU not in official ROCm support | `10.3.0`, `11.0.0`, `11.5.1` |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | UMA systems (Strix Halo, APUs) | `1` |
| `HIP_VISIBLE_DEVICES` | Pick a specific GPU | `0` |

The AMD-named alias `GGML_HIP_ENABLE_UNIFIED_MEMORY` is no longer read (dropped
in the September 2026 ROCmFPX merge); use the `GGML_CUDA_` name.

---

## Vulkan Only (No HIP Arch Needed)

Use this when ROCm/HIP is unavailable or your GPU is not yet supported by HIP:

```bash
cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_HIP=OFF \
  -DGGML_CUDA=OFF

cmake --build build-vulkan -j "$(nproc)" --target llama-cli llama-quantize

./build-vulkan/bin/llama-cli -m model.gguf -dev Vulkan0 -ngl 999 ...
```

---

## Advanced Tuning (Experts Only)

HIP micro-optimization knobs are passed via `CMAKE_HIP_FLAGS`. Defaults are
already tuned — only change these if you are running regression guards.

```bash
env CMAKE_HIP_FLAGS="-DGGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD=0" \
    CMAKE_HIP_ARCHITECTURES=gfx1151 \
    scripts/build-rocmfp4.sh
```

Full list of knobs: [`ggml/rocmfp4/README.md`](../ggml/rocmfp4/README.md)

---

## Validate Your Build

```bash
# CPU quant check (no GPU needed)
scripts/check-rocmfp4-quant-regression.sh

# Full gate (Strix defaults — override BUILD_DIR for other builds)
env HSA_OVERRIDE_GFX_VERSION=11.5.1 BUILD_DIR=build-strix-rocmfp4 \
    scripts/check-rocmfp4-all-regression.sh
```

Set `BUILD_DIR`, `BIN`, or `TEST_BACKEND_OPS_BIN` when not using the Strix
default paths. Details: [`docs/ROCmFP4-REPRODUCIBILITY.md`](ROCmFP4-REPRODUCIBILITY.md)

---

## Quick Reference

```bash
# Strix Halo (best-tested path)
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh

# Desktop AMD GPUs
env JOBS=16 scripts/build-rdna2.sh   # RX 6000
env JOBS=16 scripts/build-rdna3.sh   # RX 7000
env JOBS=16 scripts/build-rdna4.sh   # RX 9000

# Windows: ROCm 10.1 cmake command in the README quick start
```
