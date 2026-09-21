# AMD support tiers

ROCmFPX keeps normal llama.cpp Vulkan support enabled. Vulkan is the portable
default; HIP is an additional backend, and the special ROCmI4/W4A4 build is a
separate opt-in profile.

| GPU generation | Examples | Tier | Qualification expectation |
| --- | --- | --- | --- |
| RDNA 1 | gfx101x | compatible | CPU reference plus Vulkan smoke |
| RDNA 2 | gfx103x | supported | CPU, Vulkan, HIP build/runtime |
| RDNA 3 | gfx110x | supported | CPU, Vulkan, HIP build/runtime |
| RDNA 3.5 | gfx115x | primary | full correctness, Qwen, MTP, throughput |
| RDNA 4 | gfx120x | supported | CPU, Vulkan, HIP build/runtime |
| future RDNA | unknown | provisional | no claim until identified and tested |

“Compatible” is best-effort and not a promise that every ROCm release exposes a
HIP target for that device. Release notes must name the OS, driver, ROCm/Vulkan
versions, exact GPU, commit, model hash, and test matrix. Backend fallbacks must
be visible in logs; a fallback result must not be reported as a native-kernel
benchmark.

HIP builds use the State Space Duality matmul path for profitable long Mamba-2
prefills and retain the sequential scan as a fallback. Set
`GGML_CUDA_DISABLE_SSD=1` before process start to force that fallback for an A/B
or operational rollback; the environment setting is read once per process.

## Experimental Vulkan ROCmFP4 cooperative matrices

`GGML_VULKAN_ROCMFP4_COOPMAT` is a default-off CMake option. Enable it with
`-DGGML_VULKAN=ON -DGGML_VULKAN_ROCMFP4_COOPMAT=ON` to compile ROCmFP4 and
ROCmFP4-FAST CM1 shaders. An ordinary build excludes these shader variants.

The experimental build still uses the scalar path unless
`GGML_VK_ROCMFP4_COOPMAT=1` is set before launch. CM1 and F32 accumulation
support are required; otherwise the request logs a scalar fallback. CM2,
other ROCmFPX formats, HIP, GGUF layouts, and model weights are unchanged.
Unset the variable and restart the process to return to the scalar path.

Kernel tolerance tests are not a guarantee of identical generated text.
The experimental path changes numerical execution and requires fixed-model
quality and performance checks before use. Upstream updates remain supported,
but both build-option states need testing when merging Vulkan changes.

See [the activation guide and Qwen Flash PLE16 example](VULKAN-CM1.md) for build/runtime on-off controls, the measured launch flags, benchmark limits and rollback instructions.
