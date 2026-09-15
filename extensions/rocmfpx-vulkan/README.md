# ROCmFPX Charlie Vulkan backend

This optional backend preserves the proven Charlie-era ROCmFPX Vulkan path
without replacing llama.cpp's upstream Vulkan backend. It is derived from
[`charlie12345/ROCmFPX` commit `7b02624ee`](https://github.com/charlie12345/ROCmFPX/commit/7b02624ee),
which is the source identity associated with the coherent ROCmFP4 Vulkan
baseline used during qualification. The repository's MIT license applies to
this code, and the project attribution is recorded in the root `NOTICE`.

The default build does not compile or load this extension. Enable it with
`-DROCMFPX_VULKAN_PLUGIN=ON`, then opt in at runtime by setting
`ROCMFPX_PLUGIN_PATH` to the absolute path of `rocmfpx-vulkan-plugin.so`. The
loader registers the sibling backend as `ROCmFPXVulkan0`; the upstream backend
remains available as `Vulkan0`.

## Build and run

```bash
cmake -S . -B build-rocmfpx-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_VULKAN=ON \
  -DROCMFPX_VULKAN_PLUGIN=ON
cmake --build build-rocmfpx-vulkan \
  --target llama-cli rocmfpx-vulkan-plugin -j

export ROCMFPX_PLUGIN_PATH="$PWD/build-rocmfpx-vulkan/bin/rocmfpx-vulkan-plugin.so"
build-rocmfpx-vulkan/bin/llama-cli --list-devices
build-rocmfpx-vulkan/bin/llama-cli \
  -m /absolute/path/to/model.gguf -dev ROCmFPXVulkan0 -ngl 999
```

The startup log should include `registered 1 ROCmFPX Vulkan device(s)` and
`loaded rocmfpx-vulkan-charlie`. `--list-devices` should list
`ROCmFPXVulkan0`. If it does not, confirm that `ROCMFPX_PLUGIN_PATH` is an
absolute path and that both shared libraries are in the same directory.

`rocmfpx-vulkan-plugin.so` is the ROCmFPX ABI wrapper. It resolves and loads
the sibling `libggml-rocmfpx-vulkan.so`; deploy both files together. The plugin
path is processed before command-line device validation, so
`-dev ROCmFPXVulkan0` and `--spec-draft-device ROCmFPXVulkan0` work normally.

`benchmark-rocmfp4.sh base` and `benchmark-rocmfp4.sh mtp4` provide matched
ROCmFP4 decode profiles. Set `ROCMFPX_BIN_DIR` and `ROCMFPX_MODEL` first. The
MTP profile uses upstream-supported F16 draft cache types and deterministic
sampling.

## Scope and qualification

This is a preserved compatibility backend, not a claim that every newer
ROCmFPX type exists in the Charlie snapshot. Its primary qualified path is
ROCmFP4/ROCmFP4 FAST; the snapshot also contains FP2, FP3, FP6, FP7, and FP8
shader paths. ROCmFP5 and ROCmI4 are not advertised by this backend and should
continue to use a separately qualified current backend.

For a focused ROCmFP4 operation check:

```bash
cmake --build build-rocmfpx-vulkan --target test-backend-ops -j
GGML_BACKEND_PATH="$PWD/build-rocmfpx-vulkan/bin/libggml-rocmfpx-vulkan.so" \
  build-rocmfpx-vulkan/bin/test-backend-ops test \
  -b ROCmFPXVulkan0 -o MUL_MAT -p 'type_a=q4_0_rocmfp4,'
```

The extension must be rebuilt for the exact ROCmFPX/ggml commit shipped in the
same release. Do not copy a backend library between releases. The ggml backend
structure is source-compatible only when the extension is qualified against
that tree.

To disable the extension without rebuilding, unset `ROCMFPX_PLUGIN_PATH` and
select `Vulkan0`. A plugin load failure does not replace or modify the normal
Vulkan backend.

Upstream synchronization order:

1. Merge and build clean llama.cpp ancestry with the extension disabled.
2. Build the extension against the merged tree.
3. Test both `Vulkan0` and `ROCmFPXVulkan0` on the same model and prompts.
4. Publish the extension only when coherence, backend operations, and performance gates pass.

The vendored backend retains llama.cpp and ggml provenance; ROCmFPX-specific
Vulkan work is credited to Charlie12345 and Carlo Pasquale as documented by
the project.
