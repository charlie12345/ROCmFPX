# Plugin and sidecar ABI

ROCmFPX plugin ABI v1 is declared in `include/rocmfpx-plugin.h`. A shared
library exports `rocmfpx_plugin_query`, returns a size-versioned descriptor,
and may advertise backend, sidecar, KV-checkpoint, repack-cache, and
expert-streaming capabilities. The host validates ABI version and structure
size, calls `on_load`, and unloads plugins in reverse order.

The ABI has two extension patterns:

- A backend plugin is a small wrapper that calls `backend_load(path)` to
  register a standard ggml backend shared library.
- A sidecar plugin receives model lifecycle notifications and maintains its
  own model-scoped state outside the GGUF.

ABI v1 does not provide tensor, scheduler, sampler, or token callbacks.
Capability flags describe what a plugin owns; they do not activate a data path
by themselves. A compute backend must use the standard ggml backend ABI.

Plugins load only from the explicit `ROCMFPX_PLUGIN_PATH` environment variable.
It is a colon-separated list on Unix and a semicolon-separated list on Windows;
entries can be shared libraries or directories. Directories are scanned once,
non-recursively and in sorted order. Canonical paths prevent duplicate loads.
The current working directory is never searched implicitly.

The host exposes three backend-registry callbacks to v1 plugins:

- `backend_load(path)` registers a standard ggml backend shared library and
  returns its device count.
- `backend_count()` and `backend_name(index)` enumerate the active backend
  registry after registration.
- `plugin_path` is the canonical path of the plugin being queried, so relative
  sidecar assets can be resolved beside the library.

The host object passed to `rocmfpx_plugin_query` is temporary. A plugin that
needs a service later must copy the function pointer or string value during the
query; it must not retain the host-structure pointer.

## Create a plugin

Use the following rules for every ABI v1 plugin:

1. Include `include/rocmfpx-plugin.h` and export the exact
   `rocmfpx_plugin_query` symbol.
2. Validate the host ABI version and every host field before using it.
3. Return a static descriptor with `abi_version` and `struct_size` populated.
4. Keep the descriptor, its strings, and every copied host callback alive
   until `on_unload` returns.
5. Return zero from successful lifecycle callbacks. A nonzero return rejects
   `on_load` or reports a failed model notification.
6. Treat all model strings and the query-time host structure as borrowed,
   short-lived data. Copy any value needed later.

Minimal sidecar descriptor:

```c
#define ROCMFPX_PLUGIN_BUILD
#include "rocmfpx-plugin.h"

static const struct rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION, sizeof(plugin), "example", "0.1.0",
    ROCMFPX_PLUGIN_CAP_SIDECAR, 0, 0, 0, 0,
};

const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t abi, const struct rocmfpx_plugin_host_v1 * host) {
    if (abi != ROCMFPX_PLUGIN_ABI_VERSION || !host ||
        host->abi_version != ROCMFPX_PLUGIN_ABI_VERSION) {
        return 0;
    }
    return &plugin;
}
```

New v1 descriptors may also provide `on_model_open` and `on_model_close`.
`on_model_open` receives a non-owning model identifier, canonical model path,
architecture, file type, byte and element counts, plus flags for models with a
PLE lane or SSM layers. The strings and structure are valid only for the
duration of the callback. `on_model_close` is delivered in reverse plugin order
before the model is destroyed. Plugins should key sidecar state by `model_id`
and release it during close.

Older v1 plugins remain loadable because both host and plugin structures are
size-versioned. Capability bits are discovery contracts: a backend plugin
registers its ggml backend through `backend_load`, while a sidecar can prepare
model-scoped state in `on_model_open`. Plugins are native code and must be
treated as trusted.

Call `rocmfpx_plugins_shutdown()` explicitly only after all plugin-visible
models have been released. Otherwise ROCmFPX keeps plugins resident until the
llama library or process exits. This is deliberate: some llama.cpp tools call
`llama_backend_free()` before their stack-owned models are destroyed, and the
model-close callbacks must remain callable throughout that teardown.

Build and run a plugin on Linux:

```bash
cc -shared -fPIC -I/path/to/ROCmFPX/include plugin.c -o plugin.so
ROCMFPX_PLUGIN_PATH=/absolute/path/to/plugin.so ./llama-cli -m model.gguf ...
```

Use an explicit absolute path in production. A directory may be mounted
read-only into a container and assigned to `ROCMFPX_PLUGIN_PATH`.

For a CMake project, define `ROCMFPX_PLUGIN_BUILD`, include the ROCmFPX SDK
header directory, and create a module library without the Unix `lib` prefix:

```cmake
add_library(example-sidecar MODULE plugin.c)
target_compile_definitions(example-sidecar PRIVATE ROCMFPX_PLUGIN_BUILD)
target_include_directories(example-sidecar PRIVATE /path/to/ROCmFPX/include)
set_target_properties(example-sidecar PROPERTIES PREFIX "")
```

The repository's `tests/rocmfpx-test-plugin.c` demonstrates query validation,
logging, capability flags, and model callbacks. Its matching
`tests/test-rocmfpx-plugin.cpp` exercises load deduplication, model open/close,
and shutdown.

## Windows AMD multi-GPU transport boundary

Charlie12345's
[Windows AMD Multi-GPU Bridge](https://github.com/charlie12345/windows-amd-vllm-multigpu)
supports an external, source-clean ROCmFPX/llama.cpp integration. Its
[`install-llama-rocmfpx.md`](https://github.com/charlie12345/windows-amd-vllm-multigpu/blob/main/docs/install-llama-rocmfpx.md)
guide builds a six-symbol RCCL ABI shim and exposes it as a `roc::rccl` CMake
package. ROCmFPX's HIP backend must be configured with `GGML_HIP_RCCL=ON` and
linked against that package before it can use the transport.

This is deliberately separate from plugin ABI v1:

- `ROCMFPX_PLUGIN_PATH` libraries must export `rocmfpx_plugin_query`; the
  bridge's `rccl.dll` exports the NCCL/RCCL symbols consumed by ggml HIP.
- ABI v1 backend registration loads a complete standard ggml backend. It does
  not replace a dependency inside an already-built HIP backend.
- ABI v1 sidecar callbacks observe model open and close. They do not receive or
  override collective operations.

Do not rename or wrap `rccl.dll` and present it as an ABI v1 plugin. A future
runtime adapter would need either a commit-matched HIP backend bundle registered
as a standard ggml backend, or a new pre-backend collective-provider contract;
a query-symbol wrapper alone cannot remove the `GGML_HIP_RCCL` build
requirement. Until such an interface is implemented and qualified, the
external CMake package plus its PowerShell launcher is the supported clean-tree
route on Windows.

## Use the Charlie Vulkan backend plugin

The in-tree [Charlie Vulkan extension](../../extensions/rocmfpx-vulkan/README.md)
is a complete backend-plugin example with a stricter compatibility rule: its
wrapper and backend libraries must come from the same ROCmFPX build. It
registers the legacy implementation as `ROCmFPXVulkan0` without replacing
`Vulkan0`.

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

The wrapper resolves `libggml-rocmfpx-vulkan.so` beside itself and registers
it through the host's backend registry. Expected startup messages include
`registered 1 ROCmFPX Vulkan device(s)` and `loaded
rocmfpx-vulkan-charlie`. Keep the two shared libraries together, use an
absolute plugin path, and rebuild both for each ROCmFPX/ggml commit. Unset
`ROCMFPX_PLUGIN_PATH` to disable the extension and use `Vulkan0` normally.

## Sidecar state

The SSD lanes have distinct purposes: KV checkpoints persist resumable prompt
state, repack caches store device-specific transformed weights keyed by model
and kernel identity, and expert streaming supplies bounded MoE expert windows.
They must use atomic publication, verify model/tensor hashes, and never treat a
cache as the canonical GGUF.
