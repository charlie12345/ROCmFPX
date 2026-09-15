#define ROCMFPX_PLUGIN_BUILD
#include "rocmfpx-plugin.h"

#include <filesystem>
#include <string>

namespace {

rocmfpx_plugin_log_fn host_log;
rocmfpx_plugin_backend_load_fn host_backend_load;
std::string plugin_path;

const char * backend_filename() {
#if defined(_WIN32)
    return "ggml-rocmfpx-vulkan.dll";
#elif defined(__APPLE__)
    return "libggml-rocmfpx-vulkan.dylib";
#else
    return "libggml-rocmfpx-vulkan.so";
#endif
}

int on_load() {
    const std::filesystem::path backend = std::filesystem::path(plugin_path).parent_path() / backend_filename();
    const int devices = host_backend_load(backend.string().c_str());
    if (devices <= 0) {
        const std::string message = "failed to load backend " + backend.string();
        host_log(ROCMFPX_PLUGIN_LOG_ERROR, message.c_str());
        return -1;
    }
    const std::string message = "registered " + std::to_string(devices) + " ROCmFPX Vulkan device(s)";
    host_log(ROCMFPX_PLUGIN_LOG_INFO, message.c_str());
    return 0;
}

const rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION,
    sizeof(rocmfpx_plugin_v1),
    "rocmfpx-vulkan-charlie",
    "0.1.0",
    ROCMFPX_PLUGIN_CAP_BACKEND,
    on_load,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

extern "C" const rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t host_abi_version, const rocmfpx_plugin_host_v1 * host) {
    if (host_abi_version != ROCMFPX_PLUGIN_ABI_VERSION || !host ||
        host->abi_version != ROCMFPX_PLUGIN_ABI_VERSION || host->struct_size < sizeof(*host) ||
        !host->log || !host->plugin_path || !host->backend_load) {
        return nullptr;
    }
    host_log = host->log;
    host_backend_load = host->backend_load;
    plugin_path = host->plugin_path;
    return &plugin;
}
