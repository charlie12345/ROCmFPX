#include "rocmfpx-plugin.h"

static int loaded;
static int host_services_ok;
static uint64_t opened_model_id;
static rocmfpx_plugin_log_fn host_log;

static int on_load(void) {
    loaded = 1;
    return 0;
}

static void on_unload(void) {
    loaded = 0;
}

static int on_model_open(const struct rocmfpx_plugin_model_v1 * model) {
    if (!loaded || !host_services_ok || !model || !model->path || !model->architecture ||
        (model->features & ~(ROCMFPX_PLUGIN_MODEL_FEATURE_PLE | ROCMFPX_PLUGIN_MODEL_FEATURE_SSM)) != 0) {
        return -1;
    }
    opened_model_id = model->model_id;
    host_log(ROCMFPX_PLUGIN_LOG_INFO, "test plugin received model open");
    return 0;
}

static int on_model_close(uint64_t model_id) {
    if (!loaded || model_id == 0 || model_id != opened_model_id) {
        return -1;
    }
    opened_model_id = 0;
    host_log(ROCMFPX_PLUGIN_LOG_INFO, "test plugin received model close");
    return 0;
}

static const struct rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION,
    sizeof(struct rocmfpx_plugin_v1),
    "rocmfpx-test-plugin",
    "1.0.0",
    ROCMFPX_PLUGIN_CAP_SIDECAR | ROCMFPX_PLUGIN_CAP_REPACK_CACHE,
    on_load,
    on_unload,
    on_model_open,
    on_model_close,
};

const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host) {
    if (host_abi_version != ROCMFPX_PLUGIN_ABI_VERSION || !host ||
        host->abi_version != ROCMFPX_PLUGIN_ABI_VERSION || host->struct_size < sizeof(*host) ||
        !host->log || !host->plugin_path || !host->backend_load || !host->backend_count || !host->backend_name) {
        return 0;
    }
    host_log = host->log;
    host_services_ok = host->backend_count() > 0 && host->backend_name(0) != 0 && host->backend_load(0) == -1;
    return &plugin;
}
