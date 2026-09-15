#include "rocmfpx-plugin.h"

#include <cstring>
#include <cstdio>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "requirement failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char ** argv) {
    REQUIRE(argc == 2);
    REQUIRE(rocmfpx_plugins_count() == 0);
    REQUIRE(rocmfpx_plugins_load(argv[1]) == 1);
    REQUIRE(rocmfpx_plugins_load(argv[1]) == 0);
    REQUIRE(rocmfpx_plugins_count() == 1);

    const rocmfpx_plugin_v1 * plugin = rocmfpx_plugin_at(0);
    REQUIRE(plugin != nullptr);
    REQUIRE(std::strcmp(plugin->name, "rocmfpx-test-plugin") == 0);
    REQUIRE((plugin->capabilities & ROCMFPX_PLUGIN_CAP_SIDECAR) != 0);
    REQUIRE((plugin->capabilities & ROCMFPX_PLUGIN_CAP_REPACK_CACHE) != 0);
    REQUIRE(rocmfpx_plugin_at(1) == nullptr);

    const rocmfpx_plugin_model_v1 model = {
        ROCMFPX_PLUGIN_ABI_VERSION, sizeof(rocmfpx_plugin_model_v1), 17,
        "test-model.gguf", "test", 0, 0,
        ROCMFPX_PLUGIN_MODEL_FEATURE_PLE | ROCMFPX_PLUGIN_MODEL_FEATURE_SSM,
        1024, 256,
    };
    REQUIRE(rocmfpx_plugins_model_open(&model) == 1);
    REQUIRE(rocmfpx_plugins_model_close(model.model_id) == 1);

    rocmfpx_plugins_shutdown();
    REQUIRE(rocmfpx_plugins_count() == 0);
    return 0;
}
