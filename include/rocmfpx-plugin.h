#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(LLAMA_SHARED)
#    if defined(LLAMA_BUILD)
#      define ROCMFPX_API __declspec(dllexport)
#    else
#      define ROCMFPX_API __declspec(dllimport)
#    endif
#  else
#    define ROCMFPX_API
#  endif
#  if defined(ROCMFPX_PLUGIN_BUILD)
#    define ROCMFPX_PLUGIN_EXPORT __declspec(dllexport)
#  else
#    define ROCMFPX_PLUGIN_EXPORT
#  endif
#else
#  define ROCMFPX_API __attribute__((visibility("default")))
#  define ROCMFPX_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define ROCMFPX_PLUGIN_ABI_VERSION 1u

enum rocmfpx_plugin_capability {
    ROCMFPX_PLUGIN_CAP_BACKEND          = 1ull << 0,
    ROCMFPX_PLUGIN_CAP_SIDECAR          = 1ull << 1,
    ROCMFPX_PLUGIN_CAP_KV_CHECKPOINT    = 1ull << 2,
    ROCMFPX_PLUGIN_CAP_REPACK_CACHE     = 1ull << 3,
    ROCMFPX_PLUGIN_CAP_EXPERT_STREAMING = 1ull << 4,
};

enum rocmfpx_plugin_log_level {
    ROCMFPX_PLUGIN_LOG_ERROR = 0,
    ROCMFPX_PLUGIN_LOG_WARN  = 1,
    ROCMFPX_PLUGIN_LOG_INFO  = 2,
    ROCMFPX_PLUGIN_LOG_DEBUG = 3,
};

typedef void (*rocmfpx_plugin_log_fn)(int level, const char * message);
typedef int (*rocmfpx_plugin_backend_load_fn)(const char * path);
typedef size_t (*rocmfpx_plugin_backend_count_fn)(void);
typedef const char * (*rocmfpx_plugin_backend_name_fn)(size_t index);

enum rocmfpx_plugin_model_feature {
    ROCMFPX_PLUGIN_MODEL_FEATURE_PLE = 1ull << 0,
    ROCMFPX_PLUGIN_MODEL_FEATURE_SSM = 1ull << 1,
};

struct rocmfpx_plugin_model_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t model_id;
    const char * path;
    const char * architecture;
    uint32_t file_type;
    uint32_t reserved;
    uint64_t features;
    uint64_t size_bytes;
    uint64_t element_count;
};

struct rocmfpx_plugin_host_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    rocmfpx_plugin_log_fn log;
    const char * plugin_path;
    rocmfpx_plugin_backend_load_fn backend_load;
    rocmfpx_plugin_backend_count_fn backend_count;
    rocmfpx_plugin_backend_name_fn backend_name;
};

struct rocmfpx_plugin_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    const char * version;
    uint64_t capabilities;
    int  (*on_load)(void);
    void (*on_unload)(void);
    int (*on_model_open)(const struct rocmfpx_plugin_model_v1 * model);
    int (*on_model_close)(uint64_t model_id);
};

#define ROCMFPX_PLUGIN_HOST_V1_BASE_SIZE offsetof(struct rocmfpx_plugin_host_v1, plugin_path)
#define ROCMFPX_PLUGIN_V1_BASE_SIZE offsetof(struct rocmfpx_plugin_v1, on_model_open)

typedef const struct rocmfpx_plugin_v1 * (*rocmfpx_plugin_query_fn)(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host);

// Every v1 shared library exports this exact symbol.
ROCMFPX_PLUGIN_EXPORT const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host);

// Load a platform-separated list of files or directories. A directory is
// scanned non-recursively for shared libraries. Duplicate canonical paths are
// ignored. Returns the number of newly loaded plugins, or a negative value when
// the path list itself is invalid.
ROCMFPX_API int rocmfpx_plugins_load(const char * path_list);
ROCMFPX_API size_t rocmfpx_plugins_count(void);
ROCMFPX_API const struct rocmfpx_plugin_v1 * rocmfpx_plugin_at(size_t index);
ROCMFPX_API int rocmfpx_plugins_model_open(const struct rocmfpx_plugin_model_v1 * model);
ROCMFPX_API int rocmfpx_plugins_model_close(uint64_t model_id);
ROCMFPX_API void rocmfpx_plugins_shutdown(void);

#if defined(__cplusplus)
}
#endif
