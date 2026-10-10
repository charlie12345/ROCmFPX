#include "common.cuh"

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst);
// the q8_0 pool-cache GET_ROWS whose rows the indexer reads in place, or nullptr
const ggml_tensor * ggml_cuda_lightning_indexer_pool_gather(int device, const ggml_tensor * dst);
