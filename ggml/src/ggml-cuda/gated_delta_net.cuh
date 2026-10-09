#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Short batches (under 16 tokens, the recurrent kernel) read their state straight from the recurrent cache when the
// state input is reshape(get_rows(cache, ids)): returns that GET_ROWS node, or nullptr.
const ggml_tensor * ggml_cuda_gdn_state_gather(const ggml_tensor * gdn);

// the GET_ROWS at node i only feeds such delta nets, so it need not run
bool ggml_cuda_gdn_state_gather_skippable(const ggml_cgraph * cgraph, int i);
