#pragma once

#include "common.cuh"

// ROCmFPX [TAG_EXPERT_HANDOFF] the send and join ops of the expert handoff (protocol in ggml-expert-handoff.h)
void ggml_cuda_op_expert_handoff(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// host memory the device reads and writes in place at the same address (pinned, mapped, coherent), zeroed
void * ggml_cuda_host_mapped_alloc(size_t size);
void   ggml_cuda_host_mapped_free(void * ptr);
