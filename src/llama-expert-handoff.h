#pragma once

// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff: the routed experts of a MoE model stay in host memory, the most used of
// them are copied into a VRAM cache, and for small batches each layer is split between the two: the device computes
// the cached experts while a host thread pool computes the others in place from host memory (docs/expert-handoff.md).
//
// The graph side: build_moe_ffn reads the selected experts' cache slots from a slot map on the device (the experts
// not in the cache get the miss slot, which holds zeros), sends the selection to the host (ggml_expert_handoff_send),
// computes its experts from the cache banks, and joins the host's sum (ggml_expert_handoff_join). Large batches take
// the usual path: the scheduler uploads the experts they use, the cached ones are copied from the banks.

#include "llama.h"

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

struct llama_expert_handoff_params {
    int32_t     n_slots          = -1; // expert slots in VRAM in units of the largest expert, -1 = fill the free VRAM
    size_t      budget_bytes     =  0; // instead of n_slots: the VRAM cache size in bytes (--moe-cache-mib)
    int32_t     vram_reserve_mib = -1; // VRAM the automatic size leaves free, -1 = default
    int32_t     n_workers        =  0; // host threads, 0 = one per 4 logical CPUs
    uint32_t    max_tokens       = 32; // largest batch the layers are split for
    std::string profile;               // routing counts to fill the cache from
    std::string profile_save;          // where to write the routing counts at exit
    int32_t     adapt_every      =  4; // the adaptive cache: a round every N decode graphs, 0 = off
    int32_t     adapt_swaps      = 96; // most swaps per round
    float       adapt_decay      = 0.7f; // the usage counts are multiplied by this after each round
};

// how a layer's routed experts are computed, as build_moe_ffn does it (the host pool repeats it)
struct llama_expert_handoff_form {
    int32_t act           = -1;    // ggml_expert_handoff_act
    float   alpha         = 0.0f;
    float   limit         = 0.0f;
    float   beta          = 0.0f;
    float   lb            = 0.0f;
    bool    weight_before = false;
    int32_t n_used        = 0;     // only the first n_used selected experts count, 0 = all

    bool operator==(const llama_expert_handoff_form & o) const {
        return act == o.act && alpha == o.alpha && limit == o.limit && beta == o.beta && lb == o.lb &&
               weight_before == o.weight_before && n_used == o.n_used;
    }
};

class llama_expert_handoff {
public:
    llama_expert_handoff(const llama_model & model, const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_expert_handoff_params & params);
    ~llama_expert_handoff();

    // true: layer il's routed experts are split for a batch of n_tokens, computed as form says; the first call per
    // layer registers its experts with the host pool
    bool use(int32_t il, int64_t n_tokens, const llama_expert_handoff_form & form) const;

    ggml_backend_t backend() const;
    ggml_tensor *  slot_map(int32_t il) const;                // I32 [1, n_expert] on the device
    ggml_tensor *  get_experts(const ggml_tensor * w) const;  // the cache view used in place of the host experts w
                                                              // (also of their biases: [n, n_slots + 1], zeros at the miss slot)

    // what the send and join ops of layer il need
    ggml_expert_handoff_params params(int32_t il) const;

    // large batches: copy the cached experts in [e, last] of w from the banks, returns how many consecutive ones
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) const;

    void wake()  const; // a graph that may hand off experts is about to run
    void idle()  const; // the graphs are done: the host pool stops spinning
    void check() const; // before a graph: report joins that timed out, the periodic hit-rate line

    // before a graph of n_tokens: every adapt_every decode graphs, the most routed experts the cache misses replace
    // the least routed ones it holds; their copies run on a helper thread, and they serve once they have landed
    void adapt(int64_t n_tokens) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_expert_handoff_ptr = std::unique_ptr<llama_expert_handoff>;
