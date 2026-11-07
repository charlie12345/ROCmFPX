#pragma once

// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff: the layout of the mailbox the device and the host share.
//
// The mailbox is host memory both sides read and write in place (pinned, mapped into the device, coherent). For each
// MoE layer the device's send op writes the routed experts' input, their ids, cache slots and weights here and bumps
// seq_send; the host computes the experts the device does not hold (the slots equal to the layer's miss slot), writes
// their weighted sum to y and sets seq_done = seq_send; the device's join op waits for that and adds y to its own
// experts' sum. One layer is in flight at a time per mailbox, because the next layer's send needs this layer's join.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXPERT_HANDOFF_ALIGN 256

// the header, one 64-byte line per writer so neither side's writes evict the other's line
struct ggml_expert_handoff_header {
    uint32_t seq_send;      // device: bumped once the inputs of a layer are in place
    uint32_t pad_send[15];
    uint32_t seq_done;      // host: set to seq_send once y is in place
    uint32_t pad_done[15];
    uint32_t n_timeouts;    // device: joins that gave up waiting for the host (the host's experts are then missing)
    uint32_t to_want;       // device, the last join that gave up [TAG_EH_TIMEOUT_DIAG]: the send it waited for,
    uint32_t to_done;       //   the host's last answer it saw,
    int32_t  to_layer;      //   its layer
    uint32_t pad_err[12];
    int32_t  layer;         // device, written before seq_send
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  n_used;
    uint32_t pad_info[12];
};

// capacities, fixed when the mailbox is allocated (passed to the ops in their op params)
struct ggml_expert_handoff_dims {
    int32_t max_tokens;
    int32_t max_embd;
    int32_t max_used;
};

static inline size_t ggml_expert_handoff_pad(size_t n) {
    return (n + GGML_EXPERT_HANDOFF_ALIGN - 1) / GGML_EXPERT_HANDOFF_ALIGN * GGML_EXPERT_HANDOFF_ALIGN;
}

// region offsets in bytes from the start of the mailbox:
// x [max_tokens][max_embd] f32, ids [max_tokens][max_used] i32, slots (same), weights [max_tokens][max_used] f32,
// y [max_tokens][max_embd] f32
static inline size_t ggml_expert_handoff_off_x(struct ggml_expert_handoff_dims d) {
    (void) d;
    return ggml_expert_handoff_pad(sizeof(struct ggml_expert_handoff_header));
}
static inline size_t ggml_expert_handoff_off_ids(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_x(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_embd*sizeof(float));
}
static inline size_t ggml_expert_handoff_off_slots(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_ids(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(int32_t));
}
static inline size_t ggml_expert_handoff_off_weights(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_slots(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(int32_t));
}
static inline size_t ggml_expert_handoff_off_y(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_weights(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(float));
}
static inline size_t ggml_expert_handoff_size(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_y(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_embd*sizeof(float));
}

// op params of GGML_OP_EXPERT_HANDOFF (int32 slots)
enum ggml_expert_handoff_param {
    GGML_EXPERT_HANDOFF_P_MODE       = 0, // GGML_EXPERT_HANDOFF_SEND or _JOIN
    GGML_EXPERT_HANDOFF_P_PTR_LO     = 1, // the mailbox (the same address on the host and the device)
    GGML_EXPERT_HANDOFF_P_PTR_HI     = 2,
    GGML_EXPERT_HANDOFF_P_LAYER      = 3,
    GGML_EXPERT_HANDOFF_P_MAX_TOKENS = 4,
    GGML_EXPERT_HANDOFF_P_MAX_EMBD   = 5,
    GGML_EXPERT_HANDOFF_P_MAX_USED   = 6,
    GGML_EXPERT_HANDOFF_P_TIMEOUT_MS = 7, // join: give up after this long (the GPU watchdog must never see a stuck kernel)
    GGML_EXPERT_HANDOFF_P_MISS_SLOT  = 8, // send: the slot of the experts the host computes
    GGML_EXPERT_HANDOFF_P_FLAGS_LO   = 9, // device memory [n_layer] int32: the misses of each layer's last send
    GGML_EXPERT_HANDOFF_P_FLAGS_HI   = 10,
    GGML_EXPERT_HANDOFF_P_COUNTS_LO  = 11, // device memory [n_layer][stride] int32 or NULL: routing counts
    GGML_EXPERT_HANDOFF_P_COUNTS_HI  = 12,
    GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE = 13,
};

// A send whose selection has no miss writes nothing to the mailbox and does not ring it, and its join returns y
// without waiting: the host is only involved in layers it has experts to compute for. The device counts the
// misses into flags[layer] (the join reads it) and every routed expert into counts (for routing profiles).

enum ggml_expert_handoff_mode {
    GGML_EXPERT_HANDOFF_SEND = 0,
    GGML_EXPERT_HANDOFF_JOIN = 1,
};

#ifdef __cplusplus
}
#endif
