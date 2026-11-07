// ROCmFPX [TAG_EXPERT_HANDOFF] the expert handoff against a plain MoE layer.
//
// A synthetic layer with random quantized experts in host memory. The reference computes it with mul_mat_id on the
// CPU backend, the way build_moe_ffn does. The handoff graph runs on the tested backend: the experts of a
// VRAM-cache-like bank (a random subset) are computed there, the others by the host pool, and the join adds them.
// Both must agree. The cases cover the expert forms build_moe_ffn has: the activations, merged gate_up, experts
// without a gate, per-expert scales and biases, routing weights applied to the input, and a used-expert count below
// the selection.
//
// usage: test-expert-handoff [--gpu] [-v]
//        test-expert-handoff --bench [type] [workers,...]   (the host pool's speed, CPU only)
//   without --gpu the handoff graph runs on the CPU backend (the protocol and the pool, no device involved)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-expert-handoff.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct test_case {
    ggml_type type_gu;
    ggml_type type_down;
    int64_t   n_embd;
    int64_t   n_ff;
    int64_t   n_expert;
    int64_t   n_used;
    int64_t   n_tokens;
    int64_t   n_slots;        // experts in the bank
    bool      merged;         // gate_up instead of gate + up
    int       act;            // ggml_expert_handoff_act
    bool      scales;         // per-expert scales on gate/up and down
    bool      biases        = false;
    bool      weight_before = false;
    int64_t   n_count       = 0; // experts summed per token, 0 = n_used
    float     alpha = 0.0f, limit = 0.0f, beta = 0.0f, lb = 0.0f;

    bool gated() const { return act < GGML_EXPERT_HANDOFF_ACT_UP_SILU; }
    int64_t count() const { return n_count > 0 ? n_count : n_used; }
};

static const char * act_name(int act) {
    switch (act) {
        case GGML_EXPERT_HANDOFF_ACT_SILU:        return "silu";
        case GGML_EXPERT_HANDOFF_ACT_GELU:        return "gelu";
        case GGML_EXPERT_HANDOFF_ACT_RELU:        return "relu";
        case GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI:  return "oai";
        case GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP:  return "silu_clamp";
        case GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT:  return "silu_limit";
        case GGML_EXPERT_HANDOFF_ACT_SITU:        return "situ";
        case GGML_EXPERT_HANDOFF_ACT_UP_SILU:     return "up_silu";
        case GGML_EXPERT_HANDOFF_ACT_UP_GELU:     return "up_gelu";
        case GGML_EXPERT_HANDOFF_ACT_UP_RELU:     return "up_relu";
        case GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR: return "up_relu_sqr";
    }
    return "?";
}

static void fill_rand(ggml_tensor * t, std::mt19937 & rng, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = ggml_nrows(t);
    std::vector<float> f(n_per_row*nrows);
    for (auto & v : f) {
        v = nd(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else {
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

struct moe_tensors {
    ggml_tensor * gate      = nullptr;
    ggml_tensor * up        = nullptr;
    ggml_tensor * gate_up   = nullptr;
    ggml_tensor * down      = nullptr;
    ggml_tensor * gate_s    = nullptr;
    ggml_tensor * up_s      = nullptr;
    ggml_tensor * down_s    = nullptr;
    ggml_tensor * gate_b    = nullptr;
    ggml_tensor * up_b      = nullptr;
    ggml_tensor * gate_up_b = nullptr;
    ggml_tensor * down_b    = nullptr;
};

// the routed experts as build_moe_ffn computes them (without the router): mm_ids selects the weights and biases
// (the experts in the reference, their slots in the handoff graph), ids the per-expert scales
static ggml_tensor * moe_graph(ggml_context * ctx, const test_case & c, const moe_tensors & m,
        ggml_tensor * x, ggml_tensor * ids, ggml_tensor * mm_ids, ggml_tensor * weights) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, c.n_embd, 1, c.n_tokens);
    if (c.weight_before) {
        x3 = ggml_mul(ctx, ggml_repeat_4d(ctx, x3, c.n_embd, c.n_used, c.n_tokens, 1), weights);
    }

    auto scaled = [&](ggml_tensor * res, ggml_tensor * s) {
        if (s == nullptr) {
            return res;
        }
        ggml_tensor * ss = ggml_reshape_3d(ctx, s, 1, c.n_expert, 1);
        ss = ggml_repeat_4d(ctx, ss, 1, c.n_expert, c.n_tokens, 1);
        ss = ggml_get_rows(ctx, ss, ids);
        return ggml_mul(ctx, res, ss);
    };
    auto biased = [&](ggml_tensor * res, ggml_tensor * b) {
        return b ? ggml_add_id(ctx, res, b, mm_ids) : res;
    };

    ggml_tensor * g = nullptr;
    ggml_tensor * u = nullptr;
    if (c.merged) {
        ggml_tensor * gu = biased(scaled(ggml_mul_mat_id(ctx, m.gate_up, x3, mm_ids), m.up_s), m.gate_up_b);
        g = ggml_view_3d(ctx, gu, c.n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
        u = ggml_view_3d(ctx, gu, c.n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], c.n_ff*gu->nb[0]);
    } else {
        u = biased(scaled(ggml_mul_mat_id(ctx, m.up, x3, mm_ids), m.up_s), m.up_b);
        if (m.gate) {
            g = biased(scaled(ggml_mul_mat_id(ctx, m.gate, x3, mm_ids), m.gate_s), m.gate_b);
        }
    }

    ggml_tensor * h = nullptr;
    switch (c.act) {
        case GGML_EXPERT_HANDOFF_ACT_SILU:       h = ggml_swiglu_split(ctx, g, u); break;
        case GGML_EXPERT_HANDOFF_ACT_GELU:       h = ggml_geglu_split(ctx, g, u); break;
        case GGML_EXPERT_HANDOFF_ACT_RELU:       h = ggml_reglu_split(ctx, g, u); break;
        case GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI: h = ggml_swiglu_oai(ctx, g, u, c.alpha, c.limit); break;
        case GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP: h = ggml_swiglu_clamp(ctx, g, u, c.limit); break;
        case GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT: {
            ggml_tensor * uc = ggml_clamp(ctx, u, -c.limit, c.limit);
            ggml_tensor * ga = ggml_clamp(ctx, ggml_silu(ctx, g), -INFINITY, c.limit);
            h = ggml_mul(ctx, ga, uc);
        } break;
        case GGML_EXPERT_HANDOFF_ACT_SITU: {
            ggml_tensor * a = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, g, 1.0f/c.beta)), c.beta);
            a = ggml_mul(ctx, a, ggml_sigmoid(ctx, g));
            ggml_tensor * ub = c.lb > 0.0f ? ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, u, 1.0f/c.lb)), c.lb) : u;
            h = ggml_mul(ctx, a, ub);
        } break;
        case GGML_EXPERT_HANDOFF_ACT_UP_SILU:     h = ggml_silu(ctx, u); break;
        case GGML_EXPERT_HANDOFF_ACT_UP_GELU:     h = ggml_gelu(ctx, u); break;
        case GGML_EXPERT_HANDOFF_ACT_UP_RELU:     h = ggml_relu(ctx, u); break;
        case GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR: h = ggml_sqr(ctx, ggml_relu(ctx, u)); break;
    }
    ggml_tensor * d = biased(scaled(ggml_mul_mat_id(ctx, m.down, h, mm_ids), m.down_s), m.down_b); // [n_embd, n_used, n_tokens]
    if (!c.weight_before) {
        d = ggml_mul(ctx, d, weights);
    }

    ggml_tensor * out = ggml_view_2d(ctx, d, c.n_embd, c.n_tokens, d->nb[2], 0);
    for (int64_t k = 1; k < c.count(); ++k) {
        out = ggml_add(ctx, out, ggml_view_2d(ctx, d, c.n_embd, c.n_tokens, d->nb[2], k*d->nb[1]));
    }
    return ggml_cont(ctx, out);
}

struct mailbox_alloc {
    void * (*alloc)(size_t) = nullptr;
    void   (*free)(void *)  = nullptr;
};

static bool run_case(ggml_backend_t be, ggml_backend_t cpu, const mailbox_alloc & ma, const test_case & c, int n_workers, bool verbose) {
    std::mt19937 rng(1234 + (int) c.type_gu*7 + (int) c.n_tokens + c.act*31);

    // host experts and inputs (CPU buffer)
    ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_host = ggml_init(ip);
    moe_tensors hm;
    if (c.merged) {
        hm.gate_up = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, 2*c.n_ff, c.n_expert);
    } else {
        if (c.gated()) {
            hm.gate = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, c.n_ff, c.n_expert);
        }
        hm.up = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, c.n_ff, c.n_expert);
    }
    hm.down = ggml_new_tensor_3d(ctx_host, c.type_down, c.n_ff, c.n_embd, c.n_expert);
    if (c.scales) {
        hm.gate_s = hm.gate ? ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert) : nullptr;
        hm.up_s   = ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert);
        hm.down_s = ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert);
    }
    if (c.biases) {
        if (c.merged) {
            hm.gate_up_b = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, 2*c.n_ff, c.n_expert);
        } else {
            hm.gate_b = hm.gate ? ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, c.n_ff, c.n_expert) : nullptr;
            hm.up_b   = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, c.n_ff, c.n_expert);
        }
        hm.down_b = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, c.n_embd, c.n_expert);
    }
    ggml_tensor * x       = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, c.n_embd, c.n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx_host, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx_host, GGML_TYPE_F32, 1, c.n_used, c.n_tokens);
    ggml_backend_buffer_t buf_host = ggml_backend_alloc_ctx_tensors(ctx_host, cpu);

    for (ggml_tensor * t : { hm.gate, hm.up, hm.gate_up, hm.down }) {
        if (t) {
            fill_rand(t, rng, 0.05f);
        }
    }
    for (ggml_tensor * t : { hm.gate_b, hm.up_b, hm.gate_up_b, hm.down_b }) {
        if (t) {
            fill_rand(t, rng, 0.1f);
        }
    }
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::vector<float> gs(c.n_expert), us(c.n_expert), ds(c.n_expert);
    for (int64_t e = 0; e < c.n_expert; ++e) {
        gs[e] = ud(rng); us[e] = ud(rng); ds[e] = ud(rng);
    }
    if (hm.gate_s) { ggml_backend_tensor_set(hm.gate_s, gs.data(), 0, gs.size()*sizeof(float)); }
    if (hm.up_s)   { ggml_backend_tensor_set(hm.up_s,   us.data(), 0, us.size()*sizeof(float)); }
    if (hm.down_s) { ggml_backend_tensor_set(hm.down_s, ds.data(), 0, ds.size()*sizeof(float)); }
    std::vector<int32_t> iv(c.n_used*c.n_tokens);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xv(c.n_embd*c.n_tokens);
        for (auto & v : xv) { v = nd(rng); }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));

        std::vector<float>   wv(c.n_used*c.n_tokens);
        std::vector<int32_t> perm(c.n_expert);
        for (int64_t t = 0; t < c.n_tokens; ++t) {
            for (int64_t e = 0; e < c.n_expert; ++e) { perm[e] = (int32_t) e; }
            std::shuffle(perm.begin(), perm.end(), rng);
            float sum = 0.0f;
            for (int64_t k = 0; k < c.n_used; ++k) {
                iv[t*c.n_used + k] = perm[k];
                wv[t*c.n_used + k] = ud(rng);
                sum += wv[t*c.n_used + k];
            }
            for (int64_t k = 0; k < c.n_used; ++k) { wv[t*c.n_used + k] /= sum; }
        }
        ggml_backend_tensor_set(ids, iv.data(), 0, iv.size()*sizeof(int32_t));
        ggml_backend_tensor_set(weights, wv.data(), 0, wv.size()*sizeof(float));
    }

    // reference on the CPU backend
    std::vector<float> ref(c.n_embd*c.n_tokens);
    {
        ggml_init_params gp = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * out = moe_graph(ctx, c, hm, x, ids, ids, weights);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(cpu, gf);
        ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    // the banks on the tested backend: n_slots random experts, then the zeroed miss slot and padding;
    // weights [k, n, slots] and biases [n, slots], selected by slot like the expert handoff's cache
    std::vector<int32_t> slot_of(c.n_expert, (int32_t) c.n_slots);
    {
        std::vector<int32_t> perm(c.n_expert);
        for (int64_t e = 0; e < c.n_expert; ++e) { perm[e] = (int32_t) e; }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t s = 0; s < c.n_slots; ++s) { slot_of[perm[s]] = (int32_t) s; }
    }
    ggml_context * ctx_dev = ggml_init(ip);
    moe_tensors dm; // the bank views of the experts (and biases)
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> host_bank; // host tensor, bank
    auto bank_of = [&](ggml_tensor * t) -> ggml_tensor * {
        if (t == nullptr) {
            return nullptr;
        }
        ggml_tensor * b = ggml_n_dims(t) >= 3 ? ggml_new_tensor_3d(ctx_dev, t->type, t->ne[0], t->ne[1], c.n_slots + 2)
                                               : ggml_new_tensor_2d(ctx_dev, t->type, t->ne[0], c.n_slots + 2);
        host_bank.emplace_back(t, b);
        return b;
    };
    ggml_tensor * bg  = bank_of(hm.gate);
    ggml_tensor * bu  = bank_of(hm.up);
    ggml_tensor * bgu = bank_of(hm.gate_up);
    ggml_tensor * bd  = bank_of(hm.down);
    ggml_tensor * bgb = bank_of(hm.gate_b);
    ggml_tensor * bub = bank_of(hm.up_b);
    ggml_tensor * bgub = bank_of(hm.gate_up_b);
    ggml_tensor * bdb = bank_of(hm.down_b);
    ggml_tensor * slot_map = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_I32, 1, c.n_expert);
    ggml_tensor * dx  = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_F32, c.n_embd, c.n_tokens);
    ggml_tensor * dids = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * dw  = ggml_new_tensor_3d(ctx_dev, GGML_TYPE_F32, 1, c.n_used, c.n_tokens);
    ggml_tensor * dgs = hm.gate_s ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * dus = hm.up_s   ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * dds = hm.down_s ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * dflags  = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, 1);
    ggml_tensor * dcounts = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, c.n_expert);
    ggml_backend_buffer_t buf_dev = ggml_backend_alloc_ctx_tensors(ctx_dev, be);
    ggml_backend_buffer_clear(buf_dev, 0);
    for (const auto & [t, b] : host_bank) {
        const size_t es = ggml_n_dims(t) >= 3 ? t->nb[2] : t->nb[1];
        for (int64_t e = 0; e < c.n_expert; ++e) {
            if (slot_of[e] < c.n_slots) {
                ggml_backend_tensor_set(b, (const uint8_t *) t->data + e*es, slot_of[e]*es, es);
            }
        }
    }
    ggml_backend_tensor_set(slot_map, slot_of.data(), 0, slot_of.size()*sizeof(int32_t));
    ggml_backend_tensor_copy(x, dx);
    ggml_backend_tensor_copy(ids, dids);
    ggml_backend_tensor_copy(weights, dw);
    if (dgs) { ggml_backend_tensor_copy(hm.gate_s, dgs); }
    if (dus) { ggml_backend_tensor_copy(hm.up_s,   dus); }
    if (dds) { ggml_backend_tensor_copy(hm.down_s, dds); }

    // the mailbox and the pool
    const ggml_expert_handoff_dims dims = { 32, (int32_t) c.n_embd, (int32_t) c.n_used };
    const size_t mb_size = ggml_expert_handoff_size(dims);
    void * mb = ma.alloc ? ma.alloc(mb_size) : nullptr;
    std::vector<uint8_t> mb_host;
    if (!ma.alloc) {
        mb_host.assign(mb_size + 256, 0);
        mb = (void *) (((uintptr_t) mb_host.data() + 255) & ~(uintptr_t) 255);
    }
    if (mb == nullptr) {
        fprintf(stderr, "  mailbox allocation failed\n");
        return false;
    }
    ggml_expert_handoff_host_t host = ggml_expert_handoff_host_new(mb, dims.max_tokens, dims.max_embd, dims.max_used, 1, n_workers);
    ggml_expert_handoff_layer hl = {};
    hl.gate = hm.gate; hl.up = hm.up; hl.gate_up = hm.gate_up; hl.down = hm.down;
    hl.gate_scale = hm.gate_s ? gs.data() : nullptr;
    hl.up_scale   = hm.up_s   ? us.data() : nullptr;
    hl.down_scale = hm.down_s ? ds.data() : nullptr;
    hl.gate_bias  = hm.gate_b ? (const float *) hm.gate_b->data : nullptr;
    hl.up_bias    = hm.gate_up_b ? (const float *) hm.gate_up_b->data : hm.up_b ? (const float *) hm.up_b->data : nullptr;
    hl.down_bias  = hm.down_b ? (const float *) hm.down_b->data : nullptr;
    hl.act = (ggml_expert_handoff_act) c.act;
    hl.alpha = c.alpha; hl.limit = c.limit; hl.beta = c.beta; hl.lb = c.lb;
    hl.weight_before = c.weight_before;
    hl.n_used = (int32_t) c.n_count;
    hl.miss_slot = (int32_t) c.n_slots;
    if (!ggml_expert_handoff_host_set_layer(host, 0, &hl)) {
        fprintf(stderr, "  the pool rejected the layer\n");
        ggml_expert_handoff_host_free(host);
        return false;
    }

    // the handoff graph on the tested backend, run three times (the sequence numbers must advance)
    ggml_expert_handoff_params ehp = {};
    ehp.mailbox       = mb;
    ehp.layer         = 0;
    ehp.max_tokens    = dims.max_tokens;
    ehp.max_embd      = dims.max_embd;
    ehp.max_used      = dims.max_used;
    ehp.miss_slot     = (int32_t) c.n_slots;
    ehp.flags         = (int32_t *) dflags->data;
    ehp.counts        = (int32_t *) dcounts->data;
    ehp.counts_stride = (int32_t) c.n_expert;
    ehp.timeout_ms    = 2000;
    const int n_rep = 3;
    std::vector<float> got(c.n_embd*c.n_tokens);
    bool ok = true;
    double nmse = 0.0;
    int rep = 0;
    for (; rep < n_rep && ok; ++rep) {
        ggml_init_params gp = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * sent = ggml_expert_handoff_send(ctx, dx, dids, slot_map, dw, &ehp);
        auto view = [&](ggml_tensor * b) -> ggml_tensor * {
            if (b == nullptr) {
                return nullptr;
            }
            return ggml_n_dims(b) >= 3 ? ggml_view_3d(ctx, b, b->ne[0], b->ne[1], c.n_slots + 1, b->nb[1], b->nb[2], 0)
                                       : ggml_view_2d(ctx, b, b->ne[0], c.n_slots + 1, b->nb[1], 0);
        };
        dm.gate = view(bg); dm.up = view(bu); dm.gate_up = view(bgu); dm.down = view(bd);
        dm.gate_b = view(bgb); dm.up_b = view(bub); dm.gate_up_b = view(bgub); dm.down_b = view(bdb);
        dm.gate_s = dgs; dm.up_s = dus; dm.down_s = dds;
        ggml_tensor * out = moe_graph(ctx, c, dm, dx, dids, sent, dw);
        out = ggml_expert_handoff_join(ctx, out, &ehp);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_expert_handoff_host_wake(host);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "  graph compute failed\n");
            ok = false;
        }
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(ga);
        ggml_free(ctx);

        double err = 0.0, norm = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            err  += (got[i] - ref[i])*(double) (got[i] - ref[i]);
            norm += ref[i]*(double) ref[i];
        }
        nmse = norm > 0.0 ? err/norm : err;
        if (!(nmse < 5e-4)) {
            ok = false;
        }
    }

    const uint32_t timeouts = ((ggml_expert_handoff_header *) mb)->n_timeouts;
    ggml_expert_handoff_stats st = {};
    ggml_expert_handoff_host_get_stats(host, &st);
    ggml_expert_handoff_host_free(host);
    if (timeouts != 0) {
        ok = false;
    }

    // the device counted every routed expert of every run; a run without misses never reached the host; the host
    // computed the misses among the counted experts (after all runs; a failed run already says why)
    if (ok && rep == n_rep) {
        std::vector<int32_t> cnt(c.n_expert), want(c.n_expert, 0);
        ggml_backend_tensor_get(dcounts, cnt.data(), 0, cnt.size()*sizeof(int32_t));
        int64_t n_miss_all = 0, n_miss = 0;
        for (int64_t t = 0; t < c.n_tokens; ++t) {
            for (int64_t k = 0; k < c.n_used; ++k) {
                const int32_t e = iv[t*c.n_used + k];
                want[e] += n_rep;
                n_miss_all += slot_of[e] == c.n_slots;
                n_miss     += slot_of[e] == c.n_slots && k < c.count();
            }
        }
        if (cnt != want) {
            printf("  routing counts differ\n");
            ok = false;
        }
        if (n_miss_all == 0 && st.n_layers != 0) {
            printf("  a run without misses was sent to the host\n");
            ok = false;
        }
        if (st.n_host != (uint64_t) (n_miss*n_rep)) {
            printf("  the host computed %llu experts, expected %lld\n", (unsigned long long) st.n_host, (long long) (n_miss*n_rep));
            ok = false;
        }
    }

    printf("  %-7s/%-7s %-7s %-11s embd %4lld ff %4lld exp %3lld used %2lld%s tok %2lld cached %3lld%s%s%s: nmse %.2e, host %llu, timeouts %u  %s\n",
        ggml_type_name(c.type_gu), ggml_type_name(c.type_down), c.merged ? "gate_up" : (c.gated() ? "gate+up" : "up"),
        act_name(c.act), (long long) c.n_embd, (long long) c.n_ff, (long long) c.n_expert, (long long) c.n_used,
        c.n_count ? (" count " + std::to_string(c.n_count)).c_str() : "", (long long) c.n_tokens, (long long) c.n_slots,
        c.scales ? " scales" : "", c.biases ? " biases" : "", c.weight_before ? " w-before" : "",
        nmse, (unsigned long long) st.n_host, timeouts, ok ? "OK" : "FAIL");
    if (verbose && !ok) {
        for (int i = 0; i < 8 && i < (int) ref.size(); ++i) {
            printf("    [%d] ref %.6f got %.6f\n", i, ref[i], got[i]);
        }
    }

    if (ma.free) {
        ma.free(mb);
    }
    ggml_backend_buffer_free(buf_dev);
    ggml_free(ctx_dev);
    ggml_backend_buffer_free(buf_host);
    ggml_free(ctx_host);
    return ok;
}

// --bench: the host pool's speed with the experts in RAM, the device's side of the mailbox played by this thread.
// Random experts of several layers (more than the caches hold), so every expert comes from DRAM. Next to it, the
// read bandwidth of the same threads on a plain buffer.
static double bench_read_gbps(int n_threads, size_t bytes) {
    std::vector<uint64_t> buf(bytes/8);
    for (size_t i = 0; i < buf.size(); ++i) { buf[i] = i; }
    double best = 0.0;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<std::thread> th;
        std::vector<uint64_t> sums(n_threads*8);
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < n_threads; ++t) {
            th.emplace_back([&, t] {
                const size_t i0 = buf.size()*t/n_threads, i1 = buf.size()*(t + 1)/n_threads;
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t i = i0; i + 3 < i1; i += 4) { s0 += buf[i]; s1 += buf[i+1]; s2 += buf[i+2]; s3 += buf[i+3]; }
                sums[t*8] = s0 + s1 + s2 + s3;
            });
        }
        for (auto & x : th) { x.join(); }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::max(best, bytes/s/1e9);
    }
    return best;
}

static void bench_fill(ggml_tensor * t, std::mt19937 & rng) {
    // random quants with a small scale in front of every block (the types benched here start with an f16 scale)
    uint8_t * p = (uint8_t *) t->data;
    const size_t n = ggml_nbytes(t);
    for (size_t i = 0; i + 4 <= n; i += 4) { const uint32_t r = rng(); memcpy(p + i, &r, 4); }
    const size_t bs = ggml_type_size(t->type);
    const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
    for (size_t i = 0; i + bs <= n; i += bs) { memcpy(p + i, &d, sizeof(d)); }
}

// [TAG_EH_POOL_WAKE] a job that outlasts the pool's spin window, then the next layer's send at once: the workers other
// than 0 used to fall asleep past the job's last barrier while worker 0 still spun, took the send and waited at the
// next barrier for workers nothing woke (every later layer of the graph timed out; 2026-11-07, Flash-Next in prod).
// The window runs out here by idle() during the job, as it did there by a job longer than the window.
static bool test_pool_wake() {
    const ggml_type type = GGML_TYPE_Q8_0;
    const int64_t n_embd = 2560, n_ff = 640, n_expert = 16;
    const int n_tok = 32, n_used = 8;
    std::mt19937 rng(7);
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<std::vector<uint8_t>> mem;
    ggml_tensor * g = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
    ggml_tensor * u = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
    ggml_tensor * d = ggml_new_tensor_3d(ctx, type, n_ff, n_embd, n_expert);
    for (ggml_tensor * t : { g, u, d }) {
        mem.emplace_back(ggml_nbytes(t) + 64);
        t->data = (void *) (((uintptr_t) mem.back().data() + 63) & ~(uintptr_t) 63);
        bench_fill(t, rng);
    }
    ggml_expert_handoff_layer hl = {};
    hl.gate = g; hl.up = u; hl.down = d;
    hl.act = GGML_EXPERT_HANDOFF_ACT_SILU;
    hl.miss_slot = 0;

    const ggml_expert_handoff_dims dims = { 32, (int32_t) n_embd, n_used };
    std::vector<uint8_t> mb_mem(ggml_expert_handoff_size(dims) + 256, 0);
    uint8_t * mb = (uint8_t *) (((uintptr_t) mb_mem.data() + 255) & ~(uintptr_t) 255);
    auto * hdr = (ggml_expert_handoff_header *) mb;
    auto sq_load  = [](const uint32_t * p) { return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(std::memory_order_acquire); };
    auto sq_store = [](uint32_t * p, uint32_t v) { reinterpret_cast<std::atomic<uint32_t> *>(p)->store(v, std::memory_order_release); };
    {
        float * x = (float *) (mb + ggml_expert_handoff_off_x(dims));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int64_t i = 0; i < (int64_t) n_tok*n_embd; ++i) { x[i] = nd(rng); }
        int32_t * ids   = (int32_t *) (mb + ggml_expert_handoff_off_ids(dims));
        int32_t * slots = (int32_t *) (mb + ggml_expert_handoff_off_slots(dims));
        float   * w     = (float   *) (mb + ggml_expert_handoff_off_weights(dims));
        for (int t = 0; t < n_tok; ++t) {
            for (int k = 0; k < n_used; ++k) {
                ids[t*dims.max_used + k] = (t + k) % n_expert; slots[t*dims.max_used + k] = 0; w[t*dims.max_used + k] = 1.0f/n_used;
            }
        }
        hdr->layer = 0; hdr->n_tokens = n_tok; hdr->n_embd = (int32_t) n_embd; hdr->n_used = n_used;
    }

    bool ok = true;
    for (int nw : { 2, 6 }) {
        ggml_expert_handoff_host_t host = ggml_expert_handoff_host_new(mb, dims.max_tokens, dims.max_embd, dims.max_used, 1, nw);
        ggml_expert_handoff_host_set_layer(host, 0, &hl);
        int n_late = 0;
        double worst_ms = 0.0;
        for (int round = 0; round < 20; ++round) {
            // a quiet spell: the pool falls asleep (between graphs, or before a graph's first send)
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            // the first send: worker 0's poll finds it and wakes the others
            uint32_t s = sq_load(&hdr->seq_send) + 1;
            sq_store(&hdr->seq_send, s);
            // the spin window runs out while the job runs, as it did for a job longer than the window
            std::this_thread::sleep_for(std::chrono::microseconds(2500));
            ggml_expert_handoff_host_idle(host);
            while (sq_load(&hdr->seq_done) != s) { }
            // the next layers, each as soon as the answer before is in (the device's join, then its next send)
            for (int layer = 0; layer < 3; ++layer) {
                s = sq_load(&hdr->seq_send) + 1;
                sq_store(&hdr->seq_send, s);
                const auto t0 = std::chrono::steady_clock::now();
                double ms = 0.0;
                while (sq_load(&hdr->seq_done) != s && ms < 500.0) {
                    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
                worst_ms = std::max(worst_ms, ms);
                if (sq_load(&hdr->seq_done) != s) {
                    n_late++;
                    ggml_expert_handoff_host_wake(host); // unstick it, as the next graph did
                    while (sq_load(&hdr->seq_done) != s) { }
                }
            }
        }
        ggml_expert_handoff_host_free(host);
        printf("  pool wake, %d workers: %d of 60 sends after an expired spin window unanswered for 500 ms (worst %.1f ms): %s\n",
            nw, n_late, worst_ms, n_late == 0 ? "OK" : "FAIL");
        ok = ok && n_late == 0;
    }
    ggml_free(ctx);
    return ok;
}

static int run_bench(ggml_type type, int64_t n_embd, int64_t n_ff, int64_t n_expert, int n_layer,
        const std::vector<int> & workers, const std::vector<int> & miss_list, const std::vector<int> & token_list) {
    printf("bench: %s experts %lld x %lld, %lld per layer, %d layers (%.2f MiB per expert, %.1f GiB)\n",
        ggml_type_name(type), (long long) n_embd, (long long) n_ff, (long long) n_expert, n_layer,
        3.0*ggml_row_size(type, n_embd)*n_ff/1048576.0, 3.0*ggml_row_size(type, n_embd)*n_ff*n_expert*n_layer/1073741824.0);
    std::mt19937 rng(42);
    ggml_init_params ip = { (size_t) (3*n_layer + 4)*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<std::vector<uint8_t>> mem;
    std::vector<ggml_expert_handoff_layer> hls(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * g = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
        ggml_tensor * u = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
        ggml_tensor * d = ggml_new_tensor_3d(ctx, type, n_ff, n_embd, n_expert);
        for (ggml_tensor * t : { g, u, d }) {
            mem.emplace_back(ggml_nbytes(t) + 64);
            t->data = (void *) (((uintptr_t) mem.back().data() + 63) & ~(uintptr_t) 63);
            bench_fill(t, rng);
        }
        ggml_expert_handoff_layer & hl = hls[il];
        hl = {};
        hl.gate = g; hl.up = u; hl.down = d;
        hl.act = GGML_EXPERT_HANDOFF_ACT_SILU;
        hl.miss_slot = 0;
    }
    const int max_used = *std::max_element(miss_list.begin(), miss_list.end());
    const int max_tok  = *std::max_element(token_list.begin(), token_list.end());
    const ggml_expert_handoff_dims dims = { 32, (int32_t) n_embd, (int32_t) max_used };
    std::vector<uint8_t> mb_mem(ggml_expert_handoff_size(dims) + 256, 0);
    uint8_t * mb = (uint8_t *) (((uintptr_t) mb_mem.data() + 255) & ~(uintptr_t) 255);
    auto * hdr = (ggml_expert_handoff_header *) mb;
    auto sq_load  = [](const uint32_t * p) { return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(std::memory_order_acquire); };
    auto sq_store = [](uint32_t * p, uint32_t v) { reinterpret_cast<std::atomic<uint32_t> *>(p)->store(v, std::memory_order_release); };
    {
        float * x = (float *) (mb + ggml_expert_handoff_off_x(dims));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int64_t i = 0; i < (int64_t) max_tok*n_embd; ++i) { x[i] = nd(rng); }
    }
    const size_t expert_bytes = 3*ggml_row_size(type, n_embd)*n_ff;
    for (int nw : workers) {
        printf("  %2d workers, reading a plain buffer: %.1f GB/s\n", nw, bench_read_gbps(nw, (size_t) 2 << 30));
        ggml_expert_handoff_host_t host = ggml_expert_handoff_host_new(mb, dims.max_tokens, dims.max_embd, dims.max_used, n_layer, nw);
        for (int il = 0; il < n_layer; ++il) {
            ggml_expert_handoff_host_set_layer(host, il, &hls[il]);
        }
        for (int nt : token_list) {
          for (int shared = 0; shared <= (nt > 1 ? 1 : 0); ++shared) {
            for (int nm : miss_list) {
                const int n_iter = 400;
                double t_sum = 0.0, t_min = 1e9;
                size_t bytes = 0;
                int32_t * ids   = (int32_t *) (mb + ggml_expert_handoff_off_ids(dims));
                int32_t * slots = (int32_t *) (mb + ggml_expert_handoff_off_slots(dims));
                float   * w     = (float   *) (mb + ggml_expert_handoff_off_weights(dims));
                std::vector<int32_t> perm(n_expert);
                for (int it = -20; it < n_iter; ++it) {
                    const int il = (int) (rng() % n_layer);
                    std::vector<char> seen(n_expert, 0);
                    size_t uniq = 0;
                    for (int t = 0; t < nt; ++t) {
                        // shared: every token misses the same experts (a batch's tokens often route alike)
                        if (t == 0 || !shared) {
                            for (int64_t e = 0; e < n_expert; ++e) { perm[e] = (int32_t) e; }
                            for (int k = 0; k < nm; ++k) {
                                std::swap(perm[k], perm[k + rng() % (n_expert - k)]);
                            }
                        }
                        for (int k = 0; k < nm; ++k) {
                            ids[t*dims.max_used + k] = perm[k];
                            slots[t*dims.max_used + k] = 0;
                            w[t*dims.max_used + k] = 1.0f/nm;
                            uniq += !seen[perm[k]];
                            seen[perm[k]] = 1;
                        }
                    }
                    hdr->layer = il; hdr->n_tokens = nt; hdr->n_embd = (int32_t) n_embd; hdr->n_used = nm;
                    ggml_expert_handoff_host_wake(host);
                    const auto t0 = std::chrono::steady_clock::now();
                    const uint32_t s = sq_load(&hdr->seq_send) + 1;
                    sq_store(&hdr->seq_send, s);
                    while (sq_load(&hdr->seq_done) != s) { }
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    if (it >= 0) {
                        t_sum += us; t_min = std::min(t_min, us); bytes += uniq*expert_bytes;
                    }
                }
                const double t_avg = t_sum/n_iter;
                printf("    %2d tokens x %2d misses%s: %8.1f us per layer (min %7.1f), %6.1f us per expert read, %5.1f GB/s\n",
                    nt, nm, shared ? " (shared)" : "         ", t_avg, t_min, t_sum/(bytes/(double) expert_bytes), bytes/(t_sum*1e-6)/1e9);
            }
          }
        }
        ggml_expert_handoff_host_free(host);
    }
    ggml_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool use_gpu = false;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0) { use_gpu = true; }
        if (strcmp(argv[i], "-v") == 0)    { verbose = true; }
        if (strcmp(argv[i], "--check-dots") == 0) {
            // the CPU vec_dot of the ROCmFPX types against dequantize + float dot: random raw blocks (every code,
            // valid scales) and quantized random rows, rows of 1, 2, 3 and 80 blocks
            ggml_backend_load_all();
            struct fmt { ggml_type t; int n_scales; };
            const fmt fmts[] = {
                { GGML_TYPE_Q2_0_ROCMFPX, 2 }, { GGML_TYPE_Q3_0_ROCMFPX, 2 }, { GGML_TYPE_Q5_0_ROCMFPX, 2 },
                { GGML_TYPE_Q6_0_ROCMFPX, 2 }, { GGML_TYPE_Q7_0_ROCMFPX, 2 }, { GGML_TYPE_Q8_0_ROCMFPX, 1 },
                { GGML_TYPE_Q4_0_ROCMI4, 1 },  { GGML_TYPE_Q4_0_ROCMFP4, 0 }, { GGML_TYPE_Q4_0_ROCMFP4_FAST, 0 },
                { GGML_TYPE_Q4_0, 0 },
            };
            std::mt19937 rng(11);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            int n_fail = 0;
            for (const fmt & f : fmts) {
                const auto * tr   = ggml_get_type_traits_cpu(f.t);
                const auto * base = ggml_get_type_traits(f.t);
                const auto * q8   = ggml_get_type_traits(GGML_TYPE_Q8_0);
                double worst = 0.0;
                for (int raw = 0; raw < 2; ++raw) {
                    if (raw && f.n_scales == 0) {
                        continue; // only quantized rows for the formats whose scales this test does not know
                    }
                    for (int64_t n : { 32, 64, 96, 2560 }) {
                        for (int rep = 0; rep < 50; ++rep) {
                            const size_t row = ggml_row_size(f.t, n);
                            std::vector<uint8_t> xq(row);
                            std::vector<float> xf(n), yf(n);
                            for (auto & v : yf) { v = nd(rng); }
                            if (raw) {
                                for (auto & b : xq) { b = (uint8_t) rng(); }
                                const size_t bsz = ggml_type_size(f.t);
                                for (size_t o = 0; o < row; o += bsz) {
                                    for (int k = 0; k < f.n_scales; ++k) {
                                        xq[o + bsz - f.n_scales + k] = (uint8_t) (rng() % 0x7f);
                                    }
                                }
                            } else {
                                for (auto & v : xf) { v = nd(rng); }
                                tr->from_float(xf.data(), xq.data(), n);
                            }
                            std::vector<uint8_t> yq(ggml_row_size(GGML_TYPE_Q8_0, n));
                            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(yf.data(), yq.data(), n);
                            std::vector<float> xd(n), yd(n);
                            base->to_float(xq.data(), xd.data(), n);
                            q8->to_float(yq.data(), yd.data(), n);
                            double ref = 0.0, mag = 0.0;
                            for (int64_t j = 0; j < n; ++j) { ref += (double) xd[j]*yd[j]; mag += fabs((double) xd[j]*yd[j]); }
                            float got = 0.0f;
                            tr->vec_dot((int) n, &got, 0, xq.data(), 0, yq.data(), 0, 1);
                            const double err = fabs(got - ref)/(mag + 1e-30);
                            worst = std::max(worst, err);
                        }
                    }
                }
                const bool ok = worst < 1e-5;
                n_fail += !ok;
                printf("  %-20s worst relative error %.2e  %s\n", ggml_type_name(f.t), worst, ok ? "OK" : "FAIL");
            }
            printf("%s\n", n_fail == 0 ? "all passed" : "FAILED");
            return n_fail == 0 ? 0 : 1;
        }
        if (strcmp(argv[i], "--bench-gpu") == 0) {
            // the round trip on the device: 48 layers of send -> join in one graph, Flash-Next's expert shape, the
            // host computing 0, 1 or 2 missed experts per layer from RAM; time per layer against the host's own time
            ggml_backend_load_all();
            ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            if (gpu == nullptr) { fprintf(stderr, "no GPU backend\n"); return 1; }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
            auto mb_alloc = (void * (*)(size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_alloc");
            auto mb_free  = (void   (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_free");
            const int n_layer = 48, n_data = 4, n_used = 10, n_workers = i + 1 < argc ? atoi(argv[i + 1]) : 6;
            const int64_t n_embd = 2560, n_ff = 640, n_expert = 512;
            std::mt19937 rng(3);
            ggml_init_params hp = { (size_t) (3*n_data + 4)*ggml_tensor_overhead(), nullptr, true };
            ggml_context * hctx = ggml_init(hp);
            std::vector<std::vector<uint8_t>> mem;
            std::vector<ggml_expert_handoff_layer> hls(n_data);
            for (int il = 0; il < n_data; ++il) {
                ggml_tensor * g = ggml_new_tensor_3d(hctx, GGML_TYPE_Q2_0, n_embd, n_ff, n_expert);
                ggml_tensor * u = ggml_new_tensor_3d(hctx, GGML_TYPE_Q2_0, n_embd, n_ff, n_expert);
                ggml_tensor * d = ggml_new_tensor_3d(hctx, GGML_TYPE_Q2_0, n_ff, n_embd, n_expert);
                for (ggml_tensor * t : { g, u, d }) {
                    mem.emplace_back(ggml_nbytes(t) + 64);
                    t->data = (void *) (((uintptr_t) mem.back().data() + 63) & ~(uintptr_t) 63);
                    bench_fill(t, rng);
                }
                hls[il] = {};
                hls[il].gate = g; hls[il].up = u; hls[il].down = d;
                hls[il].act = GGML_EXPERT_HANDOFF_ACT_SILU;
                hls[il].miss_slot = 1;
            }
            const ggml_expert_handoff_dims dims = { 32, (int32_t) n_embd, n_used };
            void * mb = mb_alloc(ggml_expert_handoff_size(dims));
            ggml_expert_handoff_host_t host = ggml_expert_handoff_host_new(mb, dims.max_tokens, dims.max_embd, dims.max_used, n_layer, n_workers);
            for (int il = 0; il < n_layer; ++il) {
                ggml_expert_handoff_host_set_layer(host, il, &hls[il % n_data]);
            }
            ggml_init_params dp = { 16*ggml_tensor_overhead(), nullptr, true };
            ggml_context * dctx = ggml_init(dp);
            ggml_tensor * dx     = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, n_embd, 1);
            ggml_tensor * dids   = ggml_new_tensor_2d(dctx, GGML_TYPE_I32, n_used, 1);
            ggml_tensor * dw     = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, 1, n_used, 1);
            ggml_tensor * dmap   = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, n_expert);
            ggml_tensor * dflags = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, n_layer);
            ggml_tensor * dcnt   = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, n_layer*n_expert);
            ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(dctx, gpu);
            ggml_backend_buffer_clear(dbuf, 0);
            {
                std::vector<float> xv(n_embd);
                for (auto & v : xv) { v = (float) (rng() % 1000)/1000.0f - 0.5f; }
                ggml_backend_tensor_set(dx, xv.data(), 0, xv.size()*sizeof(float));
                std::vector<float> wv(n_used, 0.1f);
                ggml_backend_tensor_set(dw, wv.data(), 0, wv.size()*sizeof(float));
            }
            for (int n_miss : { 0, 1, 2 }) {
                // experts 0..9 selected; the first n_miss of them at the miss slot
                std::vector<int32_t> iv(n_used), map(n_expert, 0);
                for (int k = 0; k < n_used; ++k) { iv[k] = (int32_t) (rng() % n_expert); }
                for (int k = 0; k < n_miss; ++k) { map[iv[k]] = 1; }
                ggml_backend_tensor_set(dids, iv.data(), 0, iv.size()*sizeof(int32_t));
                ggml_backend_tensor_set(dmap, map.data(), 0, map.size()*sizeof(int32_t));
                ggml_init_params gp = { (size_t) (4*n_layer + 8)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * gctx = ggml_init(gp);
                ggml_cgraph * gf = ggml_new_graph(gctx);
                ggml_tensor * cur = dx;
                for (int il = 0; il < n_layer; ++il) {
                    ggml_expert_handoff_params ehp = {};
                    ehp.mailbox = mb; ehp.layer = il; ehp.max_tokens = dims.max_tokens; ehp.max_embd = dims.max_embd;
                    ehp.max_used = dims.max_used; ehp.miss_slot = 1; ehp.flags = (int32_t *) dflags->data;
                    ehp.counts = (int32_t *) dcnt->data; ehp.counts_stride = (int32_t) n_expert; ehp.timeout_ms = 1000;
                    ggml_tensor * s = ggml_expert_handoff_send(gctx, cur, dids, dmap, dw, &ehp);
                    ggml_build_forward_expand(gf, s);
                    cur = ggml_expert_handoff_join(gctx, cur, &ehp);
                    ggml_build_forward_expand(gf, cur);
                }
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
                ggml_gallocr_alloc_graph(ga, gf);
                ggml_expert_handoff_stats st0 = {}, st1 = {};
                const int n_iter = 50;
                double best = 1e9, sum = 0.0;
                for (int it = -5; it < n_iter; ++it) {
                    if (it == 0) { ggml_expert_handoff_host_get_stats(host, &st0); }
                    ggml_expert_handoff_host_wake(host);
                    const auto t0 = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute(gpu, gf);
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    if (it >= 0) { sum += us; best = std::min(best, us); }
                }
                ggml_expert_handoff_host_get_stats(host, &st1);
                const double host_us = st1.n_layers > st0.n_layers ? (st1.busy_ns - st0.busy_ns)*1e-3/(st1.n_layers - st0.n_layers) : 0.0;
                printf("%d workers, %d of %d experts missed per layer: %7.1f us per layer (best %7.1f); the host %6.1f us of it per sent layer\n",
                    n_workers, n_miss, n_used, sum/n_iter/n_layer, best/n_layer, host_us);
                ggml_gallocr_free(ga);
                ggml_free(gctx);
            }
            const uint32_t to = ((ggml_expert_handoff_header *) mb)->n_timeouts;
            printf("timeouts: %u\n", to);
            ggml_expert_handoff_host_free(host);
            mb_free(mb);
            ggml_backend_buffer_free(dbuf);
            ggml_free(dctx);
            ggml_free(hctx);
            ggml_backend_free(gpu);
            return 0;
        }
        if (strcmp(argv[i], "--bench-dot") == 0) {
            // one thread, one vec_dot type: rows of 2560 (gate/up) and 640 (down) from L1/L2, then from DRAM
            ggml_backend_load_all();
            ggml_type type = GGML_TYPE_Q2_0;
            if (i + 1 < argc) {
                for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                    const char * n = ggml_type_name((ggml_type) t);
                    if (n && strcmp(n, argv[i + 1]) == 0) { type = (ggml_type) t; }
                }
            }
            const auto * tr = ggml_get_type_traits_cpu(type);
            std::mt19937 rng(7);
            for (int64_t n : { 2560, 640 }) {
                for (size_t span : { (size_t) 64 << 10, (size_t) 1 << 30 }) {
                    const size_t row = ggml_row_size(type, n);
                    const size_t n_rows = std::max<size_t>(1, span/row);
                    std::vector<uint8_t> w(n_rows*row + 64);
                    for (auto & b : w) { b = (uint8_t) rng(); }
                    const size_t bs = ggml_type_size(type);
                    const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
                    for (size_t k = 0; k + bs <= w.size() - 64; k += bs) { memcpy(w.data() + k, &d, sizeof(d)); }
                    std::vector<float> xf(n);
                    for (auto & v : xf) { v = (float) (rng() % 1000)/500.0f - 1.0f; }
                    std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, n));
                    ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(xf.data(), xq.data(), n);
                    float acc = 0.0f;
                    const size_t n_calls = span < ((size_t) 1 << 20) ? 2000000 : n_rows;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (size_t c = 0; c < n_calls; ++c) {
                        float s;
                        tr->vec_dot((int) n, &s, 0, w.data() + (c % n_rows)*row, 0, xq.data(), 0, 1);
                        acc += s;
                    }
                    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    printf("%s rows of %4lld from %s: %6.1f ns per row, %5.1f GB/s (%g)\n", ggml_type_name(type), (long long) n,
                        span < ((size_t) 1 << 20) ? "cache" : "DRAM ", sec*1e9/n_calls, n_calls*row/sec/1e9, acc);
                }
            }
            return 0;
        }
        if (strcmp(argv[i], "--bench") == 0) {
            // --bench [type] [workers,...]: Flash-Next's expert shape (2560 x 640, 512 per layer)
            ggml_backend_load_all();
            ggml_type type = GGML_TYPE_Q2_0;
            std::vector<int> workers = { 2, 4, 6, 8, 11 };
            if (i + 1 < argc) {
                for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                    const char * n = ggml_type_name((ggml_type) t);
                    if (n && strcmp(n, argv[i + 1]) == 0) { type = (ggml_type) t; }
                }
            }
            if (i + 2 < argc) {
                workers.clear();
                for (const char * p = argv[i + 2]; *p; ) { workers.push_back(atoi(p)); while (*p && *p != ',') { ++p; } if (*p) { ++p; } }
            }
            return run_bench(type, 2560, 640, 512, 4, workers, { 1, 2, 4, 10 }, { 1, 4 });
        }
    }

    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t be  = cpu;
    mailbox_alloc ma;
    if (use_gpu) {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (be == nullptr) {
            fprintf(stderr, "no GPU backend\n");
            return 1;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
        ma.alloc = (void * (*)(size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_alloc");
        ma.free  = (void   (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_free");
        if (!ma.alloc || !ma.free) {
            fprintf(stderr, "%s cannot map host memory\n", ggml_backend_name(be));
            return 1;
        }
    }
    printf("expert handoff on %s\n", ggml_backend_name(be));

    const int SILU = GGML_EXPERT_HANDOFF_ACT_SILU;
    const int GELU = GGML_EXPERT_HANDOFF_ACT_GELU;
    std::vector<test_case> cases = {
        // type_gu          type_down         embd  ff  exp used tok slots merged act scales
        { GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,    256, 128, 16, 4,  1,  8, false, SILU, false },
        { GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,    256, 128, 16, 4,  5,  8, false, SILU, false },
        { GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0,    256, 128, 16, 4,  3,  0, false, SILU, false }, // nothing cached
        { GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0,    256, 128, 16, 4,  3, 16, false, SILU, false }, // everything cached
        { GGML_TYPE_Q4_K,   GGML_TYPE_Q6_K,    512, 256, 32, 8,  2, 12, false, SILU, false },
        { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS,  512, 256, 32, 8,  4, 12, false, SILU, false },
        { GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,    256, 128, 16, 8,  2,  6, true,  GELU, true  }, // gate_up, gelu, scales
        { GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K,    512, 256, 32, 8, 16, 10, false, GELU, true  },
        { GGML_TYPE_F32,    GGML_TYPE_F32,     128,  64,  8, 2,  1,  3, false, SILU, false },
        { GGML_TYPE_Q2_0,   GGML_TYPE_Q2_0,    512, 128, 32, 8,  4, 12, false, SILU, false }, // Flash-Next's experts
        { GGML_TYPE_Q2_0,   GGML_TYPE_Q2_0,    512, 128, 32, 8, 12,  4, false, SILU, false }, // tokens sharing misses
    };
    auto add = [&](test_case c) { cases.push_back(c); return &cases.back(); };
    {   // gpt-oss: swiglu_oai with biases on every projection
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 3, 6, false, GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI, false });
        c->biases = true; c->alpha = 1.702f; c->limit = 7.0f;
    }
    {   // the same with MXFP4 experts
        test_case * c = add({ GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, 256, 128, 16, 4, 2, 6, false, GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI, false });
        c->biases = true; c->alpha = 1.702f; c->limit = 7.0f;
    }
    {   // merged gate_up with its bias and scales
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 2, 5, true, SILU, true });
        c->biases = true;
    }
    {   // DeepSeek V4 / GLM-5 style clamp (a limit inside the test's value range, so it clamps; on the GPU the
        // device's own activation rounding then shows a little more)
        test_case * c = add({ GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 512, 256, 32, 8, 3, 12, false, GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP, false });
        c->limit = 0.15f;
    }
    {   // the other clamp: silu limited, up clamped
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 3, 6, false, GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT, false });
        c->limit = 0.15f;
    }
    {   // Kimi K3: situ
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 2, 6, false, GGML_EXPERT_HANDOFF_ACT_SITU, false });
        c->beta = 0.5f; c->lb = 0.3f;
    }
    add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 2, 6, false, GGML_EXPERT_HANDOFF_ACT_RELU,        false });
    add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 2, 6, false, GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR, false }); // Nemotron-H
    add({ GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, 256, 128, 16, 4, 2, 6, false, GGML_EXPERT_HANDOFF_ACT_UP_SILU,     false });
    {   // llama4: the routing weights scale the input
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 2, 3, 6, false, SILU, false });
        c->weight_before = true;
    }
    {   // fewer experts counted than selected
        test_case * c = add({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128, 16, 8, 3, 6, false, SILU, false });
        c->n_count = 6;
    }

    int n_fail = 0;
    for (int n_workers : { 1, 4 }) {
        printf("pool workers: %d\n", n_workers);
        for (const auto & c : cases) {
            if (!run_case(be, cpu, ma, c, n_workers, verbose)) {
                n_fail++;
            }
        }
    }
    // the pool itself, on the host either way [TAG_EH_POOL_WAKE]
    if (!test_pool_wake()) {
        n_fail++;
    }

    if (be != cpu) {
        ggml_backend_free(be);
    }
    ggml_backend_free(cpu);
    printf("%s\n", n_fail == 0 ? "all passed" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
