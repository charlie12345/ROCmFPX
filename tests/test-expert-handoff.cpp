// ROCmFPX [TAG_EXPERT_HANDOFF] the expert handoff against a plain MoE layer.
//
// A synthetic layer with random quantized experts in host memory. The reference computes it with mul_mat_id on the
// CPU backend. The handoff graph runs on the tested backend: the experts of a VRAM-cache-like bank (a random subset)
// are computed there, the others by the host pool, and the join adds them. Both must agree.
//
// usage: test-expert-handoff [--gpu] [-v]
//   without --gpu the handoff graph runs on the CPU backend (the protocol and the pool, no device involved)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-expert-handoff.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct test_case {
    ggml_type type_gu;
    ggml_type type_down;
    int64_t   n_embd;
    int64_t   n_ff;
    int64_t   n_expert;
    int64_t   n_used;
    int64_t   n_tokens;
    int64_t   n_slots;   // experts in the bank
    bool      merged;    // gate_up instead of gate + up
    int       act;       // ggml_expert_handoff_act
    bool      scales;    // per-expert scales on gate/up and down
};

static void fill_quant(ggml_tensor * t, std::mt19937 & rng, float sd) {
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

// the routed experts as the graph computes them (build_moe_ffn without the router)
static ggml_tensor * moe_graph(ggml_context * ctx, const test_case & c, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * gate_up,
        ggml_tensor * down, ggml_tensor * gate_s, ggml_tensor * up_s, ggml_tensor * down_s,
        ggml_tensor * x, ggml_tensor * ids, ggml_tensor * mm_ids, ggml_tensor * weights) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, c.n_embd, 1, c.n_tokens);

    auto scaled = [&](ggml_tensor * res, ggml_tensor * s) {
        if (s == nullptr) {
            return res;
        }
        ggml_tensor * ss = ggml_reshape_3d(ctx, s, 1, c.n_expert, 1);
        ss = ggml_repeat_4d(ctx, ss, 1, c.n_expert, c.n_tokens, 1);
        ss = ggml_get_rows(ctx, ss, ids);
        return ggml_mul(ctx, res, ss);
    };

    ggml_tensor * g = nullptr;
    ggml_tensor * u = nullptr;
    if (c.merged) {
        ggml_tensor * gu = scaled(ggml_mul_mat_id(ctx, gate_up, x3, mm_ids), up_s);
        g = ggml_view_3d(ctx, gu, c.n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
        u = ggml_view_3d(ctx, gu, c.n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], c.n_ff*gu->nb[0]);
    } else {
        u = scaled(ggml_mul_mat_id(ctx, up,   x3, mm_ids), up_s);
        g = scaled(ggml_mul_mat_id(ctx, gate, x3, mm_ids), gate_s);
    }
    ggml_tensor * h = c.act == GGML_EXPERT_HANDOFF_ACT_GELU ? ggml_geglu_split(ctx, g, u) : ggml_swiglu_split(ctx, g, u);
    ggml_tensor * d = scaled(ggml_mul_mat_id(ctx, down, h, mm_ids), down_s); // [n_embd, n_used, n_tokens]
    d = ggml_mul(ctx, d, weights);

    ggml_tensor * out = ggml_view_2d(ctx, d, c.n_embd, c.n_tokens, d->nb[2], 0);
    for (int64_t k = 1; k < c.n_used; ++k) {
        out = ggml_add(ctx, out, ggml_view_2d(ctx, d, c.n_embd, c.n_tokens, d->nb[2], k*d->nb[1]));
    }
    return ggml_cont(ctx, out);
}

struct mailbox_alloc {
    void * (*alloc)(size_t) = nullptr;
    void   (*free)(void *)  = nullptr;
};

static bool run_case(ggml_backend_t be, ggml_backend_t cpu, const mailbox_alloc & ma, const test_case & c, int n_workers, bool verbose) {
    std::mt19937 rng(1234 + (int) c.type_gu*7 + (int) c.n_tokens);

    // host experts and inputs (CPU buffer)
    ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_host = ggml_init(ip);
    ggml_tensor * gate = nullptr, * up = nullptr, * gate_up = nullptr;
    if (c.merged) {
        gate_up = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, 2*c.n_ff, c.n_expert);
    } else {
        gate = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, c.n_ff, c.n_expert);
        up   = ggml_new_tensor_3d(ctx_host, c.type_gu, c.n_embd, c.n_ff, c.n_expert);
    }
    ggml_tensor * down    = ggml_new_tensor_3d(ctx_host, c.type_down, c.n_ff, c.n_embd, c.n_expert);
    ggml_tensor * gate_s  = c.scales && !c.merged ? ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * up_s    = c.scales ? ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * down_s  = c.scales ? ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * x       = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, c.n_embd, c.n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx_host, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx_host, GGML_TYPE_F32, 1, c.n_used, c.n_tokens);
    ggml_backend_buffer_t buf_host = ggml_backend_alloc_ctx_tensors(ctx_host, cpu);

    for (ggml_tensor * t : { gate, up, gate_up, down }) {
        if (t) {
            fill_quant(t, rng, 0.05f);
        }
    }
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::vector<float> gs(c.n_expert), us(c.n_expert), ds(c.n_expert);
    for (int64_t e = 0; e < c.n_expert; ++e) {
        gs[e] = ud(rng); us[e] = ud(rng); ds[e] = ud(rng);
    }
    if (gate_s) { ggml_backend_tensor_set(gate_s, gs.data(), 0, gs.size()*sizeof(float)); }
    if (up_s)   { ggml_backend_tensor_set(up_s,   us.data(), 0, us.size()*sizeof(float)); }
    if (down_s) { ggml_backend_tensor_set(down_s, ds.data(), 0, ds.size()*sizeof(float)); }
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xv(c.n_embd*c.n_tokens);
        for (auto & v : xv) { v = nd(rng); }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));

        std::vector<int32_t> iv(c.n_used*c.n_tokens);
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
        ggml_tensor * out = moe_graph(ctx, c, gate, up, gate_up, down, gate_s, up_s, down_s, x, ids, ids, weights);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(cpu, gf);
        ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    // the bank on the tested backend: n_slots random experts, then the zeroed miss slot and padding
    std::vector<int32_t> slot_of(c.n_expert, (int32_t) c.n_slots);
    {
        std::vector<int32_t> perm(c.n_expert);
        for (int64_t e = 0; e < c.n_expert; ++e) { perm[e] = (int32_t) e; }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t s = 0; s < c.n_slots; ++s) { slot_of[perm[s]] = (int32_t) s; }
    }
    ggml_context * ctx_dev = ggml_init(ip);
    std::vector<ggml_tensor *> host_t, bank_t;
    for (ggml_tensor * t : { gate, up, gate_up, down }) {
        if (t) {
            host_t.push_back(t);
            bank_t.push_back(ggml_new_tensor_3d(ctx_dev, t->type, t->ne[0], t->ne[1], c.n_slots + 2));
        }
    }
    ggml_tensor * slot_map = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_I32, 1, c.n_expert);
    ggml_tensor * dx  = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_F32, c.n_embd, c.n_tokens);
    ggml_tensor * dids = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_tensor * dw  = ggml_new_tensor_3d(ctx_dev, GGML_TYPE_F32, 1, c.n_used, c.n_tokens);
    ggml_tensor * dgs = gate_s ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * dus = up_s   ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_tensor * dds = down_s ? ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, c.n_expert) : nullptr;
    ggml_backend_buffer_t buf_dev = ggml_backend_alloc_ctx_tensors(ctx_dev, be);
    ggml_backend_buffer_clear(buf_dev, 0);
    for (size_t i = 0; i < host_t.size(); ++i) {
        for (int64_t e = 0; e < c.n_expert; ++e) {
            if (slot_of[e] < c.n_slots) {
                ggml_backend_tensor_set(bank_t[i], (const uint8_t *) host_t[i]->data + e*host_t[i]->nb[2], slot_of[e]*host_t[i]->nb[2], host_t[i]->nb[2]);
            }
        }
    }
    ggml_backend_tensor_set(slot_map, slot_of.data(), 0, slot_of.size()*sizeof(int32_t));
    ggml_backend_tensor_copy(x, dx);
    ggml_backend_tensor_copy(ids, dids);
    ggml_backend_tensor_copy(weights, dw);
    if (dgs) { ggml_backend_tensor_copy(gate_s, dgs); }
    if (dus) { ggml_backend_tensor_copy(up_s,   dus); }
    if (dds) { ggml_backend_tensor_copy(down_s, dds); }

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
    hl.gate = gate; hl.up = up; hl.gate_up = gate_up; hl.down = down;
    hl.gate_scale = gate_s ? gs.data() : nullptr;
    hl.up_scale   = up_s   ? us.data() : nullptr;
    hl.down_scale = down_s ? ds.data() : nullptr;
    hl.act = (ggml_expert_handoff_act) c.act;
    hl.miss_slot = (int32_t) c.n_slots;
    if (!ggml_expert_handoff_host_set_layer(host, 0, &hl)) {
        fprintf(stderr, "  the pool rejected the layer\n");
        ggml_expert_handoff_host_free(host);
        return false;
    }

    // the handoff graph on the tested backend, run three times (the sequence numbers must advance)
    std::vector<float> got(c.n_embd*c.n_tokens);
    bool ok = true;
    double nmse = 0.0;
    for (int rep = 0; rep < 3 && ok; ++rep) {
        ggml_init_params gp = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * sl = ggml_get_rows(ctx, slot_map, ggml_reshape_1d(ctx, dids, c.n_used*c.n_tokens));
        sl = ggml_reshape_2d(ctx, sl, c.n_used, c.n_tokens);
        ggml_tensor * sent = ggml_expert_handoff_send(ctx, dx, dids, sl, dw, mb, 0, dims.max_tokens, dims.max_embd, dims.max_used);
        std::vector<ggml_tensor *> views;
        for (ggml_tensor * b : bank_t) {
            views.push_back(ggml_view_3d(ctx, b, b->ne[0], b->ne[1], c.n_slots + 1, b->nb[1], b->nb[2], 0));
        }
        size_t iv = 0;
        ggml_tensor * vg  = c.merged ? nullptr : views[iv++];
        ggml_tensor * vu  = c.merged ? nullptr : views[iv++];
        ggml_tensor * vgu = c.merged ? views[iv++] : nullptr;
        ggml_tensor * vd  = views[iv++];
        ggml_tensor * out = moe_graph(ctx, c, vg, vu, vgu, vd, dgs, dus, dds, dx, dids, sent, dw);
        out = ggml_expert_handoff_join(ctx, out, mb, 0, dims.max_tokens, dims.max_embd, dims.max_used, 2000);
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

    printf("  %-7s/%-7s %s %-4s embd %4lld ff %4lld experts %3lld used %2lld tokens %2lld cached %3lld%s: nmse %.2e, host %llu/%llu, timeouts %u  %s\n",
        ggml_type_name(c.type_gu), ggml_type_name(c.type_down), c.merged ? "gate_up" : "gate+up",
        c.act == GGML_EXPERT_HANDOFF_ACT_GELU ? "gelu" : "silu", (long long) c.n_embd, (long long) c.n_ff,
        (long long) c.n_expert, (long long) c.n_used, (long long) c.n_tokens, (long long) c.n_slots, c.scales ? " scales" : "",
        nmse, (unsigned long long) st.n_host, (unsigned long long) st.n_routed, timeouts, ok ? "OK" : "FAIL");
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

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool use_gpu = false;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0) { use_gpu = true; }
        if (strcmp(argv[i], "-v") == 0)    { verbose = true; }
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
    };

    int n_fail = 0;
    for (int n_workers : { 1, 4 }) {
        printf("pool workers: %d\n", n_workers);
        for (const auto & c : cases) {
            if (verbose) {
                printf("  case %s ...\n", ggml_type_name(c.type_gu));
            }
            if (!run_case(be, cpu, ma, c, n_workers, verbose)) {
                n_fail++;
            }
        }
    }

    if (be != cpu) {
        ggml_backend_free(be);
    }
    ggml_backend_free(cpu);
    printf("%s\n", n_fail == 0 ? "all passed" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
