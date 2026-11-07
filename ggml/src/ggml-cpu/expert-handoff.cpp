// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff, the host side (protocol in ggml-expert-handoff.h):
//  - ggml_compute_forward_expert_handoff: the send and join ops when the graph itself runs on the CPU backend
//  - ggml_expert_handoff_host_*: the pool of threads that answers a mailbox by computing the routed experts the
//    device does not hold, in place from host memory, while the device computes the ones it holds

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-expert-handoff.h"
#include "ggml-impl.h"
#include "ops.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define EH_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define EH_PAUSE() __asm__ __volatile__("yield")
#else
#define EH_PAUSE() std::this_thread::yield()
#endif

// the mailbox fields are plain uint32_t shared with the device: access them as atomics
static inline uint32_t eh_load(const uint32_t * p) {
    return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(std::memory_order_acquire);
}
static inline void eh_store(uint32_t * p, uint32_t v) {
    reinterpret_cast<std::atomic<uint32_t> *>(p)->store(v, std::memory_order_release);
}

static inline int64_t eh_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_expert_handoff_dims eh_dims(const ggml_tensor * dst) {
    ggml_expert_handoff_dims d;
    d.max_tokens = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_TOKENS);
    d.max_embd   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_EMBD);
    d.max_used   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_USED);
    return d;
}

static void * eh_ptr(const ggml_tensor * dst, int lo_i, int hi_i) {
    const uint64_t lo = (uint32_t) ggml_get_op_params_i32(dst, lo_i);
    const uint64_t hi = (uint32_t) ggml_get_op_params_i32(dst, hi_i);
    return (void *) (uintptr_t) (lo | (hi << 32));
}

static uint8_t * eh_mailbox(const ggml_tensor * dst) {
    return (uint8_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_PTR_LO, GGML_EXPERT_HANDOFF_P_PTR_HI);
}

//
// the ops on the CPU backend
//

void ggml_compute_forward_expert_handoff(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }

    uint8_t * mb = eh_mailbox(dst);
    const ggml_expert_handoff_dims d = eh_dims(dst);
    auto * hdr = (ggml_expert_handoff_header *) mb;
    const int32_t layer = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_LAYER);
    int32_t * flags = (int32_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_FLAGS_LO, GGML_EXPERT_HANDOFF_P_FLAGS_HI);

    if (ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MODE) == GGML_EXPERT_HANDOFF_SEND) {
        const ggml_tensor * x     = dst->src[0];
        const ggml_tensor * ids   = dst->src[1];
        const ggml_tensor * map   = dst->src[2];
        const ggml_tensor * w     = dst->src[3];
        int32_t * slots = (int32_t *) dst->data;

        const int64_t n_embd   = x->ne[0];
        const int64_t n_tokens = ggml_nrows(x);
        const int64_t n_used   = ids->ne[0];

        // the misses, and the routing counts
        const int32_t miss_slot = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MISS_SLOT);
        int32_t * counts = (int32_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_COUNTS_LO, GGML_EXPERT_HANDOFF_P_COUNTS_HI);
        const int64_t stride = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE);
        const int64_t n_map = ggml_nelements(map);
        int32_t n_miss = 0;
        for (int64_t i = 0; i < n_used*n_tokens; ++i) {
            const int32_t e = ((const int32_t *) ids->data)[i];
            slots[i] = e >= 0 && e < n_map ? ((const int32_t *) map->data)[e] : miss_slot;
            n_miss += slots[i] == miss_slot;
            if (counts && e >= 0 && e < stride) {
                counts[layer*stride + e]++;
            }
        }
        flags[layer] = n_miss;
        if (n_miss == 0) {
            return;
        }

        float   * mx = (float   *) (mb + ggml_expert_handoff_off_x(d));
        int32_t * mi = (int32_t *) (mb + ggml_expert_handoff_off_ids(d));
        int32_t * ms = (int32_t *) (mb + ggml_expert_handoff_off_slots(d));
        float   * mw = (float   *) (mb + ggml_expert_handoff_off_weights(d));

        for (int64_t t = 0; t < n_tokens; ++t) {
            memcpy(mx + t*d.max_embd, (const float *) x->data + t*n_embd, n_embd*sizeof(float));
            memcpy(mi + t*d.max_used, (const int32_t *) ids->data   + t*n_used, n_used*sizeof(int32_t));
            memcpy(ms + t*d.max_used, slots + t*n_used, n_used*sizeof(int32_t));
            memcpy(mw + t*d.max_used, (const float   *) w->data     + t*n_used, n_used*sizeof(float));
        }
        hdr->layer    = layer;
        hdr->n_tokens = (int32_t) n_tokens;
        hdr->n_embd   = (int32_t) n_embd;
        hdr->n_used   = (int32_t) n_used;

        eh_store(&hdr->seq_send, eh_load(&hdr->seq_send) + 1);
        return;
    }

    // join
    const ggml_tensor * y = dst->src[0];
    const int64_t n_embd   = y->ne[0];
    const int64_t n_tokens = ggml_nrows(y);

    if (flags[layer] == 0) {
        // nothing was sent
        memcpy(dst->data, y->data, ggml_nbytes(y));
        return;
    }

    const uint32_t want = eh_load(&hdr->seq_send);
    const int64_t timeout_ns = (int64_t) ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_TIMEOUT_MS)*1000000;
    const int64_t t0 = eh_now_ns();
    bool ok = true;
    uint32_t done = 0;
    while ((done = eh_load(&hdr->seq_done)) != want) {
        if (eh_now_ns() - t0 > timeout_ns) {
            ok = false;
            break;
        }
        EH_PAUSE();
    }
    if (!ok) {
        // [TAG_EH_TIMEOUT_DIAG] what it waited for and what it saw, for check()
        eh_store(&hdr->to_want, want);
        eh_store(&hdr->to_done, done);
        eh_store((uint32_t *) &hdr->to_layer, (uint32_t) layer);
        eh_store(&hdr->n_timeouts, eh_load(&hdr->n_timeouts) + 1);
    }

    const float * my = (const float *) (mb + ggml_expert_handoff_off_y(d));
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * src = (const float *) y->data + t*n_embd;
        float       * out = (float       *) dst->data + t*n_embd;
        for (int64_t i = 0; i < n_embd; ++i) {
            out[i] = src[i] + (ok ? my[t*d.max_embd + i] : 0.0f);
        }
    }
}

//
// the host pool
//

namespace {

struct eh_barrier {
    std::atomic<int> count{0};
    std::atomic<int> gen{0};
    int n = 1;
    const std::atomic<bool> * stop = nullptr; // a stopping pool leaves its barriers

    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            gen.store(g + 1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) {
                if (stop && stop->load(std::memory_order_relaxed)) {
                    return;
                }
                EH_PAUSE();
            }
        }
    }
};

struct eh_layer {
    bool    set = false;
    ggml_expert_handoff_layer l = {};
    int64_t n_embd   = 0;
    int64_t n_ff     = 0;
    int64_t n_expert = 0;
    bool    has_gate = true;
    int64_t gu_rows  = 0; // rows of the gate/up phase per expert: 2*n_ff, or n_ff without a gate
    ggml_type type_gu   = GGML_TYPE_COUNT; // gate/up weights
    ggml_type type_down = GGML_TYPE_COUNT;
    ggml_type vt_gu     = GGML_TYPE_COUNT; // their vec_dot types
    ggml_type vt_down   = GGML_TYPE_COUNT;
    std::vector<uint64_t> counts;
};

// one job: the misses of one layer
struct eh_miss {
    int32_t t;      // token
    int32_t e;      // expert
    float   w;      // routing weight
};

} // namespace

struct ggml_expert_handoff_host {
    uint8_t * mb = nullptr;
    ggml_expert_handoff_dims dims = {};
    ggml_expert_handoff_header * hdr = nullptr;

    std::vector<eh_layer> layers;

    int n_workers = 1;
    std::vector<std::thread> threads;
    eh_barrier barrier;

    std::atomic<bool> stop{false};
    std::atomic<int64_t> spin_until{0};
    std::mutex mtx;
    std::condition_variable cv;
    uint64_t wake_gen = 0; // under mtx
    // [TAG_EH_POOL_WAKE] the workers asleep in wait_work (worker 0 polls instead), and when the pool last answered
    std::atomic<int>     n_asleep{0};
    std::atomic<int64_t> t_active{0};
    static constexpr int64_t active_ns = 1000*1000*1000; // in use: answered within the last second

    // spin this long after the last work or wake; the context also calls idle() once a graph's results are in, so the
    // spinning threads do not take the CPU from whatever runs between graphs (an MTP draft's own CPU work)
    static constexpr int64_t linger_ns = 20*1000*1000;

    uint32_t seq_base = 0; // seq_send when the pool was made

    // measuring aids (environment, read when the pool is made; the answers are wrong with either):
    // ROCMFPX_EH_SIM_RESIDENT=N reads expert e's weights from expert e % N of the layer, so a model bigger than RAM
    //   runs with the host's compute and memory traffic of one whose experts all fit in RAM (no page faults);
    // ROCMFPX_EH_SIM_NOHOST=1 answers every send with zeros at once: the device's side alone
    int64_t sim_resident = 0;
    bool    sim_nohost   = false;
    int64_t de(int32_t e) const { return sim_resident > 0 ? e % sim_resident : e; }

    // the job, prepared by worker 0 between barriers
    uint32_t seq = 0;
    const eh_layer * job_layer = nullptr;
    int32_t n_tokens = 0;
    std::vector<eh_miss> misses;
    std::vector<int32_t> miss_tokens;      // the tokens with misses
    std::vector<int32_t> uniq;             // the missed experts, each once
    std::vector<int32_t> uniq_begin;       // per missed expert: its first entry in by_expert, then the end
    std::vector<int32_t> by_expert;        // miss indices grouped by expert (a batch's tokens share an expert's reads)
    std::vector<int32_t> uniq_of;          // [n_expert] scratch: index in uniq or -1
    std::vector<uint8_t> xq;               // [max_tokens] rows of the gate/up vec_dot type
    size_t xq_row = 0;
    std::vector<float>   gu;               // [n_miss][2*n_ff] gate then up outputs
    std::vector<uint8_t> hq;               // [n_miss] rows of the down vec_dot type
    size_t hq_row = 0;
    std::vector<float>   hf;               // [n_workers][n_ff] f32 scratch for the activation

    // stats
    std::atomic<uint64_t> st_layers{0};
    std::atomic<uint64_t> st_routed{0};
    std::atomic<uint64_t> st_host{0};
    std::atomic<uint64_t> st_busy_ns{0};
    int64_t t_seen = 0;
    // [TAG_EH_TIMEOUT_DIAG] worker 0 writes these; get_stats may read a job's fields torn, which only blurs a diagnostic
    std::atomic<uint64_t> st_multi{0};
    std::atomic<uint64_t> st_busy_multi{0};
    std::atomic<uint64_t> st_skipped{0};
    std::atomic<uint64_t> st_max_ns{0};
    std::atomic<int32_t>  st_max_layer{-1};
    std::atomic<int32_t>  st_max_tokens{0};
    std::atomic<int32_t>  st_max_miss{0};
    std::atomic<int32_t>  st_max_uniq{0};
    std::atomic<uint64_t> st_max_step[4] = {};
    int32_t job_il = -1;
    int64_t t_step[4] = {};      // worker 0: the end of each step of the job in run()

    void wake() {
        spin_until.store(eh_now_ns() + linger_ns, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx);
            wake_gen++;
        }
        cv.notify_all();
    }

    // keep the pool spinning until at least t
    void extend_spin(int64_t t) {
        int64_t cur = spin_until.load(std::memory_order_relaxed);
        while (cur < t && !spin_until.compare_exchange_weak(cur, t, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

    // wait for a new send; false if the pool stops. After a quiet spell the workers sleep, but worker 0 keeps looking
    // at the mailbox every millisecond and wakes them: a send can come long after the wake (the first kernels of a
    // cold device load for hundreds of milliseconds), and the device cannot wake a sleeping pool itself.
    // [TAG_EH_POOL_WAKE] A worker can also fall asleep while worker 0 still spins: a job that ends after the spin window
    // (a cold one, or one that follows a layer the device uploaded instead) lets the others pass the job's last barrier
    // with the window run out, before worker 0 extends it. Worker 0 then took the next send while spinning, without
    // waking anyone, and waited at the barrier for workers nothing woke until the context's next graph: every later
    // layer of that graph gave up after the join's timeout (2026-11-07, Flash-Next in prod: every batch of more than one
    // token). So the worker that finds a send wakes the sleepers (worker()), the workers extend the window before that
    // barrier, and a sleeper of a pool in use waits a millisecond at most before it looks at the mailbox itself.
    bool wait_work(int iw, uint32_t seen, uint64_t & my_gen, uint32_t & s) {
        for (;;) {
            s = eh_load(&hdr->seq_send);
            if (s != seen) {
                return true;
            }
            if (stop.load(std::memory_order_acquire)) {
                return false;
            }
            if (eh_now_ns() > spin_until.load(std::memory_order_acquire)) {
                if (iw == 0) {
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait_for(lk, std::chrono::milliseconds(1), [&] { return stop.load() || wake_gen != my_gen; });
                        my_gen = wake_gen;
                    }
                    if (eh_load(&hdr->seq_send) != seen) {
                        wake();
                        std::lock_guard<std::mutex> lk(mtx);
                        my_gen = wake_gen;
                    }
                } else {
                    // registered before it looks at the mailbox again, so a worker that finds a send after this look
                    // also sees it asleep and wakes it
                    n_asleep.fetch_add(1, std::memory_order_seq_cst);
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        auto ready = [&] {
                            return stop.load() || wake_gen != my_gen || eh_load(&hdr->seq_send) != seen;
                        };
                        if (eh_now_ns() - t_active.load(std::memory_order_relaxed) < active_ns) {
                            cv.wait_for(lk, std::chrono::milliseconds(1), ready);
                        } else {
                            cv.wait(lk, ready);
                        }
                        my_gen = wake_gen;
                    }
                    n_asleep.fetch_sub(1, std::memory_order_seq_cst);
                }
                continue;
            }
            for (int i = 0; i < 64; ++i) {
                EH_PAUSE();
            }
        }
    }

    // worker 0: read the send, list the misses
    void prepare(uint32_t s) {
        // [TAG_EH_TIMEOUT_DIAG] each send bumps seq_send by one: a bigger step means sends this pool never took up
        if (s - seq > 1) {
            st_skipped.fetch_add(s - seq - 1, std::memory_order_relaxed);
        }
        seq = s;
        t_seen = eh_now_ns();

        const int32_t il = hdr->layer;
        job_il = il;
        n_tokens         = hdr->n_tokens;
        const int32_t n_used = hdr->n_used;

        job_layer = (il >= 0 && il < (int32_t) layers.size() && layers[il].set) ? &layers[il] : nullptr;
        misses.clear();
        miss_tokens.clear();
        uniq.clear();
        uniq_begin.clear();
        by_expert.clear();
        if (job_layer == nullptr || hdr->n_embd != job_layer->n_embd || n_tokens > dims.max_tokens || n_used > dims.max_used) {
            if (job_layer == nullptr) {
                GGML_LOG_ERROR("%s: no experts registered for layer %d, answering zero\n", __func__, il);
            } else {
                GGML_LOG_ERROR("%s: layer %d: mailbox shape mismatch, answering zero\n", __func__, il);
            }
            job_layer = nullptr;
            return;
        }

        const int32_t * ids   = (const int32_t *) (mb + ggml_expert_handoff_off_ids(dims));
        const int32_t * slots = (const int32_t *) (mb + ggml_expert_handoff_off_slots(dims));
        const float   * w     = (const float   *) (mb + ggml_expert_handoff_off_weights(dims));
        eh_layer & L = layers[il];

        const int32_t n_count = L.l.n_used > 0 ? std::min(n_used, L.l.n_used) : n_used;
        for (int32_t t = 0; t < n_tokens; ++t) {
            bool any = false;
            for (int32_t k = 0; k < n_count; ++k) {
                const int32_t e = ids[t*dims.max_used + k];
                if (e < 0 || e >= L.n_expert) {
                    continue;
                }
                L.counts[e]++;
                if (slots[t*dims.max_used + k] == L.l.miss_slot) {
                    misses.push_back({ t, e, w[t*dims.max_used + k] });
                    any = true;
                }
            }
            if (any) {
                miss_tokens.push_back(t);
            }
        }

        // the misses by expert: each expert's rows are read once for all the tokens that miss it
        uniq_of.assign(L.n_expert, -1);
        std::vector<int32_t> cnt;
        for (const eh_miss & ms : misses) {
            if (uniq_of[ms.e] < 0) {
                uniq_of[ms.e] = (int32_t) uniq.size();
                uniq.push_back(ms.e);
                cnt.push_back(0);
            }
            cnt[uniq_of[ms.e]]++;
        }
        uniq_begin.assign(uniq.size() + 1, 0);
        for (size_t u = 0; u < uniq.size(); ++u) {
            uniq_begin[u + 1] = uniq_begin[u] + cnt[u];
        }
        by_expert.assign(misses.size(), 0);
        {
            std::vector<int32_t> fill(uniq_begin.begin(), uniq_begin.end() - 1);
            for (size_t m = 0; m < misses.size(); ++m) {
                by_expert[fill[uniq_of[misses[m].e]]++] = (int32_t) m;
            }
        }

        st_layers.fetch_add(1, std::memory_order_relaxed);
        st_routed.fetch_add((uint64_t) n_tokens*n_used, std::memory_order_relaxed);
        st_host.fetch_add(misses.size(), std::memory_order_relaxed);

        // scratch
        const eh_layer & J = *job_layer;
        xq_row = ggml_row_size(J.vt_gu, J.n_embd);
        hq_row = ggml_row_size(J.vt_down, J.n_ff);
        if (xq.size() < (size_t) n_tokens*xq_row) {
            xq.resize((size_t) n_tokens*xq_row);
        }
        if (gu.size() < misses.size()*J.gu_rows) {
            gu.resize(misses.size()*J.gu_rows);
        }
        if (hq.size() < misses.size()*hq_row) {
            hq.resize(misses.size()*hq_row);
        }
        if (hf.size() < (size_t) n_workers*J.n_ff) {
            hf.resize((size_t) n_workers*J.n_ff);
        }
    }

    // gate/up row r (0 .. gu_rows - 1: gate rows first, or the up rows alone without a gate) of expert e
    const uint8_t * gu_row(const eh_layer & J, int32_t e, int64_t r) const {
        const int64_t ed = de(e);
        if (J.l.gate_up) {
            const ggml_tensor * t = J.l.gate_up;
            return (const uint8_t *) t->data + ed*t->nb[2] + r*t->nb[1];
        }
        if (!J.has_gate) {
            const ggml_tensor * t = J.l.up;
            return (const uint8_t *) t->data + ed*t->nb[2] + r*t->nb[1];
        }
        const ggml_tensor * t = r < J.n_ff ? J.l.gate : J.l.up;
        return (const uint8_t *) t->data + ed*t->nb[2] + (r % J.n_ff)*t->nb[1];
    }

    float gu_scale(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up || !J.has_gate) {
            return J.l.up_scale ? J.l.up_scale[e] : 1.0f;
        }
        const float * s = r < J.n_ff ? J.l.gate_scale : J.l.up_scale;
        return s ? s[e] : 1.0f;
    }

    float gu_bias(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up || !J.has_gate) {
            return J.l.up_bias ? J.l.up_bias[e*J.gu_rows + r] : 0.0f;
        }
        const float * b = r < J.n_ff ? J.l.gate_bias : J.l.up_bias;
        return b ? b[e*J.n_ff + r % J.n_ff] : 0.0f;
    }

    // the activation of one expert: gu holds gate then up (or up alone), h gets n_ff values
    static void activate(const eh_layer & J, const float * gu, float * h) {
        const int64_t n_ff = J.n_ff;
        const ggml_expert_handoff_layer & l = J.l;
        auto silu    = [](float v) { return v/(1.0f + expf(-v)); };
        auto gelu    = [](float v) { return 0.5f*v*(1.0f + tanhf(0.7978845608028654f*v*(1.0f + 0.044715f*v*v))); };
        auto sigmoid = [](float v) { return 1.0f/(1.0f + expf(-v)); };
        if (!J.has_gate) {
            for (int64_t i = 0; i < n_ff; ++i) {
                const float u = gu[i];
                switch (l.act) {
                    case GGML_EXPERT_HANDOFF_ACT_UP_SILU:     h[i] = silu(u); break;
                    case GGML_EXPERT_HANDOFF_ACT_UP_GELU:     h[i] = gelu(u); break;
                    case GGML_EXPERT_HANDOFF_ACT_UP_RELU:     h[i] = std::max(u, 0.0f); break;
                    case GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR: { const float r = std::max(u, 0.0f); h[i] = r*r; } break;
                    default: h[i] = 0.0f; break;
                }
            }
            return;
        }
        const float * g = gu;
        const float * u = gu + n_ff;
        for (int64_t i = 0; i < n_ff; ++i) {
            switch (l.act) {
                case GGML_EXPERT_HANDOFF_ACT_SILU: h[i] = silu(g[i])*u[i]; break;
                case GGML_EXPERT_HANDOFF_ACT_GELU: h[i] = gelu(g[i])*u[i]; break;
                case GGML_EXPERT_HANDOFF_ACT_RELU: h[i] = std::max(g[i], 0.0f)*u[i]; break;
                case GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI: {
                    const float x = std::min(g[i], l.limit);
                    const float y = std::clamp(u[i], -l.limit, l.limit);
                    h[i] = x/(1.0f + expf(-l.alpha*x))*(y + 1.0f);
                } break;
                case GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP:
                    h[i] = silu(std::min(g[i], l.limit))*std::clamp(u[i], -l.limit, l.limit);
                    break;
                case GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT:
                    h[i] = std::min(silu(g[i]), l.limit)*std::clamp(u[i], -l.limit, l.limit);
                    break;
                case GGML_EXPERT_HANDOFF_ACT_SITU: {
                    const float a = l.beta*tanhf(g[i]/l.beta)*sigmoid(g[i]);
                    h[i] = a*(l.lb > 0.0f ? l.lb*tanhf(u[i]/l.lb) : u[i]);
                } break;
                default: h[i] = 0.0f; break;
            }
        }
    }

    void run(int iw) {
        const eh_layer & J = *job_layer;
        const auto * tr_gu   = ggml_get_type_traits_cpu(J.type_gu);
        const auto * tr_down = ggml_get_type_traits_cpu(J.type_down);
        const auto * tr_vgu  = ggml_get_type_traits_cpu(J.vt_gu);
        const auto * tr_vdn  = ggml_get_type_traits_cpu(J.vt_down);
        const float * x = (const float *) (mb + ggml_expert_handoff_off_x(dims));
        float       * y = (float       *) (mb + ggml_expert_handoff_off_y(dims));
        const int64_t n_embd = J.n_embd;
        const int64_t n_ff   = J.n_ff;
        const int64_t n_miss = (int64_t) misses.size();

        // 1. the inputs of the tokens with misses, in the gate/up dot type
        for (size_t i = iw; i < miss_tokens.size(); i += n_workers) {
            const int32_t t = miss_tokens[i];
            uint8_t * dst = xq.data() + t*xq_row;
            if (J.vt_gu == GGML_TYPE_F32) {
                memcpy(dst, x + (int64_t) t*dims.max_embd, n_embd*sizeof(float));
            } else {
                tr_vgu->from_float(x + (int64_t) t*dims.max_embd, dst, n_embd);
            }
        }
        barrier.wait();
        if (iw == 0) {
            t_step[0] = eh_now_ns(); // [TAG_EH_TIMEOUT_DIAG]
        }

        // 2. gate and up rows of every missed expert, each row against all the tokens that miss it while it is in
        //    the cache (with weight_before, the routing weight scales the input: it is linear up to here, so it
        //    scales the dot products)
        {
            const int64_t rows  = J.gu_rows;
            const int64_t total = (int64_t) uniq.size()*rows;
            const int64_t i0 = total*iw/n_workers;
            const int64_t i1 = total*(iw + 1)/n_workers;
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t u = i / rows;
                const int64_t r = i % rows;
                const int32_t e = uniq[u];
                const uint8_t * row = gu_row(J, e, r);
                const float sc = gu_scale(J, e, r);
                const float bi = gu_bias(J, e, r);
                for (int32_t k = uniq_begin[u]; k < uniq_begin[u + 1]; ++k) {
                    const int32_t m = by_expert[k];
                    const eh_miss & ms = misses[m];
                    float s;
                    tr_gu->vec_dot((int) n_embd, &s, 0, row, 0, xq.data() + ms.t*xq_row, 0, 1);
                    gu[m*rows + r] = s*sc*(J.l.weight_before ? ms.w : 1.0f) + bi;
                }
            }
        }
        barrier.wait();
        if (iw == 0) {
            t_step[1] = eh_now_ns(); // [TAG_EH_TIMEOUT_DIAG]
        }

        // 3. the activation, in the down dot type
        {
            float * h = hf.data() + (size_t) iw*n_ff;
            for (int64_t m = iw; m < n_miss; m += n_workers) {
                activate(J, gu.data() + m*J.gu_rows, h);
                uint8_t * dst = hq.data() + m*hq_row;
                if (J.vt_down == GGML_TYPE_F32) {
                    memcpy(dst, h, n_ff*sizeof(float));
                } else {
                    tr_vdn->from_float(h, dst, n_ff);
                }
            }
        }
        barrier.wait();
        if (iw == 0) {
            t_step[2] = eh_now_ns(); // [TAG_EH_TIMEOUT_DIAG]
        }

        // 4. down rows: each worker owns a range of output columns for every token (tokens without misses get
        //    zeros); each missed expert's rows in that range are read once for all the tokens that miss it
        {
            const ggml_tensor * dn = J.l.down;
            const int64_t c0 = n_embd*iw/n_workers;
            const int64_t c1 = n_embd*(iw + 1)/n_workers;
            for (int32_t t = 0; t < n_tokens; ++t) {
                memset(y + (int64_t) t*dims.max_embd + c0, 0, (c1 - c0)*sizeof(float));
            }
            for (size_t u = 0; u < uniq.size(); ++u) {
                const int32_t e = uniq[u];
                const uint8_t * base = (const uint8_t *) dn->data + de(e)*dn->nb[2];
                const float sc = J.l.down_scale ? J.l.down_scale[e] : 1.0f;
                const float * bias = J.l.down_bias ? J.l.down_bias + (int64_t) e*n_embd : nullptr;
                for (int64_t c = c0; c < c1; ++c) {
                    const uint8_t * row = base + c*dn->nb[1];
                    for (int32_t k = uniq_begin[u]; k < uniq_begin[u + 1]; ++k) {
                        const int32_t m = by_expert[k];
                        const eh_miss & ms = misses[m];
                        float s;
                        tr_down->vec_dot((int) n_ff, &s, 0, row, 0, hq.data() + m*hq_row, 0, 1);
                        s = s*sc + (bias ? bias[c] : 0.0f);
                        y[(int64_t) ms.t*dims.max_embd + c] += (J.l.weight_before ? 1.0f : ms.w)*s;
                    }
                }
            }
        }
    }

    void worker(int iw) {
        // the sequence the pool was made at, not what this thread reads when it starts: a send may already be waiting
        uint32_t seen = seq_base;
        uint64_t my_gen = 0;
        for (;;) {
            uint32_t s = 0;
            if (!wait_work(iw, seen, my_gen, s)) {
                return;
            }
            // [TAG_EH_POOL_WAKE] the barrier below needs every worker: wake those asleep
            if (n_asleep.load(std::memory_order_seq_cst) > 0) {
                wake();
            }
            // every worker saw a new send; worker 0 reads it, the others wait
            barrier.wait();
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            if (iw == 0) {
                prepare(eh_load(&hdr->seq_send));
            }
            barrier.wait();
            if (job_layer != nullptr && !misses.empty() && !sim_nohost) {
                run(iw);
            } else if (iw == 0) {
                float * y = (float *) (mb + ggml_expert_handoff_off_y(dims));
                const int64_t n_embd = job_layer ? job_layer->n_embd : hdr->n_embd;
                for (int32_t t = 0; t < n_tokens; ++t) {
                    memset(y + (int64_t) t*dims.max_embd, 0, n_embd*sizeof(float));
                }
            }
            // [TAG_EH_POOL_WAKE] the spin window counts from the end of this job for every worker, also one that took
            // longer than the window (otherwise the others sleep past this barrier while worker 0 spins)
            extend_spin(eh_now_ns() + linger_ns);
            barrier.wait();
            seen = seq;
            if (iw == 0) {
                eh_store(&hdr->seq_done, seq);
                const int64_t t_end = eh_now_ns();
                t_active.store(t_end, std::memory_order_relaxed);
                st_busy_ns.fetch_add((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
                // [TAG_EH_TIMEOUT_DIAG]
                if (n_tokens > 1) {
                    st_multi.fetch_add(1, std::memory_order_relaxed);
                    st_busy_multi.fetch_add((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
                }
                if ((uint64_t) (t_end - t_seen) > st_max_ns.load(std::memory_order_relaxed)) {
                    const bool ran = job_layer != nullptr && !misses.empty() && !sim_nohost;
                    st_max_ns.store((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
                    st_max_layer.store(job_il, std::memory_order_relaxed);
                    st_max_tokens.store(n_tokens, std::memory_order_relaxed);
                    st_max_miss.store((int32_t) misses.size(), std::memory_order_relaxed);
                    st_max_uniq.store((int32_t) uniq.size(), std::memory_order_relaxed);
                    int64_t prev = t_seen;
                    for (int k = 0; k < 4; ++k) {
                        const int64_t t = k < 3 ? (ran ? t_step[k] : prev) : t_end;
                        st_max_step[k].store((uint64_t) std::max<int64_t>(0, t - prev), std::memory_order_relaxed);
                        prev = std::max(prev, t);
                    }
                }
                spin_until.store(std::max(spin_until.load(std::memory_order_relaxed), eh_now_ns() + linger_ns), std::memory_order_release);
            }
        }
    }
};

ggml_expert_handoff_host_t ggml_expert_handoff_host_new(
        void * mailbox, int32_t max_tokens, int32_t max_embd, int32_t max_used, int32_t n_layer, int32_t n_workers) {
    GGML_ASSERT(mailbox != nullptr && n_layer > 0 && max_tokens > 0 && max_embd > 0 && max_used > 0);
    ggml_cpu_init();

    auto * h = new ggml_expert_handoff_host;
    h->mb   = (uint8_t *) mailbox;
    h->hdr  = (ggml_expert_handoff_header *) mailbox;
    h->dims = { max_tokens, max_embd, max_used };
    h->layers.resize(n_layer);
    h->n_workers = std::max(1, n_workers);
    h->barrier.n = h->n_workers;
    h->barrier.stop = &h->stop;

    // a fresh mailbox: nothing sent, nothing answered
    h->seq_base = eh_load(&h->hdr->seq_send);
    h->seq      = h->seq_base; // [TAG_EH_TIMEOUT_DIAG] prepare() counts the sends it never took up from here
    if (const char * v = getenv("ROCMFPX_EH_SIM_RESIDENT")) {
        h->sim_resident = std::max<int64_t>(0, atoll(v));
    }
    if (const char * v = getenv("ROCMFPX_EH_SIM_NOHOST")) {
        h->sim_nohost = atoi(v) != 0;
    }
    if (h->sim_resident > 0 || h->sim_nohost) {
        GGML_LOG_WARN("%s: measuring: %s%s (the answers are wrong)\n", __func__,
            h->sim_nohost ? "the host answers zeros" : "", h->sim_resident > 0 ? "the host reads the first experts of each layer only" : "");
    }
    eh_store(&h->hdr->seq_done, h->seq_base);

    h->misses.reserve((size_t) max_tokens*max_used);
    h->threads.reserve(h->n_workers);
    for (int i = 0; i < h->n_workers; ++i) {
        h->threads.emplace_back([h, i] { h->worker(i); });
    }
    return h;
}

void ggml_expert_handoff_host_free(ggml_expert_handoff_host_t h) {
    if (h == nullptr) {
        return;
    }
    h->stop.store(true, std::memory_order_release);
    h->wake();
    for (auto & t : h->threads) {
        t.join();
    }
    delete h;
}

bool ggml_expert_handoff_host_set_layer(ggml_expert_handoff_host_t h, int32_t il, const ggml_expert_handoff_layer * l) {
    if (il < 0 || il >= (int32_t) h->layers.size() || l->down == nullptr || (l->gate_up == nullptr && l->up == nullptr)) {
        return false;
    }
    // a gate (separate or merged) for the gated activations, none for the activations of up alone
    const bool has_gate = l->gate_up != nullptr || l->gate != nullptr;
    const bool up_only_act = l->act >= GGML_EXPERT_HANDOFF_ACT_UP_SILU;
    if (has_gate == up_only_act) {
        return false;
    }
    const ggml_tensor * gu = l->gate_up ? l->gate_up : (l->gate ? l->gate : l->up);
    const ggml_tensor * dn = l->down;
    if (l->gate && !l->gate_up && (l->up == nullptr || l->up->type != l->gate->type || !ggml_are_same_shape(l->up, l->gate))) {
        return false;
    }
    if ((l->act == GGML_EXPERT_HANDOFF_ACT_SITU && l->beta == 0.0f) ||
        ((l->act == GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI || l->act == GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP ||
          l->act == GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT) && !(l->limit > 0.0f))) {
        return false;
    }

    eh_layer L;
    L.l        = *l;
    L.has_gate = has_gate;
    L.n_embd   = gu->ne[0];
    L.n_ff     = l->gate_up ? gu->ne[1]/2 : gu->ne[1];
    L.n_expert = gu->ne[2];
    L.gu_rows  = has_gate ? 2*L.n_ff : L.n_ff;
    L.type_gu   = gu->type;
    L.type_down = dn->type;
    if (dn->ne[0] != L.n_ff || dn->ne[1] != L.n_embd || dn->ne[2] != L.n_expert || L.n_embd > h->dims.max_embd) {
        return false;
    }
    const auto * tg = ggml_get_type_traits_cpu(L.type_gu);
    const auto * td = ggml_get_type_traits_cpu(L.type_down);
    if (tg->vec_dot == nullptr || td->vec_dot == nullptr) {
        return false;
    }
    L.vt_gu   = tg->vec_dot_type;
    L.vt_down = td->vec_dot_type;
    if ((L.vt_gu != GGML_TYPE_F32 && ggml_get_type_traits_cpu(L.vt_gu)->from_float == nullptr) ||
        (L.vt_down != GGML_TYPE_F32 && ggml_get_type_traits_cpu(L.vt_down)->from_float == nullptr)) {
        return false;
    }
    if (L.n_embd % ggml_blck_size(L.vt_gu) != 0 || L.n_ff % ggml_blck_size(L.vt_down) != 0) {
        return false;
    }
    L.counts.assign(L.n_expert, 0);
    L.set = true;
    h->layers[il] = std::move(L);
    return true;
}

void ggml_expert_handoff_host_wake(ggml_expert_handoff_host_t h) {
    h->wake();
}

void ggml_expert_handoff_host_idle(ggml_expert_handoff_host_t h) {
    h->spin_until.store(0, std::memory_order_release);
}

void ggml_expert_handoff_host_get_stats(ggml_expert_handoff_host_t h, ggml_expert_handoff_stats * st) {
    st->n_layers = h->st_layers.load();
    st->n_routed = h->st_routed.load();
    st->n_host   = h->st_host.load();
    st->busy_ns  = h->st_busy_ns.load();
    // [TAG_EH_TIMEOUT_DIAG]
    st->n_multi       = h->st_multi.load();
    st->busy_ns_multi = h->st_busy_multi.load();
    st->n_skipped     = h->st_skipped.load();
    st->max_ns        = h->st_max_ns.load();
    st->max_layer     = h->st_max_layer.load();
    st->max_tokens    = h->st_max_tokens.load();
    st->max_miss      = h->st_max_miss.load();
    st->max_uniq      = h->st_max_uniq.load();
    for (int k = 0; k < 4; ++k) {
        st->max_step_ns[k] = h->st_max_step[k].load();
    }
}

const uint64_t * ggml_expert_handoff_host_counts(ggml_expert_handoff_host_t h, int32_t il) {
    if (il < 0 || il >= (int32_t) h->layers.size() || !h->layers[il].set) {
        return nullptr;
    }
    return h->layers[il].counts.data();
}
