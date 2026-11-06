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
        const ggml_tensor * slots = dst->src[2];
        const ggml_tensor * w     = dst->src[3];

        const int64_t n_embd   = x->ne[0];
        const int64_t n_tokens = ggml_nrows(x);
        const int64_t n_used   = ids->ne[0];

        // the misses, and the routing counts
        const int32_t miss_slot = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MISS_SLOT);
        int32_t * counts = (int32_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_COUNTS_LO, GGML_EXPERT_HANDOFF_P_COUNTS_HI);
        const int64_t stride = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE);
        int32_t n_miss = 0;
        for (int64_t i = 0; i < n_used*n_tokens; ++i) {
            n_miss += ((const int32_t *) slots->data)[i] == miss_slot;
            const int32_t e = ((const int32_t *) ids->data)[i];
            if (counts && e >= 0 && e < stride) {
                counts[layer*stride + e]++;
            }
        }
        flags[layer] = n_miss;
        memcpy(dst->data, slots->data, ggml_nbytes(slots));
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
            memcpy(ms + t*d.max_used, (const int32_t *) slots->data + t*n_used, n_used*sizeof(int32_t));
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
    while (eh_load(&hdr->seq_done) != want) {
        if (eh_now_ns() - t0 > timeout_ns) {
            ok = false;
            break;
        }
        EH_PAUSE();
    }
    if (!ok) {
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

    static constexpr int64_t linger_ns = 250*1000*1000; // spin this long after the last work or wake

    uint32_t seq_base = 0; // seq_send when the pool was made

    // the job, prepared by worker 0 between barriers
    uint32_t seq = 0;
    const eh_layer * job_layer = nullptr;
    int32_t n_tokens = 0;
    std::vector<eh_miss> misses;
    std::vector<int32_t> miss_tokens;      // the tokens with misses
    std::vector<int32_t> miss_begin;       // per token: first miss index (misses are sorted by token), n_tokens + 1
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

    void wake() {
        spin_until.store(eh_now_ns() + linger_ns, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx);
            wake_gen++;
        }
        cv.notify_all();
    }

    // wait for a new send; false if the pool stops. After a quiet spell the workers sleep, but worker 0 keeps looking
    // at the mailbox every millisecond and wakes them: a send can come long after the wake (the first kernels of a
    // cold device load for hundreds of milliseconds), and the device cannot wake a sleeping pool itself
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
                    std::unique_lock<std::mutex> lk(mtx);
                    cv.wait(lk, [&] { return stop.load() || wake_gen != my_gen; });
                    my_gen = wake_gen;
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
        seq = s;
        t_seen = eh_now_ns();

        const int32_t il = hdr->layer;
        n_tokens         = hdr->n_tokens;
        const int32_t n_used = hdr->n_used;

        job_layer = (il >= 0 && il < (int32_t) layers.size() && layers[il].set) ? &layers[il] : nullptr;
        misses.clear();
        miss_tokens.clear();
        miss_begin.assign(n_tokens + 1, 0);
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

        for (int32_t t = 0; t < n_tokens; ++t) {
            miss_begin[t] = (int32_t) misses.size();
            bool any = false;
            for (int32_t k = 0; k < n_used; ++k) {
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
        miss_begin[n_tokens] = (int32_t) misses.size();

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
        if (gu.size() < misses.size()*2*J.n_ff) {
            gu.resize(misses.size()*2*J.n_ff);
        }
        if (hq.size() < misses.size()*hq_row) {
            hq.resize(misses.size()*hq_row);
        }
        if (hf.size() < (size_t) n_workers*J.n_ff) {
            hf.resize((size_t) n_workers*J.n_ff);
        }
    }

    // gate/up row r (0 .. 2*n_ff - 1, gate rows first) of expert e
    const uint8_t * gu_row(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up) {
            const ggml_tensor * t = J.l.gate_up;
            return (const uint8_t *) t->data + e*t->nb[2] + r*t->nb[1];
        }
        const ggml_tensor * t = r < J.n_ff ? J.l.gate : J.l.up;
        return (const uint8_t *) t->data + e*t->nb[2] + (r % J.n_ff)*t->nb[1];
    }

    float gu_scale(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up) {
            return J.l.up_scale ? J.l.up_scale[e] : 1.0f;
        }
        const float * s = r < J.n_ff ? J.l.gate_scale : J.l.up_scale;
        return s ? s[e] : 1.0f;
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

        // 2. gate and up rows of every miss
        {
            const int64_t rows  = 2*n_ff;
            const int64_t total = n_miss*rows;
            const int64_t i0 = total*iw/n_workers;
            const int64_t i1 = total*(iw + 1)/n_workers;
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t m = i / rows;
                const int64_t r = i % rows;
                const eh_miss & ms = misses[m];
                float s;
                tr_gu->vec_dot((int) n_embd, &s, 0, gu_row(J, ms.e, r), 0, xq.data() + ms.t*xq_row, 0, 1);
                gu[m*rows + r] = s*gu_scale(J, ms.e, r);
            }
        }
        barrier.wait();

        // 3. the activation, in the down dot type
        {
            float * h = hf.data() + (size_t) iw*n_ff;
            for (int64_t m = iw; m < n_miss; m += n_workers) {
                const float * g = gu.data() + m*2*n_ff;
                const float * u = g + n_ff;
                if (J.l.act == GGML_EXPERT_HANDOFF_ACT_GELU) {
                    constexpr float c = 0.7978845608028654f; // sqrt(2/pi)
                    for (int64_t i = 0; i < n_ff; ++i) {
                        const float v = g[i];
                        h[i] = 0.5f*v*(1.0f + tanhf(c*v*(1.0f + 0.044715f*v*v)))*u[i];
                    }
                } else {
                    for (int64_t i = 0; i < n_ff; ++i) {
                        const float v = g[i];
                        h[i] = v/(1.0f + expf(-v))*u[i];
                    }
                }
                uint8_t * dst = hq.data() + m*hq_row;
                if (J.vt_down == GGML_TYPE_F32) {
                    memcpy(dst, h, n_ff*sizeof(float));
                } else {
                    tr_vdn->from_float(h, dst, n_ff);
                }
            }
        }
        barrier.wait();

        // 4. down rows, summed over each token's misses with their weights; tokens without misses get zeros
        {
            const ggml_tensor * dn = J.l.down;
            const int64_t total = (int64_t) n_tokens*n_embd;
            const int64_t i0 = total*iw/n_workers;
            const int64_t i1 = total*(iw + 1)/n_workers;
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t t = i / n_embd;
                const int64_t c = i % n_embd;
                float acc = 0.0f;
                for (int32_t m = miss_begin[t]; m < miss_begin[t + 1]; ++m) {
                    const eh_miss & ms = misses[m];
                    float s;
                    tr_down->vec_dot((int) n_ff, &s, 0, (const uint8_t *) dn->data + ms.e*dn->nb[2] + c*dn->nb[1], 0,
                            hq.data() + m*hq_row, 0, 1);
                    acc += ms.w*s*(J.l.down_scale ? J.l.down_scale[ms.e] : 1.0f);
                }
                y[t*dims.max_embd + c] = acc;
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
            // every worker saw a new send; worker 0 reads it, the others wait
            barrier.wait();
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            if (iw == 0) {
                prepare(eh_load(&hdr->seq_send));
            }
            barrier.wait();
            if (job_layer != nullptr && !misses.empty()) {
                run(iw);
            } else if (iw == 0) {
                float * y = (float *) (mb + ggml_expert_handoff_off_y(dims));
                const int64_t n_embd = job_layer ? job_layer->n_embd : hdr->n_embd;
                for (int32_t t = 0; t < n_tokens; ++t) {
                    memset(y + (int64_t) t*dims.max_embd, 0, n_embd*sizeof(float));
                }
            }
            barrier.wait();
            seen = seq;
            if (iw == 0) {
                eh_store(&hdr->seq_done, seq);
                st_busy_ns.fetch_add((uint64_t) (eh_now_ns() - t_seen), std::memory_order_relaxed);
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
    if (il < 0 || il >= (int32_t) h->layers.size() || l->down == nullptr || (l->gate_up == nullptr && (l->gate == nullptr || l->up == nullptr))) {
        return false;
    }
    const ggml_tensor * gu = l->gate_up ? l->gate_up : l->gate;
    const ggml_tensor * dn = l->down;
    if (!l->gate_up && (l->up->type != l->gate->type || !ggml_are_same_shape(l->up, l->gate))) {
        return false;
    }

    eh_layer L;
    L.l        = *l;
    L.n_embd   = gu->ne[0];
    L.n_ff     = l->gate_up ? gu->ne[1]/2 : gu->ne[1];
    L.n_expert = gu->ne[2];
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

void ggml_expert_handoff_host_get_stats(ggml_expert_handoff_host_t h, ggml_expert_handoff_stats * st) {
    st->n_layers = h->st_layers.load();
    st->n_routed = h->st_routed.load();
    st->n_host   = h->st_host.load();
    st->busy_ns  = h->st_busy_ns.load();
}

const uint64_t * ggml_expert_handoff_host_counts(ggml_expert_handoff_host_t h, int32_t il) {
    if (il < 0 || il >= (int32_t) h->layers.size() || !h->layers[il].set) {
        return nullptr;
    }
    return h->layers[il].counts.data();
}
