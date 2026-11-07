#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

    //
    // ROCmFPX [TAG_EXPERT_HANDOFF] the host side of the expert handoff (protocol in ggml-expert-handoff.h): a pool of
    // threads that answers a mailbox by computing the routed experts the device does not hold, from host memory
    //

    // the activation between the experts' gate/up and down projections, as build_moe_ffn applies it
    enum ggml_expert_handoff_act {
        GGML_EXPERT_HANDOFF_ACT_SILU       = 0,  // silu(gate) * up (ggml_swiglu_split)
        GGML_EXPERT_HANDOFF_ACT_GELU       = 1,  // gelu(gate) * up, tanh approximation (ggml_geglu_split)
        GGML_EXPERT_HANDOFF_ACT_RELU       = 2,  // relu(gate) * up (ggml_reglu_split)
        GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI = 3,  // g = min(gate, limit): g*sigmoid(alpha*g) * (clamp(up, +-limit) + 1)
        GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP = 4,  // silu(min(gate, limit)) * clamp(up, +-limit) (ggml_swiglu_clamp)
        GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT = 5,  // min(silu(gate), limit) * clamp(up, +-limit)
        GGML_EXPERT_HANDOFF_ACT_SITU       = 6,  // beta*tanh(gate/beta)*sigmoid(gate) * (lb > 0 ? lb*tanh(up/lb) : up)
        // experts without a gate projection: the activation of up alone
        GGML_EXPERT_HANDOFF_ACT_UP_SILU     = 7,
        GGML_EXPERT_HANDOFF_ACT_UP_GELU     = 8,
        GGML_EXPERT_HANDOFF_ACT_UP_RELU     = 9,
        GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR = 10,
    };

    struct ggml_expert_handoff_layer {
        // the routed experts in host memory, ggml layout: gate and up [n_embd, n_ff, n_expert] or the merged
        // gate_up [n_embd, 2*n_ff, n_expert] (gate rows first), or up alone; down [n_ff, n_embd, n_expert]
        const struct ggml_tensor * gate;
        const struct ggml_tensor * up;
        const struct ggml_tensor * gate_up;
        const struct ggml_tensor * down;
        // per-expert output scales [n_expert] or NULL (gate_up takes up_scale, like the graph)
        const float * gate_scale;
        const float * up_scale;
        const float * down_scale;
        // per-expert biases in host memory or NULL: gate and up [n_expert][n_ff] (gate_up: [n_expert][2*n_ff],
        // in up_bias), down [n_expert][n_embd]; added after the scale, like the graph
        const float * gate_bias;
        const float * up_bias;
        const float * down_bias;
        enum ggml_expert_handoff_act act;
        float   alpha;         // SWIGLU_OAI
        float   limit;         // SWIGLU_OAI, SILU_CLAMP, SILU_LIMIT
        float   beta;          // SITU
        float   lb;            // SITU
        bool    weight_before; // the routing weight scales the input (llama4) instead of the output
        int32_t n_used;        // only the first n_used selected experts of a token count, 0 = all
        int32_t miss_slot;     // the slot the device gives the experts the host computes
    };

    struct ggml_expert_handoff_stats {
        uint64_t n_layers;  // layers answered
        uint64_t n_routed;  // experts routed in them
        uint64_t n_host;    // of which the host computed
        uint64_t busy_ns;   // time from a send to its answer
        // [TAG_EH_TIMEOUT_DIAG] what a join that gave up needs explained
        uint64_t n_multi;        // layers answered for batches of more than one token
        uint64_t busy_ns_multi;  // their time from the send to the answer
        uint64_t n_skipped;      // sends the host never took up: the device had sent the next one before it looked
        uint64_t max_ns;         // the slowest answer so far, from the host seeing the send
        int32_t  max_layer;      // its layer, batch, misses and distinct missed experts
        int32_t  max_tokens;
        int32_t  max_miss;
        int32_t  max_uniq;
        uint64_t max_step_ns[4]; // its steps: inputs, gate/up, activation, down
    };

    typedef struct ggml_expert_handoff_host * ggml_expert_handoff_host_t;

    // mailbox: at least ggml_expert_handoff_size(dims) bytes the device reads and writes in place; n_layer: the
    // highest layer index + 1; n_workers: threads (they spin while work is expected)
    GGML_BACKEND_API ggml_expert_handoff_host_t ggml_expert_handoff_host_new(
            void * mailbox, int32_t max_tokens, int32_t max_embd, int32_t max_used, int32_t n_layer, int32_t n_workers);
    GGML_BACKEND_API void ggml_expert_handoff_host_free(ggml_expert_handoff_host_t host);

    // returns false if the host cannot compute these experts (types without CPU dot products, shapes)
    GGML_BACKEND_API bool ggml_expert_handoff_host_set_layer(ggml_expert_handoff_host_t host, int32_t layer, const struct ggml_expert_handoff_layer * l);

    // a graph with handoff layers is about to run: spin for work from now on (the pool sleeps after a quiet spell)
    GGML_BACKEND_API void ggml_expert_handoff_host_wake(ggml_expert_handoff_host_t host);
    // the graphs are done: stop spinning (one thread keeps looking at the mailbox every millisecond)
    GGML_BACKEND_API void ggml_expert_handoff_host_idle(ggml_expert_handoff_host_t host);

    GGML_BACKEND_API void ggml_expert_handoff_host_get_stats(ggml_expert_handoff_host_t host, struct ggml_expert_handoff_stats * stats);

    // how often each expert of a layer was routed since the start (hits and misses), n_expert entries or NULL
    GGML_BACKEND_API const uint64_t * ggml_expert_handoff_host_counts(ggml_expert_handoff_host_t host, int32_t layer);

#ifdef __cplusplus
}
#endif
