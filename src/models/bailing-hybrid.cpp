#include "models.h"
#include "llama-memory-recurrent.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

// ---------------------------------------------------------------------------
// LING_KDA_DEBUG=1 — compact per-(token, KDA-layer) probe of decay gate g,
// exp(g), recurrent-state L2, and KDA layer-output L2. OFF by default so
// live llama-server routes are unaffected. Decode-only (n_tokens==1).
// Line format (stderr):
//   #KDA tok=<n> il=<i> g_min= g_max= g_mean= exp_min= exp_max= exp_mean= s_l2= o_l2= frac_floor=
// ---------------------------------------------------------------------------
static bool ling_kda_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("LING_KDA_DEBUG");
        v = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
        if (v) {
            fprintf(stderr, "#KDA debug probe enabled (LING_KDA_DEBUG=1)\n");
        }
    }
    return v == 1;
}

// kind in high 8 bits: 0=gate, 1=state, 2=out; low 24 bits = layer index
enum { LING_KDA_KIND_GATE = 0, LING_KDA_KIND_STATE = 1, LING_KDA_KIND_OUT = 2 };

struct ling_kda_layer_stats {
    float g_min, g_max, g_sum;
    float e_min, e_max, e_sum;
    int64_t g_n;
    int64_t n_floor;
    float s_l2;
    float o_l2;
    bool has_g, has_s, has_o;
};

static constexpr int LING_KDA_MAX_LAYER = 64;
static ling_kda_layer_stats g_ling_kda_buf[LING_KDA_MAX_LAYER];
static std::atomic<int> g_ling_kda_tok{0};

static void ling_kda_reset_layer(ling_kda_layer_stats & s) {
    s.g_min =  std::numeric_limits<float>::infinity();
    s.g_max = -std::numeric_limits<float>::infinity();
    s.e_min =  std::numeric_limits<float>::infinity();
    s.e_max = -std::numeric_limits<float>::infinity();
    s.g_sum = 0.f;
    s.e_sum = 0.f;
    s.g_n = 0;
    s.n_floor = 0;
    s.s_l2 = 0.f;
    s.o_l2 = 0.f;
    s.has_g = s.has_s = s.has_o = false;
}

static void ling_kda_probe_fn(struct ggml_tensor * dst, const struct ggml_tensor * a,
                             int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }

    const intptr_t pack = (intptr_t) userdata;
    const int kind = (int) ((pack >> 24) & 0xff);
    const int il   = (int) (pack & 0xffffff);
    if (il < 0 || il >= LING_KDA_MAX_LAYER) {
        return;
    }

    if (a->type != GGML_TYPE_F32 || a->data == nullptr) {
        return;
    }

    // Decode-only for gate/out (state always 4D with n_seqs).
    int64_t n_tokens_guess = 1;
    if (kind == LING_KDA_KIND_GATE) {
        // 3D [head_dim,n_head,n_tokens] or 4D [head_dim,n_head,n_seq_tokens,n_seqs]
        if (a->ne[2] > 1) {
            n_tokens_guess = a->ne[2];
        }
    } else if (kind == LING_KDA_KIND_OUT) {
        // [n_embd, n_tokens]
        if (a->ne[1] > 1) {
            n_tokens_guess = a->ne[1];
        }
    }

    if (n_tokens_guess != 1 && kind != LING_KDA_KIND_STATE) {
        if (dst && dst->data && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
            ((float *) dst->data)[0] = 0.f;
        }
        return;
    }

    const float * x = (const float *) a->data;
    const int64_t n = ggml_nelements(a);
    ling_kda_layer_stats & st = g_ling_kda_buf[il];

    if (kind == LING_KDA_KIND_GATE) {
        if (!st.has_g) {
            ling_kda_reset_layer(st);
        }
        float gmin =  std::numeric_limits<float>::infinity();
        float gmax = -std::numeric_limits<float>::infinity();
        float gsum = 0.f;
        float emin =  std::numeric_limits<float>::infinity();
        float emax = -std::numeric_limits<float>::infinity();
        float esum = 0.f;
        int64_t n_floor = 0;
        constexpr float FLOOR = -4.999f;
        for (int64_t i = 0; i < n; ++i) {
            const float g = x[i];
            const float e = expf(g);
            if (g < gmin) gmin = g;
            if (g > gmax) gmax = g;
            gsum += g;
            if (e < emin) emin = e;
            if (e > emax) emax = e;
            esum += e;
            if (g <= FLOOR) ++n_floor;
        }
        st.g_min = gmin; st.g_max = gmax; st.g_sum = gsum;
        st.e_min = emin; st.e_max = emax; st.e_sum = esum;
        st.g_n = n;
        st.n_floor = n_floor;
        st.has_g = true;
    } else if (kind == LING_KDA_KIND_STATE) {
        // Skip multi-token prefill: state is always [S,S,H,n_seqs]; only emit
        // with gate+out which are decode-gated.
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double v = (double) x[i];
            acc += v * v;
        }
        st.s_l2 = (float) sqrt(acc);
        st.has_s = true;
    } else if (kind == LING_KDA_KIND_OUT) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double v = (double) x[i];
            acc += v * v;
        }
        st.o_l2 = (float) sqrt(acc);
        st.has_o = true;
    }

    if (st.has_g && st.has_s && st.has_o) {
        int tok;
        if (il == 0) {
            tok = g_ling_kda_tok.fetch_add(1) + 1;
        } else {
            tok = g_ling_kda_tok.load();
        }
        const float inv = st.g_n > 0 ? 1.f / (float) st.g_n : 0.f;
        const float frac_floor = st.g_n > 0 ? (float) st.n_floor * inv : 0.f;
        fprintf(stderr,
            "#KDA tok=%d il=%d g_min=%.4f g_max=%.4f g_mean=%.4f "
            "exp_min=%.4e exp_max=%.4e exp_mean=%.4e "
            "s_l2=%.4f o_l2=%.4f frac_floor=%.4f n=%lld\n",
            tok, il,
            st.g_min, st.g_max, st.g_sum * inv,
            st.e_min, st.e_max, st.e_sum * inv,
            st.s_l2, st.o_l2, frac_floor, (long long) st.g_n);
        fflush(stderr);
        st.has_g = st.has_s = st.has_o = false;
    }

    if (dst && dst->data && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
        ((float *) dst->data)[0] = 0.f;
    }
}

static void ling_kda_attach_probe(ggml_context * ctx0, ggml_cgraph * gf,
                                  ggml_tensor * t, int kind, int il) {
    if (!ling_kda_debug_enabled() || t == nullptr) {
        return;
    }
    const intptr_t pack = ((intptr_t) kind << 24) | (intptr_t) (il & 0xffffff);
    ggml_tensor * probe = ggml_map_custom1(ctx0, t, ling_kda_probe_fn, /*n_tasks=*/1, (void *) pack);
    ggml_format_name(probe, "ling_kda_probe_%d_%d", kind, il);
    ggml_build_forward_expand(gf, probe);
}


// Ling 3.0 flash (inclusionAI/Ling-3.0-flash) -- model_type "bailing_hybrid",
// BailingMoeV3ForCausalLM. 127.5B total / 5.1B active.
//
// 42 layers: 35 KDA (Kimi Delta Attention) + 7 gated MLA, MLA at every layer
// where (il + 1) % layer_group_size == 0 with layer_group_size 6, i.e. layers
// 5/11/17/23/29/35/41 -- MLA is LAST in each group (solar_open2 is the
// opposite, softmax-first). Layer 42 is an MTP/nextn head and is skipped.
//
// Derived from src/models/kimi-linear.cpp, which already has both halves: the
// KDA block, and MLA without Q compression at exactly this geometry
// (qk_rope 64 / qk_nope 128 / qk_head 192). The MoE router is bit-exact
// bailingmoe2 (noaux_tc grouped top-k + sigmoid + expert_bias), which
// build_moe_ffn handles from hparams with no code here.
//
// Deltas against kimi-linear, all verified against modeling_bailing_moe_v3.py
// and the fla kernels rather than inferred:
//
//   1. SAFE GATE. config kda_safe_gate=true, kda_lower_bound=-5.0 replaces
//        g = -exp(A_log) * softplus(f(x) + dt_bias)          [kimi]
//      with
//        g = lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias))
//      Confirmed identical in fla's naive reference (ops/kda/gate.py:57-69) and
//      its Triton kernel (ops/kda/gate.py:116-119). Because g is built here and
//      handed to GGML_OP_GATED_DELTA_NET as an input, the kernel is untouched --
//      same shape of fix as solar_open2's beta = 2*sigmoid(.). Note the
//      converter must store +exp(A_log), NOT kimi's -exp(A_log): the sign now
//      lives in lower_bound. Getting this wrong is a silent quality
//      regression, never a crash.
//
//   2. no_kda_lora=true -> f_proj / g_proj are FULL-RANK {n_embd, d_inner}
//      single matmuls, not kimi's low-rank f_a/f_b, g_a/g_b pairs.
//
//   3. A_log is 1-D [n_head]. Kimi's is [1,H,1,1], solar_open2's is [1,1,64,1]
//      -- third layout in three ports. The converter reshapes it.
//
//   4. MLA carries a HEAD-WISE sigmoid output gate: g_proj {n_embd, n_head},
//      one scalar per head broadcast across v_head_dim, applied to the SDPA
//      result before dense/o_proj. solar_open2's gate is elementwise and
//      full-width -- do not copy that broadcast.
//
//   5. MLA USES RoPE, unlike kimi (rotary_emb=None there). rope_interleave=true
//      resolves to llama.cpp's NORM rope: the reference de-interleaves with
//      view(d/2,2).transpose before a rotate_half, which is pairwise rotation
//      in stored order. theta 6e6 over the 64-dim rope slice only.
//
// Vestigial config fields, verified unreferenced by grepping the reference:
// expert_swiglu_limit_list / share_expert_swiglu_limit_list (BailingMoeV3MLP is
// a plain SwiGLU -- these are populated with non-zero values for the last few
// layers and are still dead), use_qk_norm, linear_silu, max_window_layers,
// mtp_use_kda, use_mla_nope, use_nGPT, scale_router_input, seq_aux.
// partial_rotary_factor is overwritten to 1.0 by the rotary module itself.

void llama_model_bailing_hybrid::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_LOWER_BOUND,             hparams.f_kda_lower_bound, false);

    // KDA layers are marked with n_head_kv == 0 (same convention as Kimi Linear /
    // Jamba); MLA layers carry the real KV head count (converter forces 1 for MLA cache).
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);

    // ROCmFPX fork: n_layer == GGUF block_count (includes MTP head).
    // Do NOT set n_layer_kv_from_start here: has_kv() would permanently exclude the
    // MTP block from every cache. Main vs MTP layer ownership is decided by
    // create_memory filters (hybrid main excludes nextn; MTP plain K-cache
    // includes only nextn) — same pattern as hyv3 / qwen35.
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer &&
                "nextn_predict_layers must be < n_layer");

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
    }

    // The safe gate is what this arch is; a GGUF without it was converted wrong.
    GGML_ASSERT(hparams.f_kda_lower_bound < 0.0f &&
                "bailing-hybrid requires a negative kda.lower_bound (safe gate); re-convert this model");

    // Ling-3.0-flash: 42 main layers (+1 MTP in block_count)
    switch (hparams.n_layer - hparams.nextn_predict_layers) {
        case 42: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bailing_hybrid::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int64_t head_dim_kda = hparams.n_embd_head_kda;   // 128
    const int64_t ssm_d_conv   = hparams.ssm_d_conv;        // 4
    const int64_t d_inner      = head_dim_kda * n_head;     // 32 * 128 = 4096

    // ROCmFPX: n_layer is GGUF block_count (includes nextn). Main stack is shorter.
    const int n_layer_main = n_layer - (int) hparams.nextn_predict_layers;

    for (int i = 0; i < n_layer; ++i) {
        // MTP block (i >= n_layer_main) is loaded fully: its decoder tensors + nextn.*
        // feed the DECODER_MTP draft graph. Main forward still stops at n_layer_main.
        const int flags = 0;

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        if (i < n_layer_main && hparams.is_recurrent(i)) {
            // ---- KDA linear-attention layer ----
            // conv1d weights are 4D in the GGUF but quantisation may drop the
            // trailing 1, so accept 3D too (same dance as kimi-linear.cpp).
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_k_conv) {
                layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_v_conv) {
                layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }

            // num_kv_heads_for_linear_attn = 0 => K is full width, like Q/V
            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            // full-rank forget/output gates (no_kda_lora = true)
            layer.ssm_f = create_tensor(tn(LLM_TENSOR_SSM_F, "weight", i), {n_embd, d_inner}, 0);
            layer.ssm_g = create_tensor(tn(LLM_TENSOR_SSM_G, "weight", i), {n_embd, d_inner}, 0);

            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // stored as +exp(A_log) by the converter; the negation lives in
            // kda.lower_bound. Converter emits ggml ne = [1, n_head, 1, 1].
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_a) {
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
            }

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim_kda}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            // ---- gated MLA layer (also the shape of the skipped MTP head) ----
            const int64_t kv_lora_rank      = hparams.n_lora_kv;
            const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();   // 192
            const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();   // 128
            const int64_t qk_rope_head_dim  = hparams.n_rot();               // 64

            // q_lora_rank is null in config => no Q compression, one wide q_proj
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, flags);

            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, flags);

            // legacy GGUFs keep kv_b fused (MLA KV cache disabled)
            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)},
                flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wkv_b) {
                layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head}, flags);
                layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);
            }

            // head-wise gate: ONE scalar per head, not per output element
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        const int64_t n_ff_exp = hparams.n_ff_exp;

        if ((uint32_t) i < hparams.n_layer_dense_lead) {
            // first_k_dense_replace = 2 -> layers 0 and 1 are dense
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, flags);

            const int64_t n_ff_shexp = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, flags);

            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }

        // MTP/nextn projections for the draft head. Ling has no nextn.embed_tokens
        // or nextn.shared_head_head -- graph_mtp falls back to main token_embd /
        // output. final_layernorm maps to NEXTN_SHARED_HEAD_NORM.
        if (i >= n_layer_main) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_bailing_hybrid::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Causal conv1d over Q/K/V. Copied from kimi-linear.cpp -- qkv selects which of
// the three conv states to read/write (0=Q, 1=K, 2=V).
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all,
        ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs,
        int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;   // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_bailing_hybrid::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // MLA layers are RoPE'd (unlike kimi-linear), so positions are needed.
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_kv      = !hparams.is_mla() ? build_inp_mem_hybrid()   : nullptr;
    auto * inp_k       =  hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs      =  hparams.is_mla() ? inp_k->get_recr()        : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn()       : nullptr;
    auto * inp_attn_k  =  hparams.is_mla() ? inp_k->get_attn()        : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head       = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();               // 192
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();               // 128
    const int64_t kv_lora_rank        = hparams.n_lora_kv;                         // 512
    const int64_t n_embd_head_qk_rope = hparams.n_rot();                           // 64
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;   // 128

    // scaling = qk_head_dim ** -0.5 over the FULL 192, not the nope part
    const float kq_scale_mla = 1.0f / sqrtf((float) n_embd_head_k_mla);

    const float kda_lower_bound = hparams.f_kda_lower_bound;

    // NORM rope over the 64-dim rope slice only -- see the header comment.
    const int    rope_type  = LLAMA_ROPE_TYPE_NORM;
    const int    n_rot      = n_embd_head_qk_rope;
    const float  freq_base  = hparams.rope_freq_base_train;
    const float  freq_scale = hparams.rope_freq_scale_train;
    const float  ext_factor = cparams.yarn_ext_factor;
    const float  attn_factor = cparams.yarn_attn_factor;
    const float  beta_fast  = cparams.yarn_beta_fast;
    const float  beta_slow  = cparams.yarn_beta_slow;
    const int    n_ctx_orig = cparams.n_ctx_orig_yarn;

    // Exclude MTP/nextn blocks from the main graph (ROCmFPX: n_layer includes them).
    const int n_layer_main = n_layer - (int) hparams.nextn_predict_layers;

    for (int il = 0; il < n_layer_main; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // ================= KDA linear-attention layer =================
            const auto * mctx_cur = inp_rs->mctx;
            const auto   kv_head  = mctx_cur->get_head();

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // *** delta 1+2 vs kimi-linear ***
            // full-rank f_proj (one matmul, not f_b(f_a(x))), then the safe gate
            //   g = lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias))
            // ssm_a already holds +exp(A_log). dt_bias is added BEFORE the
            // per-head A scaling and before the sigmoid -- fla adds the bias to
            // the raw projection, then multiplies inside the sigmoid.
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f, cur);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            // A is per-head: [1, n_head, 1] broadcast over head_dim and tokens
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            g1 = ggml_sigmoid(ctx0, g1);
            g1 = ggml_scale(ctx0, g1, kda_lower_bound);
            cb(g1, "kda_g1_safe_gate", il);
            // DEBUG: probe safe-gate g (log-space) before reshape-for-GDN
            ling_kda_attach_probe(ctx0, gf, g1, LING_KDA_KIND_GATE, il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // allow_neg_eigval is off here: plain sigmoid, no 2x (that is
            // solar_open2's delta, not this model's).
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);
            cb(beta, "kda_beta", il);

            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;
            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
            // DEBUG: probe post-update recurrent state L2
            ling_kda_attach_probe(ctx0, gf, new_state, LING_KDA_KIND_STATE, il);

            // full-rank output gate, then RMSNorm(x) * sigmoid(g)
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g2     = ggml_mul_mat(ctx0, layer.ssm_g, cur_2d);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur   = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
            // DEBUG: probe KDA layer output L2
            ling_kda_attach_probe(ctx0, gf, cur, LING_KDA_KIND_OUT, il);
        } else {
            // ================= gated MLA layer =================
            // q_proj is one wide matmul (q_lora_rank is null). Per head the
            // layout is [nope(128) | rope(64)], matching the reference's
            // split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1).
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

            // *** delta 5: kimi applies no RoPE here; this model does ***
            k_pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, k_pe), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * attn_out = nullptr;

            if (layer.wk_b && layer.wv_b) { // MLA KV cache enabled
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla),
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);

                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(Qcur->type, n_embd_head_k_mla),
                    ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
                    ggml_row_size(Qcur->type, n_embd_head_qk_nope));

                q_pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_pe), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

                // note: rope must go first for in-place context shifting
                Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                ggml_tensor * Vcur = kv_cmpr;

                // wo is applied after the head-wise gate, so pass null here
                attn_out = build_attn(inp_attn_k, nullptr, NULL, layer.wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
            } else { // MLA KV cache disabled -- fall back to MHA
                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(Qcur->type, n_embd_head_k_mla),
                    ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
                    ggml_row_size(Qcur->type, n_embd_head_qk_nope));
                q_pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_pe), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla),
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);

                // rebuild Q as [nope | rope] to match the K layout below
                Qcur = ggml_concat(ctx0, ggml_cont(ctx0, q_nope), q_pe, 0);

                ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
                const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head), 0);
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head),
                    ggml_row_size(kv->type, n_embd_head_qk_nope));
                Vcur = ggml_cont(ctx0, Vcur);

                // k_pe is shared across heads (MQA) -> broadcast before concat
                ggml_tensor * k_pe_target   = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
                ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
                ggml_tensor * Kcur = ggml_concat(ctx0, ggml_cont(ctx0, k_nope), k_pe_repeated, 0);

                attn_out = build_attn(inp_attn_kv, nullptr, NULL, layer.wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
            }
            cb(attn_out, "attn_out", il);

            // *** delta 4: HEAD-WISE sigmoid gate ***
            // g_proj is {n_embd, n_head}: one scalar per head, broadcast across
            // v_head_dim. Reshaping attn_out to [head_dim, n_head, n_tokens] and
            // the gate to [1, n_head, n_tokens] makes ggml_mul do that broadcast.
            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
            gate = ggml_sigmoid(ctx0, gate);
            gate = ggml_reshape_3d(ctx0, gate, 1, n_head, n_tokens);
            cb(gate, "attn_gate_headwise", il);

            attn_out = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v_mla, n_head, n_tokens);
            attn_out = ggml_mul(ctx0, attn_out, gate);
            attn_out = ggml_cont_2d(ctx0, attn_out, n_embd_head_v_mla * n_head, n_tokens);
            cb(attn_out, "attn_gated", il);

            cur = ggml_mul_mat(ctx0, layer.wo, attn_out);
            cb(cur, "mla_out", il);
        }

        // When embeddings_pre_norm is requested unmasked (draft-mtp target), keep
        // all token rows through final norm so t_h_pre_norm is full-width.
        if (il == n_layer_main - 1 && inp_out_ids && cparams.embeddings_pre_norm_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // noaux_tc grouped top-k + sigmoid + expert_bias: build_moe_ffn reads
            // n_expert_groups / n_group_used straight from hparams.
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // Post-final-norm hidden: MTP draft head's hnorm consumes this (vLLM-style).
    cb(cur, "h_nextn", -1);
    res->t_h_pre_norm = cur;

    if (!cparams.embeddings_pre_norm_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


// LLM_GRAPH_TYPE_DECODER_MTP draft head for Ling 3.0 flash (bailing-hybrid).
// Algorithm mirrors hyv3 / vLLM MultiTokenPredictor:
//   enorm(embed) + hnorm(prev_hidden) -> concat(e, h) -> eh_proj ->
//   one gated-MLA + MoE decoder block (blk.n_main) ->
//   final_layernorm (nextn.shared_head_norm) -> shared LM head (model.output)
//
// Ling ships no nextn.embed_tokens / nextn.shared_head_head — fall back to the
// main model's token_embd / output. The MTP block is always MoE (leading dense
// ends at layer 1; blk.42 is past that).
llama_model_bailing_hybrid::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "bailing-hybrid MTP requires nextn_predict_layers > 0");

    const int il = (int) (hparams.n_layer - hparams.nextn_predict_layers) + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.nextn_predict_layers &&
                "nextn_layer_offset out of range [0, nextn_predict_layers)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp  && "MTP block missing MoE router (expected MoE, not dense)");
    GGML_ASSERT(hparams.is_mla()    && "bailing-hybrid MTP requires MLA cache path");
    GGML_ASSERT(!hparams.is_recurrent(il) && "MTP block must be gated MLA, not KDA");

    const int64_t n_head               = hparams.n_head();
    const int64_t n_embd_head_k_mla    = hparams.n_embd_head_k_mla();               // 192
    const int64_t n_embd_head_v_mla    = hparams.n_embd_head_v_mla();               // 128
    const int64_t kv_lora_rank         = hparams.n_lora_kv;                         // 512
    const int64_t n_embd_head_qk_rope  = hparams.n_rot();                           // 64
    const int64_t n_embd_head_qk_nope  = n_embd_head_k_mla - n_embd_head_qk_rope;   // 128
    const float   kq_scale_mla         = 1.0f / sqrtf((float) n_embd_head_k_mla);

    const int    rope_type  = LLAMA_ROPE_TYPE_NORM;
    const int    n_rot      = (int) n_embd_head_qk_rope;
    const float  freq_base  = hparams.rope_freq_base_train;
    const float  freq_scale = hparams.rope_freq_scale_train;
    const float  ext_factor = cparams.yarn_ext_factor;
    const float  attn_factor = cparams.yarn_attn_factor;
    const float  beta_fast  = cparams.yarn_beta_fast;
    const float  beta_slow  = cparams.yarn_beta_slow;
    const int    n_ctx_orig = cparams.n_ctx_orig_yarn;

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    // Ling: no nextn.embed_tokens — borrow main embeddings.
    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    GGML_ASSERT(tok_embd_w && "bailing-hybrid MTP: missing token embeddings");

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    // MTP context uses a plain MLA K-cache (see create_memory mtp_on_hybrid).
    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    // ---- gated MLA decoder block (mirrors main-graph MLA path) ----
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    k_pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, k_pe), inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "mtp_k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * attn_out = nullptr;

    if (layer.wk_b && layer.wv_b) { // MLA KV cache
        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
                         ggml_row_size(Qcur->type, n_embd_head_k_mla),
                         ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);

        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(Qcur->type, n_embd_head_k_mla),
            ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
            ggml_row_size(Qcur->type, n_embd_head_qk_nope));

        q_pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_pe), inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "mtp_q_pe", il);

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "mtp_Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        // wo applied after head-wise gate
        attn_out = build_attn(inp_attn_k, nullptr, NULL, layer.wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
    } else {
        GGML_ABORT("bailing-hybrid MTP: fused wkv_b path not supported; need wk_b/wv_b");
    }
    cb(attn_out, "mtp_attn_out", il);

    // HEAD-WISE sigmoid gate (same as main graph)
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, gate);
    gate = ggml_reshape_3d(ctx0, gate, 1, n_head, n_tokens);
    cb(gate, "mtp_attn_gate_headwise", il);

    attn_out = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v_mla, n_head, n_tokens);
    attn_out = ggml_mul(ctx0, attn_out, gate);
    attn_out = ggml_cont_2d(ctx0, attn_out, n_embd_head_v_mla * n_head, n_tokens);
    cb(attn_out, "mtp_attn_gated", il);

    cur = ggml_mul_mat(ctx0, layer.wo, attn_out);
    cb(cur, "mtp_mla_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // MoE (blk.42 is past leading_dense_block_count)
    ggml_tensor * moe_out = build_moe_ffn(cur,
        layer.ffn_gate_inp,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        hparams.n_expert,
        hparams.n_expert_used,
        LLM_FFN_SILU, hparams.expert_weights_norm,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    // final_layernorm -> nextn.shared_head_norm (or main output_norm fallback)
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "bailing-hybrid MTP: missing shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_pre_norm = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    // Ling: no nextn.shared_head_head — borrow main output head.
    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "bailing-hybrid MTP: missing LM head");
    cur = layer.nextn.shared_head_head
            ? build_lora_mm(head_w, cur, layer.nextn.shared_head_head_s)
            : build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
