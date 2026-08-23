#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <cmath>
#include <stdexcept>
#include <vector>

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false) &&
        !ml.get_arr("dflash.dflash.target_layer_ids", target_layer_ids, false) &&
        !ml.get_arr("dflash.capture_layer_ids", target_layer_ids, false) &&
        !ml.get_arr("deepseek4-dflash-draft.dflash.capture_layer_ids", target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers', 'dflash.target_layer_ids', or 'dflash.capture_layer_ids' in GGUF metadata");
    }

    // The encoder fc fuses the extracted target hidden states, so its input width is
    // n_target_layers * target_hidden_size. Fall back to the draft n_embd when the key is
    // absent (draft==target, the common case) so existing GGUFs load unchanged; only when a
    // GGUF carries a differing target_hidden_size does this diverge from the old behavior.
    uint32_t n_embd_tgt = hparams.n_embd;
    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt, false);
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;
    LLAMA_LOG_INFO("%s: DFlash n_embd_tgt = %u (draft n_embd = %u)\n", __func__, n_embd_tgt, hparams.n_embd);

    LLAMA_LOG_INFO("%s: DFlash extract_layers = [", __func__);
    if (hparams.n_ctx_train == 0) hparams.n_ctx_train = 4096;
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        LLAMA_LOG_INFO("%d%s", target_layer_ids[i], i + 1 < target_layer_ids.size() ? ", " : "");
    }
    LLAMA_LOG_INFO("]\n");

    // DeepSeek-V4 DSpark uses three full, uncompressed DSV4 stages.
    // Map the upstream DSpark metadata onto this fork's existing DSV4 hparams.
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.n_hc, false);
    if (hparams.n_hc > 1) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
        if (!ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false)) {
            hparams.n_swa = 4096;
        }
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
        if (!ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,    hparams.expert_weights_norm, false)) {
            hparams.expert_weights_norm = 1.0f;
        }
        std::string gating_str;
        if (ml.get_key(LLM_KV_EXPERT_GATING_FUNC, gating_str, false) && gating_str == "softmax") {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
        } else if (ml.get_key(LLM_KV_EXPERT_GATING_FUNC, gating_str, false) && gating_str == "sigmoid") {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        } else if (ml.get_key(LLM_KV_EXPERT_GATING_FUNC, gating_str, false) && gating_str == "sqrtsoftplus") {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS;
        } else if (!ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false)) {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS;
        }
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer, false)) {
            std::fill(hparams.swiglu_clamp_exp.begin(), hparams.swiglu_clamp_exp.end(), 1.0f);
        }
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer, false)) {
            hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
        }
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, hparams.n_attn_out_groups);
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,   hparams.n_lora_o);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, hparams.hc_sinkhorn_iters);
        ml.get_key(LLM_KV_HYPER_CONNECTION_EPS,            hparams.hc_eps);

        std::vector<uint32_t> compress_ratios;
        ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, compress_ratios, false);
        if (!compress_ratios.empty() && compress_ratios.size() < hparams.n_layer) {
            throw std::runtime_error("DSpark DSV4 compress_ratios is shorter than block_count");
        }
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            const uint32_t ratio = compress_ratios.empty() ? 0 : compress_ratios[il];
            hparams.attn_compress_ratio[il] = ratio;
            if (ratio != 0) {
                throw std::runtime_error("DSpark DSV4 draft expects uncompressed attention on all stages");
            }
        }

        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS) {
            throw std::runtime_error("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        if (hparams.n_swa == 0) {
            throw std::runtime_error("DSpark DSV4 draft requires a non-zero sliding window");
        }
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0, false);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // Optional interleaved sliding-window attention for legacy DFlash drafters.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    // Which target architecture this drafter was trained against. The laguna drafters
    // carry extra tensors and a different KV-injection input; generic DFlash drafters
    // do not, so every laguna-specific behavior below is gated on this.
    std::string decoder_arch;
    ml.get_key("dflash.decoder_arch", decoder_arch, false);
    decoder_laguna = (decoder_arch == "laguna");
    LLAMA_LOG_INFO("%s: DFlash decoder_arch = %s (laguna behaviors %s)\n", __func__,
                   decoder_arch.empty() ? "(unset)" : decoder_arch.c_str(),
                   decoder_laguna ? "enabled" : "disabled");

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // DSpark adds a Markov bias and a per-position confidence head to DFlash.
    const struct ggml_tensor * markov_meta = ml.get_tensor_meta("markov_w1.weight");
    if (!markov_meta) markov_meta = ml.get_tensor_meta("dflash.dspark.markov.w1");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {dspark_markov_rank, n_vocab}, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), {dspark_markov_rank, n_vocab}, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), {n_embd + dspark_markov_rank, 1}, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   {1}, TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n",
                __func__, (long long) dspark_markov_rank);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), { n_embd }, 0);

    if (hparams.n_hc > 1) {
        const int64_t q_lora_rank     = hparams.n_lora_q;
        const int64_t n_ff_exp        = hparams.n_ff_exp;
        const int64_t n_expert_shared = hparams.n_expert_shared;
        const int64_t n_embd_head     = hparams.n_embd_head_k();
        const int64_t o_groups        = hparams.n_attn_out_groups;
        const int64_t o_lora_rank     = hparams.n_lora_o;
        const int64_t hc_mult         = hparams.n_hc;
        const int64_t hc_dim          = hc_mult * n_embd;
        const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

        output_hc_fn    = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_FN,    "weight"), {hc_dim, hc_mult}, 0);
        output_hc_base  = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_BASE,  "weight"), {hc_mult}, 0);
        output_hc_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_SCALE, "weight"), {1}, 0);

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
            layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
            layer.attn_kv        = create_tensor(tn(LLM_TENSOR_ATTN_KV,        "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head}, 0);
            layer.attn_wo_a      = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,     "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank * o_groups}, 0);
            layer.attn_wo_b      = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,     "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        }
        return;
    }

    // enc.aux_norm [n_embd, n_target_layers]: per-captured-feature RMS-norm weights,
    // applied in the encoder graph before fc (see graph<true> ctor below).
    // Session 9 declared this [n_embd, n_layer] and TENSOR_NOT_REQUIRED, so it silently
    // failed to bind and the norm was never applied -> draft acceptance 0.277.
    // Shape and usage confirmed against poolsideai/llama.cpp @04b2b72 (branch laguna).
    if (decoder_laguna) {
        aux_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_AUX_NORM, "weight"),
                                     { n_embd, (int64_t) target_layer_ids.size() }, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        // Attention output gate (ported from laguna.cpp). Only laguna drafters carry it;
        // it stays nullptr otherwise and the graph skips the gate entirely. Detect
        // per-head vs per-element width from the stored shape rather than assuming.
        if (decoder_laguna) {
            const ggml_tensor * gate_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str().c_str());
            if (gate_meta != nullptr) {
                const int64_t n_gate_per_head = n_head;
                const int64_t n_gate_per_elem = n_embd_head_k * n_head;
                const int64_t n_gate_out      = gate_meta->ne[1];
                if (n_gate_out != n_gate_per_head && n_gate_out != n_gate_per_elem) {
                    GGML_ABORT("DFlash: unexpected attention gate width %lld at layer %d "
                               "(expected %lld per-head or %lld per-element)",
                               (long long) n_gate_out, i, (long long) n_gate_per_head, (long long) n_gate_per_elem);
                }
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, n_gate_out }, 0);
            }
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.n_hc > 1) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model_df = static_cast<const llama_model_dflash &>(model);

    ggml_tensor * cur = build_inp_embd_enc();

    // Laguna drafters RMS-norm each captured target feature before concat + fc.
    // View the concatenated input as [n_feat, n_aux, n_tokens], norm over n_feat,
    // and scale by the stacked per-aux weights [n_feat, n_aux].
    // Ported from poolsideai/llama.cpp @04b2b72 (branch laguna) src/models/dflash.cpp.
    if (model_df.aux_norm_enc != nullptr) {
        const int64_t n_aux  = model_df.aux_norm_enc->ne[1];
        const int64_t n_feat = hparams.n_embd_inp_enc() / n_aux;

        cur = ggml_reshape_3d(ctx0, cur, n_feat, n_aux, n_tokens);
        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model_df.aux_norm_enc);
        cur = ggml_reshape_2d(ctx0, cur, n_feat * n_aux, n_tokens);
        cb(cur, "enc_aux_norm", -1);
    }

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_pre_norm = cur;

    ggml_build_forward_expand(gf, cur);
}

// DSpark Markov-bias and confidence heads. The chained argmax is used only to
// condition the in-graph Markov path; the final draft sampler remains external.
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && model.dspark_conf_proj && "DSpark markov/confidence weights not loaded");

    ggml_tensor * base = res->t_logits;
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    auto it = model.gguf_kv.find("dflash.block_size");
    if (it == model.gguf_kv.end()) {
        it = model.gguf_kv.find("dflash-draft.dflash.block_size");
    }
    if (it == model.gguf_kv.end()) {
        it = model.gguf_kv.find("deepseek4-dflash-draft.dflash.block_size");
    }
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark draft requires dflash.block_size metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride  = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * conf_inp = res->t_embd;
    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    for (int64_t i = 0; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);
        ggml_tensor * bias    = ggml_mul_mat(ctx0, w2, w1_prev);

        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);
        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
        ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
        if (model.dspark_conf_proj_b) {
            conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
        }
        conf = ggml_sigmoid(ctx0, conf);
        cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
    conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
    conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);
    conf = ggml_repeat(ctx0, conf, res->t_embd);
    res->t_h_pre_norm = conf;
    ggml_build_forward_expand(g.gf, conf);

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model_df = static_cast<const llama_model_dflash &>(model);
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // Laguna drafters RMS-norm the injected context features before the KV
            // projections; generic DFlash drafters project the raw features.
            // Ported from poolsideai/llama.cpp @04b2b72 src/models/dflash.cpp:194-201.
            ggml_tensor * kv_inp = inp_g;
            if (model_df.decoder_laguna) {
                kv_inp = build_norm(inp_g, layer.attn_norm, NULL, LLM_NORM_RMS, il);
                cb(kv_inp, "kv_inp_normed", il);
            }

            ggml_tensor * Kcur = build_lora_mm(layer.wk, kv_inp);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, kv_inp);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        // Gate projection on the pre-attention hidden state, ported from
        // laguna.cpp (see laguna.cpp:207-211): computed from the same input
        // as q/k/v, before QK-norm/RoPE, not from the attention output.
        const bool    gated = layer.wqkv_gate != nullptr;
        ggml_tensor * gate  = nullptr;
        if (gated) {
            gate = build_lora_mm(layer.wqkv_gate, noise_norm);
            cb(gate, "attn_gate_proj", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // o_proj is deferred (NULL wo below) so the gate can be applied to the
        // raw attention output first, exactly as laguna does.
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, NULL, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn,      NULL, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        // Softplus output gate, ported from laguna.cpp:237-257. Two shapes,
        // distinguished by the g_proj output dim (matching the load-time
        // detection in load_arch_tensors above):
        //   per-head    : gate [n_head, n_tokens] -> reshape to
        //                 [1, n_head, n_tokens] and broadcast over head_dim.
        //   per-element : gate [n_head*head_dim, n_tokens] -> direct ggml_mul.
        if (gated) {
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            if (layer.wqkv_gate->ne[1] == n_head) {
                cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head, n_head, n_tokens);
                gate = ggml_reshape_3d(ctx0, gate, 1,           n_head, n_tokens);
                cur  = ggml_mul(ctx0, cur, gate);
                cur  = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_tokens);
            } else {
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "attn_gated", il);
        }

        cur = build_lora_mm(layer.wo, cur);
        cb(cur, "attn_o_proj", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

namespace {

struct dspark_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

static ggml_tensor * dspark_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    return ggml_view_2d(ctx, scale, 1, 1, scale->nb[0], idx * scale->nb[0]);
}

static ggml_tensor * dspark_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    return ggml_view_2d(ctx, base, n, 1, base->nb[0], off * base->nb[0]);
}

static ggml_tensor * dspark_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dspark_mul_mat_hadamard(
        ggml_context * ctx,
        ggml_tensor  * cur,
        ggml_tensor  * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = ggml_is_contiguous(cur)
        ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n)
        : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

static dspark_hc_mix dspark_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int             sinkhorn_iters,
        float           hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat);
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(
            ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre  = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens,
            split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre  = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens);
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return {y, mixes, pre, post, comb};
}

static ggml_tensor * dspark_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd && residual->ne[1] == n_hc && residual->ne[2] == n_tokens);
    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static ggml_tensor * dspark_hc_head(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        float           hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat);
    pre = ggml_mul(ctx, pre, dspark_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dspark_view_base(ctx, hc_base, n_hc, 0));
    pre = dspark_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);
    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dspark_apply_rope_tail(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * inp_pos,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_tokens,
        int64_t        n_rot,
        int             rope_type,
        float           freq_base,
        bool            inverse) {
    GGML_UNUSED(n_head);
    GGML_UNUSED(n_tokens);

    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)
            : ggml_rope_ext(ctx, x, inp_pos, nullptr, n_rot, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    GGML_ASSERT(n_rot < n_embd_head);
    return ggml_dsv4_rope_tail(ctx, x, inp_pos, nullptr, n_rot, rope_type,
            0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, inverse);
}

static ggml_tensor * dspark_grouped_out(
        ggml_context * ctx,
        ggml_tensor  * out,
        ggml_tensor  * wo_a,
        ggml_tensor  * wo_b,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_groups,
        int64_t        o_lora_rank,
        int64_t        n_tokens) {
    GGML_ASSERT(n_groups > 0 && n_head % n_groups == 0);
    const int64_t group_dim = n_embd_head * (n_head / n_groups);
    out = ggml_cont(ctx, out);
    out = ggml_reshape_3d(ctx, out, group_dim, n_groups, n_tokens);
    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);
    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, out, ids);
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
    return ggml_mul_mat(ctx, wo_b, low);
}

} // namespace

// DSV4 DSpark decoder, dual-mode by batch type:
//   * embd batch  -> inject the fused target features into each stage's MLA K cache
//   * token batch -> run the three DSV4 stages, then the Markov/confidence heads
llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_hc             = hparams.n_hc;
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_lora_o         = hparams.n_lora_o;
    const int64_t n_out_groups     = hparams.n_attn_out_groups;

    GGML_ASSERT(n_hc > 1);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());
    GGML_ASSERT(n_embd_head_rope <= n_embd_head);
    GGML_ASSERT(n_out_groups > 0 && n_head % n_out_groups == 0);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_k_iswa();

    // Encoder output injection: populate the draft stage caches from target features.
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);
        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * kv = build_lora_mm(layer.attn_kv, inp_g);
            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = dspark_apply_rope_tail(ctx0, kv, inp_pos,
                    n_embd_head, 1, n_tokens, n_embd_head_rope,
                    rope_type, freq_base, false);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa) {
                kv = dspark_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }
            ggml_build_forward_expand(gf,
                    inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr &&
                "DSpark decoder requires the target model token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    cb(inpL, "inp_noise_embd", -1);
    res->add_input(std::move(inp));

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    const float kq_scale = 1.0f/std::sqrt(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        dspark_hc_mix mix = dspark_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps,
                hparams.hc_sinkhorn_iters, hparams.hc_eps);

        ggml_tensor * cur = mix.x;
        cb(cur, "hc_attn_pre", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "q_lora_norm", il);

        ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
        q = dspark_apply_rope_tail(ctx0, q, inp_pos,
                n_embd_head, n_head, n_tokens, n_embd_head_rope,
                rope_type, freq_base, false);
        cb(q, "Qcur", il);

        ggml_tensor * kv = build_lora_mm(layer.attn_kv, cur);
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
        kv = dspark_apply_rope_tail(ctx0, kv, inp_pos,
                n_embd_head, 1, n_tokens, n_embd_head_rope,
                rope_type, freq_base, false);
        cb(kv, "KVcur", il);

        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                q, kv, nullptr,
                nullptr, layer.attn_sinks, nullptr,
                kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
        cur = dspark_apply_rope_tail(ctx0, cur, inp_pos,
                n_embd_head, n_head, n_tokens, n_embd_head_rope,
                rope_type, freq_base, true);
        cur = dspark_grouped_out(ctx0, cur,
                layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head, n_head, n_out_groups, n_lora_o, n_tokens);
        cb(cur, "attn_out", il);

        inpL = dspark_hc_post(ctx0, cur, residual, mix.post, mix.comb,
                n_embd, n_hc, n_tokens);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        mix = dspark_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps,
                hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        inpL = dspark_hc_post(ctx0, cur, residual, mix.post, mix.comb,
                n_embd, n_hc, n_tokens);
        cb(inpL, "hc_ffn_post", il);
    }

    ggml_tensor * cur = dspark_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_eps);
    cb(cur, "hc_head", -1);

    // DSpark confidence is trained on the collapsed hidden state before final norm.
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr &&
                "DSpark decoder requires the target output projection");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}
