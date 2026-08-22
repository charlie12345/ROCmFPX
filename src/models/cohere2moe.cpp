#include "models.h"

// Port of ggml-org/llama.cpp cohere2moe into ROCmFPX fork.
// Adapted to fork APIs:
//   - hparams.n_layer field (not n_layer())
//   - hparams.nextn_predict_layers (not n_layer_nextn / n_layer_all)
//   - no embeddings_nextn_masked / t_h_nextn (fork lacks those result slots)
//   - SWA pattern loaded as bool array into hparams.swa_layers
// Bias tensors: HF Cohere2 MoE ships per-expert FC biases, but conversion
// drops them when zero (mainline convert raises if non-zero). Production
// North-Mini-Code GGUF has 0 bias tensors; graph therefore passes nullptr
// biases, matching mainline. Optional TENSOR_NOT_REQUIRED bias loads are
// included so a future non-zero-bias GGUF can bind without recompile.

void llama_model_cohere2moe::load_arch_hparams(llama_model_loader & ml) {
    const bool found_norm     = ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps,     false);
    const bool found_norm_rms = ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (!found_norm && !found_norm_rms) {
        throw std::runtime_error("missing Cohere2 MoE norm epsilon");
    }
    if (!found_norm_rms) {
        hparams.f_norm_rms_eps = 0.0f;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    // Prefer full per-layer bool pattern (North Mini GGUF ships this);
    // fall back to period integer used by some cohere2 exports.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer, false)) {
        uint32_t swa_period = 4;
        if (ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false)) {
            hparams.set_swa_pattern(swa_period, true);
        } else {
            hparams.set_swa_pattern(swa_period, true);
        }
    }

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    // MTP layers (if any) are the last nextn_predict_layers of n_layer, same
    // convention as glm4-moe in this fork.
    if (hparams.nextn_predict_layers > 0) {
        hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
    }

    switch (hparams.n_layer) {
        case 49: type = LLM_TYPE_30B_A3B; break; // North Mini Code (48 trunk + optional nextn)
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_cohere2moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0 for Cohere2Moe");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for Cohere2Moe");
    }

    const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        int flags = 0;
        const bool is_mtp = hparams.nextn_predict_layers > 0 &&
                            static_cast<uint32_t>(i) >= static_cast<uint32_t>(n_transformer_layers);
        if (is_mtp) {
            // Load MTP tensors but skip execution in the main graph (glm4-moe pattern).
            flags |= TENSOR_SKIP;
        }

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, flags);

        // Q: n_embd x (n_embd_head_k * n_head); K/V: n_embd x n_embd_gqa
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        // Optional attention output bias (HF may ship zeros; production GGUF has none).
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);

        if (!is_mtp && static_cast<uint32_t>(i) < hparams.n_layer_dense_lead) {
            // Leading dense FFN (layer 0 on North Mini)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);

            // Optional dense FC biases
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), { n_ff },   flags | TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), { n_ff },   flags | TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff;

            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, flags);

            // Optional per-expert FC biases (mainline convert drops zeros; bind if present).
            // Merged expert bias tensors would be named like ffn_*_exps.bias if ever exported.
            layer.ffn_gate_exps_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), { n_ff_exp, n_expert }, flags | TENSOR_NOT_REQUIRED);
            layer.ffn_up_exps_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), { n_ff_exp, n_expert }, flags | TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), { n_embd, n_expert },   flags | TENSOR_NOT_REQUIRED);
            layer.ffn_gate_inp_b  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), { n_expert },           flags | TENSOR_NOT_REQUIRED);

            if (hparams.n_expert_shared > 0) {
                const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        }

        if (is_mtp) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },              flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },              flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },     flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },     flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },              flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_cohere2moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_cohere2moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const llm_norm_type cohere2moe_norm_type = hparams.f_norm_rms_eps == 0.0f ? LLM_NORM : LLM_NORM_RMS;
    const float f_logit_scale = hparams.f_logit_scale;

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Skip MTP tail layers in the main decoder pass (glm4-moe convention).
    const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;

    for (int il = 0; il < n_transformer_layers; ++il) {
        const bool is_swa = hparams.is_swa(il);
        // Dense-prefix full-attention layers use RoPE; later layers follow SWA pattern.
        const bool force_rope = static_cast<uint32_t>(il) < hparams.n_layer_dense_lead;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, cohere2moe_norm_type, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * ffn_inp = cur;

        {
            const auto & layer = model.layers[il];

            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
                    n_embd_head, n_head, n_head_kv, il);

            if (is_swa || force_rope) {
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur     = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL    = ggml_get_rows(ctx0, inpL, inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        ggml_tensor * attn_out = cur;

        const auto & layer = model.layers[il];

        if (layer.ffn_gate_inp == nullptr) {
            cur = build_ffn(ffn_inp,
                    layer.ffn_up,   layer.ffn_up_b,   layer.ffn_up_s,
                    layer.ffn_gate, layer.ffn_gate_b, layer.ffn_gate_s,
                    layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // Prefer bias-aware overload if any expert bias is present; else weight-only path.
            if (layer.ffn_up_exps_b || layer.ffn_gate_exps_b || layer.ffn_down_exps_b || layer.ffn_gate_inp_b) {
                cur = build_moe_ffn(ffn_inp,
                        layer.ffn_gate_inp,  layer.ffn_gate_inp_b,
                        layer.ffn_up_exps,   layer.ffn_up_exps_b,
                        layer.ffn_gate_exps, layer.ffn_gate_exps_b,
                        layer.ffn_down_exps, layer.ffn_down_exps_b,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il,
                        nullptr, layer.ffn_gate_up_exps, layer.ffn_gate_up_exps_b,
                        layer.ffn_up_exps_s,
                        layer.ffn_gate_exps_s,
                        layer.ffn_down_exps_s);
            } else {
                cur = build_moe_ffn(ffn_inp,
                        layer.ffn_gate_inp,
                        layer.ffn_up_exps,
                        layer.ffn_gate_exps,
                        layer.ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il,
                        nullptr, layer.ffn_gate_up_exps,
                        layer.ffn_up_exps_s,
                        layer.ffn_gate_exps_s,
                        layer.ffn_down_exps_s);
            }
            cb(cur, "ffn_moe_out", il);

            if (layer.ffn_up_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(ffn_inp,
                        layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                        layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                        layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, cur, ffn_shexp);
                cur = ggml_scale(ctx0, cur, 0.5f);
                cb(cur, "ffn_out", il);
            }
        }

        // Parallel residual: residual + FFN + attention (Cohere-style)
        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_add(ctx0, cur, attn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, cohere2moe_norm_type, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
