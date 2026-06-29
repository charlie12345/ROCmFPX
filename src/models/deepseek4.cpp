#include "models.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,  hparams.n_lora_o);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,hparams.n_attn_out_groups);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0, false);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.compress_rope_freq_base, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k, false);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,             hparams.n_hash_layers);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,          hparams.n_hc);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, hparams.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPS,            hparams.hc_eps);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp, hparams.n_layer, false);

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS;
    }

    std::vector<uint32_t> compress_ratios;
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, compress_ratios);
    if (compress_ratios.size() < hparams.n_layer) {
        throw std::runtime_error(format("DeepSeek V4 compress ratio count mismatch: got %zu, expected %u",
                    compress_ratios.size(), hparams.n_layer));
    }
    std::copy_n(compress_ratios.begin(), hparams.n_layer, hparams.attn_compress_ratio.begin());

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const uint32_t ratio = hparams.attn_compress_ratio[il];
        if (ratio == 0) {
            continue;
        }

        const uint32_t coff = ratio == 4 ? 2 : 1;
        uint32_t state_size = coff * ratio * coff * hparams.n_embd_head_k(il);
        if (ratio == 4) {
            state_size += coff * ratio * coff * hparams.indexer_head_size;
        }
        hparams.dsv4_state_size = std::max(hparams.dsv4_state_size, state_size);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t o_lora_rank     = hparams.n_lora_o;
    const int64_t n_out_groups    = hparams.n_attn_out_groups;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_hc            = hparams.n_hc;
    const int64_t hc_dim          = n_hc * n_embd;
    const int64_t hc_mix          = (2 + n_hc) * n_hc;

    if (n_out_groups == 0) {
        throw std::runtime_error("DeepSeek V4 requires attention output groups");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), {n_embd}, 0);
    output          = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, 0);
    output_hc_base  = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_BASE,  "weight"), {n_hc}, 0);
    output_hc_fn    = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_FN,    "weight"), {hc_dim, n_hc}, 0);
    output_hc_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_SCALE, "weight"), {1}, 0);

    auto create_compressor = [&](llama_layer & layer, int bid, int64_t ratio, int64_t head_size, bool indexer) {
        const int64_t coff = ratio == 4 ? 2 : 1;
        ggml_tensor *& ape  = indexer ? layer.indexer_compressor_ape  : layer.attn_compressor_ape;
        ggml_tensor *& kv   = indexer ? layer.indexer_compressor_kv   : layer.attn_compressor_kv;
        ggml_tensor *& gate = indexer ? layer.indexer_compressor_gate : layer.attn_compressor_gate;
        ggml_tensor *& norm = indexer ? layer.indexer_compressor_norm : layer.attn_compressor_norm;

        ape  = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_APE  : LLM_TENSOR_ATTN_COMPRESSOR_APE,  "weight", bid), {coff * head_size, ratio}, 0);
        kv   = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_KV   : LLM_TENSOR_ATTN_COMPRESSOR_KV,   "weight", bid), {n_embd, coff * head_size}, 0);
        gate = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_GATE : LLM_TENSOR_ATTN_COMPRESSOR_GATE, "weight", bid), {n_embd, coff * head_size}, 0);
        norm = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_NORM : LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", bid), {head_size}, 0);
    };

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t ratio = hparams.attn_compress_ratio[i];
        const bool is_nextn = hparams.nextn_predict_layers > 0 &&
            static_cast<uint32_t>(i) >= hparams.n_layer - hparams.nextn_predict_layers;

        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix}, 0);
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,   "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,   "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
        layer.attn_kv   = create_tensor(tn(LLM_TENSOR_ATTN_KV,    "weight", i), {n_embd, n_embd_head_k}, 0);
        layer.attn_wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", i), {n_head * n_embd_head_v / n_out_groups, n_out_groups * o_lora_rank}, 0);
        layer.attn_wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", i), {n_out_groups * o_lora_rank, n_embd}, 0);

        if (ratio > 0) {
            create_compressor(layer, i, ratio, n_embd_head_k, false);
        }
        if (ratio == 4) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
            create_compressor(layer, i, ratio, hparams.indexer_head_size, true);
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        if (static_cast<uint32_t>(i) < hparams.n_hash_layers) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, 0);
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, 0);
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, TENSOR_NOT_REQUIRED);
        }

        const auto gate_exps_name = tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i);
        const auto down_exps_name = tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i);
        const auto up_exps_name   = tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i);

        const ggml_tensor * gate_exps_meta = ml.get_tensor_meta(gate_exps_name.str().c_str());
        const ggml_tensor * down_exps_meta = ml.get_tensor_meta(down_exps_name.str().c_str());
        const ggml_tensor * up_exps_meta   = ml.get_tensor_meta(up_exps_name.str().c_str());

        const bool fable_half_width_experts =
            gate_exps_meta && down_exps_meta && up_exps_meta &&
            gate_exps_meta->ne[0] == n_embd / 2 && gate_exps_meta->ne[1] == n_ff_exp     && gate_exps_meta->ne[2] == n_expert &&
            up_exps_meta->ne[0]   == n_embd / 2 && up_exps_meta->ne[1]   == n_ff_exp     && up_exps_meta->ne[2]   == n_expert &&
            down_exps_meta->ne[0] == n_ff_exp / 2 && down_exps_meta->ne[1] == n_embd     && down_exps_meta->ne[2] == n_expert;

        if (fable_half_width_experts) {
            layer.ffn_gate_exps = create_tensor(gate_exps_name, {n_embd / 2,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(down_exps_name, {n_ff_exp / 2, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(up_exps_name,   {n_embd / 2,   n_ff_exp, n_expert}, 0);
        } else {
            layer.ffn_gate_exps = create_tensor(gate_exps_name, {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(down_exps_name, {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(up_exps_name,   {n_embd,   n_ff_exp, n_expert}, 0);
        }

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);

        if (is_nextn) {
            layer.nextn.e_proj           = create_tensor(tn(LLM_TENSOR_NEXTN_E_PROJ,          "weight", i), {n_embd, n_embd}, 0);
            layer.nextn.h_proj           = create_tensor(tn(LLM_TENSOR_NEXTN_H_PROJ,          "weight", i), {n_embd, n_embd}, 0);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,           "weight", i), {n_embd}, 0);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,           "weight", i), {n_embd}, 0);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,"weight", i), {n_embd}, 0);
            layer.nextn.hc_head_base     = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE,    "weight", i), {n_hc}, 0);
            layer.nextn.hc_head_fn       = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,      "weight", i), {hc_dim, n_hc}, 0);
            layer.nextn.hc_head_scale    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE,   "weight", i), {1}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        GGML_ASSERT(hparams.nextn_predict_layers > 0 && "DeepSeek V4 MTP graph requires an appended prediction layer");
    }
    return std::make_unique<llm_build_deepseek4>(*this, params);
}
