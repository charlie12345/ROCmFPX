#include "pack.inc"
#include "prefix.h"
#include <vector>
#include <thread>
#include <cstdlib>
#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <limits>
#include "../rocmfpx-draft-vocab.h"

// Sparse draft-attention threshold for the MTP/nextn layer, in ubatch tokens.
// Below the threshold the indexer (score every block, top-k, block select) costs more than dense
// attention over the whole cache and the layer attends densely instead; the indexer cache is still
// updated, so later sparse ubatches stay correct. 0 and 1 are equivalent.
// The default here is 1 because decode-size sparse draft attention is already
// part of this build's measured baseline; 128 restores dense draft attention at decode sizes.
static int64_t qwen4exp_mtp_qsa_min_t(void) {
    static const int64_t value = [] {
        const char * env = std::getenv("LLAMA_MTP_QSA_MIN_T");
        return env ? (int64_t) std::atoll(env) : (int64_t) 1;
    }();
    return value;
}

#include <cinttypes>
#include "block-graph.inc"

// [TAG_QWEN4_REIMPLEMENT]
// TODO: this graph implementation is pending complete reimplementation - do not use it as a reference

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

// Q4_0 copy of the BF16/F16/F32 LM head for the MTP draft step. The 248K x 2560 head GEMV runs once per drafted
// token; the copy reads about a quarter of the bytes. Only draft proposals use it: the target model verifies every
// drafted token with its own full-precision head, so the accepted output is still the target's, while the quantized
// head can change which tokens get proposed (and so the acceptance rate). On Qwen3.8-Flash-Next HC-Q8 draft
// acceptance on an 8K prompt was unchanged (Q8_0, Q4_K and Q6_K were measured too; Q4_0 was the fastest).
// The copy belongs to the model that owns head_w (owner): built once per head tensor on first use (a few seconds),
// kept in owner->draft_heads and freed by ~llama_model_qwen4exp(), so it cannot outlive the tensor it was built from.
// ROCMFPX_DRAFT_HEAD_QUANT=0 disables it.
static ggml_tensor * qwen4exp_draft_head_quant(const llama_model & owner_base, ggml_tensor * head_w) {
    constexpr ggml_type qtype = GGML_TYPE_Q4_0;
    static const bool enabled = [] {
        const char * v = std::getenv("ROCMFPX_DRAFT_HEAD_QUANT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto * owner = dynamic_cast<const llama_model_qwen4exp *>(&owner_base);
    if (!enabled || owner == nullptr || head_w == nullptr || head_w->buffer == nullptr || ggml_n_dims(head_w) != 2 ||
            (head_w->type != GGML_TYPE_BF16 && head_w->type != GGML_TYPE_F16 && head_w->type != GGML_TYPE_F32) ||
            head_w->ne[0] % ggml_blck_size(qtype) != 0 || !ggml_is_contiguous(head_w)) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(owner->draft_head_mutex);
    for (const auto & e : owner->draft_heads) {
        if (e.src == head_w) {
            return e.t;
        }
    }
    const int64_t ne0 = head_w->ne[0], ne1 = head_w->ne[1];
    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, qtype, ne0, ne1);
    ggml_set_name(q, "mtp_draft_head_q4_0");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_buffer_get_type(head_w->buffer));
    if (buf == nullptr) {
        ggml_free(ctx);
        owner->draft_heads.push_back({ head_w, nullptr, nullptr, nullptr });
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto t0 = ggml_time_us();
    const int64_t rows_per_chunk = 4096;
    const size_t  src_row = ggml_row_size(head_w->type, ne0);
    const size_t  dst_row = ggml_row_size(qtype, ne0);
    std::vector<uint8_t> src(rows_per_chunk * src_row);
    std::vector<float>   f32(rows_per_chunk * ne0);
    std::vector<uint8_t> dst(rows_per_chunk * dst_row);
    const auto * traits = ggml_get_type_traits(head_w->type);
    const int n_threads = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int64_t r0 = 0; r0 < ne1; r0 += rows_per_chunk) {
        const int64_t nr = std::min(rows_per_chunk, ne1 - r0);
        ggml_backend_tensor_get(head_w, src.data(), r0 * src_row, nr * src_row);
        std::vector<std::thread> workers;
        for (int w = 0; w < n_threads; ++w) {
            workers.emplace_back([&, w]() {
                const int64_t b = nr * w / n_threads, e = nr * (w + 1) / n_threads;
                if (e <= b) return;
                if (head_w->type == GGML_TYPE_F32) {
                    std::memcpy(f32.data() + b * ne0, src.data() + b * src_row, (e - b) * src_row);
                } else {
                    traits->to_float(src.data() + b * src_row, f32.data() + b * ne0, (e - b) * ne0);
                }
                ggml_quantize_chunk(qtype, f32.data() + b * ne0, dst.data() + b * dst_row, 0, e - b, ne0, nullptr);
            });
        }
        for (auto & t : workers) t.join();
        ggml_backend_tensor_set(q, dst.data(), r0 * dst_row, nr * dst_row);
    }
    LLAMA_LOG_INFO("%s: built a Q4_0 copy of %s for MTP drafting (%.1f MiB, %.2f s)\n", __func__, head_w->name,
            ggml_nbytes(q) / 1048576.0, (ggml_time_us() - t0) / 1e6);
    owner->draft_heads.push_back({ head_w, ctx, buf, q });
    return q;
}

llama_model_qwen4exp::~llama_model_qwen4exp() {
    for (auto & e : draft_heads) {
        if (e.buf) {
            ggml_backend_buffer_free(e.buf);
        }
        if (e.ctx) {
            ggml_free(e.ctx);
        }
    }
}

static const llama_model & qwen4exp_shared_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("QWEN4EXP MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model (-md), not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("QWEN4EXP MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    if (hparams.n_layer_nextn > 1 || hparams.n_layer_nextn == hparams.n_layer_all) {
        throw std::runtime_error("qwen4exp requires trunk layers and supports at most one MTP layer");
    }
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    {   // The converted GGUF leaves the nextn/MTP layer's compress ratio at 0, but the MTP sidecar ships
        // blk.N.indexer.* and Halogen runs that layer with the same sparse attention as the trunk (one
        // k_attn_qs_bt4x call per target chunk for it), so inherit the trunk's ratio.
        if (hparams.n_layer_nextn > 0 && hparams.indexer_head_size > 0) {
            int32_t trunk_r = 0;
            for (uint32_t j = 0; j < hparams.n_layer(); ++j) {
                if (hparams.dsv4_compress_ratios[j] > 0) { trunk_r = hparams.dsv4_compress_ratios[j]; }
            }
            for (uint32_t j = hparams.n_layer(); j < hparams.n_layer_all && trunk_r > 0; ++j) {
                if (hparams.dsv4_compress_ratios[j] == 0) {
                    hparams.dsv4_compress_ratios[j] = trunk_r;
                    LLAMA_LOG_INFO("%s: nextn layer %u compress ratio 0 -> %d\n", __func__, j, trunk_r);
                }
            }
            LLAMA_LOG_INFO("%s: nextn sparse draft attention threshold LLAMA_MTP_QSA_MIN_T = %lld tokens\n",
                    __func__, (long long) qwen4exp_mtp_qsa_min_t());
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const bool mtp_only    = !ml.files.empty() && (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    if (mtp_only && !ml.load_mtp) {
        throw std::runtime_error("qwen4exp MTP-only weights must be loaded as a draft model");
    }
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);

    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat joined table or one table per n-gram head
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        int64_t table_rows = 0;
        bool has_joined = false;
        if (ml.files.empty()) {
            // llama_model_init_from_user() carries tensor overrides directly in
            // the caller-owned GGUF context and intentionally has no file-backed
            // weight map.
            const int64_t tid = gguf_find_tensor(ml.metadata, ple_name.c_str());
            if (tid >= 0) {
                table_rows = gguf_get_tensor_ne(ml.metadata, tid)[1];
                has_joined = true;
            }
        } else if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            // A file-backed PLE table may live in any shard, so use the unified
            // weight map instead of looking only in the first GGUF context.
            table_rows = ple_w->tensor->ne[1];
            has_joined = true;
        }
        if (has_joined) {
            if (table_rows < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), table_rows, ple_rows));
            }
            ple_rows = table_rows;
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
            if (!ml.files.empty()) {
                ple_reader = load_lazy_reader(ml, ple_name.c_str(), per_layer_tok_embd);
            }
        } else {
            if (!ml.files.empty()) {
                const std::vector<uint64_t> offsets(hparams.ple_head_offsets.begin(), hparams.ple_head_offsets.begin() + hparams.ple_n_heads);
                const std::vector<uint64_t> vocabs(hparams.ple_head_vocab_sizes.begin(), hparams.ple_head_vocab_sizes.begin() + hparams.ple_n_heads);
                ple_reader = load_lazy_reader_segmented(ml, ple_name.c_str(), hparams.ple_head_dim, ple_rows, offsets, vocabs);
            }
            if (!ple_reader) {
                uint32_t ple_layer = 0;
                while (ple_layer < hparams.n_layer_all && !hparams.is_ple(ple_layer)) {
                    ++ple_layer;
                }
                ple_ngram_embd.resize(hparams.ple_n_heads);
                for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
                    ple_ngram_embd[h] = create_tensor(tn(LLM_TENSOR_PLE_NGRAM_EMBD, "weight", ple_layer, h),
                                                      { hparams.ple_head_dim, (int64_t) hparams.ple_head_vocab_sizes[h] }, 0);
                }
            }
        }
    }

    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP | TENSOR_SKIP_IF_VIRTUAL : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        if (il >= n_layer && ml.files.empty() && !ml.load_mtp) {
            continue;
        }
        auto & layer = layers[il];

        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp_arr[il] ? hparams.n_ff_exp_arr[il] : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // matmuls that read the same input: try the merged [a | b] weight the converter writes, else the separate ones
        auto create_merged = [&](llm_tensor t_merged, llm_tensor t_a, llm_tensor t_b, int64_t n_in, int64_t n_a, int64_t n_b,
                ggml_tensor *& merged, ggml_tensor *& a, ggml_tensor *& b) {
            if (flags & TENSOR_SKIP) {
                const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;
                create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, skip | TENSOR_SKIP_IF_VIRTUAL);
                create_tensor(tn(t_a,      "weight", il), { n_in, n_a },       skip);
                create_tensor(tn(t_b,      "weight", il), { n_in, n_b },       skip);
                return;
            }
            merged = create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!merged) {
                a = create_tensor(tn(t_a, "weight", il), { n_in, n_a }, flags);
                b = create_tensor(tn(t_b, "weight", il), { n_in, n_b }, flags);
            }
        };

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        create_merged(LLM_TENSOR_HC_ATTN_DOWN_INJECT, LLM_TENSOR_HC_ATTN_DOWN, LLM_TENSOR_HC_ATTN_INJECT, hc_dim, hc_lr, hc,
                layer.hc_attn_down_inject, layer.hc_attn_down, layer.hc_attn_inject);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        create_merged(LLM_TENSOR_HC_FFN_DOWN_INJECT, LLM_TENSOR_HC_FFN_DOWN, LLM_TENSOR_HC_FFN_INJECT, hc_dim, hc_lr, hc,
                layer.hc_ffn_down_inject, layer.hc_ffn_down, layer.hc_ffn_inject);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            create_merged(LLM_TENSOR_ATTN_QKV_GATE, LLM_TENSOR_ATTN_QKV, LLM_TENSOR_ATTN_GATE, n_embd, key_dim * 2 + value_dim, value_dim,
                    layer.wqkv_z, layer.wqkv, layer.wqkv_gate);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            create_merged(LLM_TENSOR_SSM_BETA_ALPHA_CAT, LLM_TENSOR_SSM_BETA, LLM_TENSOR_SSM_ALPHA, n_embd, n_v_heads, n_v_heads,
                    layer.ssm_beta_alpha_cat, layer.ssm_beta, layer.ssm_alpha);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_NOT_REQUIRED | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags | TENSOR_NOT_REQUIRED);

        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    if (hc_head_norm == nullptr || (hparams.n_layer() > 0 && layers[0].hc_attn_norm == nullptr)) {
        throw std::runtime_error("this model is an MTP draft head without a trunk; "
                                 "load it as a draft of its target model (-md), not on its own");
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor *  w_down_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = nullptr;
    bool pack_di = false;
    if (inject && w_down_inject) {
        // one matmul for both: the hc_down rows, then the hc_inject rows (the converter merged the weights)
        const int64_t n_lo = w_down_inject->ne[1] - hc;
        ggml_tensor * lo_inject = build_lora_mm(w_down_inject, xn);
        // scale the combined rows before the split; both lo and inject use 1/hc
        lo_inject = ggml_scale(ctx0, lo_inject, 1.0f / (float) hc);
        lo      = ggml_view_2d(ctx0, lo_inject, n_lo, nt, lo_inject->nb[1], 0);
        *inject = ggml_view_2d(ctx0, lo_inject, hc, nt, lo_inject->nb[1], ggml_row_size(lo_inject->type, n_lo));
        cb(*inject, "hc_inject", il);
    } else {
        // GGUFs with separate weights: pack them for one matmul at prompt sizes
        pack_di = nt >= 128 && inject &&
            loras->empty() && ggml_is_quantized(w_down->type) && w_down->type == w_inject->type &&
            ggml_is_matrix(w_down) && ggml_is_matrix(w_inject) &&
            ggml_is_contiguous(w_down) && ggml_is_contiguous(w_inject) &&
            w_down->ne[0] == w_inject->ne[0];
        if (pack_di) {
            ggml_build_forward_expand(gf, xn);
            ggml_tensor * weights = ggml_concat(ctx0, w_down, w_inject, 1);
            cb(weights, "hc_down_inject_weights", il);
            ggml_tensor * projected = ggml_mul_mat(ctx0, weights, xn);
            cb(projected, "hc_down_inject", il);
            lo = ggml_cont(ctx0, ggml_view_2d(ctx0, projected, w_down->ne[1], nt, projected->nb[1], 0));
            // build_hc_combine takes inject pre-scaled by 1/hc
            *inject = ggml_scale(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, projected, w_inject->ne[1], nt,
                    projected->nb[1], w_down->ne[1] * sizeof(float))), 1.0f / (float) hc);
            ggml_build_forward_expand(gf, *inject);
        } else {
            lo = build_lora_mm(w_down, xn);
        }
        lo = ggml_scale(ctx0, lo, 1.0f / (float) hc);
    }
    lo = ggml_silu(ctx0, lo);
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject && !w_down_inject) {
        if (!pack_di) {
            *inject = ggml_scale(ctx0, build_lora_mm(w_inject, xn), 1.0f / (float) hc);
        }
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    ggml_build_forward_expand(gf, residual);
    ggml_build_forward_expand(gf, block_out);
    ggml_build_forward_expand(gf, inject);

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, inject);
    w = ggml_scale(ctx0, w, 2.0f);
    ggml_build_forward_expand(gf, w);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                model.layers[il].hc_attn_down_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                model.layers[il].hc_ffn_down_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The draft head consumes the final wide residual, before the LM-head mixer collapses it.
    // An unmasked request needs every position, so defer the output-row crop until after this
    // hand-over.
    if (cparams.embeddings_nextn) {
        res->t_h_nextn = res_hc;
        cb(res->t_h_nextn, "h_nextn", -1);
    }

    if (crop_after_nextn(inp_out_ids)) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    static const bool pad_head = [] {
        const char * value = std::getenv("ROCMFPX_QWEN4_PAD_HEAD");
        return value && value[0] == '1';
    }();
    const int64_t head_rows = cur->ne[1];
    const bool use_padded_head = pad_head && model.output->type == GGML_TYPE_BF16 &&
        ggml_is_matrix(cur) && head_rows >= 2 && head_rows <= 3;
    if (use_padded_head) {
        // Select the matrix kernel without adding transformer verification rows.
        cur = ggml_pad(ctx0, cur, 0, 4 - head_rows, 0, 0);
    }
    cur = build_lora_mm(model.output, cur, model.output_s);
    if (use_padded_head) {
        cur = ggml_view_2d(ctx0, cur, cur->ne[0], head_rows, cur->nb[1], 0);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    // token rows arrive as ids; multimodal (mtmd) rows arrive as projected embeddings, both are accepted
    GGML_ASSERT((ubatch.token || ubatch.embd) && "QWEN4EXP MTP requires token or embedding input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t n_embd_inp = hparams.n_embd_inp();
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    ggml_tensor * hc_head_norm = layer.nextn.hc_head_norm ? layer.nextn.hc_head_norm : model.hc_head_norm;
    ggml_tensor * hc_head_down = layer.nextn.hc_head_down ? layer.nextn.hc_head_down : model.hc_head_down;
    ggml_tensor * hc_head_up   = layer.nextn.hc_head_up   ? layer.nextn.hc_head_up   : model.hc_head_up;

    if (hc_head_norm == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output_hc_norm.weight");
        hc_head_norm = other.hc_head_norm;
        hc_head_down = other.hc_head_down;
        hc_head_up   = other.hc_head_up;
    }
    GGML_ASSERT(hc_head_norm && "MTP block missing hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // the embedding rows are n_embd_inp wide (mtmd projection width); the state rows are hc_dim wide
    auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd_inp, hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    if (tok_embd_w == nullptr) {
        tok_embd_w = qwen4exp_shared_model(cparams, model, "token_embd.weight").tok_embd;
    }
    ggml_tensor * tok_embd = nullptr;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        // image rows: the target encoder already projected them into the token-embedding space
        GGML_ASSERT(n_embd_inp == tok_embd_w->ne[0] && "QWEN4EXP MTP: mtmd embedding width != token embedding width");
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the MTP context is a hybrid-idx memory, so the draft head can use the same sparse
    // attention path as the trunk (this is what Halogen does: one sparse attention call per target chunk)
    llm_graph_input_attn_kv * inp_attn = nullptr;
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;
    if (hparams.indexer_head_size > 0) {
        auto * inp_hyb = build_inp_mem_hybrid();
        const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);
        if (m->get_idx() != nullptr) {
            mctx_hyb = m;
            inp_attn = inp_hyb->get_attn();
            // the MTP graph has no recurrent layers, so the hybrid input's recurrent tensors are never used and the
            // allocator skips them -- set_input would then hit a null buffer. Give them a trivial use.
            auto * rs = inp_hyb->get_recr();
            for (ggml_tensor * t : { rs->s_copy, rs->s_copy_main, rs->s_copy_tail }) {
                if (t) { ggml_build_forward_expand(gf, ggml_scale(ctx0, ggml_cast(ctx0, t, GGML_TYPE_F32), 0.0f)); }
            }
        }
    }
    if (!inp_attn) { inp_attn = build_attn_inp_kv(); }

    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // per stream, not pooled: pooling before the projection discards the hyper-connection residual.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // The eh projection sees concat as [2*n_embd, hc, n_tokens], so mul_mat batches over the
    // token axis with only hc(=4) columns per batch and dispatches a 4-column MMVQ. Flattening the
    // hc and token axes into a single column axis exposes one wide MMQ GEMM instead. The weight is
    // shared across the merged axes, so the arithmetic is identical up to the kernel's own rounding;
    // this is the MTP draft head, so it cannot change the target prefill logits.
    ggml_tensor * concat_flat = ggml_reshape_2d(ctx0, concat, concat->ne[0], concat->ne[1] * concat->ne[2]);
    ggml_tensor * res_flat    = build_lora_mm(layer.nextn.eh_proj, concat_flat, layer.nextn.eh_proj_s);
    ggml_tensor * res_hc      = ggml_reshape_3d(ctx0, res_flat, res_flat->ne[0], concat->ne[1], concat->ne[2]);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, layer.hc_attn_down_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (mctx_hyb) {
        // the converted GGUF leaves compress_ratios[nextn] = 0, but the sidecar ships blk.N.indexer.* and the
        // reference runs this layer sparsely; fall back to the trunk ratio
        const int64_t r        = hparams.dsv4_compress_ratios[il];   // patched at load for the nextn layer
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = (int64_t) hparams.indexer_top_k + r - 1;
        ggml_tensor * top_k = nullptr;
        // The private split-D decode kernel only covers 1..8 queries and the qsa3 path declines below
        // 128 queries, so 9..127 has no sparse kernel and attends densely.
        bool sparse_capable = false;
#if defined(GGML_USE_HIP)
        sparse_capable = n_tokens <= 8 && mctx_hyb->get_n_stream() == 1 &&
            cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap;
#endif
        const bool sparse = n_tokens >= qwen4exp_mtp_qsa_min_t() && (n_tokens >= 128 || sparse_capable);
        if (r > 0 && n_kv_idx > width && sparse) {
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, il);
        } else if (r > 0) {
            build_qsa_store_k(mctx_hyb, cur, il);
        }
        if (top_k) {
            cur = build_attn_qsa(inp_attn, Qcur, Kcur, Vcur, top_k, kq_scale, il);
        } else {
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
    } else {
    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, layer.hc_ffn_down_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    cur = build_hc_mix(res_hc,
            hc_head_norm, hc_head_down, hc_head_up,
            nullptr, nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // no res->t_embd: it is n_embd wide, but the context sizes that buffer by n_embd_out.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    const llama_model * head_owner = &model;
    if (head_w == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output.weight");
        head_w = other.output;
        head_s = other.output_s;
        head_owner = &other;
        GGML_ASSERT(head_w && "QWEN4EXP MTP: the target model has no LM head to borrow");
    }

    // candidate-row draft head (one row, plain head only); otherwise the Q4_0 copy of the head
    // (ROCMFPX_DRAFT_VOCAB=0, see qwen4exp_draft_head_quant) or the full head
    auto draft_vocabulary = cparams.draft_vocab;
    if (draft_vocabulary && ggml_nrows(cur) == 1 && head_s == nullptr && loras->empty() &&
            ggml_is_matrix(head_w) && ggml_is_contiguous(head_w) && ggml_is_contiguous(cur) &&
            head_w->ne[1] == draft_vocabulary->vocabulary_size()) {
        auto projection = rocmfpx::build_draft_projection(ctx0, std::move(draft_vocabulary), head_w, cur);
        cur = projection.logits;
        res->add_input(std::move(projection.input));
    } else {
        ggml_tensor * head_q = ggml_nrows(cur) == 1 && head_s == nullptr && loras->empty() ?
                qwen4exp_draft_head_quant(*head_owner, head_w) : nullptr;
        cur = head_q ? ggml_mul_mat(ctx0, head_q, cur) : build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        // qkv and z read the same input: one matmul, then the qkv rows and the z rows
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
// A batch that carries BOTH tokens and embeddings is the MTP draft head: common_speculative's draft-mtp
// impl fills batch.token and batch.embd together (the head consumes the target's h_nextn row per token).
// The original `!ubatch.embd` guard was aimed at pure-embedding (vision) batches, which are already
// excluded by requiring ubatch.token; keeping it shut the draft out of block selection, and with it out of
// the maskless/packed-key layout and the qsa3 attention kernel.
// Decode-sized ubatches (1..8 tokens) go to the bounded sparse decode kernel (ggml-cuda/qsa-decode.cuh) when
// ROCMFPX_QWEN4_QSA_DECODE=1; build_qsa_top_k and build_attn_qsa must agree on it, so both ask here.
static bool qwen4exp_qsa_decode_direct(int64_t n_tokens, int64_t n_query, int64_t n_kv) {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] {
        const char * v = getenv("ROCMFPX_QWEN4_QSA_DECODE");
        return v && strcmp(v, "1") == 0;
    }();
    return enabled && n_tokens >= 1 && n_tokens <= 8 && n_query >= 1 && n_query <= 8 && n_kv >= 8192;
#else
    GGML_UNUSED(n_tokens); GGML_UNUSED(n_query); GGML_UNUSED(n_kv);
    return false;
#endif
}

// Block selection for those decode ubatches as well: the decode kernel skips the -1 sentinels of the complete-block
// list, so it can take the same top-k-of-blocks selection prefill uses instead of the per-cell top-k over the whole
// context (per-cell score expansion, mask add and a top-k over n_kv cells per layer, plus the O(n_kv) CPU block scan
// per ubatch; ~18 ms per token at 165k on gfx1151). The masked dense attention path cannot (set_rows on -1), so this
// is tied to the decode kernel. ROCMFPX_QWEN4_QSA_DECODE_BLOCKS=0 keeps the per-cell selection.
static bool qwen4exp_qsa_decode_blocks(int64_t n_tokens, int64_t n_stream, int64_t n_kv) {
    static const bool enabled = [] {
        const char * v = getenv("ROCMFPX_QWEN4_QSA_DECODE_BLOCKS");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled && n_stream == 1 && qwen4exp_qsa_decode_direct(n_tokens, n_tokens, n_kv);
}

// the block selection of prompt ubatches lists -1 sentinels that only the maskless qsa kernel reads; it exists on
// RDNA3.5 only, so on other cards (and when any layer's device lacks it) prompts keep the masked path
static bool qwen4exp_dev_sparse_maskless(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    using fn_t = bool (*)(ggml_backend_dev_t);
    auto fn = (fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_sparse_attn_maskless");
    return fn != nullptr && fn(dev);
}

static bool qwen4exp_use_block_selection(bool blk_bias, int64_t n_stream, int64_t ratio, int64_t n_kv,
        const llama_ubatch & ubatch, const llama_cparams & cparams, const llama_hparams & hparams, bool sparse_maskless) {
    return blk_bias && n_stream==1 && ratio>1 && hparams.indexer_top_k%ratio==0 &&
        n_kv>hparams.indexer_top_k+ratio-1 && n_kv<=16777216 && ubatch.token &&
        ((ubatch.n_tokens >= 128 && sparse_maskless) || qwen4exp_qsa_decode_blocks(ubatch.n_tokens, n_stream, n_kv)) &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias==0.0f &&
        !hparams.attn_soft_cap && hparams.n_embd_head_k()==256 && hparams.n_embd_head_v()==256;
}

static int64_t qwen4exp_query_strip(int64_t n_tokens, int64_t n_stream);

// [QSA_SCORE_BOUNDS] Visible-prefix scoring (ported from 2026-09-09-sparse-prefill/qsa-score-bounds).
// For one sequence with unique non-negative positions the indexer enumerates complete groups in increasing
// logical block order, so a compressed column's ordinal cannot exceed its logical block number: every block
// visible to any query in a strip lies in the first (max_query_position + 1)/ratio columns. Holes and physical
// cell permutations do not invalidate that upper bound. Blocks past it are -inf in the visibility metadata
// anyway, so trimming them must not change the selection.
static std::vector<int64_t> qwen4exp_score_key_limits(const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch, int64_t blocks, int64_t strip, int64_t ratio, int64_t budget, bool compact) {
    // llama_context::graph_reserve builds a worst-case graph from a synthetic ubatch whose positions are all 0.
    // Bounding that graph would reserve compute buffers for a 4%-wide scorer and then execute full-width ones, so a
    // many-token ubatch whose positions are all identical is treated as synthetic and left unbounded.
    bool degenerate_pos = ubatch.pos && ubatch.n_tokens > 1;
    for (uint32_t i = 1; degenerate_pos && i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != ubatch.pos[0]) { degenerate_pos = false; }
    }
    if (!compact || ubatch.n_tokens<128 || degenerate_pos || !mctx->qsa_position_prefix(ubatch)) {
        return {};
    }
    auto limits = qsa_prefix_limits(ubatch.pos,ubatch.n_tokens,strip,ratio,blocks,budget);
    return limits;
}

// Decode-sized ubatches (fewer than 128 tokens, cell-expanded selection) read the pooled-key cache as well: without
// it every decode step gathers, pools, normalizes and rotates every indexer block of the context again, which grows
// linearly with the context (~15 ms per token at 48k on gfx1151). The cache rows are the prefix's logical blocks and
// qsa_decode_cacheable guarantees the generic block ids coincide with them. ROCMFPX_QWEN4_QSA_DECODE_CACHE=0 disables it.
static bool qwen4exp_qsa_decode_cached(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ubatch,
        uint32_t ratio, bool compact, int64_t n_stream) {
    static const bool enabled = [] {
        const char * v = getenv("ROCMFPX_QWEN4_QSA_DECODE_CACHE");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled && ratio == 4 && !compact && n_stream == 1 && ubatch.n_tokens < 128 &&
        mctx->qsa_decode_cacheable(ubatch);
}

class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, bool causal_attn) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), causal_attn(causal_attn) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        if (tail_idxs) {
            mctx->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio);
        } else {
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, causal_attn);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = ((int64_t) mctx->qsa_n_kv_window() + ratio - 1)/ratio;

        bool res = incremental_prefix == (ratio == 4 && mctx->qsa_prefix_matches(params.ubatch));
        res &= decode_cached == qwen4exp_qsa_decode_cached(mctx, params.ubatch, ratio, compact, n_stream);

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0] == (compact ? n_blocks+params.ubatch.n_tokens : (blk_bias ? n_blocks : n_kv));
        res &= compact || bias->ne[1] == params.ubatch.n_tokens/n_stream;
        const bool blocks=qwen4exp_use_block_selection(blk_bias,n_stream,ratio,n_kv,
                params.ubatch,params.cparams,params.hparams,sparse_maskless);
        const bool scalar=blocks && causal_attn && params.hparams.n_swa==0 && mctx->qsa_scalar_visibility(params.ubatch);
        res &= (tail_idxs != nullptr) == scalar;
        res &= compact == scalar;
        res &= maskless == scalar;
        if (tail_idxs) { res &= tail_idxs->ne[1] == params.ubatch.n_tokens/n_stream; }
        // [QSA_SCORE_BOUNDS] the trimmed widths are baked into the graph, so a reused graph must agree on them
        const int64_t next_strip=qwen4exp_query_strip(params.ubatch.n_tokens/n_stream,n_stream);
        const auto next_limits=qwen4exp_score_key_limits(mctx,params.ubatch,n_blocks,next_strip,ratio,
                params.hparams.indexer_top_k/ratio,scalar && compact);
        res &= score_strip==next_strip;
        res &= score_key_limits==next_limits;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    ggml_tensor * tail_idxs = nullptr;
    bool incremental_prefix = false;
    bool decode_cached = false;
    bool compact = false;
    bool maskless = false;
    bool sparse_maskless = false;   // every layer's device has the maskless qsa kernel
    int64_t score_strip = 0;
    std::vector<int64_t> score_key_limits;       // [QSA_SCORE_BOUNDS] per strip, empty = no bound

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // this is fixed for the graph's lifetime, as causal_attn is part of the reuse key (llm_graph_params::allow_reuse)
    const bool causal_attn;
};

class llama_model_qwen4exp::llm_graph_input_qsa_k : public llm_graph_input_i {
public:
    llm_graph_input_qsa_k(const llama_memory_hybrid_idx_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_qsa_k() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        return k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    ggml_tensor * k_idxs = nullptr;   // I32 [n_tokens]

    const llama_memory_hybrid_idx_context * mctx;
};

class llm_graph_input_qsa_cache : public llm_graph_input_i {
public:
    llm_graph_input_qsa_cache(const llama_memory_hybrid_idx_context * mctx, int il, bool fast) :
        mctx(mctx), il(il), fast(fast) {}
    void set_input(const llama_ubatch *) override {
        if (fast) { mctx->qsa_fill_updates(members, positions, rows); }
        mctx->qsa_commit(il);
    }
    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        return mctx->qsa_prefix_matches(params.ubatch) && mctx->qsa_fast(il, params.ubatch) == fast &&
            (!fast || rows->ne[0] == (params.ubatch.n_tokens+3)/4+1);
    }
    const llama_memory_hybrid_idx_context * mctx;
    int il;
    bool fast;
    ggml_tensor * members = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * rows = nullptr;
};

void llama_model_qwen4exp::graph::build_qsa_store_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;

    if (qsa_k_inp == nullptr) {
        auto inp = std::make_unique<llm_graph_input_qsa_k>(mctx_hyb);
        inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa_k_inp = inp.get();
        res->add_input(std::move(inp));
    }

    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, qsa_k_inp->k_idxs, il));
}

static void qwen4exp_append_strip(ggml_context * ctx, ggml_cgraph * graph,
        std::vector<ggml_tensor *> & chunks, ggml_tensor * value) {
    chunks.push_back(value);
    while (chunks.size()>1 && chunks.back()->ne[1]==chunks[chunks.size()-2]->ne[1]) {
        auto * right=chunks.back();chunks.pop_back();
        auto * left=chunks.back();chunks.pop_back();
        auto * joined=ggml_concat(ctx,left,right,1);
        chunks.push_back(joined);
        ggml_build_forward_expand(graph,joined);
    }
    ggml_build_forward_expand(graph,chunks.back());
}

static ggml_tensor * qwen4exp_finish_strips(ggml_context * ctx, ggml_cgraph * graph,
        std::vector<ggml_tensor *> & chunks) {
    while (chunks.size()>1) {
        auto * right=chunks.back();chunks.pop_back();
        auto * left=chunks.back();chunks.pop_back();
        auto * joined=ggml_concat(ctx,left,right,1);
        chunks.push_back(joined);
        ggml_build_forward_expand(graph,joined);
    }
    return chunks.front();
}

static int64_t qwen4exp_query_strip(int64_t n_tokens, int64_t n_stream) {
    return n_stream == 1 ? std::min<int64_t>(n_tokens, 512) : n_tokens;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    // blocks are keyed by position, which can run ahead of the occupied cells (see qsa_n_kv_window)
    const int64_t n_blocks = ((int64_t) mctx_hyb->qsa_n_kv_window() + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask, so it opts out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, cparams.causal_attn);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        bool sparse_maskless = true;
        for (int32_t jl = 0; jl < (int32_t) hparams.n_layer() && sparse_maskless; ++jl) {
            sparse_maskless = qwen4exp_dev_sparse_maskless(model.dev_layer(jl));
        }
        qsa->sparse_maskless = sparse_maskless;
        const bool scalar = qwen4exp_use_block_selection(blk_bias,n_stream,r,n_kv,ubatch,cparams,hparams,sparse_maskless) &&
            cparams.causal_attn && hparams.n_swa==0 && mctx_hyb->qsa_scalar_visibility(ubatch);
        qsa->incremental_prefix = r == 4 && mctx_hyb->qsa_prefix_matches(ubatch);
        qsa->compact = scalar;
        qsa->decode_cached = qwen4exp_qsa_decode_cached(mctx_hyb, ubatch, (uint32_t) r, scalar, n_stream);
        qsa->maskless = scalar;
        qsa->score_strip=qwen4exp_query_strip(n_tps,n_stream);
        qsa->score_key_limits=qwen4exp_score_key_limits(mctx_hyb,ubatch,n_blocks,qsa->score_strip,r,
                hparams.indexer_top_k/r,qsa->compact);
        qsa->bias = qsa->compact ? ggml_new_tensor_1d(ctx0,GGML_TYPE_I32,n_blocks+n_tps) :
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);
        // complete-block selection lists cells with -1 sentinels (invisible blocks, empty tail slots) that only the
        // maskless kernel understands; when the visibility is not scalar (2-D image positions in the cache, several
        // sequences) the attention takes the masked path, whose set_rows would write row -1, so the block-expanded
        // top-k with the per-block bias is used there instead
        if (scalar) {
            GGML_ASSERT(hparams.indexer_top_k % r == 0);
            qsa->tail_idxs=ggml_new_tensor_3d(ctx0,GGML_TYPE_I32,r-1,n_tps,n_stream);
            ggml_set_input(qsa->tail_idxs);
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    ggml_tensor * cached = r == 4 && (inp->compact || inp->decode_cached) && mctx_hyb->qsa_prefix_matches(ubatch)
            ? mctx_hyb->qsa_cache(ctx0, il) : nullptr;
    const bool fast_cache = cached && mctx_hyb->qsa_fast(il, ubatch);
    const int64_t prep_blocks = fast_cache ? (n_tokens+3)/4+1 : n_blocks;
    ggml_tensor * member_rows = inp->blk_cells;
    ggml_tensor * block_positions = inp->blk_pos;
    ggml_tensor * cache_rows = nullptr;
    if (cached) {
        auto cache_input = std::make_unique<llm_graph_input_qsa_cache>(mctx_hyb, il, fast_cache);
        if (fast_cache) {
            member_rows = cache_input->members = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*prep_blocks);
            block_positions = cache_input->positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*prep_blocks);
            cache_rows = cache_input->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, prep_blocks);
            ggml_set_input(member_rows); ggml_set_input(block_positions); ggml_set_input(cache_rows);
        }
        res->add_input(std::move(cache_input));
    }
    const auto prepare_keys = [&](ggml_tensor * member_rows, ggml_tensor * block_positions, int64_t prep_blocks) {
        // gathers per stream: blk_cells row s indexes stream s's own cells
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, member_rows);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, prep_blocks, n_stream);

        // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, members, idx_dim, prep_blocks, n_stream,
                            members->nb[2], members->nb[3], i*members->nb[1]));
            pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
        }
        pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
        cb(pooled, "indexer_k_pooled", il);

        // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, prep_blocks*n_stream, 1);
        pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, prep_blocks*n_stream);
        pooled = ggml_rope_multi(ctx0, pooled, block_positions, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, prep_blocks, n_stream);
        return pooled;
    };
    ggml_tensor * pooled = prepare_keys(member_rows, block_positions, prep_blocks);
    if (cached) {
        auto * prepared = ggml_reshape_2d(ctx0, pooled, idx_dim, prep_blocks);
        auto * written = fast_cache ? ggml_set_rows(ctx0, cached, prepared, cache_rows) : ggml_cpy(ctx0, prepared, cached);
        ggml_build_forward_expand(gf, written);
        pooled = ggml_reshape_3d(ctx0, written, idx_dim, n_blocks, 1);
        cb(pooled, "indexer_k_cached", il);
    }
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    const int64_t strip = qwen4exp_query_strip(n_tps, n_stream);
    std::vector<ggml_tensor *> selected;
    for (int64_t first = 0; first < n_tps; first += strip) {
        const int64_t n_query = std::min(strip, n_tps - first);
        // [QSA_SCORE_BOUNDS] trim the scorer to the blocks this strip can actually see.
        const int64_t score_blocks = inp->score_key_limits.empty()
                ? n_blocks : inp->score_key_limits[first/strip];
        ggml_tensor * score_keys = pooled;
        if (score_blocks < n_blocks) {
            score_keys = ggml_view_3d(ctx0, pooled, idx_dim, score_blocks, n_stream, pooled->nb[1], pooled->nb[2], 0);
            cb(score_keys, "indexer_k_bounded", il);
        }
        ggml_tensor * bias = inp->compact ? nullptr : ggml_view_3d(ctx0, inp->bias, inp->bias->ne[0], n_query, n_stream,
                inp->bias->nb[1], inp->bias->nb[2], first*inp->bias->nb[1]);
        ggml_tensor * query_mask = kq_mask == nullptr ? nullptr : ggml_view_4d(ctx0, kq_mask,
                n_kv, n_query, 1, n_stream, kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], first*kq_mask->nb[1]);

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * score = ggml_mul_mat(ctx0, score_keys,
                ggml_view_3d(ctx0, q, idx_dim, n_idx_h*n_query, n_stream, q->nb[1], q->nb[2]*n_tps, first*q->nb[2]));
        score = ggml_reshape_4d(ctx0, score, score_blocks, n_idx_h, n_query, n_stream);
        score = ggml_relu(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, score_blocks, n_query, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        score = summed;
        cb(score, "indexer_score", il);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (inp->compact) {
            score = qwen4exp_apply_compact_visibility(ctx0,score,inp->bias,n_blocks,first,n_query);
        } else if (blk_bias) {
            score = ggml_add(ctx0, score, bias);
        }

        if (inp->tail_idxs) {
            ggml_tensor * tail=ggml_view_4d(ctx0,inp->tail_idxs,r-1,n_query,1,n_stream,
                    inp->tail_idxs->nb[1],inp->tail_idxs->nb[2],inp->tail_idxs->nb[2],first*inp->tail_idxs->nb[1]);
            ggml_tensor * block_cells = score_blocks < n_blocks ?
                ggml_view_2d(ctx0, inp->blk_cells, r*score_blocks, n_stream, inp->blk_cells->nb[1], 0) : inp->blk_cells;
            ggml_tensor * top_k=qwen4exp_select_complete_blocks(ctx0,score,block_cells,tail,
                    hparams.indexer_top_k/r,r);
            cb(top_k,"indexer_top_k",il);
            qwen4exp_append_strip(ctx0,gf,selected,top_k);
            continue;
        }

        // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            // flash attention keeps the mask in f16; the scores are f32
            ggml_tensor * mask = query_mask->type == GGML_TYPE_F32 ? query_mask : ggml_cast(ctx0, query_mask, GGML_TYPE_F32);
            expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_query, n_stream));
        } else {
            expanded = ggml_add(ctx0, expanded, bias);
        }
        cb(expanded, "indexer_score_tokens", il);

        // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
        const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

        ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

        // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
        top_k = ggml_reshape_4d(ctx0, top_k, width, n_query, 1, n_stream);
        cb(top_k, "indexer_top_k", il);
        qwen4exp_append_strip(ctx0,gf,selected,top_k);
    }
    ggml_tensor * top_k = qwen4exp_finish_strips(ctx0,gf,selected);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot[il]) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot[il]);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot[il]);
    }

    if (inp->self_v_rot[il]) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot[il]);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * mask_all = inp->get_kq_mask();
    ggml_tensor * indices_all = top_k;
    const int64_t n_stream = mask_all->ne[3];
    const int64_t n_tps = mask_all->ne[1];
    const auto shared_qsa=qsa_inps.find((uint32_t)hparams.dsv4_compress_ratios[il]);
    const bool layout_prefill=n_tps>=128 && n_stream==1 &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias==0.0f && !hparams.attn_soft_cap &&
        shared_qsa!=qsa_inps.end() && shared_qsa->second->maskless;
    const int64_t strip=layout_prefill ? n_tps : qwen4exp_query_strip(n_tps,n_stream);
    ggml_tensor * packed_keys=nullptr;
    if (layout_prefill) {
        auto * original_keys=ggml_permute(ctx0,mctx_cur->get_k(ctx0,il),0,2,1,3);
        if (original_keys->type==GGML_TYPE_F16 && original_keys->ne[0]==256 && original_keys->ne[1]%4==0 && original_keys->ne[3]==1) {
            packed_keys=qsa_pack_keys(ctx0,original_keys);
            ggml_build_forward_expand(gf,packed_keys);
        }
    }
    // QSA attention v2 (qsa-attn/qsa2.cu) consumes V^T fragments straight from a [4 keys][256 dims] block
    // layout; build it once per graph next to the packed keys when requested.
    ggml_tensor * packed_values=nullptr;
    if (layout_prefill) {
        auto * original_values=ggml_permute(ctx0,mctx_cur->get_v(ctx0,il),0,2,1,3);
        if (original_values->type==GGML_TYPE_F16 && original_values->ne[0]==256 && original_values->ne[1]%4==0 && original_values->ne[3]==1) {
            packed_values=qsa_pack_values(ctx0,original_values);
            ggml_build_forward_expand(gf,packed_values);
        }
    }
    std::vector<ggml_tensor *> output;
    for (int64_t first = 0; first < n_tps; first += strip) {
        const int64_t n_query = std::min(strip, n_tps - first);
        ggml_tensor * kq_mask = ggml_view_4d(ctx0, mask_all, mask_all->ne[0], n_query, 1, n_stream,
                mask_all->nb[1], mask_all->nb[2], mask_all->nb[3], first*mask_all->nb[1]);
        ggml_tensor * top_k = ggml_view_4d(ctx0, indices_all, indices_all->ne[0], n_query, 1, n_stream,
                indices_all->nb[1], indices_all->nb[2], indices_all->nb[3], first*indices_all->nb[1]);

        // Private gfx1151 experiment: the existing QSA decode kernel consumes
        // the selected indices too. Keep the causal mask and fail closed on
        // unsupported shapes below. Keep the original path for short caches:
        // the sparse WMMA path changes rounding and only benefits longer scans.
        // The default path is unchanged.
        const bool decode_direct = qwen4exp_qsa_decode_direct(n_tokens, n_query, kq_mask->ne[0]);
        const bool direct_indices =
            (n_tokens >= 128 || decode_direct) &&
            n_stream == 1 && cparams.flash_attn && cparams.offload_kqv &&
            hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap;
        // a complete-block list (compact selection) carries -1 sentinels that only the index-consuming kernels skip
        GGML_ASSERT(direct_indices || shared_qsa == qsa_inps.end() || !shared_qsa->second->compact);
        ggml_tensor * kq_mask_top_k = kq_mask;
        if (!direct_indices) {
        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        // reshape KQ mask into tensor with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
        // this will be our source of zero values for unmasking top k mask elements
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        // modify KQ mask by unmasking elements that are in top_k indices
        // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
        kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // reshape to restore the original shape of KQ mask:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // combine with the original kq mask
        kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

        }

        ggml_tensor * q = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], n_query,
                q_cur->nb[1], q_cur->nb[2], first*q_cur->nb[2]);
        if (n_stream != 1) { q = q_cur; }
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);

        // TODO: enable sparse attention when we are ready
        // ref: https://github.com/ggml-org/llama.cpp/pull/27970
        //ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
        ggml_tensor * cur;
        if (direct_indices) {
            GGML_ASSERT(q->ne[0] == 256 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
            const bool v_trans = v->nb[1] > v->nb[2];
            ggml_tensor * q_view = ggml_permute(ctx0, q, 0, 2, 1, 3);
            ggml_tensor * k_view = ggml_permute(ctx0, k, 0, 2, 1, 3);
            ggml_tensor * v_view = ggml_permute(ctx0, v, 0, 2, 1, 3);
            if (v_trans) { v_view = ggml_transpose(ctx0, v_view); }
            const auto qsa_it=qsa_inps.find((uint32_t)hparams.dsv4_compress_ratios[il]);
            // without the mask, top_k is the only record of which cells are visible, and only the qsa3
            // kernel reads it: every other flash-attention path ignores src[5] and would attend to the
            // whole padded cache. qsa3 needs both packed layouts and bails below 128 queries, so the
            // mask may only be dropped where those hold - a decode ubatch keeps it.
            const bool qsa3=packed_keys && packed_values && n_query>=128;
            const bool maskless=qsa3 && qsa_it!=qsa_inps.end() && qsa_it->second->maskless;
            ggml_tensor * mask = maskless ? nullptr : (ggml_is_contiguous(kq_mask) ? kq_mask : ggml_cont(ctx0, kq_mask));
            ggml_tensor * indices = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
            GGML_ASSERT(indices->ne[1] == kq_mask->ne[1] && indices->ne[3] == kq_mask->ne[3]);
            if (decode_direct) {
                const auto device = model.dev_layer(il);
                GGML_ASSERT(device && strcmp(ggml_backend_dev_name(device), "ROCm0") == 0 &&
                    strstr(ggml_backend_dev_description(device), "8060S"));
                GGML_ASSERT(q_view->type == GGML_TYPE_F32 && q_view->ne[1] >= 1 && q_view->ne[1] <= 8);
                GGML_ASSERT(q_view->nb[0] == 4 && k_view->nb[0] == 2 && v_view->nb[0] == 2);
                GGML_ASSERT(k_view->ne[0] == 256 && v_view->ne[0] == 256);
                GGML_ASSERT(k_view->ne[1] >= 1 && k_view->ne[1] <= 262144 && v_view->ne[1] == k_view->ne[1]);
                GGML_ASSERT(q_view->ne[3] == 1 && k_view->ne[3] == 1 && v_view->ne[3] == 1);
                GGML_ASSERT(k_view->ne[2] > 0 && q_view->ne[2] % k_view->ne[2] == 0 && v_view->ne[2] == k_view->ne[2]);
                GGML_ASSERT(indices->type == GGML_TYPE_I32 && indices->ne[0] >= 1 && indices->ne[0] <= 2560);
                GGML_ASSERT(indices->ne[2] == 1 && indices->ne[3] == 1 && !packed_keys && !packed_values && mask);
                GGML_ASSERT(indices->nb[0] == 4 && indices->ne[1] >= q_view->ne[1]);
                GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->nb[0] == 2 &&
                    mask->ne[0] >= k_view->ne[1] && mask->ne[1] >= q_view->ne[1] &&
                    mask->ne[2] == 1 && mask->ne[3] == 1);
            }
            cur = ggml_flash_attn_ext(ctx0, q_view, k_view, v_view, mask, kq_scale, 0.0f, 0.0f);
            if (decode_direct) {
                GGML_ASSERT(cur->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) && !cur->src[4]);
            }
            cur->src[5] = indices;
            cur->src[6] = packed_keys;
            cur->src[7] = packed_values;
            ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(indices->ne[0]));
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            ggml_build_forward_expand(gf, cur);
        } else {
            cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
        }
        qwen4exp_append_strip(ctx0,gf,output,cur);
    }
    ggml_tensor * cur = qwen4exp_finish_strips(ctx0,gf,output);

    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot[il]) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot[il]);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // The MTP draft passes no hybrid context and attends densely.
    const bool qsa = mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = nullptr;
    if (qsa) {
        const int64_t r     = hparams.dsv4_compress_ratios[il];
        const int64_t n_kv  = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width = (int64_t) hparams.indexer_top_k + r - 1;
        if (n_kv <= width) {
            build_qsa_store_k(mctx_hyb, cur, il);
        } else {
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il);
        }
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha_cat) {
        // beta and alpha read the same input: one matmul, then the beta rows and the alpha rows
        ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);

        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        const int64_t n = (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
        return pmodel.ple_reader ? data->ne[1] == n : rows->ne[0] == n;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]
    ggml_tensor * data = nullptr;   // direct mode: staged rows [ple_head_dim, ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<uint8_t> staging;
};

// Prefetch hook: the chunk-boundary stall is the PLE row gather (257k-393k scattered 90-byte reads, ~300 ms with the
// GPU idle). Given the whole batch up front we can compute the same row indices and warm the page cache for them while
// the previous chunk is still on the GPU. It only calls posix_fadvise, so a wrong prediction costs nothing but readahead.
void qwen4exp_ple_prefetch(const llama_model & model_base, const llama_token * tokens, int32_t n_tokens, int32_t first) {
    if (!tokens || first < 0 || n_tokens <= first) { return; }
    const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model_base);
    if (!pmodel.ple_reader) { return; }
    const auto & hp = pmodel.hparams;
    const int64_t n_gram = hp.ple_ngram_size, n_heads = hp.ple_n_heads, per_gram = hp.ple_heads_per_ngram;
    const int64_t eos = hp.ple_eos_token_id, n_prev = n_gram - 1;
    if (n_heads <= 0 || n_gram < 2) { return; }
    std::vector<llama_token> toks(tokens, tokens + n_tokens);
    std::thread([&pmodel, toks = std::move(toks), n_gram, n_heads, per_gram, eos, n_prev, &hp, first]() {
        const int64_t n = (int64_t) toks.size();
        std::vector<int32_t> idx((size_t) n_heads * n);
        const int64_t i_first = std::min<int64_t>(first, n);
        std::vector<int64_t> ctx(n_gram);
        for (int64_t i = 0; i < n; ++i) {
            ctx[0] = toks[i];
            bool cut = false;
            for (int64_t sft = 1; sft < n_gram; ++sft) {
                const int64_t j = i - sft;                       // contiguous single-sequence prefill
                const llama_token t = (cut || j < 0) ? LLAMA_TOKEN_NULL : toks[j];
                cut = cut || t < 0 || t == eos;
                ctx[sft] = cut ? eos : t;
            }
            for (int64_t g = 2; g <= n_gram; ++g) {
                uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
                for (int64_t j = 1; j < g; ++j) { mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j]; }
                const int64_t base = (g - 2) * per_gram;
                for (int64_t q = 0; q < per_gram; ++q) {
                    const int64_t h_i = base + q;
                    idx[(size_t) i * n_heads + h_i] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
                }
            }
        }
        pmodel.ple_reader->prefetch(idx.data() + (size_t) i_first * n_heads, (int64_t) (n - i_first) * n_heads);
    }).detach();
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[k]);
        return is_embd ? img_tok : ubatch->token[k];
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;
    const bool split       = !pmodel.ple_ngram_embd.empty();

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                const uint64_t local = mixed % hp.ple_head_vocab_sizes[h_i];
                idx[i * n_heads + h_i] = (int32_t) (split ? local : local + hp.ple_head_offsets[h_i]);
            }
        }
    }

    if (pmodel.ple_reader) {
        staging.resize(idx.size() * pmodel.ple_reader->head_dim * sizeof(float));
        pmodel.ple_reader->gather(idx.data(), (int64_t) idx.size(), (float *) staging.data());
        ggml_backend_tensor_set(data, staging.data(), 0, staging.size());
    } else {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    }
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ggml_tensor * emb = nullptr;

    if (static_cast<const llama_model_qwen4exp &>(model).ple_reader) {
        ple_inp->data = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                           hparams.ple_head_dim, n_heads * n_tokens);
        ggml_set_input(ple_inp->data);
        ggml_tensor * data = ple_inp->data;
        res->add_input(std::move(ple_inp));

        emb = ggml_reshape_2d(ctx0, data, hparams.ple_head_dim * n_heads, n_tokens);
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));

        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        if (model.ple_ngram_embd.empty()) {
            emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
        } else {
            ggml_tensor * rows_2d = ggml_reshape_2d(ctx0, rows, n_heads, n_tokens);
            for (int64_t h = 0; h < n_heads; ++h) {
                ggml_tensor * idx = ggml_view_2d(ctx0, rows_2d, 1, n_tokens, rows_2d->nb[1], h * rows_2d->nb[0]);
                idx = ggml_reshape_1d(ctx0, ggml_cont(ctx0, idx), n_tokens);
                ggml_tensor * emb_head = ggml_get_rows(ctx0, model.ple_ngram_embd[h], idx);
                emb = emb ? ggml_concat(ctx0, emb, emb_head, 0) : emb_head;
            }
        }
        emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    }
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
