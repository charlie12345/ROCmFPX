// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration

#include "server-context-ssd-cache.h"

#include "server-context.h"
#include "server-task.h"
#include "llama.h"

#include <cinttypes>
#include <vector>

namespace llama {

uint64_t server_ssd_cache::store(uint32_t slot_id,
                                 struct llama_context* ctx,
                                 struct llama_context* ctx_dft,
                                 const common_prompt_checkpoint& ckpt,
                                 const llama_token* tokens,
                                 size_t tokens_size,
                                 uint32_t turn_id)
{
    if (!cache_ || !ctx || !ckpt.data_tgt.data()) return 0;

    // The full-state serialization below allocates host buffers proportional
    // to the sequence length (hundreds of MiB at long contexts). On machines
    // where most RAM is carved out for the iGPU this can fail — skip the
    // store instead of dying on an uncaught bad_alloc.
    try {
        // Serialize full tgt state (recurrent + KV cache) for cold-start recovery.
        const size_t tgt_size = llama_state_seq_get_size_ext(ctx, slot_id, 0);
        std::vector<uint8_t> tgt_data(tgt_size);
        if (llama_state_seq_get_data_ext(ctx, tgt_data.data(), tgt_size, slot_id, 0) != tgt_size) {
            LOG_WRN("SSD cache: tgt state serialization size mismatch (slot=%u)\n", slot_id);
            return 0;
        }

        // Serialize dft state (MTP KV cache) when ctx_dft has independent memory.
        // Skip when ctx_dft is null or shares memory with ctx (is_mem_shared models).
        std::vector<uint8_t> dft_data;
        if (ctx_dft) {
            const size_t dft_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, 0);
            if (dft_size > 0) {
                dft_data.resize(dft_size);
                if (llama_state_seq_get_data_ext(ctx_dft, dft_data.data(), dft_size, slot_id, 0) != dft_size) {
                    LOG_WRN("SSD cache: dft state serialization size mismatch (slot=%u) - skipping\n", slot_id);
                    dft_data.clear();
                }
            }
        }

        // spec_data carries speculative impl state (pending_h for MTP, boundary stash for Eagle3).
        const std::vector<uint8_t>& spec_data = ckpt.data_spec;

        return kv_ssd_store(cache_, slot_id,
                            tgt_data.data(), tgt_data.size(),
                            ckpt.pos_min, ckpt.pos_max,
                            ckpt.n_tokens, turn_id,
                            (const uint32_t*)tokens, tokens_size,
                            cache_->compat_hash,
                            dft_data.empty()  ? nullptr : dft_data.data(),  dft_data.size(),
                            spec_data.empty() ? nullptr : spec_data.data(), spec_data.size());
    } catch (const std::bad_alloc &) {
        LOG_WRN("SSD cache: out of host memory serializing checkpoint (slot=%u, n_tokens=%" PRId64 ") - skipping store\n",
                slot_id, ckpt.n_tokens);
        return 0;
    }
}

bool server_ssd_cache::load(uint64_t checkpoint_id,
                            struct llama_context* ctx,
                            struct llama_context* ctx_dft,
                            int32_t& out_pos_min,
                            int32_t& out_pos_max,
                            uint64_t& out_n_tokens,
                            std::vector<uint8_t>* out_spec_data,
                            uint32_t dest_seq_id)
{
    if (!cache_ || !ctx || checkpoint_id == 0) return false;

    const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache_, checkpoint_id);
    if (!meta) return false;

    // Use the caller-supplied destination seq_id (current slot's id) when provided.
    // Falls back to meta->slot_id for same-slot loads where they are guaranteed equal.
    const uint32_t seq_id = (dest_seq_id != UINT32_MAX) ? dest_seq_id : meta->slot_id;

    std::vector<uint8_t> tgt_data;
    std::vector<uint8_t> dft_data;
    std::vector<uint8_t> spec_data;
    try {
        if (!kv_ssd_load(cache_, checkpoint_id, tgt_data, &dft_data, &spec_data)) return false;
    } catch (const std::bad_alloc &) {
        LOG_WRN("SSD cache: out of host memory loading checkpoint %lu - falling back to prompt reprocessing\n",
                (unsigned long)checkpoint_id);
        return false;
    }

    // Restore tgt state (recurrent + KV cache) under the current slot's seq_id
    if (llama_state_seq_set_data_ext(ctx, tgt_data.data(), tgt_data.size(), (int32_t)seq_id, 0) == 0) {
        LOG_WRN("SSD cache: failed to restore tgt state for checkpoint %lu\n",
                (unsigned long)checkpoint_id);
        return false;
    }

    // Restore dft state (MTP KV cache) if it was saved and caller provided ctx_dft
    if (ctx_dft && !dft_data.empty()) {
        if (llama_state_seq_set_data_ext(ctx_dft, dft_data.data(), dft_data.size(), (int32_t)seq_id, 0) == 0) {
            LOG_WRN("SSD cache: failed to restore dft state for checkpoint %lu - MTP will catch up\n",
                    (unsigned long)checkpoint_id);
        }
    }

    if (out_spec_data) {
        *out_spec_data = std::move(spec_data);
    }

    out_pos_min  = meta->pos_min;
    out_pos_max  = meta->pos_max;
    out_n_tokens = meta->n_tokens;
    return true;
}

uint64_t server_ssd_cache::find_match(const llama_token* tokens, size_t tokens_size, uint32_t current_turn,
                                        uint64_t max_n_tokens, int32_t n_past,
                                        int32_t* out_lcp) {
    if (!cache_) return 0;
    return kv_ssd_find_match(cache_, (const uint32_t*)tokens, tokens_size, current_turn,
                             max_n_tokens, n_past, out_lcp);
}

uint64_t server_ssd_cache::find_by_slot(uint32_t slot_id, uint64_t min_tokens, uint32_t current_turn) {
    if (!cache_) return 0;
    return kv_ssd_find_by_slot(cache_, slot_id, min_tokens, current_turn);
}

void server_ssd_cache::on_turn_complete(uint32_t turn_id) {
    if (cache_) {
        kv_ssd_on_turn_complete(cache_, turn_id);
    }
}

} // namespace llama