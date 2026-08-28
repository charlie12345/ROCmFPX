// Unit tests for the prompt-cache checkpoint salvage selection used by
// server_prompt_cache::load() (issue: alternating agents on one slot with
// speculative decoding fell back to full cold prefill because every entry
// failed the spec-boundary check; the fix restores the newest checkpoint
// that carries host-memory shadows and whose captured state lies within the
// request's common prefix).
#include "common.h"

#include <cstdint>
#include <list>
#include <vector>

#include <cassert>
#include <cstdio>

namespace {

common_prompt_checkpoint make_ckpt(int64_t n_tokens, bool with_shadows, bool with_spec) {
    common_prompt_checkpoint ckpt;
    ckpt.n_tokens = n_tokens;
    if (with_shadows) {
        ckpt.data_tgt_host = {1, 2, 3};
        ckpt.data_dft_host = {4};
    }
    if (with_spec) {
        ckpt.data_spec = {5, 6, 7};
    }
    return ckpt;
}

const common_prompt_checkpoint * select(const std::list<common_prompt_checkpoint> & checkpoints, int64_t lcp) {
    return common_prompt_checkpoint_select_salvage(checkpoints, lcp);
}

} // namespace

int main() {
    {
        // empty cache entry: nothing to salvage
        std::list<common_prompt_checkpoint> none;
        assert(select(none, 1000) == nullptr);
    }
    {
        // qualifying checkpoint is returned
        std::list<common_prompt_checkpoint> cks = {make_ckpt(300, true, true)};
        auto * picked = select(cks, 400);
        assert(picked != nullptr && picked->n_tokens == 300);
    }
    {
        // checkpoint beyond the common prefix does not qualify: restoring it
        // would still require rewinding recurrent state, which is exactly what
        // the salvage path exists to avoid
        std::list<common_prompt_checkpoint> cks = {make_ckpt(500, true, true)};
        assert(select(cks, 400) == nullptr);
    }
    {
        // missing host shadows disqualify: the ON_DEVICE payloads are views
        // into live context memory and are clobbered by slot reuse
        std::list<common_prompt_checkpoint> cks = {make_ckpt(300, false, true)};
        assert(select(cks, 400) == nullptr);
    }
    {
        // missing speculative-impl state disqualifies: the restore must also
        // rewind the draft bookkeeping at the checkpoint position
        std::list<common_prompt_checkpoint> cks = {make_ckpt(300, true, false)};
        assert(select(cks, 400) == nullptr);
    }
    {
        // largest qualifying checkpoint wins
        std::list<common_prompt_checkpoint> cks = {
            make_ckpt(100, true, true),
            make_ckpt(300, true, true),
            make_ckpt(200, true, true),
        };
        auto * picked = select(cks, 400);
        assert(picked != nullptr && picked->n_tokens == 300);
    }
    {
        // a checkpoint beyond lcp is skipped even when larger, so a smaller
        // qualifying one is picked instead
        std::list<common_prompt_checkpoint> cks = {
            make_ckpt(450, true, true),
            make_ckpt(300, true, true),
        };
        auto * picked = select(cks, 400);
        assert(picked != nullptr && picked->n_tokens == 300);
    }
    {
        // tie on n_tokens: the first checkpoint in the list wins (the
        // selection never replaces an equally large pick)
        std::list<common_prompt_checkpoint> cks = {
            make_ckpt(300, true, true),
            make_ckpt(300, true, true),
        };
        auto * picked = select(cks, 400);
        assert(picked != nullptr && picked == &cks.front());
    }
    {
        // lcp exactly at the checkpoint boundary qualifies (restore covers
        // [0, n_tokens) and the request adds nothing within the prefix)
        std::list<common_prompt_checkpoint> cks = {make_ckpt(300, true, true)};
        auto * picked = select(cks, 300);
        assert(picked != nullptr && picked->n_tokens == 300);
    }
    {
        // lcp = 0 (disjoint request): only an empty checkpoint would qualify,
        // real checkpoints always cover tokens, so nothing is selected
        std::list<common_prompt_checkpoint> cks = {make_ckpt(300, true, true)};
        assert(select(cks, 0) == nullptr);
    }

    printf("test-prompt-cache-salvage: all assertions passed\n");
    return 0;
}
