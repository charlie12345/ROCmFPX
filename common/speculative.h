#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;

    // RoPE position that the target model would assign to id_last's successor.
    // Equals n_past for text-only prompts, but they diverge under M-RoPE: an
    // image chunk occupies n_tokens KV slots while advancing position by only
    // n_pos. Only the draft-mtp implementation consumes this; -1 means "caller
    // supplied no M-RoPE-aware position", and consumers fall back to n_past so
    // behaviour is byte-identical to before for every other caller and impl.
    llama_pos   pos_next = -1;

    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    int32_t n_min = -1;
    float   p_min = -1.0f;
    // optional sparse proposal distributions, one per draft token
    std::vector<common_speculative_token_dist> * dists = nullptr;

    float temperature = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target pre-norm embeddings to be extracted
bool common_speculative_need_embd_pre_norm(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);
bool common_speculative_state_required(const common_speculative * spec);

// (optional) rewind the internal state to a previously seen position, so that
// processing can resume from there after the target/draft memories rolled back
// (bounded prompt-cache boundary salvage). Returns false when the position is
// not recoverable.
bool common_speculative_rollback_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos);

// rebase per-sequence positions after the corresponding target/draft contexts shift
void common_speculative_shift_state(common_speculative * spec, llama_seq_id seq_id, llama_pos delta);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
