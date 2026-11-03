#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"
#include <cstdio>
#include "mtmd-helper.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_DECISION,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_TRANSCRIBE_RT,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_TTS,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

// [TAG_REQUEST_PRIORITY] scheduling level of a request, set with "priority" in the request body.
//   low, default, high   order the requests that wait for a slot (higher first, then by arrival)
//   xhigh                also holds running requests of a lower level while it is processed
//   low                  leaves one slot free: it starts only while another slot stays idle
// A request without the field is "default".
enum server_task_priority {
    SERVER_TASK_PRIORITY_LOW     = -1,
    SERVER_TASK_PRIORITY_DEFAULT =  0,
    SERVER_TASK_PRIORITY_HIGH    =  1,
    SERVER_TASK_PRIORITY_XHIGH   =  2,
};

inline const char * server_task_priority_name(int priority) {
    switch (priority) {
        case SERVER_TASK_PRIORITY_LOW:   return "low";
        case SERVER_TASK_PRIORITY_HIGH:  return "high";
        case SERVER_TASK_PRIORITY_XHIGH: return "xhigh";
        default:                         return "default";
    }
}

inline bool server_task_priority_from_name(const std::string & name, int32_t & priority) {
    if (name == "low")     { priority = SERVER_TASK_PRIORITY_LOW;     return true; }
    if (name == "default") { priority = SERVER_TASK_PRIORITY_DEFAULT; return true; }
    if (name == "high")    { priority = SERVER_TASK_PRIORITY_HIGH;    return true; }
    if (name == "xhigh")   { priority = SERVER_TASK_PRIORITY_XHIGH;   return true; }
    return false;
}

enum server_control_action {
    SERVER_CONTROL_ACTION_NONE,
    SERVER_CONTROL_ACTION_REASONING_END,
    SERVER_CONTROL_ACTION_AUDIO_APPEND, // push control_audio into the slot's audio stream
    SERVER_CONTROL_ACTION_AUDIO_COMMIT, // flush the audio stream, more audio can follow
    SERVER_CONTROL_ACTION_AUDIO_END,    // flush the audio stream and end it
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t priority = SERVER_TASK_PRIORITY_DEFAULT; // [TAG_REQUEST_PRIORITY]

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    server_control_action control_action = SERVER_CONTROL_ACTION_NONE;
    std::string           control_cmpl_id;
    std::vector<float>    control_audio; // PCM F32 samples for SERVER_CONTROL_ACTION_AUDIO_APPEND

    // reported in generation_settings, parsing itself is owned by the chat session
    common_chat_format      chat_format      = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_session chat_session; // owns all parsing for this generation
    common_chat_msg chat_msg;
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(common_chat_session session = {});

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;
    // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
    int32_t n_tokens_shared = 0;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_DECISION
    // where to read the model output of each option, exactly one of the two lists is used
    struct decision {
        std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
        std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
        std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
        int32_t                  column = 0;
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        int32_t                  pointer = -1;

        // first prompt position that is read, -1 if none
        int32_t pos_first() const {
            int32_t pos = pointer;
            for (const int32_t marker : markers) {
                pos = pos < 0 ? marker : std::min(pos, marker);
            }
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] != 0) {
                    return pos < 0 ? (int32_t) i : std::min(pos, (int32_t) i);
                }
            }
            return pos;
        }

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        std::vector<int32_t> order;
        int32_t              n_scores = 0;
    };
    decision decision;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    // used by SERVER_TASK_TYPE_TTS
    mtmd_helper::gen_audio::inp tts_inp;

    // [TAG_REQUEST_HISTORY] for GET /requests: unix epoch ms when the HTTP request arrived (0 = unknown), and the
    // ggml_time_us() when a slot was picked for the task (before a prompt cache restore into it)
    int64_t t_arrival_ms = 0;
    int64_t t_start_us   = 0;

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.markers.empty() || !decision.order.empty();
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.labels.empty();
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void apply_chat_session(const common_chat_session & session) {
        if (!session.has_template()) {
            return;
        }

        session.apply_sampling(params.sampling);
        params.chat_format = session.format();
        params.antiprompt.insert(params.antiprompt.end(), session.additional_stops().begin(), session.additional_stops().end());
        params.message_spans = tokens.find_message_spans(session.message_delimiters());
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot
        copy.t_arrival_ms = t_arrival_ms; // [TAG_REQUEST_HISTORY]

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }

    // embedding and rerank tasks always process their whole prompt, their state is never reused
    bool is_stateless() const {
        return type == SERVER_TASK_TYPE_EMBEDDING || type == SERVER_TASK_TYPE_RERANK;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    common_chat_input content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    common_chat_input content;
    llama_tokens      tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    bool audio_commit_drained = false; // realtime: non-final commit has fully decoded
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_tts : server_task_result {
    std::string audio; // raw bytes for this chunk (WAV or PCM, per request's out_type)
    int32_t     sample_rate = 0;
    bool        final = false; // true for the last chunk of a request

    virtual bool is_stop() override { return final; }

    virtual json to_json() override;
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_decision : server_task_result {
    std::vector<float> scores; // one raw model output per option

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    // speculative-impl state captured at the exact token boundary (e.g. the MTP
    // drafter's deferred boundary row); optional, restored on an exact-boundary hit
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

// The context checkpoint payloads may contain large host vectors. Disk-cache entries
// keep only the small scheduling metadata in RAM; the checkpoints themselves are persisted
// in full next to the entry (see save_disk) and restored as live checkpoints.
struct server_prompt_checkpoint_meta {
    int64_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
};

// one context checkpoint persisted next to a disk entry (<stem>-ckpt<k>.bin)
struct server_prompt_disk_ckpt {
    std::string path;
    size_t      size = 0;
    server_prompt_checkpoint_meta meta;
};

struct server_prompt_disk_state {
    server_tokens tokens;
    std::vector<uint8_t> spec;

    std::string path_main;
    std::string path_drft;

    // The context checkpoints persisted alongside the entry, oldest first. A recurrent/hybrid
    // state cannot be truncated, so without them the entry is usable only when the request
    // extends it exactly. With them, a request that diverges anywhere past a checkpoint
    // (a client that re-sends the last assistant turn without reasoning_content, an agent
    // that rewrote or compacted older history, ...) still restores the entry: the slot rolls
    // back to the newest checkpoint at or before the divergence, exactly as it does in-slot,
    // and re-prefills from there.
    std::vector<server_prompt_disk_ckpt> ckpts;

    // speculative-impl state blob persisted next to the entry so it survives a restart
    std::string path_spec;

    size_t size_main = 0;
    size_t size_drft = 0;

    bool has_ckpt() const { return !ckpts.empty(); }

    size_t size_ckpt() const {
        size_t res = 0;
        for (const auto & ck : ckpts) {
            res += ck.size;
        }
        return res;
    }

    // newest persisted checkpoint boundary at or before `lcp` (0 = none)
    int64_t ckpt_at_or_below(size_t lcp) const {
        int64_t res = 0;
        for (const auto & ck : ckpts) {
            if (ck.meta.n_tokens > res && (size_t) ck.meta.n_tokens <= lcp) {
                res = ck.meta.n_tokens;
            }
        }
        return res;
    }

    uint64_t id = 0;
    bool usable = true;

    size_t size() const {
        return size_main + size_drft + size_ckpt();
    }

    int n_tokens() const {
        return tokens.size();
    }
};

// [TAG_REQUEST_HISTORY] the disk tier's counters (the ones log_disk_state prints) for GET /props. The cache lives
// on the main loop; it hands a copy of these to on_disk_stats after every call that may change them
struct server_prompt_disk_stats {
    bool     enabled       = false;
    size_t   entries       = 0;
    size_t   unusable      = 0;
    size_t   bytes         = 0;
    size_t   limit_bytes   = 0;
    size_t   tokens        = 0;
    uint64_t saves         = 0;
    uint64_t loads         = 0;
    uint64_t evictions     = 0;
    uint64_t save_failures = 0;
    bool     save_disabled = false;
    uint64_t bytes_written = 0;
    uint64_t bytes_read    = 0;
    uint64_t bytes_evicted = 0;
    double   restore_ms_total = 0.0; // disk restores: reading the entry's files into the slot
    uint64_t restores_timed   = 0;

    json to_json() const;
};

struct server_prompt_cache {
    server_prompt_cache(
            int32_t limit_size_mib,
             size_t limit_tokens,
        const std::string & disk_base_path = {},
            int32_t disk_limit_size_mib = 0,
        const std::string & disk_identity = {});

    ~server_prompt_cache();

    std::list<server_prompt_cache_state> states;

    // Cold automatic cache. Entries own only token/checkpoint metadata in RAM;
    // target and draft context payloads live in owner-only files.
    std::list<server_prompt_disk_state> disk_states;

    // Identity of the model/context this cache was written for (model path + size,
    // n_ctx, KV types). Written to <run>/identity.txt; a later run adopts a stale run
    // directory's entries only when its identity matches exactly.
    std::string disk_identity;

#if defined(_WIN32)
    // Held open with deny-all sharing for the life of the run: the Windows
    // counterpart of the POSIX flock() advisory lock, so a later startup can tell a
    // live run directory from an abandoned one.
    FILE * disk_lock_file = nullptr;
#endif

    bool ram_enabled = false;

    // true when both contexts support unbounded partial sequence removal (dense KV):
    // then a cached prompt that is longer than the request can be truncated in place.
    // Recurrent/hybrid targets cannot (their saved state carries no rollback
    // snapshots), so such entries are only usable at their exact boundary or through
    // a checkpoint at or before the divergence point.
    bool partial_seq_rm = false;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // Disk fields are disabled when disk_owned_path is empty.
    std::string disk_base_path;
    std::string disk_owned_path;
    size_t disk_limit_size = 0;
    size_t disk_size_total = 0;
    int disk_lock_fd = -1;

    uint64_t disk_next_id       = 1;
    uint64_t disk_saves         = 0;
    uint64_t disk_loads         = 0;
    uint64_t disk_evictions     = 0;
    uint64_t disk_bytes_written = 0;
    uint64_t disk_bytes_read    = 0;
    uint64_t disk_bytes_evicted = 0;
    uint64_t disk_save_failures = 0;

    // [TAG_REQUEST_HISTORY] time of the successful disk restores (load_ms in the log)
    double   disk_restore_ms_total = 0.0;
    uint64_t disk_restores_timed   = 0;

    // [TAG_REQUEST_HISTORY] see server_prompt_disk_stats; called on the cache's thread at the end of save(),
    // save_prefix(), load() and update() while the disk tier is enabled
    std::function<void(const server_prompt_disk_stats &)> on_disk_stats;

    server_prompt_disk_stats disk_stats() const;

    // A durable write/removal failure opens this run-level circuit breaker.
    // Existing valid entries remain readable, but no further state files are
    // created for this server process.
    bool disk_save_disabled = false;

    // Prompts shorter than this are never written to disk: keep-alive pings, title
    // generation and similar one-shot requests would otherwise each leave an entry whose
    // size is dominated by the fixed recurrent state (~160 MiB for a few hundred tokens).
    size_t disk_min_tokens = 0;

    // max context checkpoints persisted with each entry, newest first (-1 = all, 0 = none)
    int32_t disk_max_ckpts = -1;

    // a disk entry replaces what the slot already holds only when it keeps at least this
    // many tokens more - below that the read costs more than the prefill it saves
    size_t disk_min_gain = 1024;

    bool disk_enabled() const { return !disk_owned_path.empty(); }

    size_t size() const;

    size_t n_tokens() const;

    size_t disk_size() const;

    size_t disk_n_tokens() const;

    // save the slot's state to the disk tier (if enabled) and the RAM tier (if enabled);
    // returns true when the state is now held by at least one tier
    bool save(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    server_prompt_cache_state * alloc(
        const server_prompt & prompt,
                    size_t state_size_main,
                    size_t state_size_drft,
        const std::vector<uint8_t> & state_spec);

    // Persist a shared exact-boundary entry for `tokens_prefix`. The contexts must hold
    // exactly these tokens for id_slot, i.e. this is called mid-prefill, between two batches.
    // Used for the system prompt + tools block, which many conversations have in common:
    // a new conversation then starts from the entry instead of prefilling the block again.
    // An existing entry with the same tokens is only touched (kept most-recently-used).
    bool save_prefix(
        const server_tokens & tokens_prefix,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    // true when a usable disk entry holds exactly the first n_tokens of `tokens`
    bool has_disk_prefix(const server_tokens & tokens, size_t n_tokens) const;

    // restore the most useful entry (RAM or disk) for tokens_new into the slot, if any is
    // better than what the slot already holds. On a hit, `prompt` is replaced and
    // `state_spec` (optional) receives the entry's exact-boundary speculative state.
    // `probe` marks a lookup made while the slot keeps serving the same conversation
    // (near-miss logging is then demoted to debug).
    // Returns false only when a restore was attempted and failed (the slot must be cleared).
    bool load(
              server_prompt & prompt,
        const server_tokens & tokens_new,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
                    int32_t   id_slot,
       std::vector<uint8_t> * state_spec,
                       bool   probe = false);

    void update();

private:
    bool save_disk(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    bool load_disk(
        std::list<server_prompt_disk_state>::iterator it,
        server_prompt & prompt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
         llama_seq_id   id_slot,
              size_t   lcp,
       std::vector<uint8_t> * state_spec);

    bool erase_disk_state(std::list<server_prompt_disk_state>::iterator it, bool eviction, const char * reason);

    void disable_disk_saves(const char * reason, const std::string & path);

    void update_disk();

    void log_disk_state() const;
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
