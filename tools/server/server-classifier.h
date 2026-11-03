#pragma once

// Jev-style decision classifier: typed questions about a text answered with calibrated probabilities in the TypeSafe
// System One shape, without generating text. Two families, chosen by "family" in --classifier-config:
//   pointer  (TinyJev v1 / Kev)   a small decision head on the hidden states, described below
//   letter   (Wald / StartLux-Decision ...)   no head: the options are lettered in a text prompt and the answer is the
//            model's own next-token logits of those letters at the last prompt position (see "Letter family" below)
//
// The backbone is the model llama-server already loaded (-m, any backend: CPU with -ngl 0 or a GPU). On top of it:
//   --classifier-head   FILE  the decision head, a safetensors file with q.weight/q.bias/k.weight/k.bias
//   --classifier-config FILE  JSON with everything that is prompt layout rather than weights: the delimiter tokens,
//                             how options and yes/no labels are written, limits, temperature. A tinyjev.json manifest
//                             works as-is; unset keys take the pointer-family defaults (see classifier_config below).
//
// Per question the prompt is  <state> text  <question> instructions  (<option> text <close>)*  <decide>  and the
// head scores the final normed hidden state at <decide> (query) against each <close> (keys):
//     z_i = (W_k h_close_i + b_k) . (W_q h_decide + b_q) / sqrt(head_dim) / temperature
// Only those rows are requested as outputs; the post-norm hidden state is captured from the graph ("result_norm")
// with embeddings mode off, because embeddings mode would force every token through the last layer and lm_head.
//
// Letter family: --classifier-config alone enables it (no --classifier-head). The config's "letter" object holds the
// prompt as a template, so one implementation covers the models that differ only in wording:
//     prompt        "State:\n{state}\n\nQuestion: {instructions}{options}\nAnswer: ("   ({state} may appear twice)
//     option_line   "\n({letter}) {text}"
//     variants      ["{letter}", " {letter}"]   token spellings of a letter whose probabilities are summed
//     temperature   a number, or {"choice|3-4": T, "noul": T, "default": T}: the most specific of
//                   "<type>|<option-count bucket 2, 3-4, 5-8, 9+>", "<type>", "default" divides the letter logits
// "style": "ollama" replaces the option lines by the user message Ollama's decision API builds (TinyJev v2 is trained
// on those bytes): compact JSON {"context", "schema": [every question with its lettered choices]} followed by
// "Requested field: <name>"; `prompt` then wraps it as {user}, e.g. in the model's chat template.
// The whole prompt is tokenized as one string (special tokens in the template are parsed, user text is escaped), each
// question is one pass with a single output row (the questions of a request share their common token prefix: it is
// decoded once and each question continues from a snapshot of it), and more than 26 options are read in chunks whose winners meet in a
// final read (P = P_final(chunk) * P_chunk(option)).
//
//   --classifier-cache  DIR   persistent answer cache. The model is deterministic, so an answer is a pure function of
//                             (model, head, config, state, question); each one is appended to DIR/<name>-<identity>.tsv
//                             and a repeated question about the same state is answered from memory without a decode.

#include "server-common.h"

#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_context;
struct common_params;

struct classifier_config {
    std::string name        = "classifier";
    std::string family      = "pointer";
    int         head_dim    = 0;       // 0 = from the head weights
    double      temperature = 1.0;     // divides the head logits (the model's fitted temperature)
    int         max_state   = 8192;    // tokens of state (incl. the <state> delimiter)
    int         max_branch  = 8192;    // tokens of one question row
    int         max_options = 255;
    // delimiter tokens (must each be a single token of the backbone's vocabulary)
    std::string tok_state    = "<|fim_prefix|>";
    std::string tok_question = "<|fim_middle|>";
    std::string tok_option   = "<|box_start|>";
    std::string tok_close    = "<|box_end|>";
    std::string tok_decide   = "<|fim_suffix|>";
    // user text can never forge a delimiter: "<|name|>" is rewritten to "<¦name¦>" before tokenizing
    bool        escape_special = true;
    // how a choice/boolean option is written when it has a description ("{name}" / "{desc}" placeholders)
    std::string option_format = "{name}: {desc}";
    std::string label_false   = "no";
    std::string label_true    = "yes";

    // letter family (see the header comment); the defaults are the plain Wald layout
    std::string lt_prompt          = "State:\n{state}\n\nQuestion: {instructions}{options}\nAnswer: (";
    std::string lt_style           = "lines";               // lines: the template below | ollama: see the header comment
    std::string lt_prompt_no_state = "";                    // used for a blank state when set, else {state} = lt_empty_state
    std::string lt_empty_state     = "";
    std::string lt_option_line     = "\n({letter}) {text}";
    std::vector<std::string> lt_variants = { "{letter}", " {letter}" };
    std::string lt_state_format    = "text";                // text: labelled lines, as the pointer family | json: compact JSON
    int         lt_json_index_min  = 0;                     // json: arrays of at least this many items carry "_index" (0 = off)
    std::string lt_score_format    = "{desc}";              // "{name}" is the level index
    bool        lt_true_first      = false;                 // yes/no shown as yes, no
    std::string lt_noul_instructions = "";                  // instructions of a yes/no question that has none
    // choice options whose keys only number them: none | positional (a, b, .. / o27: drop a leading "a: ")
    //                                             | bare (A, 1, (b), option_3 .., all described: show the description alone)
    std::string lt_bare_keys       = "none";
    bool        lt_same_as_name    = false;                 // a description that repeats the name is not written twice
    bool        lt_strip           = false;                 // trim instructions and descriptions
    bool        lt_share_prefix    = true;                  // decode the common prefix of a request's questions once
    std::map<std::string, double> lt_temperature;           // see the header comment; empty = `temperature`
};

struct server_classifier {
    // loads config + head and creates a private llama_context on `model` (shares the weights, own small KV cache)
    // returns false and sets `err` on failure
    bool init(llama_model * model, const common_params & params, std::string & err);
    ~server_classifier();

    // System One request {state, questions, model?} -> {model, answers, latency_ms}
    // with --classifier-cache also "cached": true when every answer came from the cache (nothing was decoded)
    // throws std::invalid_argument for bad requests (-> 422); thread-safe (requests are serialized)
    // [TAG_REQUEST_HISTORY] history (optional) receives the request's GET /requests record: the question types, the
    // chosen labels and their probabilities, the tokens decoded and whether the cache answered, never a text. Its
    // "id", "priority", "t_arrival_ms" and "t_end_ms" are null, for the caller to fill in
    json classify(const json & body, json * history = nullptr);

    // [TAG_REQUEST_HISTORY] the same record for a request that classify() did not answer (it threw)
    static json history_error(double latency_ms);

    const classifier_config & config() const { return cfg; }

private:
    struct encoded_question {
        std::string id, type;                 // type: boolean | choice | score
        std::vector<std::string> keys, opts;  // answer keys, option texts
        std::vector<llama_token> row;         // question row (after the shared state prefix)
        std::string text;                     // letter family: state and instructions as shown (cache identity)
        int decide = 0;                       // absolute position of <decide>
        std::vector<int> close_pos;           // absolute positions of each <close>
        std::string cache_key;                // "" = cache off
        json answer;                          // null until answered (from the cache or the model)
        bool cached = false;                  // [TAG_REQUEST_HISTORY] the answer came from the cache
    };

    void cache_open(const common_params & params);
    std::string cache_key(const std::vector<llama_token> & prefix, const encoded_question & e) const;
    bool cache_get(const std::string & key, json & answer);
    void cache_put(const std::string & key, const json & answer);

    // [TAG_REQUEST_HISTORY] see classify(); n_decoded = the prompt tokens the model read for the request
    json history_of(const std::vector<encoded_question> & enc, size_t n_decoded) const;

    // letter family
    json classify_letter(const json & body, json * history);
    // n_decoded (optional) is increased by the number of tokens decoded
    std::vector<std::vector<double>> letter_reads(const std::vector<std::string> & prompts, const std::vector<int> & n_opts,
                                                  size_t * n_decoded = nullptr);
    double letter_temperature(const std::string & type, size_t n) const;
    std::vector<encoded_question> encode_ollama(const json & body) const;

    json make_answer(const encoded_question & e, const std::vector<double> & p) const;
    std::string escape_user(const std::string & text) const;
    std::vector<llama_token> user_tokens(const std::string & text) const;
    std::vector<float> hidden_rows(const std::vector<llama_token> & prefix, const std::vector<encoded_question> & qs,
                                   std::vector<std::vector<int>> & out_positions);
    std::vector<float> decode_chunk(const std::vector<llama_token> & toks, int start, const std::vector<int> & want);

    classifier_config cfg;
    llama_model   * model = nullptr;
    llama_context * lctx  = nullptr;
    int n_embd = 0, n_ctx = 0;
    llama_token id_state = -1, id_question = -1, id_option = -1, id_close = -1, id_decide = -1;
    std::vector<float> wq, bq, wk, bk;  // [head_dim, n_embd] row-major, [head_dim]
    std::vector<std::vector<llama_token>> letter_ids;   // letter family: per letter A..Z, its token spellings
    std::mutex mtx;

    // answer cache: key (sha256 of identity + exact model input) -> answer JSON text, mirrored in an append-only file
    std::string cache_ident;
    std::unordered_map<std::string, std::string> cache;
    std::ofstream cache_file;
    std::mutex cache_mtx;

public:
    // graph callback state (filled by the eval callback during llama_decode)
    std::vector<float> captured;
    int64_t captured_rows = 0;
};
