#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// server_task
//

task_params server_task::params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
    task_params defaults;
    defaults.sampling      = params_base.sampling;
    defaults.speculative   = params_base.speculative;
    defaults.n_keep        = params_base.n_keep;
    defaults.n_predict     = params_base.n_predict;
    defaults.n_cache_reuse = params_base.n_cache_reuse;
    defaults.cache_prompt  = params_base.cache_prompt;
    defaults.antiprompt    = params_base.antiprompt;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose           = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream           = json_value(data,       "stream",             false);
    auto stream_opt         = json_value(data,       "stream_options",     json::object());
    params.include_usage    = json_value(stream_opt, "include_usage",      false);
    params.cache_prompt     = json_value(data,       "cache_prompt",       defaults.cache_prompt);
    params.return_tokens    = json_value(data,       "return_tokens",      false);
    params.return_progress  = json_value(data,       "return_progress",    false);
    auto max_tokens         = json_value(data,       "max_tokens",         defaults.n_predict);
    params.n_predict        = json_value(data,       "n_predict",          json_value(data, "max_completion_tokens", max_tokens));
    params.n_indent         = json_value(data,       "n_indent",           defaults.n_indent);
    params.n_keep           = json_value(data,       "n_keep",             defaults.n_keep);
    params.n_discard        = json_value(data,       "n_discard",          defaults.n_discard);
    params.n_discard        = std::max(0, params.n_discard);
    params.n_cmpl           = json_value(data,       "n_cmpl",             json_value(data, "n", 1));
    params.n_cache_reuse    = json_value(data,       "n_cache_reuse",      defaults.n_cache_reuse);
    //params.t_max_prompt_ms  = json_value(data,       "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data,       "t_max_predict_ms",   defaults.t_max_predict_ms);
    params.response_fields  = json_value(data,       "response_fields",    std::vector<std::string>());

    params.sampling.top_k              = json_value(data, "top_k",               defaults.sampling.top_k);
    params.sampling.top_p              = json_value(data, "top_p",               defaults.sampling.top_p);
    params.sampling.min_p              = json_value(data, "min_p",               defaults.sampling.min_p);
    params.sampling.top_n_sigma        = json_value(data, "top_n_sigma",         defaults.sampling.top_n_sigma);
    params.sampling.xtc_probability    = json_value(data, "xtc_probability",     defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",       defaults.sampling.xtc_threshold);
    params.sampling.typ_p              = json_value(data, "typical_p",           defaults.sampling.typ_p);
    params.sampling.temp               = json_value(data, "temperature",         defaults.sampling.temp);
    params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",      defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",   defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",       defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",      defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq       = json_value(data, "frequency_penalty",   defaults.sampling.penalty_freq);
    params.sampling.penalty_present    = json_value(data, "presence_penalty",    defaults.sampling.penalty_present);
    params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",      defaults.sampling.dry_multiplier);
    params.sampling.dry_base           = json_value(data, "dry_base",            defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length",  defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n",  defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat           = json_value(data, "mirostat",            defaults.sampling.mirostat);
    params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",        defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",        defaults.sampling.mirostat_eta);
    params.sampling.adaptive_target    = json_value(data, "adaptive_target",     defaults.sampling.adaptive_target);
    params.sampling.adaptive_decay     = json_value(data, "adaptive_decay",      defaults.sampling.adaptive_decay);
    params.sampling.seed               = json_value(data, "seed",                defaults.sampling.seed);
    params.sampling.n_probs            = json_value(data, "n_probs",             defaults.sampling.n_probs);
    params.sampling.min_keep           = json_value(data, "min_keep",            defaults.sampling.min_keep);
    params.sampling.backend_sampling   = json_value(data, "backend_sampling",    defaults.sampling.backend_sampling);
    params.post_sampling_probs         = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative = defaults.speculative;

    // per-request overrides for the draft-model based speculative parameters
    params.speculative.draft.n_min = json_value(data, "speculative.n_min", defaults.speculative.draft.n_min);
    params.speculative.draft.n_max = json_value(data, "speculative.n_max", defaults.speculative.draft.n_max);
    params.speculative.draft.p_min = json_value(data, "speculative.p_min", defaults.speculative.draft.p_min);

    params.speculative.draft.n_min = std::min(params.speculative.draft.n_max, params.speculative.draft.n_min);
    params.speculative.draft.n_min = std::max(params.speculative.draft.n_min, 0);
    params.speculative.draft.n_max = std::max(params.speculative.draft.n_max, 0);

    // TODO: to keep things simple, we disable further speculative parameter adjustments for now
#if 0
    // for debugging and research purposes
    params.speculative.type = common_speculative_type_from_name(json_value(data, "speculative.type", common_speculative_type_to_str(defaults.speculative.type)));

    params.speculative.ngram_size_n     = json_value(data, "speculative.ngram_size_n", defaults.speculative.ngram_size_n);
    params.speculative.ngram_size_m     = json_value(data, "speculative.ngram_size_m", defaults.speculative.ngram_size_m);
    params.speculative.ngram_min_hits   = json_value(data, "speculative.ngram_m_hits", defaults.speculative.ngram_min_hits);

    params.speculative.ngram_size_n     = std::max(std::min(1, (int) params.speculative.ngram_size_n),     1024);
    params.speculative.ngram_size_m     = std::max(std::min(1, (int) params.speculative.ngram_size_m),     1024);
    params.speculative.ngram_min_hits   = std::max(std::min(1, (int) params.speculative.ngram_min_hits),   1024);
#endif

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
        params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
        if (data.at("lora").is_array()) {
            params.lora = parse_lora_request(data.at("lora"));
        } else {
            throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
        }
    } else {
        params.lora = {};
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
        // note: should be the slot's context and not the full context, but it's ok
        params.sampling.penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        params.sampling.dry_penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_base < 1.0f) {
        params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema                  = json_value(data, "json_schema", json::object());
            SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
            std::string grammar_str      = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
            params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    } else {
        params.sampling.grammar = defaults.sampling.grammar;

        std::string grammar_str = json_value(data, "grammar", std::string());
        if (!grammar_str.empty()) {
            // grammar_type key is set by the server when converting chat template grammars
            std::string grammar_type = json_value(data, "grammar_type", std::string());
            if (grammar_type == "tool_calls") {
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
            } else {
                // explicit grammar from the user (API field "grammar")
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
            }
            SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
        }
        params.sampling.grammar_lazy = json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
        SRV_DBG("Grammar lazy: %s\n", params.sampling.grammar_lazy ? "true" : "false");
    }

    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.chat_parser_params.format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.chat_parser_params.format));
        } else {
            params.chat_parser_params.format = defaults.chat_parser_params.format;
        }
        common_reasoning_format reasoning_format = params_base.reasoning_format;
        if (data.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }
        params.chat_parser_params.reasoning_format = reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        params.chat_parser_params.generation_prompt = json_value(data, "generation_prompt", std::string());
        params.sampling.generation_prompt = params.chat_parser_params.generation_prompt;
        SRV_DBG("Generation prompt: '%s'\n", params.chat_parser_params.generation_prompt.c_str());
        params.chat_parser_params.parse_tool_calls = json_value(data, "parse_tool_calls", false);
        if (data.contains("chat_parser")) {
            params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }
    }

    {
        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            for (const auto & t : *preserved_tokens) {
                auto ids = common_tokenize(vocab, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    SRV_DBG("Preserved token: %d\n", ids[0]);
                    params.sampling.preserved_tokens.insert(ids[0]);
                } else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    SRV_DBG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            for (const auto & t : *grammar_triggers) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value.value;
                    auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                        params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        SRV_DBG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                    } else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                        SRV_DBG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                    } else {
                        throw std::runtime_error("Unknown grammar trigger type");
                    }
                    params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
        }
        if (params.sampling.grammar_lazy && params.sampling.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    // Parse reasoning budget sampler parameters
    {
        const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t) -1);
        const auto start_tag = json_value(data, "reasoning_budget_start_tag", std::string());
        const auto end_tag   = json_value(data, "reasoning_budget_end_tag", std::string());
        const auto message   = json_value(data, "reasoning_budget_message", std::string());
        params.sampling.reasoning_budget_tokens = budget;

        if (!start_tag.empty()) {
            params.sampling.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (!end_tag.empty()) {
            params.sampling.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            // il template re-renderizza un turno assistant passato come
            // '<think>\n' + reasoning_content|trim + '\n</think>\n\n': la sequenza forzata
            // deve portare un '\n' PRIMA dell'end tag e, quando c'e' un messaggio di
            // chiusura, anche un '\n' DOPO il messaggio (il trim dell'estratto rimuove i
            // whitespace di coda, quindi il resend ricompone esattamente '\n' + messaggio
            // + '\n' + end_tag); senza di cio' un turno tagliato dal budget non round-trippa
            // nel resend del client e la prompt cache diverge alla chiusura
            params.sampling.reasoning_budget_forced = common_tokenize(
                vocab, "\n" + message + (message.empty() ? "" : "\n") + end_tag, false, true);

            SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
        }
    }

    {
        params.sampling.logit_bias.clear();

        const auto & logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    } else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    } else if (el[0].is_string()) {
                        auto toks = common_tokenize(vocab, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    }
                }
            }
        } else if (logit_bias != data.end() && logit_bias->is_object()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : logit_bias->items()) {
                float bias;
                const auto & key = el.key();
                const auto & value = el.value();
                if (value.is_number()) {
                    bias = value.get<float>();
                } else if (value.is_boolean() && !value.get<bool>()) {
                    bias = -INFINITY;
                } else {
                    continue;
                }

                char *end;
                llama_token tok = strtol(key.c_str(), &end, 10);
                if (*end == 0) {
                    if (tok >= 0 && tok < n_vocab) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                } else {
                    auto toks = common_tokenize(vocab, key, false);
                    for (auto tok : toks) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            }
        }

        params.sampling.ignore_eos = json_value(data, "ignore_eos", params_base.sampling.ignore_eos);
        if (params.sampling.ignore_eos) {
            params.sampling.logit_bias.insert(
                    params.sampling.logit_bias.end(),
                    logit_bias_eog.begin(), logit_bias_eog.end());
        }
    }

    {
        params.antiprompt.clear();

        const auto & stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto & word : *stop) {
                if (!word.empty()) {
                    params.antiprompt.push_back(word);
                }
            }
        }
        // set reverse prompt from cli args if not set in the request
        if (params.antiprompt.empty()) {
            params.antiprompt = defaults.antiprompt;
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                params.sampling.samplers = common_sampler_types_from_names(*samplers);
            } else if (samplers->is_string()){
                params.sampling.samplers = common_sampler_types_from_chars(samplers->get<std::string>());
            }
        } else {
            params.sampling.samplers = defaults.sampling.samplers;
        }
    }

    if (params.n_cmpl > params_base.n_parallel) {
        throw std::runtime_error("n_cmpl cannot be greater than the number of slots, please increase -np");
    }

    return params;
}

//
// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
namespace {

namespace fs = std::filesystem;

constexpr const char * SERVER_PROMPT_CACHE_DISK_NAMESPACE = ".llama-prompt-cache-v1";
constexpr const char * SERVER_PROMPT_CACHE_OWNER_MAGIC = "llama.cpp automatic prompt cache v1";

static std::string server_prompt_cache_disk_path_utf8(const fs::path & path) {
#if defined(__cpp_lib_char8_t)
    const std::u8string value = path.u8string();
    return std::string(reinterpret_cast<const char *>(value.data()), value.size());
#else
    return path.u8string();
#endif
}

static bool server_prompt_cache_disk_owned(const fs::path & path) {
    std::ifstream owner(path / ".owner");
    std::string magic;
    return owner.good() && std::getline(owner, magic) && magic == SERVER_PROMPT_CACHE_OWNER_MAGIC;
}

static bool server_prompt_cache_disk_remove_file(const std::string & path) {
    if (path.empty()) {
        return true;
    }

    std::error_code ec;
    fs::remove(fs::u8path(path), ec);
    if (ec) {
        SRV_WRN("prompt cache disk cleanup failed: path=%s error=%s\n", path.c_str(), ec.message().c_str());
        return false;
    }

    // A missing file already satisfies the desired postcondition. This also
    // lets a later retry finish a pair after an earlier partial removal.
    return true;
}

static bool server_prompt_cache_disk_size_exact(
        const std::string & path,
                    size_t expected,
                    size_t * actual_out = nullptr) {
    if (path.empty()) {
        if (actual_out != nullptr) {
            *actual_out = 0;
        }
        return expected == 0;
    }

    std::error_code ec;
    const uintmax_t actual = fs::file_size(fs::u8path(path), ec);
    if (ec || actual > std::numeric_limits<size_t>::max()) {
        if (actual_out != nullptr) {
            *actual_out = 0;
        }
        return false;
    }

    if (actual_out != nullptr) {
        *actual_out = (size_t) actual;
    }
    return (size_t) actual == expected;
}

// llama_state_seq_save_file() closes the file before returning. Reopen it to
// force dirty pages to stable storage and immediately mark the cold state as
// reclaimable. This avoids replacing anonymous cache pressure with several GiB
// of sticky buffered page cache on UMA systems.
static bool server_prompt_cache_disk_flush_and_drop(const std::string & path, bool durable) {
#if !defined(_WIN32)
    const int fd = open(path.c_str(), (durable ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd < 0) {
        SRV_ERR("prompt cache disk open failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    bool ok = true;
    if (durable) {
#if defined(__APPLE__)
        const int sync_result = fsync(fd);
#else
        const int sync_result = fdatasync(fd);
#endif
        if (sync_result != 0) {
            SRV_ERR("prompt cache disk file sync failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
            ok = false;
        }
    }

#if defined(POSIX_FADV_DONTNEED)
    const int err = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (err != 0) {
        SRV_WRN("prompt cache disk fadvise failed: path=%s error=%s\n", path.c_str(), std::strerror(err));
    }
#endif

    close(fd);
    return ok;
#else
    GGML_UNUSED(path);
    GGML_UNUSED(durable);
    return true;
#endif
}

static bool server_prompt_cache_disk_sync_dir(const std::string & path) {
#if !defined(_WIN32)
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        SRV_ERR("prompt cache disk directory open failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    const bool ok = fsync(fd) == 0;
    if (!ok) {
        SRV_ERR("prompt cache disk directory fsync failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
    }
    close(fd);
    return ok;
#else
    GGML_UNUSED(path);
    return true;
#endif
}

static bool server_prompt_cache_tokens_equal(const server_tokens & expected, const llama_tokens & actual) {
    return expected.get_tokens() == actual;
}

} // namespace

server_prompt_cache::server_prompt_cache(
        int32_t limit_size_mib,
         size_t limit_tokens,
    const std::string & disk_base_path,
        int32_t disk_limit_size_mib) {
    ram_enabled       = limit_size_mib != 0;
    limit_size        = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens;

    if (disk_base_path.empty() || disk_limit_size_mib <= 0) {
        return;
    }

    disk_limit_size = 1024ull*1024ull*disk_limit_size_mib;

    std::error_code ec;
    fs::path base = fs::absolute(fs::u8path(disk_base_path), ec);
    if (ec) {
        throw std::runtime_error("unable to resolve prompt cache disk path '" + disk_base_path + "': " + ec.message());
    }

    fs::create_directories(base, ec);
    if (ec || !fs::is_directory(base)) {
        throw std::runtime_error("unable to create prompt cache disk path '" + server_prompt_cache_disk_path_utf8(base) + "': " + ec.message());
    }

    const fs::path cache_root = base / SERVER_PROMPT_CACHE_DISK_NAMESPACE;
    fs::create_directories(cache_root, ec);
    if (ec || !fs::is_directory(cache_root)) {
        throw std::runtime_error("unable to create prompt cache namespace '" + server_prompt_cache_disk_path_utf8(cache_root) + "': " + ec.message());
    }
    fs::permissions(cache_root, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) {
        throw std::runtime_error("unable to secure prompt cache namespace '" + server_prompt_cache_disk_path_utf8(cache_root) + "': " + ec.message());
    }

    // An OOM/SIGKILL cannot run the destructor. Each run therefore holds an
    // advisory lock in a magic-marked directory. A later server removes only
    // marked run-* directories whose lock is no longer held.
    for (const auto & entry : fs::directory_iterator(cache_root, ec)) {
        if (ec) {
            break;
        }
        const auto name = server_prompt_cache_disk_path_utf8(entry.path().filename());
        const bool is_run_dir      = name.rfind("run-", 0) == 0;
        const bool is_deleting_dir = name.rfind(".deleting-run-", 0) == 0;
        if (!entry.is_directory() || (!is_run_dir && !is_deleting_dir) || !server_prompt_cache_disk_owned(entry.path())) {
            continue;
        }

#if !defined(_WIN32)
        const fs::path lock_path = entry.path() / ".lock";
        const int fd = open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        const bool stale = flock(fd, LOCK_EX | LOCK_NB) == 0;
        if (stale) {
            flock(fd, LOCK_UN);
        }
        close(fd);
        if (!stale) {
            continue;
        }
#else
        // Without an advisory-lock primitive, preserve old directories rather
        // than risk deleting a live cache owned by another process.
        continue;
#endif

        const auto stale_path = server_prompt_cache_disk_path_utf8(entry.path());
        std::error_code rm_ec;
        const auto removed = fs::remove_all(entry.path(), rm_ec);
        if (!rm_ec) {
            SRV_INF("prompt cache disk stale cleanup: path=%s files=%zu\n", stale_path.c_str(), (size_t) removed);
        }
    }

    const auto stamp = (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
#if !defined(_WIN32)
    const auto pid = (uint64_t) getpid();
#else
    const uint64_t pid = 0;
#endif

    fs::path owned;
    for (uint32_t suffix = 0; suffix < 1000; ++suffix) {
        owned = cache_root / ("run-" + std::to_string(pid) + "-" + std::to_string(stamp) + "-" + std::to_string(suffix));
        if (fs::create_directory(owned, ec)) {
            break;
        }
        if (ec && ec != std::errc::file_exists) {
            throw std::runtime_error("unable to create owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
        }
        ec.clear();
        owned.clear();
    }
    if (owned.empty() || !fs::is_directory(owned)) {
        throw std::runtime_error("unable to allocate a unique prompt cache run directory below '" + server_prompt_cache_disk_path_utf8(cache_root) + "'");
    }

    fs::permissions(owned, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) {
        fs::remove_all(owned);
        throw std::runtime_error("unable to secure owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
    }

#if !defined(_WIN32)
    // Publish and hold the lock before publishing .owner. Stale cleanup only
    // considers magic-marked directories, so another startup can never see an
    // owned directory in the window before this process has acquired its lock.
    const fs::path lock_path = owned / ".lock";
    disk_lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (disk_lock_fd < 0 || flock(disk_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        if (disk_lock_fd >= 0) {
            close(disk_lock_fd);
            disk_lock_fd = -1;
        }
        fs::remove_all(owned);
        throw std::runtime_error("unable to lock owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "'");
    }
#else
    {
        std::ofstream lock(owned / ".lock", std::ios::out | std::ios::trunc);
        if (!lock.good()) {
            fs::remove_all(owned);
            throw std::runtime_error("unable to create prompt cache lock file in '" + server_prompt_cache_disk_path_utf8(owned) + "'");
        }
    }
#endif

    {
        std::ofstream owner(owned / ".owner", std::ios::out | std::ios::trunc);
        owner << SERVER_PROMPT_CACHE_OWNER_MAGIC << '\n'
              << "pid=" << pid << '\n'
              << "created=" << stamp << '\n';
        owner.flush();
        if (!owner.good()) {
#if !defined(_WIN32)
            flock(disk_lock_fd, LOCK_UN);
            close(disk_lock_fd);
            disk_lock_fd = -1;
#endif
            fs::remove_all(owned);
            throw std::runtime_error("unable to write prompt cache ownership manifest in '" + server_prompt_cache_disk_path_utf8(owned) + "'");
        }
    }
    fs::permissions(owned / ".owner", fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    if (ec) {
#if !defined(_WIN32)
        flock(disk_lock_fd, LOCK_UN);
        close(disk_lock_fd);
        disk_lock_fd = -1;
#endif
        fs::remove_all(owned);
        throw std::runtime_error("unable to secure prompt cache ownership manifest in '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
    }

    this->disk_base_path  = server_prompt_cache_disk_path_utf8(base);
    this->disk_owned_path = server_prompt_cache_disk_path_utf8(owned);

    SRV_INF("prompt cache disk enabled: path=%s owned_path=%s limit_mib=%d\n",
            this->disk_base_path.c_str(), this->disk_owned_path.c_str(), disk_limit_size_mib);
}

server_prompt_cache::~server_prompt_cache() {
    if (disk_owned_path.empty()) {
        return;
    }

    SRV_INF("prompt cache disk cleanup: path=%s entries=%zu bytes=%zu saves=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 "\n",
            disk_owned_path.c_str(), disk_states.size(), disk_size_total, disk_saves, disk_loads, disk_evictions);

    fs::path cleanup_path = fs::u8path(disk_owned_path);
    std::error_code ec;
    const fs::path trash_path = cleanup_path.parent_path() /
        fs::u8path(".deleting-" + server_prompt_cache_disk_path_utf8(cleanup_path.filename()));
    fs::rename(cleanup_path, trash_path, ec);
    if (!ec) {
        cleanup_path = trash_path;
    } else {
        ec.clear();
    }

#if !defined(_WIN32)
    if (disk_lock_fd >= 0) {
        flock(disk_lock_fd, LOCK_UN);
        close(disk_lock_fd);
        disk_lock_fd = -1;
    }
#endif

    fs::remove_all(cleanup_path, ec);
    if (ec) {
        const std::string cleanup_path_utf8 = server_prompt_cache_disk_path_utf8(cleanup_path);
        SRV_WRN("prompt cache disk cleanup failed: path=%s error=%s\n", cleanup_path_utf8.c_str(), ec.message().c_str());
    }
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::disk_size() const {
    return disk_size_total;
}

size_t server_prompt_cache::disk_n_tokens() const {
    size_t res = 0;
    for (const auto & state : disk_states) {
        res += state.n_tokens();
    }
    return res;
}

void server_prompt_cache::disable_disk_saves(const char * reason, const std::string & path) {
    disk_save_failures++;
    if (disk_save_disabled) {
        return;
    }

    disk_save_disabled = true;
    SRV_ERR("prompt cache disk writes disabled: reason=%s failures=%" PRIu64 " entries=%zu accounted_bytes=%zu path=%s cache_path=%s\n",
            reason, disk_save_failures, disk_states.size(), disk_size_total,
            path.empty() ? "-" : path.c_str(), disk_owned_path.c_str());
}

bool server_prompt_cache::save(
        const server_prompt & prompt,
              llama_context * ctx_main,
              llama_context * ctx_drft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec) {
    bool saved = false;

    if (!disk_owned_path.empty()) {
        saved = save_disk(prompt, ctx_main, ctx_drft, id_slot, state_spec) || saved;
    }

    if (!ram_enabled) {
        return saved;
    }

    const size_t state_size_main = llama_state_seq_get_size_ext(ctx_main, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t state_size_drft = ctx_drft ? llama_state_seq_get_size_ext(ctx_drft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

    auto * cur = alloc(prompt, state_size_main, state_size_drft, state_spec);
    if (cur == nullptr) {
        return saved;
    }

    const size_t n_main = llama_state_seq_get_data_ext(
        ctx_main, cur->data.main.data(), state_size_main, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_main != state_size_main) {
        SRV_ERR("failed to save RAM prompt cache target state: expected=%zu saved=%zu\n", state_size_main, n_main);
        states.pop_back();
        return saved;
    }

    if (ctx_drft) {
        const size_t n_drft = llama_state_seq_get_data_ext(
            ctx_drft, cur->data.drft.data(), state_size_drft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_drft != state_size_drft) {
            SRV_ERR("failed to save RAM prompt cache draft state: expected=%zu saved=%zu\n", state_size_drft, n_drft);
            states.pop_back();
            return saved;
        }
    }

    return true;
}

bool server_prompt_cache::save_disk(
        const server_prompt & prompt,
              llama_context * ctx_main,
              llama_context * ctx_drft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec) {
    if (disk_owned_path.empty() || disk_limit_size == 0 || prompt.tokens.empty()) {
        return false;
    }

    if (prompt.tokens.has_mtmd) {
        SRV_WRN("prompt cache disk skip: reason=multimodal tokens=%zu path=%s\n",
                prompt.tokens.size(), disk_owned_path.c_str());
        return false;
    }

    // If a usable cached prompt already contains the current stateless prompt,
    // retain the more useful state without rewriting the SSD. Stateful MTP
    // blobs are valid only at their exact token boundary, so they may touch an
    // equal-token entry but never a longer containing entry.
    for (auto it = disk_states.begin(); it != disk_states.end();) {
        if (!it->usable) {
            ++it;
            continue;
        }

        const int lcp = it->tokens.get_common_prefix(prompt.tokens);
        const bool exact_tokens = lcp == (int) prompt.tokens.size() && it->tokens.size() == prompt.tokens.size();
        const bool can_touch = state_spec.empty()
            ? lcp == (int) prompt.tokens.size()
            : exact_tokens;
        if (!can_touch) {
            ++it;
            continue;
        }

        const bool pair_shape_ok = !it->path_main.empty() && it->size_main > 0 &&
            ((ctx_drft != nullptr) == (!it->path_drft.empty() && it->size_drft > 0));
        const bool spec_shape_ok = state_spec.empty() || !it->spec.empty();
        size_t actual_main = 0;
        size_t actual_drft = 0;
        const bool files_ok = pair_shape_ok && spec_shape_ok &&
            server_prompt_cache_disk_size_exact(it->path_main, it->size_main, &actual_main) &&
            server_prompt_cache_disk_size_exact(it->path_drft, it->size_drft, &actual_drft);
        if (!files_ok) {
            SRV_WRN("prompt cache disk touch rejected: entry=%" PRIu64 " reason=unusable-pair target_bytes=%zu target_actual=%zu draft_bytes=%zu draft_actual=%zu spec_bytes=%zu path=%s\n",
                    it->id, it->size_main, actual_main, it->size_drft, actual_drft, it->spec.size(), disk_owned_path.c_str());
            auto bad = it++;
            bad->usable = false;
            if (!erase_disk_state(bad, false, "touch-unusable")) {
                disable_disk_saves("touch-unusable-removal", disk_owned_path);
            }
            continue;
        }

        {
            const auto id = it->id;
            disk_states.splice(disk_states.end(), disk_states, it);
            SRV_INF("prompt cache disk touch: entry=%" PRIu64 " lcp=%d tokens=%zu exact=%s stateful=%s safe_to_clear=true path=%s\n",
                    id, lcp, prompt.tokens.size(), exact_tokens ? "true" : "false",
                    state_spec.empty() ? "false" : "true", disk_owned_path.c_str());
            return true;
        }
    }

    if (disk_save_disabled) {
        SRV_DBG("prompt cache disk save skip: reason=circuit-open tokens=%zu path=%s\n",
                prompt.tokens.size(), disk_owned_path.c_str());
        return false;
    }

    const auto & tokens = prompt.tokens.get_tokens();
    const size_t token_bytes = tokens.size()*sizeof(llama_token);
    const size_t file_overhead = 3*sizeof(uint32_t) + token_bytes;
    const size_t state_size_main = llama_state_seq_get_size_ext(ctx_main, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t state_size_drft = ctx_drft ? llama_state_seq_get_size_ext(ctx_drft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    const size_t predicted_main = state_size_main + file_overhead;
    const size_t predicted_drft = ctx_drft ? state_size_drft + file_overhead : 0;
    const size_t predicted_total = predicted_main + predicted_drft;

    if (predicted_total > disk_limit_size) {
        SRV_WRN("prompt cache disk skip: reason=oversize target_bytes=%zu draft_bytes=%zu total_bytes=%zu limit_bytes=%zu tokens=%zu path=%s\n",
                predicted_main, predicted_drft, predicted_total, disk_limit_size, tokens.size(), disk_owned_path.c_str());
        return false;
    }

    const uint64_t entry_id = disk_next_id++;
    const fs::path owned = fs::u8path(disk_owned_path);
    const std::string stem = "state-" + std::to_string(entry_id);
    const fs::path path_main_tmp = owned / (stem + "-target.bin.tmp");
    const fs::path path_main     = owned / (stem + "-target.bin");
    const fs::path path_drft_tmp = owned / (stem + "-draft.bin.tmp");
    const fs::path path_drft     = owned / (stem + "-draft.bin");
    const std::string path_main_tmp_utf8 = server_prompt_cache_disk_path_utf8(path_main_tmp);
    const std::string path_main_utf8     = server_prompt_cache_disk_path_utf8(path_main);
    const std::string path_drft_tmp_utf8 = server_prompt_cache_disk_path_utf8(path_drft_tmp);
    const std::string path_drft_utf8     = server_prompt_cache_disk_path_utf8(path_drft);

    const auto cleanup_temps = [&]() -> bool {
        const bool main_ok = server_prompt_cache_disk_remove_file(path_main_tmp_utf8);
        const bool drft_ok = server_prompt_cache_disk_remove_file(path_drft_tmp_utf8);
        return main_ok && drft_ok;
    };
    const auto fail_io = [&](const char * reason, const std::string & path) -> bool {
        const bool cleanup_ok = cleanup_temps();
        disable_disk_saves(reason, path);
        if (!cleanup_ok) {
            disable_disk_saves("temporary-cleanup", disk_owned_path);
        }
        return false;
    };

    const int64_t t_start = ggml_time_us();

    const size_t n_main = llama_state_seq_save_file(
        ctx_main, path_main_tmp_utf8.c_str(), id_slot, tokens.data(), tokens.size());
    size_t actual_main = 0;
    if (n_main == 0 ||
        !server_prompt_cache_disk_size_exact(path_main_tmp_utf8, n_main, &actual_main) ||
        !server_prompt_cache_disk_flush_and_drop(path_main_tmp_utf8, true)) {
        SRV_ERR("prompt cache disk save failed: entry=%" PRIu64 " component=target path=%s\n",
                entry_id, path_main_tmp_utf8.c_str());
        return fail_io("target-save", path_main_tmp_utf8);
    }

    size_t n_drft = 0;
    if (ctx_drft) {
        n_drft = llama_state_seq_save_file(
            ctx_drft, path_drft_tmp_utf8.c_str(), id_slot, tokens.data(), tokens.size());
        size_t actual_drft = 0;
        if (n_drft == 0 ||
            !server_prompt_cache_disk_size_exact(path_drft_tmp_utf8, n_drft, &actual_drft) ||
            !server_prompt_cache_disk_flush_and_drop(path_drft_tmp_utf8, true)) {
            SRV_ERR("prompt cache disk save failed: entry=%" PRIu64 " component=draft path=%s\n",
                    entry_id, path_drft_tmp_utf8.c_str());
            return fail_io("draft-save", path_drft_tmp_utf8);
        }
    }

    const size_t actual_total = n_main + n_drft;
    if (actual_total > disk_limit_size) {
        const bool cleanup_ok = cleanup_temps();
        SRV_WRN("prompt cache disk skip: reason=actual-oversize entry=%" PRIu64 " target_bytes=%zu draft_bytes=%zu total_bytes=%zu limit_bytes=%zu path=%s\n",
                entry_id, n_main, n_drft, actual_total, disk_limit_size, disk_owned_path.c_str());
        if (!cleanup_ok) {
            disable_disk_saves("actual-oversize-cleanup", disk_owned_path);
        }
        return false;
    }

    std::error_code ec;
    fs::permissions(path_main_tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    if (ec) {
        SRV_ERR("prompt cache disk permissions failed: entry=%" PRIu64 " component=target path=%s error=%s\n",
                entry_id, path_main_tmp_utf8.c_str(), ec.message().c_str());
        return fail_io("target-permissions", path_main_tmp_utf8);
    }
    if (ctx_drft) {
        fs::permissions(path_drft_tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
        if (ec) {
            SRV_ERR("prompt cache disk permissions failed: entry=%" PRIu64 " component=draft path=%s error=%s\n",
                    entry_id, path_drft_tmp_utf8.c_str(), ec.message().c_str());
            return fail_io("draft-permissions", path_drft_tmp_utf8);
        }
    }

    // The complete target/draft temporary pair is durable. Commit it before
    // touching older entries so a rename or directory-sync failure cannot
    // destroy a previously usable cache. This permits one incoming entry of
    // transient staging headroom above the configured payload limit.
    ec.clear();
    fs::rename(path_main_tmp, path_main, ec);
    if (ec) {
        SRV_ERR("prompt cache disk atomic rename failed: entry=%" PRIu64 " component=target path=%s error=%s\n",
                entry_id, path_main_utf8.c_str(), ec.message().c_str());
        return fail_io("target-rename", path_main_utf8);
    }

    if (ctx_drft) {
        ec.clear();
        fs::rename(path_drft_tmp, path_drft, ec);
        if (ec) {
            const bool main_cleanup_ok = server_prompt_cache_disk_remove_file(path_main_utf8);
            const bool temp_cleanup_ok = cleanup_temps();
            SRV_ERR("prompt cache disk atomic rename failed: entry=%" PRIu64 " component=draft path=%s error=%s\n",
                    entry_id, path_drft_utf8.c_str(), ec.message().c_str());
            disable_disk_saves("draft-rename", path_drft_utf8);
            if (!main_cleanup_ok || !temp_cleanup_ok) {
                disable_disk_saves("draft-rename-cleanup", disk_owned_path);
            }
            return false;
        }
    }

    if (!server_prompt_cache_disk_sync_dir(disk_owned_path) ||
        !server_prompt_cache_disk_flush_and_drop(path_main_utf8, false) ||
        (ctx_drft && !server_prompt_cache_disk_flush_and_drop(path_drft_utf8, false))) {
        const bool main_cleanup_ok = server_prompt_cache_disk_remove_file(path_main_utf8);
        const bool drft_cleanup_ok = server_prompt_cache_disk_remove_file(path_drft_utf8);
        disable_disk_saves("commit-sync", disk_owned_path);
        if (!main_cleanup_ok || !drft_cleanup_ok) {
            disable_disk_saves("commit-sync-cleanup", disk_owned_path);
        }
        return false;
    }

    server_prompt_disk_state state;
    state.tokens    = prompt.tokens.clone();
    state.path_main = path_main_utf8;
    state.path_drft = ctx_drft ? path_drft_utf8 : std::string();
    state.size_main = n_main;
    state.size_drft = n_drft;
    state.spec      = state_spec;
    state.id        = entry_id;
    state.usable    = true;
    state.checkpoints.reserve(prompt.checkpoints.size());
    for (const auto & ckpt : prompt.checkpoints) {
        state.checkpoints.push_back({ckpt.n_tokens, ckpt.pos_min, ckpt.pos_max});
    }

    disk_states.push_back(std::move(state));
    disk_size_total    += actual_total;
    disk_saves++;
    disk_bytes_written += actual_total;

    auto new_entry = std::prev(disk_states.end());
    bool reclaim_ok = true;

    // Stateless entries can supersede shorter prefixes. Stateful MTP blobs
    // remain independently useful exact-boundary states.
    if (state_spec.empty()) {
        for (auto it = disk_states.begin(); it != new_entry;) {
            const int lcp = it->tokens.get_common_prefix(prompt.tokens);
            if (lcp == (int) it->tokens.size()) {
                auto obsolete = it++;
                if (!erase_disk_state(obsolete, false, "obsolete-prefix")) {
                    disable_disk_saves("obsolete-reclaim", disk_owned_path);
                    reclaim_ok = false;
                    break;
                }
            } else {
                ++it;
            }
        }
    }

    while (reclaim_ok && disk_size_total > disk_limit_size) {
        if (disk_states.begin() == new_entry) {
            SRV_ERR("prompt cache disk reclaim failed: entry=%" PRIu64 " reason=no-old-victim accounted_bytes=%zu limit_bytes=%zu path=%s\n",
                    entry_id, disk_size_total, disk_limit_size, disk_owned_path.c_str());
            disable_disk_saves("room-not-reclaimed", disk_owned_path);
            reclaim_ok = false;
            break;
        }
        if (!erase_disk_state(disk_states.begin(), true, "lru-limit")) {
            disable_disk_saves("lru-reclaim", disk_owned_path);
            reclaim_ok = false;
            break;
        }
    }

    if (!reclaim_ok) {
        SRV_WRN("prompt cache disk committed over limit: entry=%" PRIu64 " accounted_bytes=%zu limit_bytes=%zu save_disabled=true path=%s\n",
                entry_id, disk_size_total, disk_limit_size, disk_owned_path.c_str());
    }

    const double t_ms = (ggml_time_us() - t_start)/1000.0;
    SRV_INF("prompt cache disk save: entry=%" PRIu64 " tokens=%zu checkpoints=%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu save_ms=%.2f path=%s\n",
            entry_id, tokens.size(), prompt.checkpoints.size(), n_main, n_drft, state_spec.size(), actual_total, t_ms, disk_owned_path.c_str());
    log_disk_state();

    return true;
}

server_prompt * server_prompt_cache::alloc(
        const server_prompt & prompt,
                    size_t state_size_tgt,
                    size_t state_size_dft,
        const std::vector<uint8_t> & state_spec) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);
        const bool exact_tokens = cur_lcp_len == (int) prompt.tokens.size() &&
            it->tokens.size() == prompt.tokens.size();
        const bool cached_boundary = state_spec.empty()
            ? cur_lcp_len == (int) prompt.tokens.size()
            : exact_tokens && !it->data.spec.empty();

        if (cached_boundary) {
            SRV_INF("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + state_spec.size() + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // Stateful speculative blobs are exact-boundary states. Keep shorter
    // boundaries instead of treating them as obsolete prefixes.
    if (state_spec.empty()) {
        for (auto it = states.begin(); it != states.end();) {
            const int len = it->tokens.get_common_prefix(prompt.tokens);

            if (len == (int) it->tokens.size()) {
                SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

                it = states.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
            /*.spec =*/ state_spec,
        },
        /*.checkpoints =*/ prompt.checkpoints,
    });

    return &states.back();
}

bool server_prompt_cache::load_disk(
        std::list<server_prompt_disk_state>::iterator it,
        server_prompt & prompt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
         llama_seq_id   id_slot,
              size_t   lcp,
            uint64_t * entry_id_out) {
    if (entry_id_out != nullptr) {
        *entry_id_out = 0;
    }

    const uint64_t entry_id = it->id;
    const size_t target_bytes = it->size_main;
    const size_t draft_bytes  = it->size_drft;
    const size_t spec_bytes   = it->spec.size();
    const size_t total_bytes  = it->size();
    const size_t n_tokens_expected = it->tokens.size();
    const size_t n_checkpoints = it->checkpoints.size();
    const std::string path_main = it->path_main;
    const std::string path_drft = it->path_drft;

    const auto reject_entry = [&](const char * reason) -> bool {
        it->usable = false;
        if (!erase_disk_state(it, false, reason)) {
            disable_disk_saves("invalid-entry-removal", disk_owned_path);
        }
        log_disk_state();
        return false;
    };

    // Validate the entire pair before mutating either context.
    size_t actual_main = 0;
    size_t actual_drft = 0;
    if (path_main.empty() || target_bytes == 0 ||
        !server_prompt_cache_disk_size_exact(path_main, target_bytes, &actual_main)) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=target reason=size-mismatch expected_bytes=%zu actual_bytes=%zu path=%s\n",
                entry_id, target_bytes, actual_main, path_main.c_str());
        return reject_entry("target-size-mismatch");
    }
    if (!path_drft.empty()) {
        if (ctx_dft == nullptr || draft_bytes == 0 ||
            !server_prompt_cache_disk_size_exact(path_drft, draft_bytes, &actual_drft)) {
            SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft reason=size-mismatch expected_bytes=%zu actual_bytes=%zu path=%s\n",
                    entry_id, draft_bytes, actual_drft, path_drft.c_str());
            return reject_entry("draft-size-mismatch");
        }
    } else if (ctx_dft != nullptr || draft_bytes != 0) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft reason=missing-draft-file expected_bytes=%zu path=%s\n",
                entry_id, draft_bytes, disk_owned_path.c_str());
        return reject_entry("missing-draft-file");
    }

    const int64_t t_start = ggml_time_us();

    llama_tokens tokens_main(n_tokens_expected);
    size_t n_tokens_main = 0;
    const size_t nread_main = llama_state_seq_load_file(
        ctx_tgt, path_main.c_str(), id_slot,
        tokens_main.data(), tokens_main.size(), &n_tokens_main);
    tokens_main.resize(n_tokens_main);
    server_prompt_cache_disk_flush_and_drop(path_main, false);

    if (nread_main != target_bytes || !server_prompt_cache_tokens_equal(it->tokens, tokens_main)) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=target expected_bytes=%zu read_bytes=%zu expected_tokens=%zu restored_tokens=%zu path=%s\n",
                entry_id, target_bytes, nread_main, n_tokens_expected, n_tokens_main, path_main.c_str());
        return reject_entry("corrupt-target");
    }

    size_t nread_drft = 0;
    if (!path_drft.empty()) {
        llama_tokens tokens_drft(n_tokens_expected);
        size_t n_tokens_drft = 0;
        nread_drft = llama_state_seq_load_file(
            ctx_dft, path_drft.c_str(), id_slot,
            tokens_drft.data(), tokens_drft.size(), &n_tokens_drft);
        tokens_drft.resize(n_tokens_drft);
        server_prompt_cache_disk_flush_and_drop(path_drft, false);

        if (nread_drft != draft_bytes ||
            !server_prompt_cache_tokens_equal(it->tokens, tokens_drft) ||
            tokens_drft != tokens_main) {
            SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft expected_bytes=%zu read_bytes=%zu expected_tokens=%zu restored_tokens=%zu path=%s\n",
                    entry_id, draft_bytes, nread_drft, n_tokens_expected, n_tokens_drft, path_drft.c_str());
            return reject_entry("corrupt-draft");
        }
    }

    server_prompt restored;
    restored.tokens = it->tokens.clone();
    restored.data.spec = it->spec;
    // Intentionally do not recreate common_prompt_checkpoint payloads. The
    // disk entry retained only their small positions, not cloned host/device
    // state. Fresh checkpoints are created as processing continues.
    prompt = std::move(restored);

    disk_bytes_read += nread_main + nread_drft;

    const double t_ms = (ggml_time_us() - t_start)/1000.0;
    SRV_INF("prompt cache disk load: entry=%" PRIu64 " lcp=%zu tokens=%zu checkpoints=%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu read_bytes=%zu load_ms=%.2f path=%s\n",
            entry_id, lcp, n_tokens_expected, n_checkpoints, target_bytes, draft_bytes, spec_bytes, total_bytes,
            nread_main + nread_drft, t_ms, disk_owned_path.c_str());

    if (entry_id_out != nullptr) {
        *entry_id_out = entry_id;
    }
    return true;
}

bool server_prompt_cache::erase_disk_state(
        std::list<server_prompt_disk_state>::iterator it,
        bool eviction,
        const char * reason) {
    const uint64_t entry_id    = it->id;
    const size_t target_bytes  = it->size_main;
    const size_t draft_bytes   = it->size_drft;
    const size_t spec_bytes    = it->spec.size();
    const size_t total_bytes   = it->size();
    const size_t tokens        = it->tokens.size();
    const std::string path_main = it->path_main;
    const std::string path_drft = it->path_drft;

    // Quarantine before touching either component. If only one unlink works,
    // retain the full conservative accounting and metadata for a later retry.
    it->usable = false;
    const bool main_ok = server_prompt_cache_disk_remove_file(path_main);
    const bool drft_ok = server_prompt_cache_disk_remove_file(path_drft);
    if (!main_ok || !drft_ok) {
        SRV_ERR("prompt cache disk removal failed: entry=%" PRIu64 " reason=%s target_removed=%s draft_removed=%s accounted_bytes=%zu path=%s\n",
                entry_id, reason, main_ok ? "true" : "false", drft_ok ? "true" : "false",
                disk_size_total, disk_owned_path.c_str());
        return false;
    }

    if (total_bytes > disk_size_total) {
        SRV_ERR("prompt cache disk accounting invariant failed: entry=%" PRIu64 " entry_bytes=%zu accounted_bytes=%zu path=%s\n",
                entry_id, total_bytes, disk_size_total, disk_owned_path.c_str());
        return false;
    }
    disk_size_total -= total_bytes;

    if (eviction) {
        disk_evictions++;
        disk_bytes_evicted += total_bytes;
        SRV_INF("prompt cache disk eviction: entry=%" PRIu64 " reason=%s tokens=%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu remaining_bytes=%zu path=%s\n",
                entry_id, reason, tokens, target_bytes, draft_bytes, spec_bytes, total_bytes, disk_size_total, disk_owned_path.c_str());
    } else {
        SRV_INF("prompt cache disk remove: entry=%" PRIu64 " reason=%s tokens=%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu remaining_bytes=%zu path=%s\n",
                entry_id, reason, tokens, target_bytes, draft_bytes, spec_bytes, total_bytes, disk_size_total, disk_owned_path.c_str());
    }

    disk_states.erase(it);
    return true;
}

void server_prompt_cache::accept_disk_load(uint64_t entry_id) {
    if (entry_id == 0) {
        return;
    }

    for (auto it = disk_states.begin(); it != disk_states.end(); ++it) {
        if (it->id != entry_id || !it->usable) {
            continue;
        }

        disk_loads++;
        disk_states.splice(disk_states.end(), disk_states, it);
        SRV_INF("prompt cache disk load accepted: entry=%" PRIu64 " reusable=true path=%s\n",
                entry_id, disk_owned_path.c_str());
        log_disk_state();
        return;
    }
}

void server_prompt_cache::reject_disk_load(uint64_t entry_id, const char * reason) {
    if (entry_id == 0) {
        return;
    }

    for (auto it = disk_states.begin(); it != disk_states.end(); ++it) {
        if (it->id != entry_id) {
            continue;
        }

        it->usable = false;
        SRV_WRN("prompt cache disk load rejected: entry=%" PRIu64 " reason=%s reusable=false path=%s\n",
                entry_id, reason, disk_owned_path.c_str());
        if (!erase_disk_state(it, false, reason)) {
            disable_disk_saves("rejected-load-removal", disk_owned_path);
        }
        log_disk_state();
        return;
    }
}

void server_prompt_cache::update_disk() {
    while (!disk_states.empty() && disk_size_total > disk_limit_size) {
        if (!erase_disk_state(disk_states.begin(), true, "lru-update-limit")) {
            disable_disk_saves("update-limit-removal", disk_owned_path);
            break;
        }
    }

    log_disk_state();
}

void server_prompt_cache::log_disk_state() const {
    if (disk_owned_path.empty()) {
        return;
    }

    const size_t unusable = std::count_if(disk_states.begin(), disk_states.end(),
        [](const server_prompt_disk_state & state) { return !state.usable; });
    SRV_INF("prompt cache disk state: entries=%zu unusable=%zu bytes=%zu limit_bytes=%zu over_limit=%s tokens=%zu saves=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 " save_disabled=%s save_failures=%" PRIu64 " bytes_written=%" PRIu64 " bytes_read=%" PRIu64 " bytes_evicted=%" PRIu64 " path=%s\n",
            disk_states.size(), unusable, disk_size_total, disk_limit_size,
            disk_size_total > disk_limit_size ? "true" : "false", disk_n_tokens(),
            disk_saves, disk_loads, disk_evictions, disk_save_disabled ? "true" : "false", disk_save_failures,
            disk_bytes_written, disk_bytes_read, disk_bytes_evicted,
            disk_owned_path.c_str());
}

bool server_prompt_cache::load(
              server_prompt & prompt,
        const server_tokens & tokens_new,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
                    int32_t   id_slot,
                       bool   spec_state_required,
                       bool   spec_trailing_rm,
                       bool * cache_hit,
                   uint64_t * disk_entry_id) {
    if (cache_hit != nullptr) {
        *cache_hit = false;
    }
    if (disk_entry_id != nullptr) {
        *disk_entry_id = 0;
    }

    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    // With speculative decoding, an entry whose token sequence extends past the
    // common prefix (lcp < cached_tokens, e.g. a client re-sending a normalized
    // conversation) can still be salvaged when the target and draft memories
    // support removing the diverging tail: the slot code then performs a bounded
    // trailing rollback and reprocesses only the new tokens.
    const auto spec_boundary_valid = [&](size_t cached_tokens, int lcp) {
        if (!spec_state_required || lcp == (int) cached_tokens) {
            return true;
        }
        if (!spec_trailing_rm || lcp < 0 || cached_tokens <= (size_t) lcp) {
            return false;
        }
        const size_t delta = cached_tokens - (size_t) lcp;
        // dense KV (n_rs_seq == 0) supports removing any tail; bounded RS state only up to the snapshot bound
        uint32_t n_rs_min = UINT32_MAX;
        if (const uint32_t n_rs = llama_n_rs_seq(ctx_tgt); n_rs > 0) {
            n_rs_min = std::min(n_rs_min, n_rs);
        }
        if (ctx_dft) {
            if (const uint32_t n_rs = llama_n_rs_seq(ctx_dft); n_rs > 0) {
                n_rs_min = std::min(n_rs_min, n_rs);
            }
        }
        return n_rs_min == UINT32_MAX || delta <= (size_t) n_rs_min;
    };

    const bool base_boundary_valid = spec_boundary_valid(prompt.tokens.size(), lcp_best);
    float f_keep_best = base_boundary_valid && prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = base_boundary_valid ? float(lcp_best) / std::max<size_t>(1, tokens_new.size()) : -1.0f;

    if (spec_state_required && !prompt.tokens.empty() && !base_boundary_valid) {
        SRV_INF("prompt cache skip: reason=spec-boundary-mismatch source=slot lcp=%d cached_tokens=%zu request_tokens=%zu\n",
                lcp_best, prompt.tokens.size(), tokens_new.size());
    }

    SRV_INF(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best_ram  = states.end();
    auto it_best_disk = disk_states.end();
    size_t lcp_selected = 0;
    size_t spec_boundary_best = base_boundary_valid ? prompt.tokens.size() : 0;
    bool ram_loaded = false;

    // Find the most similar RAM prompt first. On an equal match, the hot RAM
    // copy wins and avoids SSD I/O.
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        if (!spec_boundary_valid(it->tokens.size(), lcp_cur)) {
            SRV_INF("prompt cache skip: reason=spec-boundary-mismatch source=ram lcp=%d cached_tokens=%zu request_tokens=%zu spec_bytes=%zu\n",
                    lcp_cur, it->tokens.size(), tokens_new.size(), it->data.spec.size());
            continue;
        }
        if (spec_state_required && it->data.spec.empty()) {
            SRV_INF("prompt cache skip: reason=spec-state-missing source=ram cached_tokens=%zu request_tokens=%zu\n",
                    it->tokens.size(), tokens_new.size());
            continue;
        }

        const float f_keep_cur = float(lcp_cur) / std::max<size_t>(1, it->tokens.size());
        const float sim_cur    = float(lcp_cur) / std::max<size_t>(1, tokens_new.size());

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        const bool is_better = spec_state_required
            ? it->tokens.size() > spec_boundary_best
            : f_keep_best < f_keep_cur && sim_best < sim_cur;
        if (is_better) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;
            spec_boundary_best = it->tokens.size();

            it_best_ram  = it;
            it_best_disk = disk_states.end();
            lcp_selected = lcp_cur;
        }
    }

    for (auto it = disk_states.begin(); it != disk_states.end(); ++it) {
        if (!it->usable) {
            continue;
        }

        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        if (!spec_boundary_valid(it->tokens.size(), lcp_cur)) {
            SRV_INF("prompt cache skip: reason=spec-boundary-mismatch source=disk entry=%" PRIu64 " lcp=%d cached_tokens=%zu request_tokens=%zu spec_bytes=%zu\n",
                    it->id, lcp_cur, it->tokens.size(), tokens_new.size(), it->spec.size());
            continue;
        }
        if (spec_state_required && it->spec.empty()) {
            SRV_INF("prompt cache skip: reason=spec-state-missing source=disk entry=%" PRIu64 " cached_tokens=%zu request_tokens=%zu\n",
                    it->id, it->tokens.size(), tokens_new.size());
            continue;
        }

        const float f_keep_cur = float(lcp_cur) / std::max<size_t>(1, it->tokens.size());
        const float sim_cur    = float(lcp_cur) / std::max<size_t>(1, tokens_new.size());

        if (f_keep_cur < 0.25f) {
            continue;
        }

        const bool is_better = spec_state_required
            ? it->tokens.size() > spec_boundary_best
            : f_keep_best < f_keep_cur && sim_best < sim_cur;
        if (is_better) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;
            spec_boundary_best = it->tokens.size();

            it_best_ram  = states.end();
            it_best_disk = it;
            lcp_selected = lcp_cur;
        }
    }

    if (it_best_disk != disk_states.end()) {
        SRV_INF(" - found better disk prompt with f_keep = %.3f, sim = %.3f, lcp = %zu\n",
                f_keep_best, sim_best, lcp_selected);
        const bool loaded = load_disk(it_best_disk, prompt, ctx_tgt, ctx_dft, id_slot, lcp_selected, disk_entry_id);
        if (loaded && cache_hit != nullptr) {
            *cache_hit = true;
        }
        return loaded;
    }

    if (it_best_ram != states.end()) {
        SRV_INF(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        {
            auto & data = it_best_ram->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best_ram->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(*it_best_ram);

        states.erase(it_best_ram);

        if (cache_hit != nullptr) {
            *cache_hit = true;
        }
        ram_loaded = true;
    }

    return base_boundary_valid || ram_loaded;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_INF(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_INF("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }

    update_disk();
}
