#include "server-classifier.h"

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"
#include "hash/hash.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

// Mirrors tinyjev's pointer family (tinyjev/families/pointer.py) and System One answer shapes (tinyjev/agent.py),
// which follow Kev (Apache-2.0, Jared Palmer); verified against that Python implementation on OpenDecision cases.
// The letter family follows the option-letter readout of Wald (org2ai/Wald-4B, wald_serve) and StartLux-Decision
// (startlux_decision/jevfmt.py); their differences are prompt wording, which lives in the config.

#define CLS_INF(fmt, ...) LOG_INF("classifier: " fmt, __VA_ARGS__)

// ---------------------------------------------------------------------------------------------------------------
// text rendering: state / instructions / option descriptions may be strings, numbers, lists or objects

static std::string lstrip(const std::string & s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) { i++; }
    return s.substr(i);
}

static std::string render(const json & v, int indent = 0) {
    const std::string pad(2 * indent, ' ');
    if (v.is_null())    { return ""; }
    if (v.is_string())  { return v.get<std::string>(); }
    if (v.is_boolean()) { return v.get<bool>() ? "True" : "False"; }   // Python str(bool)
    if (v.is_number())  { return v.dump(); }
    std::string out;
    if (v.is_array()) {
        for (size_t i = 0; i < v.size(); i++) {
            if (i) { out += "\n"; }
            out += pad + "- " + lstrip(render(v.at(i), indent + 1));
        }
        return out;
    }
    bool first = true;
    for (const auto & [k, x] : v.items()) {
        if (!first) { out += "\n"; }
        first = false;
        if (x.is_object() || x.is_array()) {
            out += pad + k + ":\n" + render(x, indent + 1);
        } else {
            out += pad + k + ": " + render(x);
        }
    }
    return out;
}

static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) { s.replace(p, from.size(), to); }
    return s;
}

static std::string strip(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) { a++; }
    while (b > a && std::isspace((unsigned char) s[b - 1])) { b--; }
    return s.substr(a, b - a);
}

// "{name}" placeholders in one pass, so text that was substituted is never scanned again
static std::string expand(const std::string & tpl, const std::vector<std::pair<std::string, const std::string *>> & vars) {
    std::string out;
    for (size_t i = 0; i < tpl.size();) {
        bool hit = false;
        if (tpl[i] == '{') {
            for (const auto & [k, v] : vars) {
                if (tpl.compare(i + 1, k.size(), k) == 0 && i + 1 + k.size() < tpl.size() && tpl[i + 1 + k.size()] == '}') {
                    out += *v;
                    i += k.size() + 2;
                    hit = true;
                    break;
                }
            }
        }
        if (!hit) { out += tpl[i++]; }
    }
    return out;
}

// compact JSON as Python's json.dumps(ensure_ascii=False) writes it (", " and ": "), optionally with the element
// index added to the items of long arrays
static std::string render_json(const json & v, int index_min) {
    if (v.is_array()) {
        std::string out = "[";
        const bool indexed = index_min > 0 && (int) v.size() >= index_min;
        for (size_t i = 0; i < v.size(); i++) {
            if (i) { out += ", "; }
            if (!indexed) {
                out += render_json(v.at(i), index_min);
            } else if (v.at(i).is_object()) {
                const std::string inner = render_json(v.at(i), index_min);
                out += "{\"_index\": " + std::to_string(i) + (inner.size() > 2 ? ", " + inner.substr(1) : "}");
            } else {
                out += "{\"_index\": " + std::to_string(i) + ", \"value\": " + render_json(v.at(i), index_min) + "}";
            }
        }
        return out + "]";
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (const auto & [k, x] : v.items()) {
            if (!first) { out += ", "; }
            first = false;
            out += json(k).dump() + ": " + render_json(x, index_min);
        }
        return out + "}";
    }
    return v.dump();
}

static std::string norm_label(const std::string & s) {
    static const std::regex re("[\\s_\\-]+");
    std::string out = strip(std::regex_replace(s, re, " "));
    for (auto & c : out) { c = (char) std::tolower((unsigned char) c); }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// config + head loading

static void load_config(const std::string & path, classifier_config & c) {
    if (path.empty()) { return; }
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open --classifier-config " + path); }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const json j = json::parse(text);
    auto str = [&](const json & o, const char * k, std::string & dst) { if (o.contains(k) && o.at(k).is_string()) { dst = o.at(k).get<std::string>(); } };
    auto num = [&](const json & o, const char * k, auto & dst) { if (o.contains(k) && o.at(k).is_number()) { dst = o.at(k).get<typename std::decay<decltype(dst)>::type>(); } };
    str(j, "name", c.name);
    str(j, "family", c.family);
    num(j, "max_state", c.max_state);
    num(j, "max_branch", c.max_branch);
    num(j, "max_options", c.max_options);
    if (j.contains("head")) { num(j.at("head"), "head_dim", c.head_dim); num(j.at("head"), "temperature", c.temperature); }
    if (j.contains("layout")) {
        const json & l = j.at("layout");
        if (l.contains("tokens")) {
            const json & t = l.at("tokens");
            str(t, "state", c.tok_state); str(t, "question", c.tok_question); str(t, "option", c.tok_option);
            str(t, "close", c.tok_close); str(t, "decide", c.tok_decide);
        }
        if (l.contains("escape_special") && l.at("escape_special").is_boolean()) { c.escape_special = l.at("escape_special").get<bool>(); }
        str(l, "option_format", c.option_format);
        if (l.contains("boolean_labels")) { str(l.at("boolean_labels"), "false", c.label_false); str(l.at("boolean_labels"), "true", c.label_true); }
    }
    if (j.contains("letter")) {
        const json & l = j.at("letter");
        auto flag = [&](const char * k, bool & dst) { if (l.contains(k) && l.at(k).is_boolean()) { dst = l.at(k).get<bool>(); } };
        str(l, "style", c.lt_style);
        str(l, "prompt", c.lt_prompt); str(l, "prompt_no_state", c.lt_prompt_no_state); str(l, "empty_state", c.lt_empty_state);
        str(l, "option_line", c.lt_option_line); str(l, "state_format", c.lt_state_format); str(l, "score_format", c.lt_score_format);
        str(l, "noul_instructions", c.lt_noul_instructions); str(l, "bare_keys", c.lt_bare_keys);
        num(l, "json_index_min", c.lt_json_index_min);
        flag("true_first", c.lt_true_first); flag("same_as_name", c.lt_same_as_name); flag("strip", c.lt_strip);
        flag("share_prefix", c.lt_share_prefix);
        if (l.contains("variants") && l.at("variants").is_array()) {
            c.lt_variants.clear();
            for (const auto & v : l.at("variants")) { c.lt_variants.push_back(v.get<std::string>()); }
        }
        if (l.contains("temperature")) {
            const json & t = l.at("temperature");
            if (t.is_number()) {
                c.lt_temperature["default"] = t.get<double>();
            } else if (t.is_object()) {
                for (const auto & [k, v] : t.items()) { if (v.is_number()) { c.lt_temperature[k] = v.get<double>(); } }
            }
        }
    }
    if (c.family == "kev") { c.family = "pointer"; }
    if (c.family != "pointer" && c.family != "letter") {
        throw std::runtime_error("unsupported classifier family '" + c.family + "' (supported: pointer, letter)");
    }
    if (c.family == "letter") {
        if (c.lt_style != "lines" && c.lt_style != "ollama") { throw std::runtime_error("letter.style must be lines or ollama"); }
        if (c.lt_style == "ollama" && c.lt_prompt.find("{user}") == std::string::npos) {
            throw std::runtime_error("letter classifier: with style ollama, letter.prompt needs {user}");
        }
        if (c.lt_style == "lines" && (c.lt_prompt.find("{options}") == std::string::npos || c.lt_option_line.find("{letter}") == std::string::npos)) {
            throw std::runtime_error("letter classifier: letter.prompt needs {options} and letter.option_line needs {letter}");
        }
        if (c.lt_state_format != "text" && c.lt_state_format != "json") { throw std::runtime_error("letter.state_format must be text or json"); }
        if (c.lt_bare_keys != "none" && c.lt_bare_keys != "positional" && c.lt_bare_keys != "bare") {
            throw std::runtime_error("letter.bare_keys must be none, positional or bare");
        }
    }
}

// minimal safetensors reader: u64 header size, JSON header {name: {dtype, shape, data_offsets}}, raw little-endian data
static void load_head(const std::string & path, std::vector<float> & wq, std::vector<float> & bq,
                      std::vector<float> & wk, std::vector<float> & bk, int & head_dim, int n_embd) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open --classifier-head " + path); }
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char *>(&hlen), 8);
    if (!f || hlen == 0 || hlen > (1u << 24)) { throw std::runtime_error("not a safetensors file: " + path); }
    std::string htext(hlen, '\0');
    f.read(htext.data(), (std::streamsize) hlen);
    const json header = json::parse(htext);
    const std::streamoff base = 8 + (std::streamoff) hlen;

    auto tensor = [&](const std::string & name, std::vector<int64_t> & shape) {
        if (!header.contains(name)) { throw std::runtime_error("classifier head is missing tensor '" + name + "'"); }
        const json & t = header.at(name);
        const std::string dtype = t.at("dtype").get<std::string>();
        shape.clear();
        int64_t n = 1;
        for (size_t i = 0; i < t.at("shape").size(); i++) { shape.push_back(t.at("shape").at(i).get<int64_t>()); n *= shape.back(); }
        const uint64_t a = t.at("data_offsets").at(0).get<uint64_t>(), b = t.at("data_offsets").at(1).get<uint64_t>();
        std::vector<uint8_t> raw(b - a);
        f.seekg(base + (std::streamoff) a);
        f.read(reinterpret_cast<char *>(raw.data()), (std::streamsize) raw.size());
        std::vector<float> out((size_t) n);
        if (dtype == "F32" && raw.size() == (size_t) n * 4) {
            std::memcpy(out.data(), raw.data(), raw.size());
        } else if (dtype == "F16" && raw.size() == (size_t) n * 2) {
            for (int64_t i = 0; i < n; i++) { ggml_fp16_t h; std::memcpy(&h, raw.data() + 2 * i, 2); out[i] = ggml_fp16_to_fp32(h); }
        } else if (dtype == "BF16" && raw.size() == (size_t) n * 2) {
            for (int64_t i = 0; i < n; i++) { uint32_t u = (uint32_t) (raw[2 * i] | (raw[2 * i + 1] << 8)) << 16; std::memcpy(&out[i], &u, 4); }
        } else {
            throw std::runtime_error("classifier head tensor '" + name + "': unsupported dtype " + dtype);
        }
        return out;
    };
    std::vector<int64_t> s_wq, s_bq, s_wk, s_bk;
    wq = tensor("q.weight", s_wq); bq = tensor("q.bias", s_bq);
    wk = tensor("k.weight", s_wk); bk = tensor("k.bias", s_bk);
    if (s_wq.size() != 2 || s_wq != s_wk || s_wq[1] != n_embd || s_bq.size() != 1 || s_bq[0] != s_wq[0] || s_bk != s_bq) {
        throw std::runtime_error(string_format("classifier head shapes do not match the backbone (n_embd %d): q.weight [%lld, %lld]",
                                               n_embd, (long long) s_wq[0], (long long) (s_wq.size() > 1 ? s_wq[1] : 0)));
    }
    if (head_dim == 0) { head_dim = (int) s_wq[0]; }
}

// ---------------------------------------------------------------------------------------------------------------
// graph capture: the post-norm hidden state of the output rows

static bool classifier_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_norm = std::strcmp(ggml_get_name(t), "result_norm") == 0;
    if (ask) { return is_norm; }
    if (!is_norm) { return true; }
    auto * self = static_cast<server_classifier *>(user_data);
    if (t->type != GGML_TYPE_F32) { return true; }
    const int64_t rows = ggml_nrows(t);
    const size_t off = self->captured.size();
    self->captured.resize(off + (size_t) (rows * t->ne[0]));
    ggml_backend_tensor_get(t, self->captured.data() + off, 0, ggml_nbytes(t));
    self->captured_rows += rows;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// answer cache: one append-only file per identity, lines of "<sha256 hex>\t<answer JSON>", all of it held in memory

static const uintmax_t CACHE_MAX_BYTES = 64u << 20;   // over this at startup, the older half is dropped

static std::string file_stamp(const std::string & path) {
    std::error_code e1, e2;
    const auto size = std::filesystem::file_size(path, e1);
    const auto time = std::filesystem::last_write_time(path, e2);
    return path + "|" + (e1 ? "?" : std::to_string(size)) + "|" + (e2 ? "?" : std::to_string((long long) time.time_since_epoch().count()));
}

void server_classifier::cache_open(const common_params & params) {
    if (params.classifier_cache.empty()) { return; }
    const std::string ident = "v1\n" + file_stamp(params.model.path) + "\n" + file_stamp(params.classifier_head) + "\n" +
                              file_stamp(params.classifier_config);
    const std::string ident_hash = hash_sha256_hex(ident.data(), ident.size());
    std::string name = cfg.name;
    for (auto & c : name) { if (!std::isalnum((unsigned char) c) && c != '.' && c != '-' && c != '_') { c = '_'; } }

    std::error_code ec;
    const std::filesystem::path dir(params.classifier_cache);
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path file = dir / (name + "-" + ident_hash.substr(0, 16) + ".tsv");

    std::vector<std::string> lines;
    uintmax_t bytes = 0;
    {
        std::ifstream in(file, std::ios::binary);
        for (std::string line; std::getline(in, line);) {
            if (line.size() < 66 || line[64] != '\t') { continue; }   // e.g. a line cut short by a kill
            bytes += line.size() + 1;
            lines.push_back(std::move(line));
        }
    }
    if (bytes > CACHE_MAX_BYTES) {
        lines.erase(lines.begin(), lines.begin() + (std::ptrdiff_t) (lines.size() / 2));
        const std::filesystem::path tmp = file.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            for (const auto & l : lines) { out << l << '\n'; }
        }
        std::filesystem::rename(tmp, file, ec);
    }
    for (const auto & l : lines) { cache[l.substr(0, 64)] = l.substr(65); }

    cache_file.open(file, std::ios::binary | std::ios::app);
    if (!cache_file) {
        LOG_WRN("classifier: cannot write %s, answer cache disabled\n", file.string().c_str());
        cache.clear();
        return;
    }
    cache_ident = ident_hash;
    CLS_INF("answer cache: %zu answers in %s\n", cache.size(), file.string().c_str());
}

// the key covers exactly what the answer depends on: the files (identity), the tokens the model sees, and the
// answer keys / option texts that are echoed into the answer
std::string server_classifier::cache_key(const std::vector<llama_token> & prefix, const encoded_question & e) const {
    std::string h = cache_ident + "\n" + e.type + "\n" + std::to_string(prefix.size()) + " " + std::to_string(e.row.size()) + "\n";
    for (const auto & k : e.keys) { h += k; h += '\x1f'; }
    h += '\n';
    for (const auto & o : e.opts) { h += o; h += '\x1f'; }
    h += '\n';
    h += e.text;   // letter family: the state and instructions as the prompt shows them
    h.append(reinterpret_cast<const char *>(prefix.data()), prefix.size() * sizeof(llama_token));
    h.append(reinterpret_cast<const char *>(e.row.data()), e.row.size() * sizeof(llama_token));
    return hash_sha256_hex(h.data(), h.size());
}

bool server_classifier::cache_get(const std::string & key, json & answer) {
    std::lock_guard<std::mutex> lock(cache_mtx);
    const auto it = cache.find(key);
    if (it == cache.end()) { return false; }
    answer = json::parse_no_throw(it->second);
    return !answer.is_discarded();
}

void server_classifier::cache_put(const std::string & key, const json & answer) {
    const std::string text = answer.dump();
    std::lock_guard<std::mutex> lock(cache_mtx);
    cache[key] = text;
    cache_file << key << '\t' << text << '\n';
    cache_file.flush();
}

// ---------------------------------------------------------------------------------------------------------------

bool server_classifier::init(llama_model * m, const common_params & params, std::string & err) {
    try {
        model = m;
        load_config(params.classifier_config, cfg);
        n_embd = llama_model_n_embd(model);
        const bool letter = cfg.family == "letter";
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (letter) {
            if (!params.classifier_head.empty()) { throw std::runtime_error("the letter family has no head: drop --classifier-head"); }
            for (char L = 'A'; L <= 'Z'; L++) {
                const std::string name(1, L);
                std::vector<llama_token> ids;
                for (const auto & v : cfg.lt_variants) {
                    const auto t = common_tokenize(vocab, expand(v, { { "letter", &name } }), false, false);
                    if (t.size() == 1 && std::find(ids.begin(), ids.end(), t[0]) == ids.end()) { ids.push_back(t[0]); }
                }
                if (ids.empty()) { throw std::runtime_error("letter '" + name + "' is not a single token of this model's vocabulary"); }
                letter_ids.push_back(ids);
            }
        } else {
            load_head(params.classifier_head, wq, bq, wk, bk, cfg.head_dim, n_embd);
            auto special = [&](const std::string & s) {
                auto t = common_tokenize(vocab, s, false, true);
                if (t.size() != 1) { throw std::runtime_error("delimiter '" + s + "' is not a single token of this model's vocabulary"); }
                return t[0];
            };
            id_state = special(cfg.tok_state); id_question = special(cfg.tok_question); id_option = special(cfg.tok_option);
            id_close = special(cfg.tok_close); id_decide = special(cfg.tok_decide);
        }

        llama_context_params cp = common_context_params_to_llama(params);
        n_ctx = params.n_ctx > 0 ? params.n_ctx : 4096;   // model default (32k) would be a needlessly large KV cache
        cp.n_ctx      = n_ctx;
        cp.n_batch    = n_ctx;
        cp.n_ubatch   = std::min<uint32_t>(cp.n_ubatch, (uint32_t) n_ctx);
        cp.n_seq_max  = 1;
        cp.n_outputs_max         = 0;   // one output per option + <decide>; the server's generation caps do not apply
        cp.n_outputs_max_per_seq = 0;
        cp.embeddings = false;   // outputs only for the rows the head reads (see header)
        if (!letter) {   // the letter family reads ordinary logits
            cp.cb_eval           = classifier_eval_cb;
            cp.cb_eval_user_data = this;
        }
        lctx = llama_init_from_model(model, cp);
        if (!lctx) { throw std::runtime_error("failed to create the classifier context"); }
        if (letter) {
            CLS_INF("%s ready: family letter, %zu temperature entries, n_ctx %d\n", cfg.name.c_str(), cfg.lt_temperature.size(), n_ctx);
        } else {
            CLS_INF("%s ready: family %s, head_dim %d, temperature %.4g, n_ctx %d, n_embd %d\n", cfg.name.c_str(), cfg.family.c_str(),
                    cfg.head_dim, cfg.temperature, n_ctx, n_embd);
        }
        cache_open(params);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

server_classifier::~server_classifier() {
    if (lctx) { llama_free(lctx); }
}

std::string server_classifier::escape_user(const std::string & text) const {
    if (!cfg.escape_special) { return text; }
    static const std::regex re("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(text, re, "<\xC2\xA6$1\xC2\xA6>");   // "<|x|>" -> "<¦x¦>"
}

std::vector<llama_token> server_classifier::user_tokens(const std::string & text) const {
    return common_tokenize(llama_model_get_vocab(model), escape_user(text), false, false);
}

// decode `toks` as positions start.. of sequence 0 and return the captured hidden rows for the sorted positions `want`
std::vector<float> server_classifier::decode_chunk(const std::vector<llama_token> & toks, int start, const std::vector<int> & want) {
    common_batch batch(lctx);
    for (size_t i = 0; i < toks.size(); i++) {
        const int pos = start + (int) i;
        batch.add(toks[i], pos, 0, std::binary_search(want.begin(), want.end(), pos));
    }
    captured.clear();
    captured_rows = 0;
    const int rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) { throw std::runtime_error(string_format("llama_process failed (%d)", rc)); }
    if (captured_rows != (int64_t) want.size()) {
        throw std::runtime_error(string_format("captured %lld hidden rows, expected %zu", (long long) captured_rows, want.size()));
    }
    return captured;
}

json server_classifier::classify(const json & body) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!body.is_object() || !body.contains("state") || !body.contains("questions")) {
        throw std::invalid_argument("request needs {\"state\", \"questions\"}");
    }
    const json & qs = body.at("questions");
    if (!qs.is_object() || qs.size() == 0) { throw std::invalid_argument("questions must be a non-empty object"); }
    if (cfg.family == "letter") {
        json res = classify_letter(body);
        res["latency_ms"] = std::round(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() * 100.0) / 100.0;
        return res;
    }

    // --- encode (pointer family) ---
    std::vector<llama_token> prefix = { id_state };
    { auto st = user_tokens(render(body.at("state"))); prefix.insert(prefix.end(), st.begin(), st.end()); }
    if ((int) prefix.size() > cfg.max_state) { throw std::invalid_argument(string_format("state exceeds %d tokens", cfg.max_state)); }

    auto option_text = [&](const std::string & name, const json & desc) {
        if (desc.is_null() || (desc.is_string() && desc.get<std::string>().empty())) { return name; }
        return replace_all(replace_all(cfg.option_format, "{name}", name), "{desc}", render(desc));
    };
    std::vector<encoded_question> enc;
    for (const auto & [qid, q] : qs.items()) {
        if (!q.is_object()) { throw std::invalid_argument("question '" + qid + "' must be an object"); }
        std::string type = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";
        if (type == "noul") { type = "boolean"; }
        encoded_question e;
        e.id = qid; e.type = type;
        const json crit = q.contains("criteria") ? q.at("criteria") : json();
        if (type == "boolean") {
            if (!crit.is_null() && !crit.is_object()) { throw std::invalid_argument(qid + ": boolean criteria may only hold false/true"); }
            e.keys = { "false", "true" };
            e.opts = { option_text(cfg.label_false, crit.is_object() && crit.contains("false") ? crit.at("false") : json()),
                       option_text(cfg.label_true,  crit.is_object() && crit.contains("true")  ? crit.at("true")  : json()) };
        } else if (type == "choice") {
            if (!crit.is_object() || crit.size() < 1 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: choice criteria must hold 1..%d options", qid.c_str(), cfg.max_options));
            }
            for (const auto & [k, v] : crit.items()) { e.keys.push_back(k); e.opts.push_back(option_text(k, v)); }
        } else if (type == "score") {
            if (!crit.is_array() || crit.size() < 2 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: score criteria must be a list of 2..%d levels", qid.c_str(), cfg.max_options));
            }
            for (size_t i = 0; i < crit.size(); i++) { e.keys.push_back(std::to_string(i)); e.opts.push_back(render(crit.at(i))); }
        } else {
            throw std::invalid_argument(qid + ": unsupported question type '" + type + "' (choice, noul, score)");
        }
        e.row.push_back(id_question);
        { auto it = user_tokens(render(q.contains("instructions") ? q.at("instructions") : json())); e.row.insert(e.row.end(), it.begin(), it.end()); }
        for (const auto & o : e.opts) {
            e.row.push_back(id_option);
            auto ot = user_tokens(o);
            e.row.insert(e.row.end(), ot.begin(), ot.end());
            e.row.push_back(id_close);
            e.close_pos.push_back((int) (prefix.size() + e.row.size() - 1));
        }
        e.row.push_back(id_decide);
        e.decide = (int) (prefix.size() + e.row.size() - 1);
        if ((int) e.row.size() > cfg.max_branch - (int) prefix.size()) {
            throw std::invalid_argument(string_format("%s: question row too long (%zu tokens)", qid.c_str(), e.row.size()));
        }
        if ((int) (prefix.size() + e.row.size()) > n_ctx) {
            throw std::invalid_argument(string_format("%s: state + question is %zu tokens, over the classifier context of %d",
                                                      qid.c_str(), prefix.size() + e.row.size(), n_ctx));
        }
        enc.push_back(std::move(e));
    }

    // --- answer cache: only the questions it does not hold reach the model ---
    size_t n_miss = enc.size();
    if (!cache_ident.empty()) {
        for (auto & e : enc) {
            e.cache_key = cache_key(prefix, e);
            if (cache_get(e.cache_key, e.answer)) { n_miss--; } else { e.answer = json(); }
        }
    }

    // --- backbone: one pass for a single question; otherwise the state prefix once, then each row continues from it ---
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    llama_memory_t mem = nullptr;
    const bool shared = n_miss > 1;
    if (n_miss > 0) {
        lock.lock();
        mem = llama_get_memory(lctx);
        llama_memory_clear(mem, true);
        if (shared) {
            decode_chunk(prefix, 0, { (int) prefix.size() - 1 });   // one output keeps llama_decode on its ordinary path
        }
    }
    const float scale = 1.0f / std::sqrt((float) cfg.head_dim) / (float) cfg.temperature;
    for (auto & e : enc) {
        if (!e.answer.is_null()) { continue; }   // cached
        std::vector<int> want = e.close_pos;
        want.push_back(e.decide);
        std::sort(want.begin(), want.end());
        std::vector<float> h;
        if (shared) {
            llama_memory_seq_rm(mem, 0, (llama_pos) prefix.size(), -1);
            h = decode_chunk(e.row, (int) prefix.size(), want);
        } else {
            std::vector<llama_token> all = prefix;
            all.insert(all.end(), e.row.begin(), e.row.end());
            h = decode_chunk(all, 0, want);
        }
        auto row_of = [&](int pos) { return h.data() + (size_t) (std::lower_bound(want.begin(), want.end(), pos) - want.begin()) * n_embd; };

        // --- head: query from <decide>, one key per <close> ---
        const int dp = cfg.head_dim;
        std::vector<float> qv(dp);
        const float * hd = row_of(e.decide);
        for (int d = 0; d < dp; d++) {
            double acc = bq[d];
            const float * w = wq.data() + (size_t) d * n_embd;
            for (int k = 0; k < n_embd; k++) { acc += (double) w[k] * hd[k]; }
            qv[d] = (float) acc;
        }
        std::vector<double> z(e.close_pos.size());
        for (size_t i = 0; i < e.close_pos.size(); i++) {
            const float * hc = row_of(e.close_pos[i]);
            double dot = 0;
            for (int d = 0; d < dp; d++) {
                double acc = bk[d];
                const float * w = wk.data() + (size_t) d * n_embd;
                for (int k = 0; k < n_embd; k++) { acc += (double) w[k] * hc[k]; }
                dot += acc * qv[d];
            }
            z[i] = dot * scale;
        }
        const double zmax = *std::max_element(z.begin(), z.end());
        double sum = 0;
        for (auto & v : z) { v = std::exp(v - zmax); sum += v; }
        std::vector<double> p(z.size());
        for (size_t i = 0; i < z.size(); i++) { p[i] = z[i] / sum; }

        const json a = make_answer(e, p);
        if (!e.cache_key.empty()) { cache_put(e.cache_key, a); }
        e.answer = a;
    }
    if (lock.owns_lock()) { lock.unlock(); }

    json answers = json::object();
    for (auto & e : enc) { answers[e.id] = e.answer; }

    json res = json::object();
    res["model"] = body.contains("model") && body.at("model").is_string() ? body.at("model").get<std::string>() : cfg.name;
    res["answers"] = answers;
    if (!cache_ident.empty()) { res["cached"] = n_miss == 0; }
    res["latency_ms"] = std::round(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() * 100.0) / 100.0;
    return res;
}

// System One answer of one question from its probabilities (in the order of e.keys)
json server_classifier::make_answer(const encoded_question & e, const std::vector<double> & p) const {
    auto r2 = [](double x) { return std::round(x * 100.0) / 100.0; };
    auto r4 = [](double x) { return std::round(x * 10000.0) / 10000.0; };
    json probs = json::object();
    for (size_t i = 0; i < p.size(); i++) { probs[e.keys[i]] = p[i]; }
    json a = json::object();
    if (e.type == "boolean") {
        const size_t yes = (size_t) (std::find(e.keys.begin(), e.keys.end(), "true") - e.keys.begin());
        a["type"] = "noul";
        a["noul"] = r4(p[yes]);
    } else if (e.type == "choice") {
        const size_t best = (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
        const double K = (double) p.size();
        a["type"] = "choice";
        a["choice"] = e.keys[best];
        a["confidence"] = p.size() == 1 ? 1.0 : r2((p[best] - 1.0 / K) / (1.0 - 1.0 / K));
        a["probabilities"] = probs;
    } else {
        const size_t mode = (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
        double score = 0, spread = 0;
        for (size_t i = 0; i < p.size(); i++) { score += (double) i * p[i]; spread += p[i] * std::fabs((double) i - (double) mode); }
        json legend = json::object();
        for (size_t i = 0; i < e.opts.size(); i++) { legend[e.keys[i]] = e.opts[i]; }
        a["type"] = "score";
        a["score"] = r4(score);
        a["legend"] = legend;
        a["probabilities"] = probs;
        a["confidence"] = r2(1.0 - spread / (double) (p.size() - 1));
    }
    return a;
}

// ---------------------------------------------------------------------------------------------------------------
// letter family

static const int LETTERS = 26;

// a string as Go's encoding/json writes it: <, >, & and U+2028/2029 escaped, non-ASCII kept
static std::string go_str(const std::string & s) {
    std::string out = "\"";
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char ch = (unsigned char) s[i];
        if (ch == '"')       { out += "\\\""; }
        else if (ch == '\\') { out += "\\\\"; }
        else if (ch == '\n') { out += "\\n"; }
        else if (ch == '\r') { out += "\\r"; }
        else if (ch == '\t') { out += "\\t"; }
        else if (ch < 0x20 || ch == '<' || ch == '>' || ch == '&') { out += string_format("\\u%04x", ch); }
        else if (ch == 0xE2 && i + 2 < s.size() && (unsigned char) s[i + 1] == 0x80 && ((unsigned char) s[i + 2] == 0xA8 || (unsigned char) s[i + 2] == 0xA9)) {
            out += (unsigned char) s[i + 2] == 0xA8 ? "\\u2028" : "\\u2029";
            i += 2;
        } else { out += (char) ch; }
    }
    return out + "\"";
}

// text of a state / instructions / description: a string as it is, anything else as the compact JSON a client sends
// (ASCII-escaped, as Python clients encode it and as the model was trained)
static std::string go_content(const json & v) {
    if (v.is_string()) { return v.get<std::string>(); }
    if (v.is_null())   { return ""; }
    const std::string s = v.dump();
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (n == 1 || i + (size_t) n > s.size()) { out += (char) c; i++; continue; }
        uint32_t cp = c & (0xFF >> (n + 1));
        for (int k = 1; k < n; k++) { cp = (cp << 6) | ((unsigned char) s[i + (size_t) k] & 0x3F); }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out += string_format("\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
        } else {
            out += string_format("\\u%04x", cp);
        }
        i += (size_t) n;
    }
    return out;
}

// the prompts of Ollama's decision API (ollama/decision/systemone.go, mirrored from tinyjev training/v2/ollama_format.py):
// every question sees the whole schema, and is asked for by name
std::vector<server_classifier::encoded_question> server_classifier::encode_ollama(const json & body) const {
    std::vector<encoded_question> enc;
    std::string schema;
    for (const auto & [qid, q] : body.at("questions").items()) {
        if (!q.is_object()) { throw std::invalid_argument("question '" + qid + "' must be an object"); }
        std::string type = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";
        encoded_question e;
        e.id = qid;
        e.type = type == "noul" ? "boolean" : type;
        const json crit = q.contains("criteria") ? q.at("criteria") : json();
        std::vector<std::string> values;   // as JSON
        if (type == "noul" || type == "boolean") {
            e.type = "boolean";
            e.keys = { "false", "true" };
            values = { "false", "true" };
            e.opts = { crit.is_object() && crit.contains("false") ? go_content(crit.at("false")) : "No",
                       crit.is_object() && crit.contains("true")  ? go_content(crit.at("true"))  : "Yes" };
        } else if (type == "choice" && crit.is_object()) {
            for (const auto & [k, v] : crit.items()) { e.keys.push_back(k); values.push_back(go_str(k)); e.opts.push_back(v.is_null() ? k : go_content(v)); }
        } else if (type == "score" && crit.is_array()) {
            for (size_t i = 0; i < crit.size(); i++) { e.keys.push_back(std::to_string(i)); values.push_back(go_str(e.keys.back())); e.opts.push_back(go_content(crit.at(i))); }
        } else {
            throw std::invalid_argument(qid + ": unsupported question type or criteria (choice, noul, score)");
        }
        if (e.opts.size() < 2 || (int) e.opts.size() > LETTERS) { throw std::invalid_argument(string_format("%s: 2..%d options", qid.c_str(), LETTERS)); }
        if (!schema.empty()) { schema += ","; }
        schema += "{\"name\":" + go_str(qid) + ",\"description\":" + go_str(go_content(q.contains("instructions") ? q.at("instructions") : json())) + ",\"choices\":[";
        for (size_t i = 0; i < e.opts.size(); i++) {
            schema += std::string(i ? "," : "") + "{\"code\":\"" + (char) ('A' + i) + "\",\"value\":" + values[i] + ",\"description\":" + go_str(e.opts[i]) + "}";
        }
        schema += "]}";
        enc.push_back(std::move(e));
    }
    const std::string data = "{\"context\":" + go_str(go_content(body.at("state"))) + ",\"schema\":[" + schema + "]}";
    for (auto & e : enc) {
        const std::string user = data + "\n\nRequested field: " + go_str(e.id);
        e.text = expand(cfg.lt_prompt, { { "user", &user } });   // the whole prompt, which is also the cache identity
    }
    return enc;
}

double server_classifier::letter_temperature(const std::string & type, size_t n) const {
    if (cfg.lt_temperature.empty()) { return cfg.temperature; }
    const std::string t = type == "boolean" ? "noul" : type;
    const char * bucket = n <= 2 ? "2" : n <= 4 ? "3-4" : n <= 8 ? "5-8" : "9+";
    for (const std::string & k : { t + "|" + bucket, t, std::string("default") }) {
        const auto it = cfg.lt_temperature.find(k);
        if (it != cfg.lt_temperature.end()) { return it->second; }
    }
    return cfg.temperature;
}

// the prompts of one request start alike (the state, with style ollama also the whole schema): below this many
// common tokens a shared prefix is not worth its snapshot
static const size_t SHARE_MIN_TOKENS = 16;

// -> for each prompt, the softmax of the letter logits over its first n_opts letters.
// Several prompts share their common token prefix: it is decoded once, the sequence state is saved, and every prompt
// continues from a restored copy. A state snapshot rather than removing the tail from the memory, because the
// recurrent layers of hybrid models (Qwen3.5) cannot be rolled back to a position.
std::vector<std::vector<double>> server_classifier::letter_reads(const std::vector<std::string> & prompts, const std::vector<int> & n_opts) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<std::vector<llama_token>> toks;
    for (const auto & p : prompts) {
        toks.push_back(common_tokenize(vocab, p, false, true));   // template special tokens parse; user text is escaped
        if ((int) toks.back().size() > n_ctx) {
            throw std::invalid_argument(string_format("state + question is %zu tokens, over the classifier context of %d", toks.back().size(), n_ctx));
        }
    }
    size_t common = 0;
    if (toks.size() > 1 && cfg.lt_share_prefix) {
        common = toks[0].size() - 1;   // every prompt keeps at least its last token for itself
        for (size_t r = 1; r < toks.size(); r++) {
            size_t n = 0;
            while (n < common && n + 1 < toks[r].size() && toks[r][n] == toks[0][n]) { n++; }
            common = n;
        }
        if (common < SHARE_MIN_TOKENS) { common = 0; }
    }

    std::vector<std::vector<double>> out;
    std::lock_guard<std::mutex> lock(mtx);
    // positions [from, size) of prompt r, with an output row on the last one
    auto decode = [&](size_t r, size_t from, size_t to) {
        common_batch batch(lctx);
        for (size_t i = from; i < to; i++) { batch.add(toks[r][i], (llama_pos) i, 0, i + 1 == to); }
        const int rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) { throw std::runtime_error(string_format("llama_process failed (%d)", rc)); }
    };
    std::vector<uint8_t> prefix_state;
    if (common > 0) {
        llama_memory_clear(llama_get_memory(lctx), true);
        decode(0, 0, common);
        prefix_state.resize(llama_state_seq_get_size(lctx, 0));
        if (prefix_state.empty() || llama_state_seq_get_data(lctx, prefix_state.data(), prefix_state.size(), 0) == 0) {
            common = 0;   // no snapshot on this memory type: every prompt is decoded in full
        }
    }
    for (size_t r = 0; r < toks.size(); r++) {
        if (common == 0) {
            llama_memory_clear(llama_get_memory(lctx), true);
        } else if (r > 0 && llama_state_seq_set_data(lctx, prefix_state.data(), prefix_state.size(), 0) == 0) {
            throw std::runtime_error("failed to restore the shared prompt prefix");
        }
        decode(r, common, toks[r].size());
        const float * logits = llama_get_logits_ith(lctx, -1);
        if (!logits) { throw std::runtime_error("no logits at the answer position"); }
        std::vector<double> z((size_t) n_opts[r]);
        for (int L = 0; L < n_opts[r]; L++) {   // a letter's spellings add up: log sum exp
            double m = -INFINITY;
            for (auto id : letter_ids[(size_t) L]) { m = std::max(m, (double) logits[id]); }
            double acc = 0;
            for (auto id : letter_ids[(size_t) L]) { acc += std::exp((double) logits[id] - m); }
            z[(size_t) L] = m + std::log(acc);
        }
        const double zmax = *std::max_element(z.begin(), z.end());
        double sum = 0;
        for (auto & v : z) { v = std::exp(v - zmax); sum += v; }
        for (auto & v : z) { v /= sum; }
        out.push_back(std::move(z));
    }
    return out;
}

json server_classifier::classify_letter(const json & body) {
    const json & qs = body.at("questions");

    // --- the texts the prompt shows ---
    std::string state;
    {
        const json & st = body.at("state");
        const bool none = st.is_null() || ((st.is_object() || st.is_array()) && st.empty());
        state = st.is_string() ? st.get<std::string>() : none ? "" : cfg.lt_state_format == "json" ? render_json(st, cfg.lt_json_index_min) : render(st);
        state = escape_user(state);
    }
    const bool blank = strip(state).empty();

    auto described = [&](const std::string & name, const json & desc, const std::string & format) {
        std::string d = render(desc);
        if (cfg.lt_strip) { d = strip(d); }
        if (d.empty() || (cfg.lt_same_as_name && norm_label(d) == norm_label(name))) { return name; }
        return expand(format, { { "name", &name }, { "desc", &d } });
    };

    const bool ollama = cfg.lt_style == "ollama";
    std::vector<encoded_question> enc = ollama ? encode_ollama(body) : std::vector<encoded_question>();
    for (const auto & [qid, q] : qs.items()) {
        if (ollama) { break; }
        if (!q.is_object()) { throw std::invalid_argument("question '" + qid + "' must be an object"); }
        std::string type = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";
        if (type == "noul") { type = "boolean"; }
        encoded_question e;
        e.id = qid; e.type = type;
        const json crit = q.contains("criteria") ? q.at("criteria") : json();
        std::string instr = render(q.contains("instructions") ? q.at("instructions") : json());
        if (cfg.lt_strip) { instr = strip(instr); }
        if (type == "boolean") {
            if (!crit.is_null() && !crit.is_object()) { throw std::invalid_argument(qid + ": boolean criteria may only hold false/true"); }
            const std::string no  = described(cfg.label_false, crit.is_object() && crit.contains("false") ? crit.at("false") : json(), cfg.option_format);
            const std::string yes = described(cfg.label_true,  crit.is_object() && crit.contains("true")  ? crit.at("true")  : json(), cfg.option_format);
            if (cfg.lt_true_first) { e.keys = { "true", "false" }; e.opts = { yes, no }; } else { e.keys = { "false", "true" }; e.opts = { no, yes }; }
            if (instr.empty()) { instr = cfg.lt_noul_instructions; }
        } else if (type == "choice") {
            if (!crit.is_object() || crit.size() < 1 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: choice criteria must hold 1..%d options", qid.c_str(), cfg.max_options));
            }
            for (const auto & [k, v] : crit.items()) { e.keys.push_back(k); e.opts.push_back(described(k, v, cfg.option_format)); }
            if (cfg.lt_bare_keys == "bare") {
                static const std::regex bare("^(?:\\(?[A-Za-z][\\).]?|\\(?\\d{1,3}[\\).]?|option[_ ]?\\d{1,3}|opt[_ ]?\\d{1,3})$", std::regex::icase);
                bool hide = true;
                std::vector<std::string> alone;
                for (const auto & [k, v] : crit.items()) {
                    const std::string d = strip(render(v));
                    hide = hide && std::regex_match(k, bare) && !d.empty();
                    alone.push_back(d);
                }
                if (hide) { e.opts = alone; }
            } else if (cfg.lt_bare_keys == "positional") {
                // as wald_serve.prompt: the shown keys are the option texts when those are all distinct slugs, else a, b, ..
                static const std::regex slug("[a-z0-9][a-z0-9_.-]{0,39}"), positional("[a-z]|o[0-9]+");
                bool slugs = true;
                for (size_t i = 0; i < e.opts.size(); i++) {
                    const auto end = e.opts.begin() + (std::ptrdiff_t) i;
                    slugs = slugs && std::regex_match(e.opts[i], slug) && std::find(e.opts.begin(), end, e.opts[i]) == end;
                }
                std::vector<std::string> shown;
                for (size_t i = 0; i < e.opts.size(); i++) {
                    shown.push_back(slugs ? e.opts[i] : e.opts.size() <= 26 ? std::string(1, (char) ('a' + i)) : "o" + std::to_string(i + 1));
                }
                if (std::all_of(shown.begin(), shown.end(), [&](const std::string & k) { return std::regex_match(k, positional); })) {
                    for (size_t i = 0; i < e.opts.size(); i++) {
                        const std::string lead = shown[i] + ": ";
                        if (e.opts[i].compare(0, lead.size(), lead) == 0) { e.opts[i] = e.opts[i].substr(lead.size()); }
                    }
                }
            }
        } else if (type == "score") {
            if (!crit.is_array() || crit.size() < 2 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: score criteria must be a list of 2..%d levels", qid.c_str(), cfg.max_options));
            }
            for (size_t i = 0; i < crit.size(); i++) {
                e.keys.push_back(std::to_string(i));
                e.opts.push_back(described(e.keys.back(), crit.at(i), cfg.lt_score_format));
            }
        } else {
            throw std::invalid_argument(qid + ": unsupported question type '" + type + "' (choice, noul, score)");
        }
        if ((int) e.opts.size() > LETTERS * LETTERS) { throw std::invalid_argument(string_format("%s: at most %d options", qid.c_str(), LETTERS * LETTERS)); }
        for (auto & o : e.opts) { o = escape_user(o); }
        e.text = state + '\x1f' + escape_user(instr);
        enc.push_back(std::move(e));
    }

    auto prompt_of = [&](const encoded_question & e, const std::vector<int> & idx) {
        if (ollama) { return e.text; }
        const std::string instr = e.text.substr(state.size() + 1);
        std::string options;
        for (size_t j = 0; j < idx.size(); j++) {
            const std::string letter(1, (char) ('A' + j));
            options += expand(cfg.lt_option_line, { { "letter", &letter }, { "text", &e.opts[(size_t) idx[j]] } });
        }
        const bool alt = blank && !cfg.lt_prompt_no_state.empty();
        const std::string & shown = blank ? cfg.lt_empty_state : state;
        return expand(alt ? cfg.lt_prompt_no_state : cfg.lt_prompt, { { "state", &shown }, { "instructions", &instr }, { "options", &options } });
    };
    // more than 26 options: near-equal consecutive chunks, whose winners meet in a final read
    auto chunks_of = [](int n) {
        const int k = (n + LETTERS - 1) / LETTERS, base = n / k, extra = n % k;
        std::vector<std::vector<int>> out;
        for (int c = 0, i = 0; c < k; c++) {
            std::vector<int> g;
            for (int m = base + (c < extra ? 1 : 0); m > 0; m--) { g.push_back(i++); }
            out.push_back(g);
        }
        return out;
    };

    // --- answer cache ---
    size_t n_miss = enc.size();
    if (!cache_ident.empty()) {
        for (auto & e : enc) {
            e.cache_key = cache_key({}, e);
            if (cache_get(e.cache_key, e.answer)) { n_miss--; } else { e.answer = json(); }
        }
    }

    // --- reads ---
    struct read { size_t q; std::vector<int> idx; };
    std::vector<read> reads;
    for (size_t qi = 0; qi < enc.size(); qi++) {
        if (!enc[qi].answer.is_null()) { continue; }
        for (auto & g : chunks_of((int) enc[qi].opts.size())) { reads.push_back({ qi, std::move(g) }); }
    }
    auto run = [&](const std::vector<read> & rs) {
        std::vector<std::string> prompts;
        std::vector<int> n;
        for (const auto & r : rs) { prompts.push_back(prompt_of(enc[r.q], r.idx)); n.push_back((int) r.idx.size()); }
        return letter_reads(prompts, n);
    };
    const auto first = run(reads);

    std::vector<read> finals;
    for (size_t qi = 0; qi < enc.size(); qi++) {
        if (!enc[qi].answer.is_null() || (int) enc[qi].opts.size() <= LETTERS) { continue; }
        read f{ qi, {} };
        for (size_t r = 0; r < reads.size(); r++) {
            if (reads[r].q == qi) { f.idx.push_back(reads[r].idx[(size_t) (std::max_element(first[r].begin(), first[r].end()) - first[r].begin())]); }
        }
        finals.push_back(std::move(f));
    }
    const auto last = finals.empty() ? std::vector<std::vector<double>>() : run(finals);

    for (size_t qi = 0; qi < enc.size(); qi++) {
        auto & e = enc[qi];
        if (!e.answer.is_null()) { continue; }
        std::vector<double> p(e.opts.size(), 0.0);
        const std::vector<double> * top = nullptr;
        for (size_t f = 0; f < finals.size(); f++) { if (finals[f].q == qi) { top = &last[f]; } }
        for (size_t r = 0, c = 0; r < reads.size(); r++) {
            if (reads[r].q != qi) { continue; }
            for (size_t j = 0; j < reads[r].idx.size(); j++) { p[(size_t) reads[r].idx[j]] = first[r][j] * (top ? (*top)[c] : 1.0); }
            c++;
        }
        // temperature on the finished distribution: softmax(log p / T), which leaves the order as it is
        const double T = letter_temperature(e.type, p.size());
        double zmax = -INFINITY, sum = 0;
        for (auto & v : p) { v = std::log(std::max(v, 1e-12)) / T; zmax = std::max(zmax, v); }
        for (auto & v : p) { v = std::exp(v - zmax); sum += v; }
        for (auto & v : p) { v /= sum; }

        e.answer = make_answer(e, p);
        if (!e.cache_key.empty()) { cache_put(e.cache_key, e.answer); }
    }

    json answers = json::object();
    for (auto & e : enc) { answers[e.id] = e.answer; }
    json res = json::object();
    res["model"] = body.contains("model") && body.at("model").is_string() ? body.at("model").get<std::string>() : cfg.name;
    res["answers"] = answers;
    if (!cache_ident.empty()) { res["cached"] = n_miss == 0; }
    return res;
}
