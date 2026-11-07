# Jev-style decision classifier in llama-server

llama-server can answer typed questions about a text with calibrated probabilities instead of generated text:
the TypeSafe *System One* API made popular by Jev, served by an open decision model such as
[StartLux-Decision](https://huggingface.co/startlux-models), Wald or [TinyJev](https://github.com/ankit-aglawe/tinyjev).

```
POST /v1/classifier            (alias: /v1/systemone)
{"state": "Arr, me parrot ate the treasure map again.",
 "questions": {
   "author": {"type": "choice", "instructions": "Who most likely wrote this?",
              "criteria": {"pirate": "A pirate", "grandma": "A grandmother", "robot": "A robot"}},
   "sea":    {"type": "noul",   "instructions": "Does the text mention the sea, ships or water?"},
   "chaos":  {"type": "score",  "instructions": "How chaotic is this?", "criteria": ["calm", "mild", "wild"]}}}

-> {"model": "tinyjev-4b", "latency_ms": 974,
    "answers": {"author": {"type": "choice", "choice": "pirate", "confidence": 0.6,
                           "probabilities": {"pirate": 0.735, "grandma": 0.118, "robot": 0.147}},
                "sea":    {"type": "noul", "noul": 0.0374},
                "chaos":  {"type": "score", "score": 1.9, "legend": {"0": "calm", "1": "mild", "2": "wild"},
                           "probabilities": {...}, "confidence": 0.95}}}
```

The answer shown is TinyJev-4B v1's; StartLux-Decision and the other models answer in the same shape.

The routes sit behind the normal API key check. State, instructions and option descriptions may be strings, numbers,
lists or objects. The classifier is the model loaded with `-m`, on any backend; for pure CPU use `-dev none -ngl 0`
(without `-dev none` llama.cpp may still offload large matmuls to a GPU). Two model families are served, chosen by
`"family"` in `--classifier-config FILE`.

## Letter family (no head)

Most current decision models (StartLux-Decision, Wald, decider, TinyJev v2): the options are lettered in a text
prompt and the answer is the model's next-token logits of those letters.

```
llama-server -m StartLux-Decision-v1.1-2B-Instant-BF16.gguf --classifier-config classifier.json -dev none -ngl 0 -c 4096 -b 4096 -ub 4096
```

```json
{"name": "startlux-decision-v1.1-2b", "family": "letter",
 "letter": {
   "prompt": "<|im_start|>system\nApply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.<|im_end|>\n<|im_start|>user\nEvidence:\n{state}\n\nQuestion: {instructions}\nOptions:{options}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
   "empty_state": "(none)", "option_line": "\n{letter}) {text}", "variants": ["{letter}"],
   "state_format": "json", "json_index_min": 8, "score_format": "{name}: {desc}", "true_first": true,
   "noul_instructions": "Which answer fits the evidence?", "bare_keys": "bare", "same_as_name": true, "strip": true,
   "share_prefix": false,
   "temperature": {"noul": 1.2, "choice": 1.2937, "score": 1.2902}}}
```

| Key | Meaning |
|---|---|
| `prompt` | the whole prompt of one question; `{state}` may appear twice. Special tokens written here (a chat template) are parsed, user text is escaped |
| `prompt_no_state`, `empty_state` | for a blank state: another prompt, or the text that stands in for the state |
| `option_line` | one option, `{letter}` is A, B, .. |
| `variants` | token spellings of a letter whose probabilities are added |
| `state_format` | `text` (labelled lines) or `json` (compact JSON; `json_index_min` adds `_index` to the items of long arrays) |
| `score_format`, `true_first`, `noul_instructions`, `same_as_name`, `strip` | how levels and yes/no are written |
| `bare_keys` | choice keys that only number the options: `positional` drops a leading `a: `, `bare` shows the description alone |
| `temperature` | a number, or a map looked up as `<type>\|<bucket>` (option count 2, 3-4, 5-8, 9+), `<type>`, `default` |
| `share_prefix` | `false` reads every question whole. Required for a GGUF with `attention.causal = false`, which also needs `-b` and `-ub` equal to `-c` |
| `style` | `ollama` (TinyJev v2): the options become the user message of Ollama's decision API, which `prompt` wraps as `{user}` |

Unless `share_prefix` is false, the questions of a request share their common token prefix (the state): it is decoded
once and each question continues from a restored copy. More than 26 options are read in chunks whose winners meet in a
final read.

## Pointer family (with a head)

TinyJev v1, Kev: add `--classifier-head FILE`, a safetensors file with `q.weight`, `q.bias`, `k.weight`, `k.bias`.
A `tinyjev.json` manifest works as the config; every key is optional:

```json
{"name": "tinyjev-4b", "family": "pointer",
 "head": {"head_dim": 256, "temperature": 1.0},
 "max_state": 8192, "max_branch": 8192, "max_options": 255,
 "layout": {
   "tokens": {"state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option": "<|box_start|>",
              "close": "<|box_end|>", "decide": "<|fim_suffix|>"},
   "escape_special": true, "option_format": "{name}: {desc}",
   "boolean_labels": {"false": "no", "true": "yes"}}}
```

The prompt is `<state> text <question> instructions (<option> option <close>)* <decide>`, and the head scores the
hidden state at `<decide>` against the one at each `<close>`. With `escape_special`, `<|name|>` in user text becomes
`<¦name¦>` so it cannot forge a delimiter.

## Answer cache

`--classifier-cache DIR` keeps answers on disk: an answer depends only on the model, head and config files and on
the tokens of the state and the question, so a repeated question about the same state is answered without a decode.
Caching is per question; the response carries `"cached": true` when nothing had to be decoded. Changing the model,
head or config starts a new file; a file over 64 MiB loses its older half at the next start.

## Recommended model

For a classifier that runs on the CPU next to a chat model: StartLux-Decision v1.1 2B Instant, and on AMD Ryzen CPUs in
**BF16** rather than Q8_0. Ryzen 9 7900, 12 threads, OpenDecision 500 and 115 requests of our own:

| Model | Family | OpenDecision 500 | Own requests | Latency |
|---|---|---|---|---|
| TinyJev-4B v1 | pointer | 476 | 105 | 0.86 - 0.89 s |
| **StartLux-Decision v1.1 2B Instant** | letter | 484 | 112 | 0.48 s |
| Wald-4B v1.2 | letter | 488 | 111 | 1.1 - 1.25 s |
| TinyJev-4B v2 | letter, `ollama` | 490 | 110 | 1.9 s |

StartLux-Decision v1.1 2B Instant BF16 answers a yes/no question in 0.21 - 0.31 s and a 21-option choice in 0.89 s,
0.48 s usage-weighted. The other rows were measured on the 2026-10-06 runtime.

BF16 against Q8_0 on the 2B size (v1, 2026-10-06 runtime): a yes/no question 0.39 s against 0.75 s, a 21-option choice
1.05 s against 2.0 s, at 3.8 GB against 2.0 GB. The weights are CC BY-NC 4.0; Wald-4B is the Apache-2.0 alternative.

No BF16 GGUF is published. Convert the checkpoint; the converter writes `attention.causal = false` into the file:

```
hf download startlux-models/StartLux-Decision-v1.1-2B-Instant --local-dir StartLux-Decision-v1.1-2B-Instant
python convert_hf_to_gguf.py StartLux-Decision-v1.1-2B-Instant --outtype bf16 --no-mtp --outfile StartLux-Decision-v1.1-2B-Instant-BF16.gguf
```

## Router mode

One public port for chat and classifier models: the router starts each model as a child from a `--models-preset`
INI and routes by the request's `"model"` (name or alias). Classifier routes whose `"model"` is missing or unknown go
to the first model with `classifier-head` or `classifier-config` in its preset; `--models-default NAME` sends other
requests without a known `"model"` to NAME, like a single-model server.

```ini
version = 1

[qwen/qwen3.8-27b]
model = Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
load-on-startup = true
gpu-keepalive-ms = 2000

[startlux-decision-v1.1-2b]
model = StartLux-Decision-v1.1-2B-Instant-BF16.gguf
classifier-config = classifier.json
load-on-startup = true
dev = none
ngl = 0
c = 4096
b = 4096
ub = 4096
```

```
llama-server --models-preset models.ini --models-max 0 --models-default qwen/qwen3.8-27b --port 1234
```

`--models-api-loopback-only` keeps model management (load, unload, download, delete) to loopback clients, so the API
key can be shared with outside callers. `--gpu-keepalive-ms N` touches a small buffer on each used GPU every N ms:
some Windows drivers evict an idle GPU's VRAM to system RAM once it has no display work.
