# ROCmFPX MTP Serving

This guide covers request-level MTP controls for ROCmFPX serving:

```text
speculative.n_max
speculative.n_min
speculative.p_min
```

These fields let each request lower the active draft policy without restarting
`llama-server`. The server startup value for `--spec-draft-n-max` remains the
allocation cap, so start the server at the highest draft depth you plan to test
and clamp individual requests downward.

## Build

Build `llama-server` from this ROCmFPX tree:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh llama-server llama-bench
```

The default Strix build output is:

```text
build-strix-rocmfp4/bin/llama-server
```

Set `BUILD_DIR` or `BIN` when using a different build directory.

## Start A ROCmFPX MTP Server

The helper script starts a single-slot OpenAI-compatible server with metrics and
request-level draft overrides available. By default it first checks whether the
GGUF actually exposes MTP metadata/tensors. If the model does not support MTP,
it prints a warning and starts without `draft-mtp` instead of passing invalid
speculative flags.

```bash
MODEL=/path/to/model.gguf \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.10 \
scripts/run-rocmfpx-mtp-server.sh
```

Set `REQUIRE_MTP=1` when a benchmark must fail hard if the model is not
MTP-capable:

```bash
REQUIRE_MTP=1 MODEL=/path/to/mtp-model.gguf scripts/run-rocmfpx-mtp-server.sh
```

Use the capability helper directly when preparing model-specific launchers:

```bash
scripts/rocmfpx-model-capabilities.py /path/to/model.gguf
scripts/rocmfpx-model-capabilities.py /path/to/model.gguf --has-mtp --quiet
scripts/rocmfpx-model-capabilities.py /path/to/model.gguf --server-args
```

The helper emits a validated serving profile for known local families. For
example, Qwen/Qwable MTP models get `draft-mtp` with `n-max 4`, while Nemotron
3 Nano 30B A3B ROCmFPX agent models are marked non-MTP and use the validated
ROCm/f16-KV profile.

The capability JSON includes a `model_kind` field so regular GGUF, MTP,
diffusion, QAT, and agent-oriented ROCmFPX models do not receive the same
startup assumptions:

| Kind | Detection | Default serving stance |
|---|---|---|
| `regular` | no MTP, diffusion, QAT, or agent markers | normal ROCm/Vulkan offload, no speculative or diffusion flags |
| `mtp` | MTP metadata/tensor markers such as `nextn_predict_layers` | enable `draft-mtp`, then sweep `n-max` and draft KV |
| `diffusion` | diffusion / DiffusionGemma / LLaDA markers | enable diffusion KV/cache controls, benchmark denoising settings separately |
| `qat` | QAT / quantization-aware markers or filename | keep native QAT tensor layout; do not re-quantize without A/B quality checks |
| `agent` | ROCmFPX agent/coherent markers or filename | preserve K/V quality first; lower cache precision only after JSON/tool/coding smoke tests |

This matters because speed features are not interchangeable. MTP only helps
models with MTP layers. Diffusion models need diffusion decode controls rather
than speculative draft flags. QAT models are already trained around their
quantized tensor layout, so ROCmFPX should treat them as a fast serving target
or a carefully tested source, not as a blind re-quant candidate. Agent models
can use the same ROCmFPX math, but the serving defaults should protect
structured output before chasing smaller K/V cache.

The helper also reports `supports_dflash` when a GGUF or filename exposes
DFlash/DDFlash markers. Treat this as a capability signal, not a forced launch
flag: this branch currently exposes MTP, Eagle3, and n-gram speculative types
in the public `--spec-type` path. Add DFlash launch flags only after the binary
advertises the spec type and a smoke test proves it on ROCm and Vulkan.

For production wrappers, run the preflight before launching:

```bash
MODEL=/path/to/model.gguf \
BIN=build-strix-rocmfp4/bin/llama-server \
DEVICE=ROCm0 \
scripts/rocmfpx-production-preflight.sh
```

The preflight validates that the binary and model exist, reports whether MTP is
actually supported, prints the recommended serving profile, includes a complete
`launch_command` array for wrappers, and can enforce hard requirements:

```bash
REQUIRE_MTP=1 MODEL=/path/to/model.gguf scripts/rocmfpx-production-preflight.sh
REQUIRE_PROFILE=1 MODEL=/path/to/model.gguf scripts/rocmfpx-production-preflight.sh
```

To write a reusable launch wrapper from the detected profile:

```bash
MODEL=/path/to/model.gguf \
BIN=build-strix-rocmfp4/bin/llama-server \
WRAPPER_OUT="$HOME/.local/bin/my-rocmfpx-model-server" \
scripts/rocmfpx-production-preflight.sh
```

The wrapper is made executable and appends any extra arguments you pass at run
time, so ports and host bindings can stay outside the model profile:

```bash
my-rocmfpx-model-server --host 127.0.0.1 --port 18180
```

Useful defaults:

```text
DEVICE=Vulkan0
BATCH_SIZE=2048
UBATCH_SIZE=512
PERF_PRESET=balanced
PARALLEL=1
POLL=50
POLL_BATCH=1
PRIO=0
PRIO_BATCH=0
N_GPU_LAYERS=999
FLASH_ATTN=on
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16
CACHE_TYPE_K_DRAFT=f16
CACHE_TYPE_V_DRAFT=f16
THREADS=16
THREADS_BATCH=32
STRICT_BENCH=1
```

`STRICT_BENCH=1` disables prompt-cache reuse and sets slot prompt similarity to
zero so benchmark rows are easier to compare. For interactive serving, set
`STRICT_BENCH=0` if you want normal prompt-cache behavior.

## Backend Utilization Knobs

`scripts/run-rocmfpx-mtp-server.sh` exposes low-risk ROCm/Vulkan utilization
controls through environment variables. These do not change ROCmFPX quant math;
they only change scheduling, placement, and batching:

```bash
PERF_PRESET=latency MODEL=/path/to/model.gguf DEVICE=ROCm0 scripts/run-rocmfpx-mtp-server.sh
PERF_PRESET=throughput PARALLEL=2 MODEL=/path/to/model.gguf DEVICE=Vulkan0 scripts/run-rocmfpx-mtp-server.sh
```

Presets:

| Preset | Behavior |
|---|---|
| `balanced` | stable default, single slot, conservative polling |
| `latency` | keeps one slot and raises polling to reduce wakeup overhead |
| `throughput` | uses two slots by default for continuous batching / multi-client use |

Safe sweep variables:

```text
DEVICE=ROCm0 or Vulkan0
PERF_PRESET=balanced|latency|throughput
PARALLEL=1..4
BATCH_SIZE=512,1024,2048
UBATCH_SIZE=256,512,1024
POLL=50..100
POLL_BATCH=0|1
N_GPU_LAYERS=999 or all
FLASH_ATTN=on|auto
FIT_TARGET=1024
FIT_CTX=4096
NO_HOST=1
```

Start with ROCm for Nemotron-style non-MTP models and Vulkan or ROCm for
Qwable/Qwen MTP models, then benchmark both. Keep `PARALLEL=1` for single-user
decode-speed measurements; use `PARALLEL=2` or higher only when testing
continuous batching or multiple clients.

## Imatrix Quantization

ROCmFPX supports imatrix-weighted scale search for ROCmFP3, ROCmFP6, and
ROCmFP8, and ROCmFP4 has its own imatrix path. Use imatrix especially for
ROCmFP3/low-bit agent quants where the quality gap is easiest to see:

```bash
IMATRIX=/path/to/imatrix.gguf \
FORMAT=rocmfp3 \
PROFILE=agent \
SRC=/path/to/source-bf16.gguf \
OUT=/path/to/model-Q3_0_ROCMFPX_AGENT.gguf \
scripts/quantize-rocmfpx-agent.sh
```

The wrapper passes `--imatrix` through to `llama-quantize`. The reference test
also verifies weighted reconstruction improves over plain quantization for
ROCmFP3, ROCmFP6, and ROCmFP8.

## TurboQuant Asymmetric KV

TurboQuant is already available in this ROCmFPX tree through the `turbo3` and
`turbo4` runtime cache types. It is separate from the ROCmFPX model-weight
formats.

For agentic serving, prefer asymmetric K/V instead of compressing both sides:

```bash
MODEL=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
DEVICE=Vulkan0 \
PORT=18180 \
CTX_SIZE=32768 \
scripts/run-rocmfpx-turboquant-asym-server.sh
```

That wrapper starts `scripts/run-rocmfpx-mtp-server.sh` with:

```text
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=turbo4
CACHE_TYPE_K_DRAFT=q8_0
CACHE_TYPE_V_DRAFT=turbo4
```

Use this when you want TurboQuant memory savings without spending the highest
quality risk on K cache. Symmetric sweeps are still possible with
`CACHE_TYPE_K=turbo4 CACHE_TYPE_V=turbo4`, but they should be reported as a
quality/speed experiment rather than the default Hermes/OpenClaw profile.

For symmetric sweeps, boundary-layer K protection is available as an opt-in:

```bash
LLAMA_KV_TURBO_BOUNDARY_LAYERS=2 \
CACHE_TYPE_K=turbo4 \
CACHE_TYPE_V=turbo4 \
MODEL=/path/to/model.gguf \
scripts/run-rocmfpx-mtp-server.sh
```

That keeps K cache at `q8_0` for the first and last two model layers and uses
the requested TurboQuant type for the middle layers. V boundary protection is
off by default; set `LLAMA_KV_TURBO_BOUNDARY_V=1` only for controlled
experiments.

## Per-Request Overrides

Use the request keys on `/completion`:

```bash
curl -sS http://127.0.0.1:18180/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a concise technical note about ROCmFPX MTP serving.",
    "n_predict": 512,
    "temperature": 0,
    "ignore_eos": true,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

The response `generation_settings` should echo the effective values. If
`speculative.n_max` is higher than the server cap, it is clamped to the cap.
`speculative.n_min` is clamped to `0..n_max`, and `speculative.p_min` is
clamped to `0.0..1.0`.

OpenAI chat-compatible requests use the same keys in the top-level payload:

```bash
curl -sS http://127.0.0.1:18180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rocmfpx-mtp",
    "messages": [
      {"role": "user", "content": "Summarize the request-level MTP knobs."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

## Dynamic Drafting

ROCmFPX supports Dynamic Drafting as a request-layer policy over llama-server's
per-request speculative controls. Start the server with a high enough draft cap,
then let the client choose safer or more aggressive values for each prompt.

Use `scripts/rocmfpx-draft-profile.py` when a client only needs the static
prompt-length policy:

```bash
scripts/rocmfpx-draft-profile.py --profile fp3-mtp --prompt-tokens 4096 --pretty
scripts/rocmfpx-draft-profile.py \
  --profile dense-coder \
  --base-url http://127.0.0.1:18180 \
  --prompt-file /path/to/prompt.txt \
  --pretty
```

The helper emits only request JSON fields. It does not change the server cap,
so the server still needs to start with a high enough `--spec-draft-n-max`.
Available profiles are `fp3-mtp`, `fp4-general`, and `dense-coder`.

Use `scripts/rocmfpx-dynamic-draft.py` for full Dynamic Drafting. It injects
`speculative.n_max`, `speculative.n_min`, `speculative.p_min`, and
`speculative.p_split` into each request, then optionally records
`draft_n/draft_n_accepted` feedback from responses and adjusts later requests.
The wrapper scopes learned feedback by profile, inferred workload mode, backend,
model, and context size so a Vulkan JSON run does not drag down a ROCm coding
run. Modes are inferred automatically, or can be pinned with `--mode coding`,
`--mode json`, `--mode tool`, `--mode chat`, or `--mode completion`.

```bash
scripts/rocmfpx-dynamic-draft.py \
  --base-url http://127.0.0.1:18180 \
  --endpoint /completion \
  --profile dense-coder \
  --backend ROCm0 \
  --model-key qwable-agent \
  --ctx-size 65536 \
  --mode coding \
  --state-file /tmp/rocmfpx-dd-state.json \
  --json '{
    "prompt": "Write a small JSON parser in Python.",
    "n_predict": 512,
    "temperature": 0
  }'
```

For OpenAI-compatible chat:

```bash
scripts/rocmfpx-dynamic-draft.py \
  --base-url http://127.0.0.1:18180 \
  --endpoint /v1/chat/completions \
  --profile fp4-general \
  --backend ROCm0 \
  --model-key rocmfpx-agent \
  --ctx-size 65536 \
  --mode tool \
  --state-file /tmp/rocmfpx-dd-state.json \
  --json '{
    "model": "rocmfpx-mtp",
    "messages": [{"role": "user", "content": "Return a tool-call shaped JSON object."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The state file stores an exponential moving average of acceptance. If
acceptance falls below the low threshold, the wrapper lowers `n_max` and raises
`p_min`; if acceptance is high, it cautiously raises `n_max` up to
`--max-n-max`. Use `--dry-run --pretty` to inspect the exact request before it
is sent.

The adaptive policy is throughput-first, not acceptance-only. It keeps per
`n_max` throughput buckets and occasionally explores a neighboring draft depth
with `--explore-rate` so it can learn whether `n_max=5` is actually faster than
`n_max=6` for the current model/backend/context. Set `--explore-rate 0` for
fully repeatable benchmark sweeps. When llama-server exposes per-position draft
acceptance telemetry, the wrapper also stores it and caps future `n_max` at the
first consistently weak draft position instead of waiting for whole-request
acceptance to fall.

For Qwen/Qwable-style templates, the wrapper also strips literal
`<think>...</think>` blocks and `reasoning_content` fields from returned JSON by
default. Disable that only for debugging with `--no-strip-thinking`. The wrapper
also requests `reasoning_format=deepseek` by default for OpenAI-compatible chat
calls, which keeps Hermes-style clients from seeing raw reasoning fields when
the backend supports the parser.

Mode defaults:

| Mode | Default adjustment | Use when |
|---|---|---|
| `coding` | slightly deeper draft cap, moderate split | Qwable/code models with deterministic code output |
| `json` | lower draft cap, `p_min>=0.25`, low split | strict JSON responses and schema-constrained output |
| `tool` | lower draft cap, `p_min>=0.25`, low split | agent tool-call plans and function-call shaped output |
| `chat` | profile default | normal chat |
| `completion` | profile default | raw `/completion` prompts |

## Suggested Starting Points

These are starting points for sweeps, not universal defaults:

| Model path | Startup settings | Request settings |
|---|---|---|
| FP3 MTP speed sweep | `n_max=4`, `p_split=0.10`, target/draft KV `f16` | short context: `n_max=4`, `p_min=0.75`; long context: `n_max=2`, `p_min=0.0` |
| ROCmFP4 MTP general | `n_max=4`, `p_split=0.10`, target/draft KV `f16` | start with `n_max=4`, `p_min=0.0` and sweep `p_min=0.25` |
| ROCmFP4 dense coder | `n_max=6`, `p_split=0.20`, target KV `q8_0`, draft KV `f16` | start with `n_max=6`, `p_min=0.0` |
| ROCmFPX TurboQuant asymmetric | `n_max=4`, `p_split=0.10`, target/draft KV `q8_0/turbo4` | start with `n_max=4`, `p_min=0.0`; compare against `q8_0/q8_0` |

Example dense-coder server:

```bash
MODEL=/path/to/coder-rocmfp4-agent.gguf \
ALIAS=rocmfpx-coder-mtp \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_DEVICE=Vulkan0 \
BATCH_SIZE=2048 \
UBATCH_SIZE=512 \
CACHE_TYPE_K=q8_0 \
CACHE_TYPE_V=q8_0 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
STRICT_BENCH=1 \
scripts/run-rocmfpx-mtp-server.sh
```

Use this request policy with that cap:

```json
{
  "speculative.n_max": 6,
  "speculative.n_min": 0,
  "speculative.p_min": 0.0
}
```

## Validation

Before reporting a serving result, record:

- model path and alias
- server binary path and commit
- backend device
- context allocation
- prompt tokens and generated tokens
- target and draft KV cache types
- batch and ubatch
- startup MTP cap and per-request speculative fields
- prompt-cache setting
- decode tok/s, prompt tok/s, TTFP, total time
- draft accepted and draft generated counters

For quick server checks:

```bash
curl -sS http://127.0.0.1:18180/health
curl -sS http://127.0.0.1:18180/props | jq '.default_generation_settings'
curl -sS http://127.0.0.1:18180/metrics | head
```

Use served API rows or a CLI guard with draft counters for headline MTP speed.
Do not use standalone `llama-bench` TG as the headline for MTP serving.
