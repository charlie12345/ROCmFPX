# ROCmFPX MTP Serving

> **Note (2026-11-07):** Written in June 2026 for Strix Halo (gfx1151) on
> Linux. Since the 2026-09-15 merge of the ROCmFPX org repo the server ignores
> the per-request `speculative.*` fields and rejects `turbo3`/`turbo4` caches.
> The scripts below no longer pass the removed `--checkpoint-every-n-tokens`
> and `--mmap` flags (updated 2026-11-07); for serving RDNA-FAST today see
> [Flags and switches](FLAGS.md).

This guide covers request-level MTP controls for ROCmFPX serving:

```text
speculative.n_max
speculative.n_min
speculative.p_min
```

These fields let each request lower the active draft policy without restarting
`llama-server` (currently ignored, see [Per-Request Overrides](#per-request-overrides)).
The server startup value for `--spec-draft-n-max` remains the
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

The helper script starts a single-slot OpenAI-compatible server with metrics.
By default it first checks whether the
GGUF actually exposes MTP metadata/tensors. If the model does not support MTP,
it prints a warning and starts without `draft-mtp` instead of passing invalid
speculative flags.

`--checkpoint-every-n-tokens` was removed on 2026-09-15. The script no longer
passes it; a `CPENT` of 0 or more now goes to `--checkpoint-min-step` (minimum
spacing between checkpoints), and the default `CPENT=-1` passes nothing.

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
example, Qwen MTP ROCmFP4/ROCmFPX models get the Strix prompt-fast
Vulkan profile with `draft-mtp`, `n-max 4`, q4 target/draft K/V, and a
`4096/512` batch shape, while Nemotron 3 Nano 30B A3B ROCmFPX agent models are
marked non-MTP and use the validated ROCm/f16-KV profile.

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
flag. `--spec-type` now also takes `draft-dflash` and `draft-dspark`, which need
a DFlash draft model; smoke-test such a launch on ROCm and Vulkan before
relying on it.

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

The helper's profiles pass `--load-mode mmap` (`--mmap` was removed on
2026-10-02), so `launch_command` and generated wrappers use the current flag.

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
SPEC_DRAFT_N_MAX=6
SPEC_DRAFT_N_MIN=0
SPEC_DRAFT_P_MIN=0.6
SPEC_DRAFT_P_SPLIT=0.10
STRICT_BENCH=0
```

For ROCmFP4/ROCmFPX prompt-heavy work on Strix, use the prompt-fast preset:

```bash
PERF_PRESET=prompt-fast \
MODEL=/path/to/qwen-rocmfpx-mtp.gguf \
scripts/run-rocmfpx-mtp-server.sh
```

This leaves the GGUF unchanged and only changes the serving shape:

```text
DEVICE=Vulkan0
BATCH_SIZE=4096
UBATCH_SIZE=512
CACHE_TYPE_K=q4_0
CACHE_TYPE_V=q4_0
CACHE_TYPE_K_DRAFT=q4_0
CACHE_TYPE_V_DRAFT=q4_0
FLASH_ATTN=on
PARALLEL=1
POLL=50
```

For single-user decode on Qwen MTP ROCmFP4/ROCmFPX models, use the
decode-fast preset:

```bash
PERF_PRESET=decode-fast \
MODEL=/path/to/qwen-rocmfpx-mtp.gguf \
scripts/run-rocmfpx-mtp-server.sh
```

This is still a serving policy change, not a GGUF or quantization change. It
uses Vulkan0, q8 accepted K/V, q4 draft K/V, `512/512` batch shape, single
slot, `n-max 6` and a `p-min 0.6` confidence gate so low-confidence draft tokens
do not burn verify work. Until 2026-07-03 it used `n-max 4` and `p-min 0.75`;
the matrix below predates that change. On the local Qwen3.6 27B MTP ROCmFP4
STRIX_LEAN smoke matrix, the relevant results were:

| Profile | Backend | KV | `n-max` | `p-min` | Short decode tok/s | Sustained decode tok/s |
|---|---|---|---:|---:|---:|---:|
| default q4 | ROCm0 | q4 main / q4 draft | 4 | 0.00 | 10.6 | 17.4 |
| gated q4 | ROCm0 | q4 main / q4 draft | 4 | 0.75 | 28.6 | 21.7 |
| reduced-depth q4 | ROCm0 | q4 main / q4 draft | 2 | 0.00 | 30.9 | 17.8 |
| gated q4 | Vulkan0 | q4 main / q4 draft | 4 | 0.75 | 30.3 | 22.2 |
| decode-fast shape | Vulkan0 | q8 main / q4 draft | 4 | 0.75 | 30.5 | 24.4 |

If memory headroom is tight at very long context, keep q4 accepted K/V and
retain `p-min 0.75`; that preserved most of the decode gain in the local test
without the q8 accepted-KV memory cost.

On Qwen3.6 35B A3B MTP ROCmFP4 coherent, this profile measured about
`556 tok/s` prompt and `42 tok/s` decode, versus the same ROCmFP4 GGUF on ROCm
at about `292 tok/s` prompt for the best tested batch shape. The quality gap
remains the quantization gap, not a serving-profile coherency issue: the
ROCmFP4 coherent GGUF was about 7% smaller than Q4_K_M, with WikiText-2 PPL
`5.7358` versus Q4_K_M `5.4800` on the same clone build.

`STRICT_BENCH=1` disables prompt-cache reuse and sets slot prompt similarity to
zero so benchmark rows are easier to compare. The default, `STRICT_BENCH=0`
since 2026-06-25, keeps normal prompt-cache behavior.

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
| `prompt-fast` | Strix ROCmFP4/ROCmFPX prefill profile; Vulkan0, q4 K/V, `4096/512`, single slot |
| `decode-fast` | Strix Qwen MTP decode profile; Vulkan0, q8 accepted K/V, q4 draft K/V, `n-max 6`, `p-min 0.6`, `512/512`, single slot |
| `latency` | keeps one slot and raises polling to reduce wakeup overhead |
| `throughput` | uses two slots by default for continuous batching / multi-client use |

Safe sweep variables:

```text
DEVICE=ROCm0 or Vulkan0
PERF_PRESET=balanced|prompt-fast|decode-fast|latency|throughput
PARALLEL=1..4
BATCH_SIZE=512,1024,2048,4096
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
Qwen MTP models, then benchmark both. Keep `PARALLEL=1` for single-user
decode-speed measurements; use `PARALLEL=2` or higher only when testing
continuous batching or multiple clients.

The prompt-fast profile is the safe family-wide serving fix for ROCmFP4 and
ROCmFPX GGUFs where Vulkan supports the model ops. A HIP-side family fix should
target the large-prefill `MUL_MAT`/MMQ path for ROCmFP4/ROCmFPX so ROCm0 can
match the Vulkan shader path; changing quant presets is not the first lever
when the same GGUF is already coherent and faster on Vulkan.

For HIP-side MMQ work, use the focused sweep harness before changing defaults:

```bash
OUT_DIR=/tmp/rocmfp4-prefill-mmq-sweep scripts/sweep-rocmfp4-prefill-mmq.sh
```

The harness rebuilds selected MMQ compile-time variants, runs a focused
ROCmFP4 backend guard, and benchmarks both the ROCmFP4 target GGUF and the
Q4_K_M baseline from the same build. On Qwen3.6 35B A3B MTP STRIX_LEAN, the
tested `GGML_ROCMFP4_Q8_1_MMQ_VDR=4` / `GGML_ROCMFP4_FAST_Q8_1_MMQ_VDR=4`
candidate improved decode but did not confirm as a prompt-processing win on a
repeat run, so the default MMQ VDR settings remain unchanged.

For broader ROCm/Vulkan speed checks, use the ROCmFPX speed harness:

```bash
OUT_DIR=/tmp/rocmfpx-rocm-ai-speed scripts/sweep-rocmfpx-rocm-ai-speed.sh
```

The harness can compare the default build with opt-in ROCm variants such as
rocWMMA FlashAttention while keeping GGUF formats unchanged. Every variant
must pass the ROCmFPX backend-op family gates before its full-model benchmark
numbers are accepted. On the clean Strix Halo Qwen3.6 35B A3B run, rocWMMA
FlashAttention compiled and passed the family gates, but the default build was
faster for ROCmFP4 prompt processing, so rocWMMA remains opt-in rather than a
default. The `GGML_HIP_ROCWMMA_FATTN` CMake option has since been removed
(2026-09-15), so the harness no longer builds a rocWMMA variant and ignores the
rocWMMA field of `VARIANTS`.

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

Not available since 2026-09-15: `-ctk`/`-ctv` and `-ctkd`/`-ctvd` accept only
`f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0` and `q5_1`, so
the wrapper below now exits at once saying so, and
`LLAMA_KV_TURBO_BOUNDARY_LAYERS` / `LLAMA_KV_TURBO_BOUNDARY_V` are no longer
read. Kept as a record.

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

The server currently ignores these keys: their handlers are disabled (`#if 0`)
in `tools/server/server-schema.cpp` since the 2026-09-15 merge, and
`generation_settings` reports only `speculative.types`. The startup flags
`--spec-draft-n-max`, `--spec-draft-n-min` and `--spec-draft-p-min` set the
policy for every request. The examples and clamping rules below describe the
code when the keys were honoured.

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

Neither tool below works today: `scripts/rocmfpx-dynamic-draft.py` was removed
from the tree on 2026-07-12, and the request fields
`scripts/rocmfpx-draft-profile.py` emits are ignored (see above). Kept as a
record.

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
  --model-key qwen-agent \
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

For Qwen-style templates, the wrapper also strips literal
`<think>...</think>` blocks and `reasoning_content` fields from returned JSON by
default. Disable that only for debugging with `--no-strip-thinking`. The wrapper
also requests `reasoning_format=deepseek` by default for OpenAI-compatible chat
calls, which keeps Hermes-style clients from seeing raw reasoning fields when
the backend supports the parser.

Mode defaults:

| Mode | Default adjustment | Use when |
|---|---|---|
| `coding` | slightly deeper draft cap, moderate split | code models with deterministic code output |
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
