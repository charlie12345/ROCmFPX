# DeepSeek V4 ROCmFP4 Port Status

## Status (2026-11-07)

The rest of this page is the June 2026 port log from a Linux Strix Halo machine
with 110+ GiB of free RAM. The `/home/caf` and `/mnt/ai-models` paths, the Pi,
OpenCode and Kilo profiles, and the measurements belong to that machine. What
has changed since:

- DeepSeek V4 is in the main tree as architecture `deepseek4`
  (`src/models/deepseek4.cpp`, converter class `DeepseekV4ForCausalLM` in
  `conversion/deepseek.py`). No separate feature tree is needed.
- The DeepSeek helper operations (`DSV4_HC_COMB`, `DSV4_HC_PRE`, `DSV4_HC_POST`,
  `LIGHTNING_INDEXER`) have native HIP kernels since the 2026-09-15 merge and
  Vulkan kernels since 2026-10-02. The CPU fallback that made FlashAttention
  auto-disable on `ROCm0` no longer applies. The GPU numbers below have not been
  re-measured.
- `--deepseek4-include-mtp` and `--deepseek4-max-layers` are gone from
  `convert_hf_to_gguf.py`. The base conversion drops the MTP tensors, and
  `--mtp` exports the MTP head as a separate `mtp-` GGUF, to load as the draft
  (`-md`) with `--spec-type draft-mtp`. The two-file path has not been tested
  on DeepSeek V4 here.
- `--no-mmap` became `-lm none` (`--load-mode none`) on 2026-10-02.
- The `scripts/run-deepseek-v4-*.sh` helpers no longer pass the removed flags
  (updated 2026-11-07). With a third argument of `1` the full-conversion wrapper
  now writes the `mtp-` draft with `--mtp` (run it with `0` for the target), the
  ROCmFP3 pipeline writes the target and the `mtp-` draft, and the fixture
  script, which needs the removed `--deepseek4-max-layers`, stops with a
  message. Most helpers still expect a `build-deepseek4-rocmfp4` build
  directory, so they fail on the current tree until that path is changed.
- The promoted MTP artifact is about 92 GB, far beyond a 20 GB card with 32 GB
  of system RAM.

## Target

Convert the original checkpoint:

- Hugging Face: <https://huggingface.co/0xSero/DeepSeek-V4-Flash-180B>
- Local source: `/mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B`

The target runtime was this isolated special llama ROCmFP4 feature tree (DeepSeek
V4 now lives in the main tree):

- `/home/caf/strix-fp4/llama.cpp-deepseek-v4-rocmfp4-modern-port`

Do not treat a downloadable GGUF or a standalone DeepSeek V4 runtime as the
final artifact. They are references only.

## Current Result

The conversion and runtime pipeline is proven with fixtures generated from the
original 46-shard checkpoint:

| Fixture | Intermediate GGUF | Mixed ROCmFP4 GGUF | Purpose |
| --- | --- | --- | --- |
| One layer | `/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-1layer-mxfp4-f16.gguf` | `/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-1layer-rocmfp4-mxfp4.gguf` | Minimal loader and graph gate |
| Three layers | `/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-3layer-mxfp4-f16.gguf` | `/home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-3layer-rocmfp4-mxfp4.gguf` | First compressed-attention layer, ratios `[0, 0, 4]` |

The three-layer mixed fixture is about `7.1G` and contains:

| Tensor type | Count | Purpose |
| --- | ---: | --- |
| `Q4_0_ROCMFP4` | 36 | Dense matrices on the special ROCmFP4 path |
| `MXFP4` | 9 | Source-native routed expert matrices |
| `I32` | 3 | Token-id-to-expert-id hash routing tables |
| `Q5_K` | 2 | Shared expert quality-sensitive matrices |
| `Q6_K` | 3 | Output matrices |
| `F32` | 35 | Small scales, norms, routing data, and compressor data |

The routed experts must remain `MXFP4`. Requantizing them expands work,
introduces avoidable quality risk, and defeats the checkpoint's native packed
representation. The hash-routing tables must remain `I32`.

## Reproduce Fixtures

Run the default one-layer fixture:

```bash
scripts/run-deepseek-v4-rocmfp4-fixture.sh
```

Run the compressed-attention fixture:

```bash
scripts/run-deepseek-v4-rocmfp4-fixture.sh \
  /mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B \
  /home/caf/strix-fp4/runtime-fixtures/deepseek-v4 \
  3
```

Run the one-layer-plus-MTP fixture:

```bash
scripts/run-deepseek-v4-rocmfp4-fixture.sh \
  /mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B \
  /home/caf/strix-fp4/runtime-fixtures/deepseek-v4 \
  1 \
  1
```

The script converts the requested layer count, performs a guarded quantizer
dry-run, writes the mixed ROCmFP4/MXFP4 fixture, and validates critical tensor
types. For three or more layers it also validates compressor and indexer
tensors from the first compressed-attention block.

## Runtime Gate

The three-layer compressed-attention fixture generates eight forced completion
tokens on CPU, ROCm, and Vulkan:

| Backend | Prompt processing | Decode | Notes |
| --- | ---: | ---: | --- |
| CPU | `81.70 tok/s` | `61.28 tok/s` | Operation offload disabled |
| `ROCm0` | `124.10 tok/s` | `78.65 tok/s` | Functional; FlashAttention auto-disables because DeepSeek helper operations still use CPU fallback |
| `Vulkan0` | `205.34 tok/s` | `101.28 tok/s` | Functional with current Vulkan backend coverage |

These are truncated-fixture smoke measurements, not full-model performance
claims.

Run the bounded smoke harness:

```bash
scripts/run-deepseek-v4-rocmfp4-smoke.sh \
  /home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-3layer-rocmfp4-mxfp4.gguf \
  ROCm0
```

Replace `ROCm0` with `CPU` or `Vulkan0` as needed.

## Full Conversion

The guarded full base-model conversion is allowed after the compressed runtime
gate passes. Run:

```bash
scripts/run-deepseek-v4-rocmfp4-full-conversion.sh
```

The script writes `.partial` artifacts under:

```text
/mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B
```

It preserves source-native `MXFP4` routed experts, quantizes supported dense
matrices to `Q4_0_ROCMFP4`, validates representative tensor types, and promotes
the final filename only after validation.

Full base-model conversion completed successfully. The guarded validator
promoted the archival mixed artifact after checking representative tensor
types:

```text
/mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B/DeepSeek-V4-Flash-180B-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf
```

The promoted artifact contains `1328` tensors: `596 Q4_0_ROCMFP4`,
`129 MXFP4`, `3 I32`, `535 F32`, `36 Q5_K`, and `29 Q6_K`.

## Runtime Storage Policy

Use `/mnt/ai-models` for source checkpoints, conversion intermediates, and
archival copies only. Do not run full-model generation tests from the external
model drive.

Copy the promoted artifact to the internal NVMe before runtime testing:

```text
/home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf
```

The DeepSeek V4 smoke harness refuses model paths under `/mnt/ai-models` to
prevent accidental external-drive runtime tests. Store runtime fixtures under
`/home/caf/strix-fp4/runtime-fixtures/deepseek-v4`, not RAM-backed `/tmp`. The
harness also disables mmap by default because mmap prepopulation is not
appropriate for this full artifact on the unified-memory target.

## Full Internal Runtime Gate

The promoted base artifact loads and generates from the internal NVMe on both
GPU backends. These bounded measurements used a seven-token prompt, eight
forced completion tokens, context `512`, and `--no-mmap --no-repack`:

| Backend | Startup to generation | Prompt processing | Decode | Notes |
| --- | ---: | ---: | ---: | --- |
| `ROCm0` | `32.1s` | `22.94 tok/s` | `6.91 tok/s` | Functional; FlashAttention auto-disables because DeepSeek helper operations still use CPU fallback |
| `Vulkan0` | `43.7s` | `26.26 tok/s` | `9.53 tok/s` | Functional with current Vulkan backend coverage |

Run the verified internal-NVMe smoke:

```bash
NO_REPACK=1 N_PREDICT=8 \
  scripts/run-deepseek-v4-rocmfp4-smoke.sh \
  /home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf \
  ROCm0
```

Replace `ROCm0` with `Vulkan0` for the Vulkan path. The harness adds
`--load-mode none` by default (`--no-mmap` in June). Set `NO_MMAP=0` only
for controlled loader diagnostics.

## Runtime Notes

The isolated tree included the DeepSeek V4 metadata, tensor loader, hybrid
compressed-attention cache, graph, and CPU helper operations. In June, ROCm and
Vulkan could execute the compressed fixture, but the DeepSeek-specific helper
operations had no native GPU kernels, and the ROCm path disabled FlashAttention
automatically when the scheduler detected CPU fallback. The main tree now has
HIP and Vulkan kernels for these operations (see Status).

The HIP/CUDA concat backend also needed `F16` support for the compact compressed
KV-cache merge. This was a generic backend fix, to be regression-tested against
the existing ROCmFP4 architectures before the feature tree was merged.

The first full-model attempts timed out before graph execution because the
default mmap path issued:

```text
mmap(..., 96429582304, MAP_SHARED|MAP_POPULATE, ...)
```

That eagerly prepopulated the entire mixed GGUF before GPU placement on the
unified-memory target. A bounded internal syscall trace confirmed the cause.
Disabling repack alone did not change the stall. `--no-mmap` avoids the eager
prepopulation and allows both GPU backends to load and generate normally. This
was a loader-policy issue, not an external-drive speed issue or a failed
conversion.

## MTP Phase

MTP remains opt-in and separate from the proven base-model artifact. In June the
route was `--deepseek4-include-mtp` on `convert_hf_to_gguf.py` (now removed; the
current converter uses `--mtp`, see Status), or the guarded full conversion
wrapper with a third argument of `1`:

```bash
scripts/run-deepseek-v4-rocmfp4-full-conversion.sh \
  /mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B \
  /mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B \
  1
```

The MTP conversion writes distinct `-MTP-` filenames and does not overwrite the
proven base artifact. The guarded full-conversion wrapper also enables
`--use-temp-file` for MTP and spills temporary tensor data under the external
archival directory. This avoids retaining every merged routed-expert tensor in
RAM until final write.

A one-layer-plus-MTP fixture converted and generated
successfully on CPU, `ROCm0`, and `Vulkan0` using native `draft-mtp` staging:

| Backend | Prompt processing | Decode |
| --- | ---: | ---: |
| CPU | `109.0 tok/s` | `123.2 tok/s` |
| `ROCm0` | `328.6 tok/s` | `290.6 tok/s` |
| `Vulkan0` | `195.3 tok/s` | `67.8 tok/s` |

These are bounded fixture smoke measurements, not full-model performance
claims. Run the MTP smoke harness with:

```bash
scripts/run-deepseek-v4-mtp-rocmfp4-smoke.sh \
  /home/caf/strix-fp4/runtime-fixtures/deepseek-v4/deepseek-v4-flash-1layer-mtp-rocmfp4-mxfp4.gguf \
  ROCm0
```

Replace `ROCm0` with `CPU` or `Vulkan0` as needed.

The guarded full MTP archival conversion completed successfully on
2026-06-02. Its validator checked all `1360` tensors and promoted:

```text
/mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf
```

The promoted archival artifact is about `92G`. It was copied and byte-verified
on the internal NVMe at:

```text
/home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf
```

The first internal full-model `ROCm0` attempt on 2026-06-02 was stopped by a
global host OOM before generation. The model was on the internal NVMe, but
`/tmp` is RAM-backed on this machine and still held about `35G` of disposable
fixture GGUF files. Do not use `/tmp` for multi-gigabyte fixtures on this host.
Use a persistent internal-NVMe scratch directory for runtime fixtures and the
external drive for archival-only artifacts.

The DeepSeek smoke wrappers now refuse full-model launches unless at least
`110 GiB` of RAM is available and `/tmp` usage is at most `2 GiB`. The MTP
wrapper also accepts `N_GPU_LAYERS` and `SPEC_DRAFT_N_GPU_LAYERS` so
internal-drive backend validation can increase offload gradually instead of
immediately requesting full offload.

For full-model MTP testing, use the capped launcher. It adds a user-level
`108G` memory cgroup with swap disabled around the smoke process:

```bash
N_PREDICT=8 \
scripts/run-deepseek-v4-mtp-rocmfp4-capped-smoke.sh \
  /home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf \
  ROCm0
```

Replace `ROCm0` with `Vulkan0` to test Vulkan. The internal full-model artifact
generates successfully with native MTP staging on both GPU backends:

| Backend | Main offload | MTP draft offload | Tokens | Prompt processing | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ROCm0` | `999` | `0` | `8` | `31.4 t/s` | `11.2 t/s` |
| `ROCm0` | `999` | `all` | `8` | `31.8 t/s` | `11.2 t/s` |
| `Vulkan0` | `999` | `all` | `8` | `25.1 t/s` | `9.6 t/s` |

These are bounded smoke measurements at context `128`, not long-context
throughput claims. On this short sample, native MTP draft GPU offload is
functional but does not improve ROCm decode speed over keeping the draft layer
on CPU.

The DeepSeek-specific cache follow-up on 2026-06-02 kept native MTP speculative
decoding enabled and changed only the main and draft KV-cache types. Both
quantized-cache attempts were rejected for promotion:

| Main KV | MTP draft KV | Result |
| --- | --- | --- |
| `f16` | `f16` | pass; promoted safe default |
| `q4_0` | `q4_0` | `libllama.so` segmentation fault under the `108G` cgroup |
| `q8_0` | `q8_0` | `libllama.so` segmentation fault under the `108G` cgroup |

Do not use `q4_0` or `q8_0` KV-cache flags for this DeepSeek port until the
quantized-cache crash is fixed. Other ROCmFP4 model families have separately
validated quantized-cache profiles; this restriction is specific to the
current DeepSeek V4 Flash 180B port.

For an interactive terminal chat using the promoted safe flags:

```bash
cd /home/caf/strix-fp4/llama.cpp-deepseek-v4-rocmfp4-modern-port
scripts/run-deepseek-v4-mtp-rocmfp4-chat.sh
```

The launcher uses the internal-NVMe artifact, `ROCm0`, native MTP speculative
decoding, `f16` main and draft KV caches, an initial context of `4096`, and the
same RAM, `/tmp`, and `108G` cgroup safeguards as the bounded smoke wrapper.
The `4096` value is a conservative boot default, not the model limit. The
source model metadata declares a native context of `1048576`; the source
checkpoint author validated `200000` with a different vLLM FP8-KV runtime on
one DGX Spark. Long-context fit in this llama.cpp port still requires its own
validation.

The launcher defaults to non-thinking chat, temperature `0.6`, and an English
system prompt. Override those with `REASONING`, `TEMPERATURE`, or
`SYSTEM_PROMPT`. It also passes
`models/templates/deepseek-ai-DeepSeek-V4.jinja` explicitly.
The current full artifact predates that restored template metadata; without the
override, interactive mode does not encode the required `<｜User｜>` and
`<｜Assistant｜>` markers. Future DeepSeek V4 conversions embed the same
template fallback automatically.

After releasing any resident large model, run the contained long-context
allocation ladder with:

```bash
scripts/run-deepseek-v4-mtp-context-ladder.sh
```

It tests `32768`, `65536`, `131072`, `200000`, `262144`, `524288`, and
`1048576` in order and stops at the first failed allocation. To open an
interactive chat at a validated rung:

```bash
CONTEXT=131072 scripts/run-deepseek-v4-mtp-rocmfp4-chat.sh
```

All seven configured-window acceptance smokes passed on `ROCm0` with native
MTP on 2026-06-02. These shallow tests prove that the graph accepts the native
window and executes; they do not prove that a prompt filling the entire window
fits in physical memory. The recorded table is:

```text
/home/caf/strix-fp4/ROCMFP4-HANDOFF/DEEPSEEK-V4-CONTEXT-RESULTS.md
```

If `/exit` leaves a HIP worker resident during teardown, release it with:

```bash
scripts/stop-deepseek-v4-mtp-rocmfp4-chat.sh
```

The promoted local-agent profile is:

```text
deepseek-v4-flash-180b-mtp-rocmfp4-rocm-best
```

It is available to Pi directly through `pi-llama-strix` on port `8234`, and to
OpenCode and Kilo through the local switching proxy on port `8233`. The proxy
is enabled as the user-login service `kilo-llama-switch.service`; it does not
load the 180B model until a client selects the profile. To start the backend
explicitly:

```bash
/home/caf/.local/bin/pi-llama-strix switch \
  deepseek-v4-flash-180b-mtp-rocmfp4-rocm-best
```

To release model RAM:

```bash
/home/caf/.local/bin/pi-llama-strix stop
```

## References

- <https://github.com/antirez/llama.cpp-deepseek-v4-flash>
- <https://github.com/nisparks/llama.cpp/tree/wip/deepseek-v4-support>
- <https://github.com/ggml-org/llama.cpp/issues/24082>
