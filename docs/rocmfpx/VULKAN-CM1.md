# Experimental Vulkan ROCmFP4 CM1 acceleration

This optional path runs ROCmFP4 and ROCmFP4-FAST matrix operations through Vulkan cooperative matrices (CM1) with F32 accumulation. It changes kernel execution, not the GGUF format, quantization, model weights, context window or chat template. It is separate from MTP and n-gram speculative decoding.

Both the build option and runtime selection are off by default. These instructions require a source revision containing `GGML_VULKAN_ROCMFP4_COOPMAT`; an older public binary does not acquire the feature by setting an environment variable. Check the source and startup message instead of assuming an unknown CMake argument enabled it.

## Build and select the path

Use the normal [Vulkan build prerequisites](../build.md#vulkan), including a GLSL compiler with KHR cooperative-matrix support. Keep the experimental build in a separate directory so the existing executable remains available.

```bash
cmake -S . -B build-vulkan-fp4-cm1 -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_HIP=OFF -DLLAMA_BUILD_UI=OFF -DGGML_VULKAN_ROCMFP4_COOPMAT=ON
cmake --build build-vulkan-fp4-cm1 --config Release -j 4 --target llama-server llama-cli llama-bench
```

| Selection | Before starting the model process |
| --- | --- |
| Enable the experimental path | `GGML_VK_ROCMFP4_COOPMAT=1` |
| Use the normal scalar path in the same build | `GGML_VK_ROCMFP4_COOPMAT=0`, or unset the variable |
| Exclude the experimental shaders from a build | `-DGGML_VULKAN_ROCMFP4_COOPMAT=OFF` |

Restart the model process after changing the runtime variable. For an enabled launch, confirm this log text:

```text
ROCmFP4 CM1 request: enabled (experimental, F32 accumulation)
```

`scalar fallback (build or device unsupported)` means the request did not activate this path. It requires CM1 and F32 accumulation support and does not replace the CM2 path. Validation was performed on AMD Strix Halo/gfx1151 with 128 GB unified memory. This is not a HIP optimization, a change to other ROCmFPX formats, or a performance guarantee for other GPUs.

## Worked example: Qwen3.8 Flash Next, split PLE16

The measured pair was:

- Main: `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf` (93,484,237,760 bytes).
- Matching native MTP head: `Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf` (2,444,519,296 bytes).

These identify the qualified artifacts, not a download URL or permission to redistribute weights. Obtain the matching files under their model license; renaming a different quant does not make it equivalent. Other mixtures or MTP files need their own measurements. The pair alone is about 89.34 GiB, before context, compute buffers and other services; do not start a second copy alongside an already resident model on a memory-constrained host.

The following is one Linux launch line. Replace both model paths, choose a free port, and run the server in a managed service or detached terminal. It binds only to localhost. This is the measured thinking-off performance profile, not the later custom-template/reasoning experiment.

```bash
RADV_PERFTEST=sam MALLOC_ARENA_MAX=2 GGML_VK_ROCMFP4_COOPMAT=1 ./build-vulkan-fp4-cm1/bin/llama-server -m "/path/to/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf" -md "/path/to/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf" --host 127.0.0.1 --port 8080 -dev Vulkan0 -ngl 99 -ngld 99 -c 262144 -np 1 -b 2048 -ub 512 -t 16 -tb 16 -fa on -ctk f16 -ctv f16 --fit off --jinja --reasoning off --reasoning-budget 0 --temp 0 --top-k 1 --seed 42 --repeat-penalty 1 --cache-reuse 0 --cache-ram 0 --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 --no-spec-draft-backend-sampling --spec-mtp-strict-qwen --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 64
```

For the scalar control, restart the same command with `GGML_VK_ROCMFP4_COOPMAT=0`; leave the remaining flags unchanged. This turns off CM1, not MTP or n-gram speculation. For a separate no-speculation comparison, use `--spec-type none` and remove the draft-model and other `--spec-*` arguments. Do not mix that experiment with the kernel on/off comparison.

`-c 262144` is the context capacity, not the output length. Each HTTP request has a separate response cap (`max_tokens` on the chat API or `n_predict` on `/completion`). A reasoning budget and custom chat template affect task behavior and latency independently; they do not inherit these throughput numbers automatically.

## Measured results, not a general chat speed promise

These are medians from three samples per kernel mode on the same qualified machine and files. The full sample values, workload parameters and warm-up prompts are in [the qualification data](benchmarks/qwen4exp-cm1-20260905.json).

| Workload / measurement | CM1 off | CM1 on |
| --- | ---: | ---: |
| Repetitive copy, 69 input / 384 output tokens: decode | 79.06 tok/s | 89.18 tok/s |
| Same short prompt: streamed time to first content | 0.772 s | 0.672 s |
| 8192-token uncached input / 128 output tokens: prefill | 300.45 tok/s | 346.30 tok/s |
| Same 8192-token input: streamed time to first content | 27.278 s | 23.668 s |

The copy test favors speculative acceptance, including the n-gram proposer. All six copy outputs had the same SHA256, but this does not prove general equivalence between kernel paths. The 8192-token outputs were not universally identical. Floating-point execution changes can change generated text, even when operator tests pass.

The older 106-107 tok/s results used a different experimental tree with additional Vulkan tuning. They were not reproduced by this isolated CM1 patch and are not advertised as its result. Ordinary chat, coding, long context and tool use can be substantially slower than repeated text.

### Repeating the copy workload

Use `/completion`, not chat-template wrapping, for this specific throughput measurement. Each recorded sample used a fresh server process, the four warm-ups in the data file (marker, count, code, prose; up to 256 output tokens each), then this request. The data file contains the exact JSON request. A direct request is:

```bash
curl --max-time 240 -N http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d '{"prompt":"Output only the passage below exactly eight times, preserving every word and punctuation mark. Put one blank line between copies. Do not add a title, explanation, numbering, or quotation marks.\n\nThe lighthouse keeper records the wind, checks the brass clock, closes the blue shutters, and carries a warm cup of tea upstairs before midnight.\n","n_predict":384,"temperature":0,"top_k":1,"seed":42,"repeat_penalty":1,"ignore_eos":true,"cache_prompt":false,"stream":true}'
```

Check the final timing event: `prompt_n=69`, `predicted_n=384`, `cache_n=0`. Decode is `predicted_per_second`, not output tokens divided by total wall time. Measure time to first nonempty content at the streaming client; server prompt-processing time is not the same measurement. Report all repeats, their median and the exact build identity.

`ignore_eos=true` and the fixed output cap are benchmark-only settings. The recorded output continued beyond the requested eight copies and ended at the cap, so this is a throughput probe, not a passed instruction-following task. Do not apply that setting to normal chat just to display a higher token count.

## Compatibility and rollback

Normal builds remain default-off, and an opt-in build still defaults to scalar execution without the runtime switch. The accompanying scalar-registration correction keeps ROCmFPX fallback shaders available independently of BF16 cooperative-matrix support; it does not alter BF16's own selection rule.

The existing upstream structure and commit ancestry remain intact. Vulkan updates can still require conflict resolution and requalification; no patch can guarantee conflict-free future merges. The reference CI gate compiles/links the option on and then off in the same directory to detect stale generated shaders. Physical GPU operator, model-quality and performance checks remain separate from CPU-only CI.

To roll back operationally, restart with the runtime variable unset or `0`, or return to the previous executable and launcher. Keep normal model weights unchanged. No Hermes/Telegram configuration, template, reasoning default or output cap is changed by this kernel feature.
