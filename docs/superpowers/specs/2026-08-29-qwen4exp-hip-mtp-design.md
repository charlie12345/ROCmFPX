# Qwen4exp HIP + MTP Design

## Goal

Add external Qwen3.8 Flash-Next MTP draft-head support to kingjones30/ROCmFPX on the Windows HIP backend for Radeon 8060S/gfx1151, while preserving the existing non-MTP performance and leaving the currently deployed HIP runtime untouched.

## Source baselines

- HIP/ROCmFPX target: `kingjones30/ROCmFPX` commit `36e9acd40e10a87cd3c3ef8ec734668757dc8520`.
- MTP reference implementation: `LaurentZuijdwijk/llama.cpp` commit `b98aa9847b0ced5d783e1b3287bc790de12f5fed`.
- Existing production comparison runtime: `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-gfx1151`.
- Existing Vulkan MTP runtime is reference-only and must not be copied over the HIP build.

## Architecture

Port only the runtime behavior from `b98aa9847`: qwen4exp draft tensor loading, the MTP graph, publication of the trunk hidden state to the draft context, and qwen4exp inclusion in the plain-attention MTP memory path. Resolve the patch semantically against the newer kingjones30 tree instead of merging the Laurent branch. Do not port `ae96a0dc0`, because the selected Q8 draft GGUF is already converted and contains the final draft block at `blk.48.*`.

The build remains a native Windows HIP build using TheRock Clang 23 and `gfx1151`. It is configured in a new build directory and packaged in a new llama-hub runtime directory only after source, executable, device, model-load, and HTTP generation checks pass.

## Models and fixed launch settings

- Main model: `C:\models\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf`.
- Draft model: `C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf`.
- Device: `ROCm0`.
- Context: `131072`.
- Flash attention: on.
- Batch / micro-batch: `2048 / 512`.
- Parallel slots: `1`.
- MTP mode: `--spec-type draft-mtp --spec-draft-n-max 3`.
- Generation: temperature `0`, prompt cache disabled, thinking disabled, 512 requested output tokens.
- Windows PLE control: apply the existing 8 GiB hard working-set limit after the server becomes healthy; this is held identical across comparison runs.

## Benchmark design

Build deterministic Chinese multi-turn conversation prompts from a fixed natural-language corpus. Tokenize through the target server and trim/pad each conversation to measured prompt lengths of approximately 1K, 2K, 4K, 8K, 16K, 32K, and 64K tokens. The final user turn asks for a structured synthesis requiring a long answer so the server normally reaches the 512-token generation cap.

Run three matrices under otherwise identical settings:

1. Existing HIP runtime, MTP disabled.
2. New HIP+MTP runtime, MTP disabled.
3. New HIP+MTP runtime, Q8 MTP enabled with `n=3`.

Warm each loaded server before measurement. Execute context lengths in ascending order with `cache_prompt=false`. Capture server-reported prompt tokens, prompt milliseconds/tokens per second, predicted tokens, predicted milliseconds/tokens per second, TTFT when available, wall time, stop reason, and speculative drafted/accepted counts from response timings, metrics, slots, or logs. Preserve raw requests, responses, and server logs.

## Acceptance criteria

- The new executable lists `ROCm0: AMD Radeon(TM) 8060S Graphics` and loads both the main model and the external `blk.48.*` Q8 MTP model without a missing-tensor error.
- A real chat completion succeeds with MTP disabled and enabled.
- The non-MTP weighted generation throughput of the new runtime is not below the existing HIP runtime. Any individual row slower by more than 3% is rerun; unresolved regressions block deployment as the recommended runtime.
- MTP improvement is reported per context and as token-weighted aggregate, together with acceptance rate. No speedup claim is made from startup logs or draft-token counts alone.
- The existing runtime directory is never overwritten. The new package uses `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-gfx1151`.

## Failure handling

- A compile conflict is resolved against the kingjones30 APIs, retaining its ROCmFPX operators and quantization paths.
- A missing tensor in the draft model is diagnosed against actual GGUF tensor names before changing tensor flags or indices.
- If the MTP server crashes or produces invalid tokens, preserve the logs and verify the same request without MTP before attributing the failure to the main model.
- If the 8 GiB working-set cap materially destabilizes a run, record that result and repeat both compared modes under one identical, explicitly stated memory policy.

## Deliverables

- A dedicated Git branch/worktree containing the minimal source port.
- A native Windows HIP/gfx1151 runtime package in the new llama-hub directory.
- A reproducible PowerShell benchmark harness and fixed prompt corpus under the benchmark results directory.
- Raw logs plus CSV/JSON summary for all 21 context/mode combinations and a concise conclusion.
