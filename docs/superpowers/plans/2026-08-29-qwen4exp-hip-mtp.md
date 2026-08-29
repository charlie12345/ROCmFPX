# Qwen4exp HIP + MTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and benchmark a native Windows HIP/gfx1151 kingjones30/ROCmFPX runtime that supports the external Qwen3.8 Flash-Next MTP draft head without slowing its non-MTP path.

**Architecture:** Semantically port Laurent commit `b98aa9847` onto kingjones30 commit `36e9acd`, retaining the target tree's ROCmFPX implementation. Build and package in new directories, then compare the existing and new binaries with one deterministic natural Chinese multi-turn workload at seven prompt lengths.

**Tech Stack:** C++17, llama.cpp/ROCmFPX, CMake/Ninja, TheRock Clang/HIP 7.14, gfx1151, PowerShell, OpenAI-compatible HTTP API, GGUF.

**Spec:** `docs/superpowers/specs/2026-08-29-qwen4exp-hip-mtp-design.md`

## Global Constraints

- Base source is exactly `36e9acd40e10a87cd3c3ef8ec734668757dc8520`; reference runtime patch is exactly `b98aa9847b0ced5d783e1b3287bc790de12f5fed`.
- Do not merge the Laurent Vulkan branch and do not port conversion commit `ae96a0dc0`.
- Do not alter or overwrite `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-gfx1151`.
- Build HIP only for `gfx1151` with `GGML_HIP_ROCWMMA_FATTN=OFF`.
- Main model, Q8 MTP model, context/batch settings, generation settings, and 8 GiB working-set policy must match the design spec exactly.
- Preserve raw logs and JSON results. Every speed claim must come from server timings for a real completion.
- New non-MTP weighted TG must not be below the existing HIP weighted TG. Rerun rows with more than 3% apparent regression before deciding.
- Package only after version, device, no-MTP generation, MTP generation, and benchmark gates pass.

---

### Task 1: Port qwen4exp MTP Runtime Support

**Files:**
- Modify: `src/llama-model.cpp`
- Modify: `src/models/models.h`
- Modify: `src/models/qwen4exp.cpp`
- Reference: `C:\Users\james\OneDrive\文档\llamacpp\LaurentZuijdwijk-llama.cpp` commit `b98aa9847`

**Interfaces:**
- Consumes: qwen4exp trunk graph, `LLM_GRAPH_TYPE_DECODER_MTP`, `llama_model_loader::load_mtp`, and the existing nextn tensor naming API.
- Produces: qwen4exp MTP-only tensor loading, one-block draft graph, trunk `t_h_nextn`, and MTP-compatible memory selection.

- [ ] **Step 1: Record the expected pre-port failure**

Run the existing HIP binary with the main and Q8 draft GGUF on a disposable port and preserve the `missing tensor 'blk.0.hc_attn_norm.weight'` failure in the task report.

- [ ] **Step 2: Compare the target APIs with the reference patch**

Run `git show b98aa9847 -- src/llama-model.cpp src/models/models.h src/models/qwen4exp.cpp` in the Laurent repository and map every changed symbol to the target tree. Record any target-only behavior that must be preserved.

- [ ] **Step 3: Implement the minimal semantic port**

Add qwen4exp to the MTP-on-hybrid-Qwen memory condition; load `LLM_KV_NEXTN_PREDICT_LAYERS`; detect an MTP-only file by the absence of `blk.0.hc_attn_norm.weight`; make trunk tensors optional for that file; load the draft block at `hparams.n_layer()` with nextn tensors; dispatch `LLM_GRAPH_TYPE_DECODER_MTP` to a one-block qwen4exp draft graph; publish the trunk hyper-connection state as `res->t_h_nextn`. Preserve the kingjones30 PLE and ROCmFPX code unchanged outside these integration points.

- [ ] **Step 4: Review the source delta**

Run `git diff --check`, inspect `git diff --stat`, and compare the resulting semantics line-by-line with `b98aa9847`. No converter, Vulkan, quantization, or unrelated source files may be changed.

- [ ] **Step 5: Commit the port**

Commit the three runtime source files with message `feat: add qwen4exp MTP draft support`.

### Task 2: Build and Validate the HIP/gfx1151 Binary

**Files:**
- Create: `C:\Users\james\OneDrive\文档\llamacpp\build-kingjones30-qwen4exp-hip-mtp`
- Test: `bin\llama-server.exe`, `bin\llama-cli.exe`, `bin\llama-bench.exe`

**Interfaces:**
- Consumes: Task 1 source and TheRock toolchain.
- Produces: Release HIP executables plus matching runtime DLLs and rocBLAS gfx1151 data.

- [ ] **Step 1: Configure a fresh build**

Run from PowerShell with `ROCM_PATH=C:\TheRock\build`, an empty `HIP_PATH`, Ninja, `CMAKE_C_COMPILER=C:/TheRock/build/lib/llvm/bin/clang.exe`, `CMAKE_CXX_COMPILER=C:/TheRock/build/lib/llvm/bin/clang++.exe`, `CMAKE_BUILD_TYPE=Release`, `GGML_HIP=ON`, `GGML_VULKAN=OFF`, `GPU_TARGETS=gfx1151`, `CMAKE_HIP_ARCHITECTURES=gfx1151`, `GGML_HIP_ROCWMMA_FATTN=OFF`, and `LLAMA_CURL=OFF`.

- [ ] **Step 2: Compile bounded parallel targets**

Run `cmake --build <build-dir> --config Release --target llama-server llama-cli llama-bench -j 8`. Preserve the complete configure and build logs.

- [ ] **Step 3: Supply the matching HIP runtime**

Copy the same TheRock runtime DLL set used by the verified HIP4 build and copy `C:\TheRock\build\lib\rocblas\library` into `bin\rocblas\library`. Hash-check `amdhip64_7.dll`, `amd_comgr.dll`, and `rocm_kpack.dll` against `C:\TheRock\build\bin`.

- [ ] **Step 4: Run static/device gates**

Run `llama-server.exe --version`, `llama-cli.exe --version`, and `llama-cli.exe --list-devices`; require the branch commit and non-zero `ROCm0` Radeon 8060S capacity.

- [ ] **Step 5: Commit build documentation changes if any**

Do not commit generated build products. Commit only source-controlled build scripts or notes created to make the build reproducible.

### Task 3: Prove Main-Model and External-MTP Runtime Behavior

**Files:**
- Create: `C:\Users\james\OneDrive\文档\llamacpp\test-logs\qwen4exp-hip-mtp-smoke`
- Consume: main and Q8 MTP GGUF paths from the spec.

**Interfaces:**
- Consumes: Task 2 runtime.
- Produces: healthy no-MTP and MTP servers, coherent chat completions, and speculative statistics.

- [ ] **Step 1: Select a free disposable port and record owners**

Inspect listeners and stop only a llama-server process started by this task. Never terminate llama-hub or an unrelated server.

- [ ] **Step 2: Start and test without MTP**

Launch the new server with the fixed 131072/2048/512/parallel-1 HIP settings, wait for `/health`, apply the 8 GiB working-set cap, and send one temperature-0 Chinese chat request with `cache_prompt=false`.

- [ ] **Step 3: Start and test with Q8 MTP n=3**

Restart with `--spec-type draft-mtp --spec-draft-model <Q8> --spec-draft-n-max 3`, wait for health, apply the same memory cap, and send the same request.

- [ ] **Step 4: Validate speculative decoding evidence**

Require successful `blk.48.*` draft loading, `speculative: true` or equivalent slot/log state, non-zero drafted tokens, and a real completion. Record accepted/drafted counts and acceptance rate.

- [ ] **Step 5: Preserve and stop the smoke server**

Save stdout/stderr, request/response JSON, process command lines, and memory counters, then stop only the disposable server and confirm the port is released.

### Task 4: Implement the Reproducible Conversation Benchmark

**Files:**
- Create: `scripts/benchmark-qwen4exp-hip-mtp.ps1`
- Create: `scripts/data/qwen4exp-conversation-corpus-zh.txt`
- Create at runtime: `C:\Users\james\OneDrive\文档\llamacpp\test-logs\qwen4exp-hip-mtp-benchmark-<timestamp>`

**Interfaces:**
- Consumes: existing HIP runtime, new HIP runtime, fixed model/settings, fixed corpus.
- Produces: exact-length multi-turn prompts, raw logs/responses, JSONL and CSV summaries, rerun support.

- [ ] **Step 1: Add argument and safety validation**

The script accepts runtime/mode/output paths but defaults to the exact spec paths. It validates files, records listener ownership, uses a task-specific port, and stops only its own child PID.

- [ ] **Step 2: Add deterministic conversation construction**

Build alternating Chinese user/assistant turns from the fixed corpus. Use the server tokenizer endpoint or token-count response to converge within 1% of targets `1024, 2048, 4096, 8192, 16384, 32768, 65536`; append a fixed final synthesis request.

- [ ] **Step 3: Add server lifecycle and memory policy**

Start each of `existing-no-mtp`, `new-no-mtp`, and `new-mtp-q8-n3` once; wait for health; apply and verify the 8 GiB hard working-set limit; warm the server; then run ascending contexts with prompt caching disabled.

- [ ] **Step 4: Capture trustworthy timings and MTP evidence**

Store actual prompt/predicted token counts, prompt/predicted milliseconds and rates, TTFT/wall time, finish reason, output, drafted/accepted counts, working set/private bytes, and log paths. Prefer the largest valid cumulative timing sample and retain `predicted_ms` as fallback.

- [ ] **Step 5: Add comparison and rerun logic**

Calculate per-row new-vs-existing no-MTP and MTP-vs-new-no-MTP percentages plus token-weighted TG/PP. Flag more than 3% no-MTP regression and support rerunning only flagged rows before final aggregation.

- [ ] **Step 6: Validate the harness cheaply**

Run script syntax parsing, `-WhatIf`/validation mode, and a short 1K/32-output dry benchmark before starting the full matrix. Commit the script and corpus with message `test: add qwen4exp HIP MTP benchmark`.

### Task 5: Run the 1K-64K Benchmark and Adjudicate Regression

**Files:**
- Generate: timestamped raw logs, JSONL, CSV, and summary JSON under the benchmark output directory.

**Interfaces:**
- Consumes: Task 4 harness and both runtime packages.
- Produces: 21 primary measurements plus targeted reruns and final comparison.

- [ ] **Step 1: Establish idle baseline**

Confirm no unrelated llama-server occupies the GPU test environment, record GPU/system memory and listener state, and avoid running other GPU workloads during the matrix.

- [ ] **Step 2: Run existing HIP without MTP**

Measure 1K, 2K, 4K, 8K, 16K, 32K, and 64K prompts with 512 requested output tokens.

- [ ] **Step 3: Run new HIP without MTP**

Repeat the exact generated requests and settings against the new runtime.

- [ ] **Step 4: Rerun potential non-MTP regressions**

For every row where the new runtime is more than 3% slower, rerun both old and new binaries for that same saved request. Use the rerun pair for adjudication and preserve both original values.

- [ ] **Step 5: Run new HIP with Q8 MTP n=3**

Repeat the seven saved requests, capturing TG/PP and accepted/drafted counts.

- [ ] **Step 6: Produce the final benchmark summary**

Report each context's three TG/PP values, no-MTP regression percentage, MTP speedup percentage, acceptance rate, finish reason, and actual token counts. Also report weighted aggregate and explicitly state any failed row.

### Task 6: Package, Deploy in Isolation, and Audit Completion

**Files:**
- Create: `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-gfx1151`
- Create: `README-HIP-MTP.txt` and `set-llama-working-set-8g.ps1` in the package.

**Interfaces:**
- Consumes: verified build and benchmark verdict.
- Produces: reusable llama-hub runtime that does not replace the existing HIP runtime.

- [ ] **Step 1: Gate packaging on all required checks**

Require successful build, ROCm0 enumeration, main-model health, external-MTP health, real MTP statistics, and no unresolved non-MTP weighted regression.

- [ ] **Step 2: Copy and hash-check the package**

Copy executables, dependent DLLs, and rocBLAS gfx1151 data into the new directory. Compare SHA-256 hashes with build outputs and TheRock sources.

- [ ] **Step 3: Add runtime notes and working-set helper**

Document the exact no-MTP and MTP llama-hub arguments, the required `rocm0` device name, Q8 draft path, `n=3`, and the fact that the 8 GiB limiter must be reapplied after every restart.

- [ ] **Step 4: Verify from the deployed directory**

Run version, device listing, health, one no-MTP completion, and one MTP completion using the packaged executable. Confirm the old runtime still exists and its hashes were not changed.

- [ ] **Step 5: Complete the requirement audit**

Re-read the design and this plan, list every acceptance criterion with fresh evidence, and provide exact source/build/deployment/results paths.
