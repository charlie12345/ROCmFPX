# Qwen4Exp Windows PLE Direct Pager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, genuinely unbuffered Windows SSD reader for the 50.664 GiB Qwen4Exp PLE table while preserving deterministic output, HIP throughput, and external Q8 MTP behavior.

**Architecture:** Keep the existing mmap graph as the default. In direct mode, capture the PLE tensor's GGUF source extent, skip normal tensor materialization, read only requested aligned sectors with bounded overlapped Windows I/O, dequantize Q8_0 rows into a small F32 graph input, and expose cumulative pager statistics. Disable whole-file prefetch and trim loading pages only for direct mode.

**Tech Stack:** C++17, llama.cpp/ROCmFPX, Win32 `CreateFileW`/`ReadFile` overlapped unbuffered I/O, ggml Q8_0 type traits, HIP 7.14, gfx1151, CMake/Ninja, PowerShell, OpenAI-compatible HTTP API.

**Spec:** `docs/superpowers/specs/2026-08-30-qwen4exp-windows-ple-direct-pager-design.md`

## Global Constraints

- Work only on branch `hip-qwen4exp-mtp-ple-pager`, based on `aff23ac` plus design commit `972521b`.
- Do not overwrite `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-gfx1151`.
- The existing mmap path is the default and must remain behaviorally unchanged when `--ple-ssd off`.
- Direct mode is Windows-only and Q8_0-only in this implementation; unsupported requests fail explicitly.
- Direct mode must use `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS`; no silent buffered fallback.
- Pager allocations are bounded by `--ple-buffer-mib`, with a default of 32 MiB.
- Preserve the existing Qwen4Exp PLE hash/history semantics and MTP changes through `aff23ac`.
- Compile and model tests are serialized. No other llama-server or GPU workload may overlap an A/B performance run.
- A runtime is deployable only when deterministic tokens match, uncapped working set meets the spec, weighted no-MTP throughput is not below baseline, and MTP accepts real draft tokens.

---

### Task 1: Add and Validate the PLE Storage Parameters

**Files:**
- Modify: `include/llama.h`
- Modify: `common/common.h`
- Modify: `common/common.cpp`
- Modify: `common/arg.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `tests/test-arg-parser.cpp`

**Interfaces:**
- Produces: `llama_ple_storage_type`, `llama_model_params::ple_storage`, `ple_io_depth`, and `ple_buffer_size`.
- Produces CLI: `--ple-ssd {off,direct}`, `--ple-io-depth N`, `--ple-buffer-mib N`.
- Defaults: off, 32 operations, 32 MiB.

- [ ] **Step 1: Write parser tests before adding production fields**

Add focused cases to `tests/test-arg-parser.cpp`:

```cpp
{
    common_params params;
    auto argv = list_str_to_char({"llama-server", "--ple-ssd", "direct",
                                  "--ple-io-depth", "64", "--ple-buffer-mib", "48"});
    assert(common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.ple_storage == LLAMA_PLE_STORAGE_DIRECT);
    assert(params.ple_io_depth == 64);
    assert(params.ple_buffer_size == 48ull*1024*1024);
}
```

Add separate failure cases for `--ple-ssd invalid`, I/O depth zero, and a buffer below 1 MiB.

- [ ] **Step 2: Run the parser target and verify RED**

Use the existing test-capable build and run:

```powershell
cmake --build C:\Users\james\OneDrive\文档\llamacpp\build-kingjones30-qwen4exp-hip-mtp --target test-arg-parser -j 8
```

Expected: compilation fails because the PLE parameter fields and enum do not exist.

- [ ] **Step 3: Implement the minimal parameter plumbing**

Add the public enum and fields:

```cpp
enum llama_ple_storage_type {
    LLAMA_PLE_STORAGE_OFF    = 0,
    LLAMA_PLE_STORAGE_DIRECT = 1,
};

enum llama_ple_storage_type ple_storage;
uint32_t                    ple_io_depth;
size_t                      ple_buffer_size;
```

Mirror these in `common_params`, register the three common/server arguments, reject invalid values in the argument callbacks, copy them in `common_model_params_to_llama`, and initialize defaults in `llama_model_default_params()`.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run `test-arg-parser.exe` directly or through CTest. Require all existing parser cases and the four new PLE cases to pass.

- [ ] **Step 5: Commit the parameter surface**

Commit with message `feat: add qwen4exp PLE storage options`.

### Task 2: Implement the Pure PLE Read Planner and Q8_0 Conversion

**Files:**
- Create: `src/llama-ple-pager.h`
- Create: `src/llama-ple-pager.cpp`
- Modify: `src/CMakeLists.txt`
- Create: `tests/test-ple-pager.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `llama_ple_source`, `llama_ple_read_plan`, `llama_ple_pager_stats`.
- Produces: `llama_ple_plan_rows(...)` and `llama_ple_dequantize_q8_0_rows(...)` for production and unit tests.

- [ ] **Step 1: Add planner tests and declarations first**

Create `tests/test-ple-pager.cpp` with deterministic assertions for the real geometry:

```cpp
const llama_ple_source source {
    /* tensor_offset = */ 547040384,
    /* tensor_bytes  = */ 54400261120,
    /* row_count     = */ 320001536,
    /* row_elements  = */ 160,
    /* row_bytes     = */ 170,
    /* alignment     = */ 4096,
};

const auto plan = llama_ple_plan_rows(source, {0, 5, 5, 6}, 32*1024*1024);
assert(plan.rows.size() == 4);
assert(plan.sectors.size() < 5); // repeated and neighbouring rows deduplicate sectors
assert(plan.peak_bytes <= 32*1024*1024);
```

Add tests for a boundary-crossing row, `row_count`, `UINT64_MAX` overflow, one-sector budget waves, and preservation of duplicate row output order.

Quantize one 160-float row with ggml Q8_0 traits, dequantize through the new helper, and compare with the type trait output element by element.

- [ ] **Step 2: Verify RED**

Register `test-ple-pager` with `llama_build_and_test`, add `${PROJECT_SOURCE_DIR}/src` to its include path, and build it. Expected: unresolved pager declarations or missing implementation.

- [ ] **Step 3: Implement checked planning**

Implement safe helpers:

```cpp
bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out);
bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out);
uint64_t align_down(uint64_t value, uint64_t alignment);
uint64_t align_up_checked(uint64_t value, uint64_t alignment);
```

For each row, calculate the absolute byte span, split it into aligned sectors, sort/deduplicate sector offsets, retain row-to-sector slice references, and partition the plan into waves whose live sector buffers do not exceed the configured budget.

- [ ] **Step 4: Implement Q8_0 dequantization**

Use `ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float` for each compact 170-byte row. Reject null traits, wrong row size, and destination size mismatch with exceptions that include the row geometry.

- [ ] **Step 5: Verify GREEN and commit**

Run `test-ple-pager` and `test-arg-parser`, then `git diff --check`. Commit with message `feat: plan bounded PLE row reads`.

### Task 3: Implement Real Windows Unbuffered Overlapped I/O

**Files:**
- Modify: `src/llama-ple-pager.h`
- Modify: `src/llama-ple-pager.cpp`
- Modify: `tests/test-ple-pager.cpp`

**Interfaces:**
- Produces: `llama_ple_pager::open_from_file_id(...)`.
- Produces: `llama_ple_pager::read_rows(const std::vector<int32_t> &, float *, size_t)`.
- Owns: reopened direct handle, aligned arena, events/OVERLAPPED structures, execution mutex, and atomic statistics.

- [ ] **Step 1: Write the Windows integration test**

Under `_WIN32`, create a normal temporary file on the system drive, fill a multiple-of-sector-size payload with a deterministic ramp, close the writer, reopen through `open_from_file_id`, request rows that are duplicated and cross sectors, and assert exact reconstructed bytes/dequantized floats. Add an injected invalid row test and verify the next valid call still succeeds.

- [ ] **Step 2: Verify RED on Windows**

Build and run `test-ple-pager`. Expected: the direct reader entry point is absent.

- [ ] **Step 3: Implement strict direct-handle setup**

From the loader's CRT file descriptor, use `_get_osfhandle`, `GetFinalPathNameByHandleW`, and `CreateFileW` with:

```cpp
GENERIC_READ,
FILE_SHARE_READ,
OPEN_EXISTING,
FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS
```

Query alignment using `GetFileInformationByHandleEx(FileAlignmentInfo)` and retain file size and last-write time. Allocate arena memory with `VirtualAlloc`, which satisfies page and sector alignment. Perform one aligned startup probe and fail construction on any error.

- [ ] **Step 4: Implement bounded overlapped waves**

Create at most `ple_io_depth` reusable event/OVERLAPPED lanes. Submit one aligned sector per free lane, accept both synchronous completion and `ERROR_IO_PENDING`, wait with `GetOverlappedResult`, then scatter row fragments into the compact row buffer. Cancel and drain outstanding operations before propagating an error.

- [ ] **Step 5: Add statistics and cleanup**

Update rows, sectors, deduplicated sectors, bytes, operations, failures, cumulative/max read microseconds, dequant microseconds, and peak arena bytes. In the destructor call `CancelIoEx`, drain operations, close every event and handle, and `VirtualFree` the arena.

- [ ] **Step 6: Verify GREEN and commit**

Run the real Windows direct-reader test repeatedly (`ctest -R test-ple-pager --repeat until-fail:20`) and require zero handle leaks or intermittent failures. Commit with message `feat: read PLE rows with Windows direct IO`.

### Task 4: Connect the Pager to Qwen4Exp Tensor Loading and the Graph

**Files:**
- Modify: `src/llama-model-loader.h`
- Modify: `src/llama-model-loader.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `src/models/models.h`
- Modify: `src/models/qwen4exp.cpp`
- Create: `scripts/tests/qwen4exp-ple-direct-smoke.ps1`

**Interfaces:**
- Produces: `llama_model_loader::get_tensor_source(const char *)` returning file descriptor, source index, absolute offset, length, shape, and type while the loader is alive.
- `llama_model_qwen4exp` owns `std::unique_ptr<llama_ple_pager> ple_pager`.
- Direct graph input is F32 `[ple_head_dim * ple_n_heads, n_tokens]`.

- [ ] **Step 1: Add the failing real-model smoke script**

The script launches a supplied server on an owned disposable port with `--ple-ssd direct`, waits for health, sends a deterministic 32-token completion, requires a `PLE direct pager active` log line, then stops only its child PID. Run it against the current binary and preserve the expected unknown-argument failure.

- [ ] **Step 2: Expose immutable tensor source metadata**

Add:

```cpp
struct llama_tensor_source {
    int         file_id;
    uint16_t    file_index;
    uint64_t    offset;
    uint64_t    size;
    ggml_type   type;
    int64_t     ne0;
    int64_t     ne1;
};
```

`get_tensor_source` reads the existing `weights_map` entry and its ggml tensor metadata. It does not transfer ownership; the pager reopens the file immediately.

- [ ] **Step 3: Skip only the Qwen4Exp PLE tensor in direct mode**

In `llama_model_qwen4exp::load_arch_tensors`, inspect and validate `per_layer_token_embd.weight`, construct the pager from its source, then call `create_tensor` with `TENSOR_SKIP`. Leave `per_layer_tok_embd` null in direct mode. Mmap mode retains the current code exactly.

Require shape `[160, 320001536]` only through model metadata values rather than hard-coding the row count; require `GGML_TYPE_Q8_0`, `ne0 == ple_head_dim`, and `ne1` equal to the metadata-derived table rows.

- [ ] **Step 4: Replace the direct-mode graph gather with a small F32 input**

Extend `llm_graph_input_ple` with `ggml_tensor * embeddings`. In direct mode create:

```cpp
embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
        hparams.ple_head_dim * hparams.ple_n_heads, n_tokens);
ggml_set_input(embeddings);
```

After the existing hash/history loop, call the pager with the computed row IDs and upload the gathered F32 data through `ggml_backend_tensor_set`. `build_ple` consumes `embeddings` directly. The old `rows` input and `ggml_get_rows` remain exclusive to mmap mode.

- [ ] **Step 5: Disable whole-file prefetch for direct mode**

Change the model loader call to:

```cpp
ml.init_mappings(params.ple_storage != LLAMA_PLE_STORAGE_DIRECT,
                 use_mlock ? &pimpl->mlock_mmaps : nullptr);
```

Reject `use_mlock` combined with direct PLE mode because locking the mapped model contradicts the SSD policy.

- [ ] **Step 6: Build and run the first GREEN model smoke**

Build `llama-server`, run the direct smoke script on the real model without MTP at 1K context, and require coherent output, direct-pager startup evidence, nonzero rows/bytes at shutdown, and no crash.

- [ ] **Step 7: Commit graph integration**

Run source tests and `git diff --check`. Commit with message `feat: page qwen4exp PLE directly from SSD`.

### Task 5: Add Loading-Page Trim, Stable Statistics, and Regression Tests

**Files:**
- Modify: `src/llama-ple-pager.h`
- Modify: `src/llama-ple-pager.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `src/models/qwen4exp.cpp`
- Modify: `scripts/tests/qwen4exp-ple-direct-smoke.ps1`
- Create: `scripts/tests/qwen4exp-ple-token-equivalence.ps1`

**Interfaces:**
- Produces: one-time `llama_trim_process_working_set()` on Windows direct mode.
- Produces: stable startup/shutdown summary fields suitable for log parsing.

- [ ] **Step 1: Extend smoke assertions before implementation**

Require the log to contain:

```text
PLE direct pager active: alignment=... io_depth=... budget_mib=...
PLE direct pager stats: rows=... sectors=... dedup=... bytes=... reads=... failures=0 ...
```

Also require the process working set to fall after loading and remain below the script's configurable smoke threshold.

- [ ] **Step 2: Verify RED**

Run the smoke against the Task 4 binary. Expected: the final field set or working-set trim assertion is missing.

- [ ] **Step 3: Implement one-time Windows trim**

After successful model data loading in direct mode, call `EmptyWorkingSet(GetCurrentProcess())`, log success or a Win32 warning, and never install a hard cap. Keep this helper a no-op on non-Windows builds.

- [ ] **Step 4: Implement stable summary formatting**

Snapshot atomic counters in a value struct and emit one startup line and one destruction line. Use integer bytes/microseconds and fixed field names; avoid localized number formatting.

- [ ] **Step 5: Add deterministic token-equivalence test**

Start the same candidate binary once with `--ple-ssd off` and once with `--ple-ssd direct`. Submit three saved temperature-zero, seed-42, `cache_prompt=false` requests and compare returned token ID arrays, not only text. Include a multi-ubatch prompt and EOS-separated conversation. Require exact equality.

- [ ] **Step 6: Verify GREEN and commit**

Run unit tests, smoke, and token equivalence. Commit with message `test: verify qwen4exp direct PLE correctness`.

### Task 6: Produce the Isolated HIP/gfx1151 Candidate Build

**Files:**
- Create build directory: `C:\Users\james\OneDrive\文档\llamacpp\build-qwen4exp-hip-mtp-ple-pager`
- Create test package: `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-ple-pager-gfx1151-test`

**Interfaces:**
- Produces Release `llama-server.exe`, `llama-cli.exe`, tests, matching HIP DLLs, and rocBLAS gfx1151 data.

- [ ] **Step 1: Configure a fresh HIP-only build**

Use Ninja, `C:\TheRock\build`, AMD Clang 23, `GGML_HIP=ON`, `GGML_VULKAN=OFF`, `GPU_TARGETS=gfx1151`, `CMAKE_HIP_ARCHITECTURES=gfx1151`, `GGML_HIP_ROCWMMA_FATTN=OFF`, `LLAMA_CURL=OFF`, and `LLAMA_BUILD_TESTS=ON`. Preserve configure output.

- [ ] **Step 2: Compile serially with bounded CPU parallelism**

Build `llama-server`, `llama-cli`, `test-arg-parser`, and `test-ple-pager` with `-j 8`. Preserve complete logs and stop on the first compile error.

- [ ] **Step 3: Run static and unit gates**

Require clean `--version`, `--list-devices` showing ROCm0 Radeon 8060S, parser tests, pager tests, repeated Windows I/O tests, and no `git diff --check` errors.

- [ ] **Step 4: Assemble an isolated test package**

Copy only required executables, matching TheRock DLLs, and `rocblas\library`. Record SHA-256 for every copied file and verify no file under the existing installed runtime changed.

- [ ] **Step 5: Run packaged smoke gates**

From the test package run mmap no-MTP, direct no-MTP, and direct MTP n=3 smoke tests. For MTP require `/slots` speculative state, nonzero drafted and accepted counters, real output, post-request health, and clean crash-log inspection.

### Task 7: Extend the Benchmark for Direct PLE Memory and I/O Evidence

**Files:**
- Create: `scripts/benchmark-qwen4exp-ple-pager.ps1`
- Create: `scripts/tests/benchmark-qwen4exp-ple-pager.tests.ps1`
- Reuse: `scripts/data/qwen4exp-conversation-corpus-zh.txt`
- Generate: `test-logs/qwen4exp-ple-pager-<timestamp>`

**Interfaces:**
- Modes: `baseline-mmap-capped`, `candidate-mmap-capped`, `candidate-direct-uncapped`, `candidate-direct-mtp-q8-n3-uncapped`.
- Prompt targets: 1024, 2048, 4096, 8192, 16384, 32768, 65536.
- Three primary repetitions per mode and prompt target.

- [ ] **Step 1: Write harness tests before the benchmark implementation**

Test argument validation, owned-process stopping, exact saved-prompt reuse, three-repeat aggregation, 3% rerun detection, pager-log parsing, token-weighted PP/TG, and the rule that only the baseline/candidate mmap modes receive the 8 GiB external cap.

- [ ] **Step 2: Verify RED**

Run the PowerShell test script. Expected: benchmark functions or direct-mode fields are missing.

- [ ] **Step 3: Implement the four-mode lifecycle**

Adapt the existing MTP benchmark's safe listener ownership and fixed corpus. Save executable hashes, complete arguments, model hashes, disk identity, working-set policy, raw request/response JSON, stdout/stderr, and listener state before and after every child.

- [ ] **Step 4: Capture memory and I/O evidence**

At startup, after warmup, before each request, after each request, and before shutdown, record process working set/private bytes, `Win32_Process.ReadOperationCount`, `ReadTransferCount`, and system available RAM. Parse the final PLE summary and require direct modes to report nonzero rows and bytes with zero failures.

- [ ] **Step 5: Implement acceptance calculations**

Calculate three-run medians per row, weighted PP/TG, direct-vs-baseline percentages, candidate-mmap-vs-baseline control, maximum working set, load time, and MTP accepted/drafted totals. Flag a row beyond 3% and run a five-run paired confirmation before adjudication.

- [ ] **Step 6: Verify GREEN cheaply and commit**

Run syntax/unit tests and a 1K/32-output smoke matrix. Commit with message `test: benchmark Windows direct PLE paging`.

### Task 8: Run the Full A/B Matrix and Decide Deployment

**Files:**
- Generate timestamped JSON, CSV, logs, hashes, and summary under `test-logs/qwen4exp-ple-pager-<timestamp>`.

**Interfaces:**
- Consumes the installed current runtime and isolated candidate package.
- Produces an acceptance verdict for correctness, memory, load time, no-MTP performance, and MTP behavior.

- [ ] **Step 1: Establish a clean device baseline**

Record all llama-server processes and listeners. Stop only the currently active llama-hub model process when the benchmark is ready to begin; leave llama-hub Java running. Confirm no other GPU workload is active.

- [ ] **Step 2: Run all four modes serially**

Run all seven prompt lengths three times per mode with identical saved prompts, 512 requested output tokens, temperature zero, seed 42, `cache_prompt=false`, batch 2048, ubatch 512, parallel 1, and matching context/checkpoint settings.

- [ ] **Step 3: Confirm or reject flagged performance rows**

For every direct no-MTP row more than 3% below baseline, run five paired baseline/direct repetitions using the same request. Preserve primary and confirmation values separately.

- [ ] **Step 4: Apply every acceptance criterion**

Require exact token equivalence, direct-mode evidence, maximum uncapped direct working set at or below 6 GiB or within the spec's 1 GiB fallback band, weighted no-MTP throughput not below baseline, load regression at or below 10%, nonzero MTP acceptance, server liveness, and clean logs.

- [ ] **Step 5: Produce the final report**

Report PP, TG, TTFT, load time, working set, available RAM, SSD bytes/read latency, and MTP acceptance for every context. Distinguish measured benefit from expectation and state explicitly whether the candidate passed deployment.

### Task 9: Deploy Only a Passing Candidate and Verify in llama-hub

**Files:**
- Create on pass: `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-ple-pager-gfx1151`
- Preserve: `C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-gfx1151`
- Modify on pass: the llama-hub model runtime/extra-argument configuration selected through its normal configuration path.

**Interfaces:**
- Produces a rollback-safe llama-hub runtime using `--ple-ssd direct --ple-io-depth 32 --ple-buffer-mib 32`.

- [ ] **Step 1: Gate deployment**

If any Task 8 criterion fails, do not modify llama-hub configuration. Keep the candidate branch/package and report the failed measurements.

- [ ] **Step 2: Install and hash-check a passing build**

Copy the verified test package to the final new runtime directory. Compare every executable/DLL hash with the tested package and verify the old runtime hashes remain unchanged.

- [ ] **Step 3: Select the runtime in llama-hub**

Use I/O depth 32 and a 32 MiB pager budget, the values covered by the full acceptance matrix. Keep the existing main model, Q8 MTP path, n=3, context, batch, ubatch, and parallel settings. Remove the external 8 GiB helper only after uncapped behavior is reproduced through llama-hub.

- [ ] **Step 4: Verify the actual hub path**

Start the MTP profile through llama-hub, inspect the live process executable and command line, wait for model-port health, send one real text request, verify accepted drafts, inspect logs, and confirm the process remains alive. Retain the packaged standalone no-MTP verification from Task 6 as the no-MTP deployment gate.

- [ ] **Step 5: Reproduce the memory claim in llama-hub**

Run at least an 8K and 64K saved request through the hub path without applying the external hard cap. Record working set, available RAM, pager statistics, PP/TG, and liveness. If the hub path violates the memory gate, re-enable the 8 GiB helper and mark uncapped deployment unsuccessful.

- [ ] **Step 6: Complete the audit**

Run the verification-before-completion checklist, re-read the design and plan, list evidence for every acceptance criterion, provide exact source/build/package/results paths, commit final source-controlled scripts or documentation, and report rollback instructions.
