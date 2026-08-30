# Qwen4Exp Windows PLE Direct Pager Design

## Purpose

Add an opt-in Windows SSD path for the Qwen3.8/Qwen4Exp PLE n-gram table in the existing HIP + ROCmFPX + MTP llama.cpp runtime. The new path must prevent the 50.664 GiB PLE table from accumulating in the process working set while preserving model output and the existing HIP and MTP performance envelope.

This design does not replace the HIP backend, implement MoE expert paging, or change speculative decoding. It borrows three ideas from MoE4All: explicit positioned file reads, fixed memory budgets, and observable paging statistics.

## Current State

The target model file is 113.47 GiB. Its `per_layer_token_embd.weight` tensor is a 50.664 GiB Q8_0 matrix with shape `[160, 320001536]`. Qwen4Exp PLE uses 16 table rows per token because the model has n-gram size 3 and eight heads per n-gram order.

The current graph computes the row IDs on the host and calls `ggml_get_rows` against the complete PLE tensor. On Windows the complete GGUF is memory mapped. The `--direct-io` option reaches `llama_file`, but the Windows implementation opens a normal CRT file and ignores the direct-I/O request. `PrefetchVirtualMemory` may also request the whole mapped file during model loading.

The external working-set script limits the model process to 8 GiB and trims it after startup. This keeps the machine usable, but it is process-wide pressure rather than an explicit PLE storage policy.

## Design Goals

1. Read requested PLE rows directly from the GGUF on the NVMe drive without populating the Windows file cache.
2. Keep all pager-owned memory bounded and reported.
3. Preserve token identity against the existing mmap path under deterministic decoding.
4. Preserve the existing no-MTP throughput. A candidate with a meaningful no-MTP regression is not deployable.
5. Preserve external Q8 MTP support and prove real draft acceptance after the change.
6. Keep the current path unchanged unless the new option is explicitly enabled.
7. Fail clearly when direct PLE paging cannot be established; never silently claim direct mode while using buffered I/O.

## Non-Goals

- Paging MoE expert tensors between SSD, RAM, and GPU.
- Replacing ROCm/HIP with Vulkan.
- Fixing the separate MTP plus multimodal draft-context crash.
- Adding a multi-gigabyte persistent PLE cache. PLE hashes distribute requests over approximately 320 million rows, so reuse is expected to be too low to justify that RAM.
- Supporting non-Q8_0 PLE tensors in the first implementation.
- Enabling the feature by default for other architectures or model files.

## User Interface

The server and other common-argument applications gain these options:

```text
--ple-ssd {off,direct}
--ple-io-depth N
--ple-buffer-mib N
```

Defaults:

- `--ple-ssd off`
- `--ple-io-depth 32`
- `--ple-buffer-mib 32`

`direct` is accepted only on Windows for a Qwen4Exp model with a Q8_0 `per_layer_token_embd.weight`. Unsupported combinations terminate model loading with an actionable error. The existing mmap behavior remains the default.

The option is a target-model option only. The external MTP GGUF does not contain the trunk PLE table and continues to load normally.

## Components

### Model Parameters and Argument Plumbing

Add an internal PLE storage-mode enum plus I/O depth and memory budget to `llama_model_params` and `common_params`. Parse and validate the public options in `common/arg.cpp`, then copy them into model parameters in `common/common.cpp`.

Validation rejects zero I/O depth, buffers smaller than the minimum aligned working set, and values that overflow byte calculations. Direct mode remains opt-in.

### PLE Tensor Source Description

During Qwen4Exp tensor construction, inspect the original GGUF weight metadata before skipping normal PLE tensor materialization. Capture:

- the source-file index;
- the absolute tensor offset in that source file;
- tensor byte length;
- row count;
- row byte length;
- quantization type.

The loader exposes a narrow method that duplicates the source file handle and returns immutable source metadata. The pager never depends on the lifetime of `llama_model_loader`.

In direct mode, `per_layer_token_embd.weight` is created with `TENSOR_SKIP`, so it consumes neither a CPU tensor allocation nor a normal mmap-backed graph input. Its metadata remains available for bounds and type validation.

### Windows Direct Reader

Introduce a focused PLE reader implementation compiled on Windows. It opens the GGUF path or duplicates/reopens the source with:

```text
FILE_FLAG_NO_BUFFERING
FILE_FLAG_OVERLAPPED
FILE_FLAG_RANDOM_ACCESS
```

The reader queries the volume sector size and allocation granularity. Every read offset and length is aligned to the required sector size, and every destination comes from a page-aligned allocation. The reader verifies at startup that a small aligned read succeeds. Failure is fatal for requested direct mode.

For a requested PLE row, the reader calculates the absolute byte span, then the aligned sectors covering that span. A row may cross a sector boundary. All arithmetic is checked against the captured tensor and file bounds.

### Batch Read Planner

For each llama.cpp ubatch:

1. Reuse the existing PLE hash calculation to produce row IDs in model order.
2. Convert every row span to one or two aligned sector requests.
3. Sort requests by file offset and deduplicate identical sectors within the ubatch.
4. Submit at most `ple_io_depth` overlapped reads concurrently.
5. Copy each requested 170-byte Q8_0 row from completed sector buffers into a compact raw-row buffer.
6. Dequantize rows in original model order into the F32 PLE graph-input buffer.

The fixed arena contains only the overlapped sector buffers, request metadata, compact raw rows, and the F32 graph input. If the ubatch exceeds the configured budget, the planner processes it in bounded waves. It never allocates proportional to the 50.664 GiB table.

### Graph Integration

In mmap mode the graph remains unchanged:

```text
row IDs -> ggml_get_rows(complete Q8_0 PLE table) -> PLE embedding
```

In direct mode `llm_graph_input_ple` owns an F32 input tensor shaped to match the gathered PLE embedding. Its `set_input` implementation asks the pager to fill this tensor. The graph consumes that input directly and skips `ggml_get_rows` over the complete table.

The existing PLE history, EOS reset, image placeholder handling, convolution state, and downstream projections remain unchanged.

### Loading-Phase Page Cleanup

Direct mode disables whole-file mmap prefetch for the target model. After target weights have been uploaded and model initialization is complete, Windows performs a one-time working-set trim to release file pages touched while copying non-PLE weights to the GPU.

This is not a hard ongoing working-set limit. The existing 8 GiB script remains available as a fallback during validation. The direct PLE path itself must not require that script to prevent PLE residency growth.

### Statistics

The pager records and logs:

- PLE rows requested;
- aligned sectors requested and sectors deduplicated;
- bytes read from SSD;
- read operation count;
- read failures;
- cumulative and maximum read latency;
- time spent dequantizing;
- configured and peak arena bytes.

The final server timing log prints a compact PLE pager summary. Metrics are cumulative and thread-safe. No per-row log messages are emitted.

## Data Integrity and Error Handling

- Reject a row ID outside `[0, row_count)` before file I/O.
- Reject an aligned request outside the source file.
- Treat short reads, cancelled overlapped operations, or failed completion status as inference errors with the Windows error code and file offset.
- Keep the file identity stable for the process lifetime. Capture size and last-write timestamp at open and verify them after an I/O failure before reporting corruption.
- Do not fall back to mmap after direct mode has been announced. The request fails so users cannot mistake buffered execution for SSD-direct execution.
- The pager destructor cancels outstanding I/O, waits for completion, closes events and handles, and releases aligned buffers.

## Concurrency

The current target configuration has `--parallel 1`, but the pager must serialize mutation of its reusable arena and counters so a future multi-slot configuration is correct. A single pager instance belongs to the target model. Each fill operation holds an execution mutex for the bounded read/dequantization phase; no graph may reuse the arena until the previous fill completes.

This first version does not implement cross-request asynchronous prefetch. It uses overlap only within the set of rows already known for the current ubatch.

## Testing Strategy

### Unit Tests

Add tests that fail before implementation for:

- sector alignment when the PLE tensor starts at a non-aligned offset;
- rows fully inside one sector and rows crossing a boundary;
- sort and deduplication while preserving row output order;
- bounded wave planning when the request set exceeds the arena;
- checked rejection of out-of-range rows and overflow;
- Q8_0 row dequantization equivalence with the existing ggml conversion;
- cleanup after an injected read failure;
- parameter parsing and unsupported-platform/type errors.

Windows-only reader tests use a temporary aligned file with deterministic byte patterns and real overlapped unbuffered reads.

### Model Correctness Tests

Run the target model in mmap mode and direct mode with identical deterministic parameters. Compare generated token IDs for at least three prompts, including a prompt crossing multiple ubatches and a conversation containing EOS boundaries. All compared token sequences must match.

### Performance and Memory Tests

All GPU/model tests run serially. Before each run, stop competing llama-server processes and record the exact executable hash, command line, model hashes, disk identity, available RAM, and working-set policy.

Test prompt lengths 1K, 2K, 4K, 8K, 16K, 32K, and 64K. Generate a fixed number of output tokens using the same prompt corpus and sampling settings. Run three repetitions per point and report medians for:

- TTFT;
- prompt-processing throughput;
- token-generation throughput;
- process working set and private bytes;
- system available RAM;
- PLE bytes/read count/latency;
- MTP proposed and accepted tokens when enabled.

Run the matrix first without MTP and then with the existing Q8 MTP model at `n=3`.

## Acceptance Criteria

The candidate is deployable only if all of the following hold:

1. Unit, parser, and model correctness tests pass.
2. Deterministic direct-mode outputs match mmap-mode token IDs.
3. Direct mode is confirmed through startup logs and nonzero pager read statistics.
4. Without the external 8 GiB hard cap, working set remains at or below 6 GiB through the 64K run, or is no more than 1 GiB above the freshly measured capped baseline if non-PLE runtime buffers make 6 GiB unattainable.
5. The weighted no-MTP throughput across 1K through 64K is not below the freshly measured baseline. A repeatable per-point regression greater than 3% fails deployment unless it is traced to measurement noise and disappears in a five-run confirmation.
6. MTP mode reports `speculative=true`, accepts draft tokens, completes real generation, remains alive afterward, and does not regress more than the same measurement tolerance.
7. Model load time does not regress by more than 10%.
8. No access violation, HIP error, short read, silent fallback, or new warning appears in post-run logs.

If the candidate fails a deployment criterion, keep it on the optimization branch and leave the installed llama-hub runtime unchanged.

## Deployment and Rollback

Build into a new directory and install into a new llama-hub runtime directory. Do not overwrite `rocmfpx-qwen4exp-hip-mtp-gfx1151` during testing. Record SHA-256 hashes for the candidate binaries and DLLs.

After acceptance, update the llama-hub model configuration to select the new runtime and append the direct-PLE arguments. Keep the prior runtime directory and model configuration available for immediate rollback. The external working-set script remains available until the uncapped memory result has been reproduced in llama-hub, not only in a standalone server.

## Reference Boundary

MoE4All is a Rust, Vulkan-first engine. This implementation uses its architectural ideas but does not copy its Vulkan pager or MTP code. The Windows HIP implementation follows llama.cpp ownership, graph-input, logging, and argument patterns.
