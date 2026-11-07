# ROCmFP4 Upstream Integration Tree

> **Note (2026-11-07):** Historical: the legacy ROCmFPX integration workspace on
> a Linux Strix Halo machine (May-June 2026, llama.cpp `b9438`); its paths,
> branch and build directory are not part of this RDNA-FAST checkout. The
> Step 3.7 scripts below need updating before they run: `--mmap`/`--no-mmap`
> gave way to `--load-mode` (2026-10-02), and the `step35` pre-tokenizer alias
> is gone (2026-09-15), so such files now need
> `--override-kv tokenizer.ggml.pre=str:deepseek-v3`.

This checkout is the active upstream-integration workspace.

## Do Not Confuse These Trees

| Purpose | Path |
|---|---|
| Protected clean ROCmFP4 package | `/home/caf/strix-fp4/rocmfp4` |
| Previous ROCmFP4 working fork | `/home/caf/strix-fp4/llama.cpp-mtp-rocmfp4` |
| Active upstream integration | `/home/caf/strix-fp4/llama.cpp-rocmfp4-upstream-integration` |

The protected clean package and previous working fork must remain untouched while
this integration is validated.

## Integration Baseline

- Branch: `rocmfp4-upstream-b9438-integration`
- Upstream baseline: official llama.cpp `b9438`, commit `22cadc194`
- Default build directory: `build-strix-rocmfp4-upstream-integration`
- Build command: `scripts/build-strix-rocmfp4-mtp.sh`

## Included ROCmFP4 Work

- `Q4_0_ROCMFP4` and `Q4_0_ROCMFP4_FAST` GGUF tensor formats.
- Tensor-aware ROCmFP4 presets and quantization tooling.
- CPU reference, HIP/ROCm, and Vulkan runtime support.
- ROCmFP4 KV-cache FlashAttention handling.
- ROCmFP4 regression scripts.
- Promoted MTP host-path embedding-fetch cleanup.
- Promoted Vulkan ROCmFP4 exact scale-search pruning.
- StepFun Step 3.7 Flash conversion support for three appended MTP predictors.
- StepFun Step 3.7 Flash runtime graph selection for multi-head native MTP.

## Validation Status

Passed on 2026-05-31:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
scripts/check-rocmfp4-quant-regression.sh
scripts/check-rocmfp4-rocm-cpy-regression.sh
scripts/check-rocmfp4-rocm-fattn-regression.sh
scripts/check-rocmfp4-rocm-runtime-regression.sh
scripts/check-rocmfp4-vulkan-cpy-regression.sh
scripts/check-rocmfp4-vulkan-runtime-regression.sh
```

The Qwen native-MTP compatibility smoke is deterministic on this branch:

```bash
scripts/check-rocmfp4-qwen-mtp-regression.sh
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
```

The sustained floors are calibrated for official b9438. Older stochastic runs
were content-sensitive because draft acceptance changes with generated text.

The StepFun modular converter audit also passed with:

```text
step35.block_count = 48
step35.nextn_predict_layers = 3
tokenizer.ggml.tokens = 128896 entries
tokenizer.ggml.eos_token_id = 128007
indexed text tensors = 804
unmapped text tensors = 0
```

The corrected converter writes `tokenizer.ggml.pre = deepseek-v3`. The first
completed experimental ROCmFP4 artifact retained the equivalent early
`step35` label, so the isolated runtime accepts that label as a compatibility
alias.

## StepFun Conversion

The StepFun Step 3.7 Flash BF16+MTP conversion completed on May 31, 2026.

Output:

```text
/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf
```

The previous partial file was preserved as:

```text
/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf.incomplete-20260530
```

A second partial was stopped and preserved after its tokenizer metadata audit
showed that EOS ID `128007` had been omitted:

```text
/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf.tokenizer-incomplete-20260531
```

The completed conversion header must pass:

```bash
scripts/check-step37-mtp-gguf-metadata.sh
```

The first automatic smoke attempt was stopped by the kernel OOM killer because
it used `--no-mmap`, a 32K context, and 512-token batches for this 99 GiB
artifact. This was a memory-pressure failure, not an external-drive throughput
failure. The dedicated StepFun smoke runner now starts with `--mmap`,
`--fit off`, a 512-token context, and small batches.

Historical pipeline log:

```text
/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-MTP.pipeline.log
```

## StepFun NVMe Finalization

The BF16 intermediate must remain on the 8TB disk because it is about `398.9G`.
After conversion finishes, quantize directly into the distinct NVMe model
directory:

```bash
cd /home/caf/strix-fp4/llama.cpp-rocmfp4-upstream-integration
scripts/finalize-step37-mtp-rocmfp4.sh
```

The completed MTP runtime artifact is:

```text
/home/caf/strix-fp4/models/Step-3.7-Flash-MTP-GGUF/Step-3.7-flash-BF16-MTP-to-ROCmFP4-STRIX_LEAN.gguf
```

Native ROCm MTP validation passed on May 31, 2026:

```text
draft-mtp initialized
prompt eval = 47.1 tok/s
decode = 32.8 tok/s
draft acceptance = 4 accepted / 8 generated
answer = "17 plus 25 is 42."
```

The validated conservative smoke profile is implemented by:

```bash
scripts/run-step37-mtp-rocmfp4-smoke.sh
```

It uses `--mmap`, `--fit off`, a 512-token context, q4 KV caches, and a
user-level `MemoryMax=112G` cgroup scope. Increase context only in staged tests.

Vulkan native MTP also passed from the same internal artifact. A directional
64-token sweep at context 512 measured the following while swap still
contained residue from the earlier OOM:

| Backend | n-max 1 | n-max 2 | n-max 3 |
| --- | ---: | ---: | ---: |
| ROCm0 | 35.7 tok/s | 35.8 tok/s | 33.5 tok/s |
| Vulkan0 | 39.3 tok/s | 40.1 tok/s | 35.1 tok/s |

Use the dedicated interactive launcher for terminal chats:

```bash
scripts/run-step37-mtp-rocmfp4-chat.sh
```

It defaults to Vulkan0 with context 32768, q8/q8 KV caches, and
`--spec-draft-n-max 2`. The correctness smoke keeps ROCm0 with
`--spec-draft-n-max 3` to exercise all three appended MTP heads.

At configured native context 262144, Vulkan `n-max 2` measured q4/q4 at
`40.1 tok/s` and q8/q8 at `41.2 tok/s`. The q8/q8 profile passed the
configured-window sweep from 32K through 262K on both backends and is the
promoted interactive default.

The GGUF metadata declares `step35.context_length = 262144` and
`step35.attention.sliding_window = 512`. Run the guarded q4 Vulkan allocation
ladder after resetting swap:

```bash
sudo swapoff -a && sudo swapon -a
scripts/run-step37-mtp-context-ladder.sh
```

It tests contexts from 2K through the native 262K limit and stops at the first
guarded failure. Passing the ladder proves allocation and short generation at
each requested context; targeted filled-context runs are still required to
measure sustained long-prompt behavior.

Measured tables and long-prompt fill logs are recorded in:

```text
/home/caf/strix-fp4/ROCMFP4-HANDOFF/STEP37-MTP-BENCHMARKS.md
```

The existing internal no-MTP artifact may now be removed after confirming it is
not needed as a local fallback:

```text
/home/caf/strix-fp4/models/Step-3.7-flash-BF16-to-ROCmFP4-STRIX_LEAN.gguf
```

The same no-MTP artifact is retained on the 8TB disk, so the internal copy can
be removed after the new MTP artifact is validated:

```text
/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-to-ROCmFP4-STRIX_LEAN.gguf
```
