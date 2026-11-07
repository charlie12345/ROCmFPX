# Native draft vocabulary

This path adds a draft-vocabulary selector, its state, and graph integration. It is on by default for qwen4exp (Qwen3.8 Flash Next) MTP draft contexts with 16384 candidate rows; `ROCMFPX_DRAFT_VOCAB=<rows>` changes the budget and `ROCMFPX_DRAFT_VOCAB=0` turns it off. For qwen35 (Qwen3.5/3.8 dense, such as Swift 1.5) it is off unless `ROCMFPX_DRAFT_VOCAB` is set; the production launcher sets 24576 (see [Environment variables](../../README.md#environment-variables) in the top-level README). qwen35moe accepts the variable, but its draft graph still runs the full head. Target verification still evaluates the full vocabulary, so accepted output is the target's; the candidate rows only change which tokens get proposed.

Each MTP draft context owns its candidate state, so independent draft drivers (for example two server instances on one model) never share candidates. Candidates are kept per sequence: with `-np > 1` every slot selects from its own last target logits, and the draft graph uploads the rows of the sequence it decodes. The candidate-row head runs for single-row draft steps; when several sequences draft in one step the full head is used.

The selector uses a sampled cutoff, SIMD screening, exact comparison-based partitioning, and ordered bitset emission. Sampling only reduces work: if the candidate pool is too small, a complete scan supplies the exact partition. Scratch storage is reused between decode steps. Signed zero, equal scores, NaNs, infinities, and token-ID tie order have explicit compatibility tests.

The draft context owns the candidate state; the draft driver holds an ownership lease on it and graph inputs retain it while graphs exist. Updated row IDs are uploaded into fixed-shape graph inputs, which support graph reuse for one-token decoding. Indirect ggml matrix multiplication computes selected head rows; masked scattering preserves the full-vocabulary output interface. Tied or non-finite winning logits use the existing sampler.

## Provenance and scope

The selector, state, and graph integration are original code under the repository MIT license. The lower half of the candidate budget covers low token IDs, a valid previous token is reserved, and the remaining slots follow target scores. Selected rows are projected for the draft head, and masked scattering keeps the full-vocabulary output interface.

The implementation and tests were written with AI assistance and require normal human review before upstream submission. Existing upstream ggml operators and the repository MIT license remain in use; model licensing is unchanged.

## Validation

`test-rocmfpx-draft-select` compares selection with a complete-sort oracle across deterministic fixtures, changing sizes, ties, adversarial score layouts, and unusual IEEE values. `test-rocmfpx-draft-vocab` checks F32/BF16/Q8_0 selected projection, masks across updates, per-sequence candidates and reset, graph reuse eligibility, winner mapping, fallback behavior, driver ownership, and resource lifetime. The latter test is registered only for builds with a directly linked CPU backend.

A faster selector microbenchmark is not an end-to-end decoding measurement.

## Enabling the fast Qwen3.8 Flash Next path

Every switch below except `ROCMFPX_DRAFT_VOCAB` defaults off; `ROCMFPX_HIP_GRAPH_COPY_KERNEL` is also on by default when `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is set. A normal build and launch keep their current kernels on RDNA 3, RDNA 3.5, and RDNA 4.

Set these only for the measured Flash Next server (RDNA 3.5, Strix Halo):

```text
ROCMFPX_DRAFT_VOCAB=16384
ROCMFPX_QWEN4_QSA_DECODE=1
ROCMFPX_QWEN4_PAD_HEAD=1
ROCMFPX_NGRAM_WINDOW=1
ROCMFPX_NGRAM_FLOOR4=1
ROCMFPX_HIP_GRAPH_COPY_KERNEL=1
ROCMFPX_MOE_SORT_IDS=1
ROCMFPX_ROCMI4_MOE_RPB8=1
```

`ROCMFPX_QWEN4_QSA_DECODE=1` requires the Radeon 8060S as `ROCm0` and stops if the device or tensor shape is outside the kernel it was measured on. `ROCMFPX_HIP_GRAPH_COPY_KERNEL`, `ROCMFPX_MOE_SORT_IDS` and `ROCMFPX_ROCMI4_MOE_RPB8` act only on RDNA 3.5. The signed ROCmI4 dot is compiled only for gfx1151; other GPU targets keep the previous dot.

On a 20 GB discrete card such as the RX 7900 XT, Flash Next runs through the [expert handoff](../expert-handoff.md) instead.

`GGML_ROCMI4_Q8_1_MMVQ_VDR=4` is an optional gfx1151 compile definition. CMake does not set it. Leaving it unset preserves the existing ROCmI4 mat-vec width on every architecture.
