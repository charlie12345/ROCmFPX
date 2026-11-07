# ROCmFPX upstream credits

This branch carries ROCmFPX-specific adaptations of selected upstream
`ggml-org/llama.cpp` work and of other outside code. Keep this file updated
when upstream code is merged or manually adapted. Manual ports of single
upstream commits are recorded in [UPSTREAM-ATTRIBUTION.md](UPSTREAM-ATTRIBUTION.md).

## DFlash speculative decoding

- Source: `ggml-org/llama.cpp` PR #22105, commit `d1b34251bc57b696a5c91968069f8a0e6be13ef4`
- Title: `spec : add DFlash support`
- Author: Ruixiang Wang
- Co-author: Kashif Rasul

## DFlash draft conversion refactor

- Source: `ggml-org/llama.cpp` PR #25110, commit `fa72bc6826a5ff30dda3abd1e2fd87ba91df5762`
- Title: `dflash: refactor draft model conversion`
- Author: Ruixiang Wang

## DeepSeek V3.2 and generic DSA runtime

- Source: `ggml-org/llama.cpp` PR #23346, commit `1f0aa2a6964091c33827b1daae1e2b74118c6fa7`
- Title: `model : support for DeepseekV32ForCausalLM with generic DeepSeek Sparse Attention (DSA) implementation`
- Author: fairydreaming
- Co-authors: Stanislaw Szymczyk, Sigbjorn Skjaeret, ggerganov

## TurboQuant K/V-cache quantization

- Source: ROCmFPX local integration commit `d859c9e67b0ba6cae4856be1a096ee368f746782`
- Title: `kv-cache: integrate user TurboQuant KV for deep-context TPS`
- Contributor: Tom Turney / `PlunderStruck` / Aydan S.
- Scope: TurboQuant `turbo3`/`turbo4` K/V-cache quantization paths for ROCm/HIP and Vulkan.
- Status (2026-11-07): since the 2026-09-15 merge of the ROCmFPX org repo,
  `-ctk`/`-ctv` no longer accept `turbo3`/`turbo4`. The kernels remain in
  `ggml/src` (CPU, CUDA/HIP, Vulkan).

## WMMA prompt GEMM

- Source: `adelj88/rocm_wmma_gemm` (https://github.com/adelj88/rocm_wmma_gemm), commit `ea3aa74`
- Author: Adel Johar
- License: MIT, in `vendor/rocm_wmma_gemm/LICENSE` and `licenses/LICENSE-rocm_wmma_gemm`
- Local integration: commit `96c837311` (`[TAG_WMMA_GEMM]`), vendored in
  `vendor/rocm_wmma_gemm`; the included parts and local changes are listed in
  its README, and `NOTICE` carries the attribution.
- Scope: prompt-sized matrix products on gfx11 through `ggml/src/ggml-cuda/wmma-gemm.cu`.
