# ROCmFPX upstream credits

This branch carries ROCmFPX-specific adaptations of selected upstream
`ggml-org/llama.cpp` work. Keep this file updated when upstream code is
merged or manually adapted.

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
