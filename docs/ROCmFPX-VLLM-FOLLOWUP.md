# ROCmFPX vLLM Follow-Up

This is a parked follow-up, not part of the current llama.cpp ROCm/Vulkan kernel work.

## Goal

Evaluate whether ROCmFPX formats can run faster under vLLM on AMD hardware than the
current llama.cpp paths, especially for parallel serving and long-running agent loads.

## Starting Points

- Keep GGUF/llama.cpp compatibility as the reference implementation.
- Prototype vLLM support separately so ROCmFPX GGUF users are not blocked by vLLM changes.
- Prefer native ROCm kernels through AITER, Triton, or HIP extensions instead of
  converting ROCmFPX tensors back to generic K-quants at load time.

## Work Items

- Define a loader path for ROCmFPX tensor metadata and per-tensor routing.
- Map ROCmFP3, ROCmFP4, ROCmFP6, and ROCmFP8 blocks into vLLM linear kernels.
- Add decode-first kernels for single-token matvec, then verify-batch matmul for MTP.
- Check PagedAttention and KV-cache type support for ROCmFPX serving profiles.
- Validate MTP/speculative decode compatibility separately from normal decode.

## Bench Gate

A vLLM prototype should be considered useful only if it beats the tuned llama.cpp
ROCmFPX build on at least one real serving case:

- higher sustained decode throughput at the same parallel count,
- lower first-token latency at the same context size,
- or better total tokens/sec on 16-32 parallel workers without worse coherency.
