# Quant mixing policy

ROCmFPX mixing is tensor-level: each tensor remains a normal GGML type, while a
model preset chooses different types for tensors with different sensitivity.
That preserves simple loading and lets existing backends dispatch per tensor.

The implemented `*_AGENT` presets protect token embeddings, output weights,
attention projections, and selected FFN down/gate tensors. FP2 agent routing
uses FP3 for those transformer tensors and fast FP4 for embeddings/output; FP5
promotes to FP6; FP7 promotes to FP8. Existing FP3, FP6, and FP8 agent routes
retain their established policy. The straight presets remain available when
exact size is the priority.

| Base | Implemented agent promotion | Additional candidate to measure |
| --- | --- | --- |
| ROCmFP2 | attention/selected FFN to FP3; embeddings/output to fast FP4 | selectively promote the first/last FFN-down layers to FP4 |
| ROCmFP3 | established Q5_K/Q6_K and ROCmFP6 routing | use imatrix ranking to reduce the promoted layer count |
| ROCmFP5 | sensitive tensors to ROCmFP6 | compare FP6_K where a backend lacks a native FP6 kernel |
| ROCmFP6 | established ROCmFP8 or Q6_K routing, including lean variants | separately tune dense and MoE expert tensors |
| ROCmFP7 | sensitive tensors to ROCmFP8 | compare F16 only for small normalization-adjacent projections |
| ROCmFP8 | selected tensors to stock Q8_0 | retain F16 for logits-critical tensors only when measurements justify it |

“Candidate” means an experiment, not a release default. Every candidate is
expressible today through `--tensor-type`/`--tensor-type-file`, so the format
does not need irregular per-block metadata.

Recommended evaluation loop:

1. Generate a representative importance matrix from BF16/F16 using
   `llama-imatrix`, including coding, tool-call, JSON, multilingual, and normal
   chat samples for an agent model.
2. Quantize a straight baseline and an agent mixture from the same source.
3. Use `--tensor-type` or `--tensor-type-file` for controlled promotions based
   on per-layer importance statistics; never re-quantize a low-bit model for a
   release artifact.
4. Compare file size, perplexity/KLD, fixed-prompt logits, tool-call validity,
   long-context behavior, and prompt/decode throughput on each backend.
5. Accept a promotion only when its measured quality gain justifies both its
   bytes and any slower kernel path.

Research supports activation- or Hessian-informed sensitivity ranking, but the
hardware cost of irregular mixtures matters. Therefore v1 limits candidates to
whole-tensor promotions already expressible in GGUF instead of introducing
per-channel type metadata. Qwen is the first qualification family; results do
not automatically generalize to every architecture.

Useful primary references: llama.cpp's
[`tools/imatrix/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md),
[AWQ](https://arxiv.org/abs/2306.00978),
[APTQ](https://arxiv.org/abs/2402.14866), and
[HAWQ](https://arxiv.org/abs/1905.03696).
