# Flags and switches

The flags and environment variables of RDNA-FAST's [quick start](../README.md#quick-start), the flags this
runtime adds to llama.cpp's, and its switches. For llama.cpp's own flags, see `llama-server --help`.

All measurements on the RX 7900 XT with Swift 1.5 IQ4_XS unless a row says
otherwise.

## The quick start's flags

| flag | what it does | where it helps |
|---|---|---|
| `-c 98304`, `-ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0` | 96k tokens of context, with a 4-bit KV cache (~18 KB per token) for the model and its draft head | the most a 20 GB card holds without paging: 17.9 GB at load, 18.6 GB with both slots full (58k and 39k tokens). The ROCmFP4-FP6 quant takes 0.5 GB more. Windows silently pages whatever does not fit to system RAM: 0.9 GB at `-c 131072` |
| `-np 2 --kv-unified --kv-unified-reserve 12288` | two slots on one shared KV pool; a second request starts only while both can still generate 12k tokens, otherwise it waits in the queue | two agents on one card, e.g. an orchestrator and a worker. [More](../README.md#several-slots-on-one-gpu) |
| `--no-cache-idle-slots` | keeps an idle conversation in its slot; the default saves and clears it at every new task | the next turn of either agent starts without a restore |
| `-b 2048 -ub 2048` | prompts in batches of 2048 tokens | prompts 3-4% faster than at 1024. It fits since [rollback by replay](../README.md#rollback-by-replay). 1536 is slower, as the WMMA GEMM is tuned for 1024 and 2048 rows; below 512 the WMMA GEMM never runs |
| `--no-mmproj-offload` | the 0.9 GB vision projector runs on the CPU | VRAM for context at two slots. Images take 7-12 s to the first token instead of 3-5 s. The BF16 projector encodes 8% faster than f16 on CPUs with AVX-512 BF16 (Ryzen 7000 and newer) |
| `--spec-type draft-mtp`, `--spec-draft-n-max 4 --spec-draft-p-min 0.60` | the model's own MTP head drafts up to 4 tokens per pass | models that ship an MTP head (Qwen3.5/3.6/3.8): decode +45% on prose, +112% on code, for ~0.7 GB of VRAM and 3% of prefill speed |
| `--spec-type "...,ngram-map-k4v"` | text copy boost: where the last tokens repeat an earlier place in the context, the continuation found there is drafted, up to 48 tokens once six 4-token probes in a row would have been accepted | answers that copy the prompt, such as code edits and quoted files: up to 315 t/s. On two days of agent traffic it saved 4.6-6.0% of decode time. Quote the value in a .cmd file, as cmd splits arguments at commas |
| `--ctx-checkpoints 8 --checkpoint-min-step 2048` | up to 8 checkpoints of the model state per slot (~200 MiB each, host RAM), at least 2048 tokens apart. When the list is full, the one whose removal leaves the smallest gap goes; the one at the first user message stays | hybrid models such as Qwen3.8, whose recurrent layers cannot roll back: a prompt that changes late resumes from the nearest checkpoint instead of from scratch |
| `--cache-ram 0 --cache-disk PATH --cache-disk-limit 65536 --cache-disk-checkpoints 4` | conversations and their checkpoints go to SSD (64 GB limit, 4 checkpoints per entry) instead of RAM | more agents than slots, and restarts. [More](../README.md#the-ssd-prompt-cache) |
| `--load-mode dio` | reads the model with direct I/O instead of memory-mapping it | keeps the 15 GB model file out of system RAM |
| `--gpu-keepalive-ms 2000` | a tiny clear on the GPU every 2 s | AMD on Windows: a card on a nearly full VRAM budget that sits idle loses its resident memory, and the next request crawls |
| `--chat-template-kwargs "{\"reasoning_effort\":\"low\"}"` | Swift 1.5's low thinking level | agent turns; medium and xhigh did no better in our agent bench |
| `--reasoning-budget 4096`, `--reasoning-budget-message` | caps the thinking of one answer at 4096 tokens; then the server inserts the message and ends the thinking, and the model answers | long re-planning loops. In 567 agent calls the median answer thought 86 tokens; the cap reached about 1% (up to 12.5k tokens, three and a half minutes) and cut 9% of all thinking. Per request: `reasoning_budget_tokens` |
| `--reasoning-preserve` | every past assistant turn keeps its think block, empty when the client does not send it back (~4 tokens) | clients that do not echo thinking, such as Hermes: without it, the previous turn renders differently once a new user message arrives, and a tool-using turn was prefilled again (median 5k tokens) on every exchange |
| `--temp 1` | sampling temperature 1 | Qwen3.8, which degrades under greedy decoding |
| `-dev ROCm0 -ngl 999 -fa on --jinja`, `--alias`, `--host`, `--port`, `--api-key-file` | standard llama.cpp: the whole model on the first GPU, flash attention, the model's own chat template, the served name and the listening address | |

## Flags this runtime adds

Beyond the ones in the quick start:

| flag | default | what it does | where it helps |
|---|---|---|---|
| `--shared-pass`, `--no-shared-pass` | on | slots that decode at the same moment go through the model in one pass | two busy slots on Qwen3.5/3.8: 46 to 67 t/s in total. [More](../README.md#several-slots-on-one-gpu) |
| `--spec-ngram-map-ramp`, `--no-spec-ngram-map-ramp` | on | the probe gate of the text copy boost (`ngram-map-k`, `ngram-map-k4v`). Without it every n-gram draft has full length from the first match | without: 268 t/s on code edits and 180 t/s on the copy at 41k, but slower on answers that only share phrases with the context, where a wrong long draft costs a checkpoint restore and a replay pass |
| `--priority-trickle-ms N` | 2000 | while an `xhigh` request runs, the held ones still join one pass every N ms; 0 = none | see [Request priority](../README.md#request-priority) |
| `--classifier-head FILE`, `--classifier-config FILE`, `--classifier-cache DIR` | off | serve the loaded model as a decision classifier on `/v1/classifier`, with a decision head and its prompt layout; the cache answers a repeated question about the same text from disk | yes/no, choice and score decisions from a small model, on the CPU. [More](../README.md#decision-classifier) |
| `--models-default NAME` | none | router mode: the model that serves requests without a known `"model"` | clients that send no model name, or another server's |
| `--models-api-loopback-only` | off | router mode: loading and unloading models is accepted only from this machine; other addresses get 403 | a router port reachable from a LAN or the internet |
| `--spec-mtp-strict-qwen` | off | MTP verification that keeps greedy output exact at draft boundaries (from the ROCmFPX project) | reproducible temperature-0 runs with MTP on Qwen3.5/3.8 and Qwen3.8-Flash-Next, one slot only |
| `--cache-disk-min-tokens`, `--cache-disk-prefix-step` | 2048, 4096 | see [The SSD prompt cache](../README.md#the-ssd-prompt-cache) | |

## Expert handoff

For mixture-of-experts models bigger than the card: all experts stay in RAM and the most used of them in a VRAM
cache. [More](expert-handoff.md)

| flag | default | what it does |
|---|---|---|
| `--expert-cache N\|auto` | off | turns expert handoff on with a VRAM cache of `N` experts (in units of the model's largest expert), or `auto` for the free VRAM less `--vram-reserve-mib` |
| `--moe-cache-mib N` | off | the same with a cache of `N` MiB |
| `--vram-reserve-mib N` | 1024 | VRAM that `auto` leaves free beyond the compute buffers, which are sized first |
| `--pool-workers N` | one per 4 logical CPUs, at least 2 | CPU threads that compute the experts the cache misses |
| `--adapt-every N` | 4 | every `N` decode steps, the most used experts outside the cache replace the least used ones in it; 0 keeps the cache as it started |
| `--adapt-swaps N` | 96 | the most experts swapped per round |
| `--adapt-decay F` | 0.7 | how fast old use is forgotten: usage is multiplied by `F` after each round |
| `--expert-profile FILE` | none | starts the cache with the experts this file ranks most used ([profiles](expert-handoff.md#profiles)) |
| `--expert-profile-save FILE` | none | writes this run's routing counts (plus those of `--expert-profile`) to `FILE` every 30 s of use and at exit |

## Environment variables

**Set in the quick start:**

| variable | what it does | where it helps |
|---|---|---|
| `GGML_CUDA_NO_PINNED=1` | no pinned host memory | a nearly full card on Windows: pinned memory makes more of the model page out, for no speed gain |
| `LLAMA_MAX_QUEUED=3` | a request is turned away once three are already waiting | clients that can fall back to another server instead of queueing |
| `ROCMFPX_DRAFT_VOCAB=24576` | the MTP draft head scores 24,576 candidate tokens instead of all 248,320: half fixed, half the main model's own top tokens at the previous position, per slot. The main model still verifies every drafted token against the full vocabulary | MTP on Qwen3.5/3.8, where it is off unless set (Qwen3.8-Flash-Next uses 16,384 by default): decode +9% at a short context, +3-5% at 41k. 24,576 accepts slightly more drafted tokens than 16,384, at the same speed |

**On by default.** `=0` turns each off, for comparisons or a card where one
misbehaves:

| variable | what it does | where it helps |
|---|---|---|
| `GGML_HIP_FA_GQA6` | decode attention with one GPU block per KV head carrying all six of its query heads, for batches of 1 to 5 tokens, instead of three groups of two and a batch padded to 8 | Qwen3.5/3.8-27B (head size 256, 24 query heads on 4 KV heads) on RDNA: decode +10-12% at 41k, +2-6% at 17k, nothing at a short context |
| `GGML_HIP_FA_GQA7` | the same for seven query heads per KV head at head size 128 | models of that shape on RDNA, e.g. Nanbeige4.3-33B |
| `GGML_HIP_FA_Q4_TILE` (`_MIN`, `_MAX`: 1, 8 tokens) | decode attention reads a q4_0 KV cache as stored, once per group of query heads, instead of converting the cache to f16 on every call | a q4_0 KV cache at head size 256 (or 128 with seven query heads per KV head), batches of 1 to 8 tokens, on RDNA; `=1` also on other HIP cards |
| `ROCMFPX_WMMA_GEMM` | prompt batches of 512 tokens or more through an f16 WMMA GEMM instead of the int8 path | RDNA3 and RDNA3.5 with the GEMM library built. Weights: ROCmFP4/FP6 and the stock Q4_0, Q4_1, Q5_1, Q8_0, Q2_K-Q6_K, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS and IQ2 types. A cold 17k prompt: IQ4_XS 781 to 864 t/s, ROCmFP4 682 to 923. [More](../README.md#prompts-through-a-wmma-gemm) |
| `ROCMFPX_WMMA_GEMM_MIN_BATCH`, `ROCMFPX_WMMA_GEMM_SCRATCH_MB`, `ROCMFPX_WMMA_GEMM_DIR` | the smallest batch for it (512), its f16 weight scratch (sized automatically), and where its library is loaded from (next to the binaries) | tuning, and installs that keep the library elsewhere |
| `GGML_HIP_MM_SKINNY` | an F32 matmul kernel for weights of 9 to 256 rows at prompt and multi-slot batch sizes, instead of rocBLAS | Qwen3.5/3.8's DeltaNet gate weights, kept F32 by IQ4_XS and K-quant GGUFs, and F32 MoE routers, on RDNA: 1024-token prompts +3%, 10 to 16 parallel decodes +25-30% |
| `GGML_HIP_GDN_CONV_SNAP` | the concat, the conv, the SILU and the rollback snapshots of a DeltaNet layer in one kernel | Qwen3.5/3.6/3.8 on HIP and CUDA. With `GGML_HIP_MMVQ_FUSE_VERIFY`: decode +2.3-3.9%, two slots +6.8%, short prompts +9-17% |
| `GGML_HIP_MMVQ_FUSE_VERIFY` | the matmul kernel adds the residual itself in verify batches of 2 to 8 tokens | MTP and n-gram drafting, and two busy slots, on RDNA3.0 and RDNA4 |
| `GGML_CUDA_GDN_CHUNKED` | the DeltaNet part of a prompt through a chunked WMMA kernel (from the ROCmFPX project) | Qwen3.5/3.6/3.8 and Qwen3-Next prompts of one sequence, 64 tokens or more, on RDNA3 |
| `ROCMFPX_RS_REPLAY` | [rollback by replay](../README.md#rollback-by-replay) instead of a copy of the recurrent state per drafted token | Qwen3.5/3.6/3.8 and Qwen3-Next with drafting: 0.8 GB less VRAM at two slots, decode equal |
| `ROCMFPX_MTP_LEAN_RESERVE` | the MTP draft context reserves memory for the two shapes it runs, 80 MiB instead of 784 MiB | dense Qwen3.5/3.8 with one MTP block; it lets `-ub 2048` fit beside `-c 98304` |
| `ROCMFPX_NGRAM_SHADOW`, `ROCMFPX_NGRAM_PROBES` | the text copy boost scores its probes in the background instead of sending them (`=0` sends them); probes in a row before long drafts start (6) | with the ramp: 5-6% of the requests came out slower, against 12-40% when the probes were sent |
| `LLAMA_KV_LANES_MERGE` | the largest batch, in tokens, that a shared pass takes (8); 0 = never share | the shared pass of two busy slots |
| `GGML_CUDA_MMVQ_Q8_CACHE` | matmuls that read the same activation quantize it once (from the ROCmFPX project) | written for Qwen3.8-Flash-Next |
| `GGML_CUDA_GRAPH_CHUNK_SIZE` | GPU graphs are captured in chunks of 512 nodes (from llama.cpp); 0 = one graph | |

**Off by default:**

| variable | what it does | where it helps |
|---|---|---|
| `ROCMFPX_FP8_GEMM=1` | prompt batches through an FP8 GEMM: FP8 WMMA on RDNA4, an f16 stand-in on RDNA3 that is only there to check it. `ROCMFPX_FP8_GEMM_TILE=1` picks 4-wave tiles instead of 8; `ROCMFPX_FP8_GEMM_CHECK=N` (RDNA3) compares the first N products with an emulation in the log | RDNA4, not yet run on one. With every token through FP8, mean KL divergence rises by about 0.005. See [Other cards](../README.md#other-cards) |
| `GGML_HIP_MMVQ_FUSE_VERIFY_GLU=1` | also runs gate, up and GLU as one kernel in verify batches of 2 to 8 tokens | not measured on RDNA4; on the RX 7900 XT the same text in 2.2 times the time |
| `GGML_CUDA_DQ_MMV`, `GGML_CUDA_DQ_Q6K` (`GGML_CUDA_DQ_ROWS`: 1, 2, 4 or 8 rows per block) | one-token decode of Q4_K and Q5_K (with `DQ_Q6K` also Q6_K) in floats instead of quantizing the activation | K-quant models on RDNA3.5, where it is on by default; `=1` on the RX 7900 XT is 6-24% slower per kernel |
| `GGML_CUDA_FA_WMMA_MAX_HEAD` | the largest head size on the WMMA attention kernel: 576 on RDNA4, 256 elsewhere | heads of 320, 512 or 576 (Mistral Small 4, DeepSeek, GLM) run there on RDNA4; elsewhere for tests |
| `ROCMFPX_MMQ_FUSION` | `=1` also runs the fused gate/up/GLU prompt kernel for MoE experts on RDNA3.0 (by default NVIDIA only), `=0` turns it off everywhere | correct but 3-16% slower on the RX 7900 XT (Qwen3-30B-A3B IQ4_XS) |
| `GGML_CUDA_SHARED_EXPERT_FUSION` | llama.cpp's fused routed and shared expert matmul, off on HIP (`=1` turns it on) | MoE models with a shared expert. On gfx1151 MTP verify was 3-7% slower with it |
| `LLAMA_SPEC_LOOKUP_EXTEND=1` (`LLAMA_SPEC_LOOKUP_MATCH`, `LLAMA_SPEC_LOOKUP_MAX`: 8, 16 tokens) | after a model drafter (MTP, DFlash2, EAGLE3) proposes its tokens, the draft is extended with what followed the same tail earlier in the prompt | the older form of the text copy boost; `ngram-map-k4v` is the measured one |
| `GGML_CUDA_AR_P2P=1` (`_ISSUER`: device 1) | the all-reduce of two GPUs sends large tensors by peer copy instead of through host memory | two AMD cards whose peer access works both ways; not tested here (one card) |

**From the ROCmFPX project**, for its models, quants and cards; not tuned or
measured here:

| variable | what it does | where it helps |
|---|---|---|
| `ROCMFPX_QWEN4_QSA_DECODE=1`, `ROCMFPX_QWEN4_QSA_DECODE_BLOCKS`, `ROCMFPX_QWEN4_QSA_DECODE_CACHE` | sparse attention for decode batches at 8k cells and more, block selection as in the prompt, and a cache of the pooled keys (the last two on by default) | Qwen3.8-Flash-Next (qwen4exp) on HIP |
| `LLAMA_MTP_QSA_MIN_T` (1) | below this batch size the MTP layer attends densely | Qwen3.8-Flash-Next MTP |
| `ROCMFPX_DRAFT_HEAD_QUANT` (on), `ROCMFPX_QWEN4_PAD_HEAD=1` | draft steps use a Q4_0 copy of the output head; 2-3-row head products padded to 4 rows | Qwen3.8-Flash-Next MTP |
| `ROCMFPX_MTP_CUM_P` (`ROCMFPX_MTP_CUM_MIN`, `ROCMFPX_MTP_CUM_MAX`: 1, 5) | stops drafting once the product of the drafted tokens' probabilities falls below P | MTP with an adaptive draft length; off by default |
| `ROCMFPX_NGRAM6`, `ROCMFPX_NGRAM_WINDOW`, `ROCMFPX_NGRAM_FLOOR4` | a 6-token table behind the main one, an adaptive draft window, and a window floor of 4 instead of 8 | `--spec-type ngram-mod` |
| `ROCMFPX_MOE_SORT_IDS=1`, `ROCMFPX_ROCMI4_MOE_RPB8=1` | MoE decode kernels with sorted experts and with 8 rows per block | ROCmI4 and ROCmFP4_FAST experts on RDNA3.5 |
| `GGML_CUDA_MMQ_MOE_TILE_MULT` (1) | weights the tokens per expert when the MoE prompt kernel picks its tile | MoE prompts on RDNA3 and RDNA4 |
| `GGML_CUDA_NORM_GATED_MIN_ROWS` (4096), `GGML_CUDA_HC_COMBINE_MIN_ROWS` (512), `GGML_CUDA_HC_UP_PRE` (on) | fusions of the gated norm and of the hyper-connection kernels, and their thresholds | Qwen3.8-Flash-Next, DeepSeek 4, GLM-5-Next |
| `ROCMFPX_HIP_GRAPH_COPY_KERNEL` | copies inside HIP graphs run as kernels (on by default only with unified memory) | RDNA3.5, where graph copies left stale recurrent state |
| `GGML_SCHED_UMA_RING` (2) | copies of the graph inputs, so the host writes the next while the GPU reads the last | iGPUs and other shared-memory setups |
| `GGML_SCHED_PIN_ASYNC_READS` (on) | keeps tensors that another backend still reads out of memory reuse | several backends in one graph |
| `GGML_CUDA_DISABLE_SSD` | turns off the matmul path of the Mamba-2 scan for long sequences | Mamba-2 models |
| `LLAMA_LAZY_PREFETCH_DRAFT=1` | with `--lazy-mode`, warms the per-layer embedding rows of drafted tokens | Qwen3.8-Flash-Next with lazy loading |
| `GGML_VK_MMID_TILED` (on, `_MIN` 1024), `GGML_VK_ROCMFP4_COOPMAT=1` | Vulkan: a tiled int8 kernel for IQ3_S MoE experts in prompts; a ROCmFP4 coopmat matmul (built with `-DGGML_VULKAN_ROCMFP4_COOPMAT=ON`) | Vulkan |
| `ROCMFPX_PLUGIN_PATH` | ROCmFPX plugins to load | see [Plugin system](../README.md#plugin-system) |

**HIP runtime.** Read by AMD's runtime, not by this code:

| variable | what it does | where it helps |
|---|---|---|
| `GPU_MAX_HW_QUEUES=1` | all GPU work of the process goes through one hardware queue instead of four | Linux (amdgpu): a server with `--mmproj` that keeps the GPU at full load and clocks while idle. Not needed on Windows. Work on separate GPU streams can no longer overlap |

**Diagnostics.** None of these is needed in normal use:

| variable | what it does |
|---|---|
| `ROCMFPX_WMMA_GEMM_FP8_EMU=1`, `2`, `3` | rounds the weights, the activations or both to E4M3 inside the f16 WMMA GEMM, to measure what FP8 costs in quality on any gfx11 card (changes the output) |
| `GGML_CUDA_OP_PROFILE=N` | GPU time per kind of op, every N graphs; slows every pass |
| `LLAMA_DECODE_PROFILE=N` | host time per phase of a ubatch, and the server loop's phases |
| `LLAMA_NODE_DUMP=1` | a fingerprint of every named tensor after each ubatch, to find where two runs part |
| `GGML_CUDA_FA_DEBUG`, `GGML_HIP_FA_PARALLEL_BLOCKS=N` | logs attention launch shapes; forces N blocks along the KV axis |
| `GGML_CUDA_FUSE_MASK`, `GGML_CUDA_FUSE_MASK0` | bit masks of the fusion groups, to bisect a fusion |
| `GGML_CUDA_GRAPH_VERIFY_UID=1` | checks every reused GPU graph node by node |
| `GGML_SCHED_SANITIZE=1` or `2`, `GGML_SCHED_SANITIZE_NONFATAL=1` | the scheduler's race detector, fatal or counting |
| `ROCMFPX_TRACE_NGRAM`, `ROCMFPX_MMQ_NO_DEDUP` | logs n-gram accepts; turns off the once-per-token quantization of MoE prompt inputs |
