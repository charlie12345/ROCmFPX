# Expert handoff: MoE models bigger than the card

Mixture-of-experts models such as Qwen3.6-35B-A3B, Qwen3.8-Flash-Next or Gemma 4 26B-A4B use only a few of their
experts for each token. Expert handoff keeps all experts in system RAM and the most used of them in a VRAM cache. For
every token and layer, the GPU computes the experts it holds while the CPU computes the others from RAM at the same
time, and the two results are added. Expert weights don't cross PCIe while the model writes. The cache follows the
conversation: experts that keep being used move into VRAM. It works similar to Strata.

## Results

RX 7900 XT 20 GB, Ryzen 9 7900 (12 cores), 32 GB DDR5, Windows 11, ROCm 10.1. Decode speed through `llama-server`,
greedy, three chat prompts of up to 300 tokens each.

**Qwen3.6-35B-A3B UD-Q4_K_S** (20 GB of weights, more than the card holds next to its context):

| | Decode | 4k-token prompt |
| --- | ---: | ---: |
| `--n-cpu-moe 8` (the usual way: 8 layers' experts on the CPU) | 66-71 t/s | 3178 t/s |
| Expert handoff, `--expert-cache auto` | **103-105 t/s** | 2737 t/s |

The cache holds 92% of the experts and serves 99.9% of the ones the model routes to. Prompts ran with `-ub 4096`.

**Gemma 4 26B-A4B Q4_0** with only half or a quarter of its experts in VRAM, as on a smaller card (it fits whole in
20 GB: 101-102 t/s):

| Experts in VRAM, cache starts from | Cache stays as it started | Cache follows the conversation |
| --- | ---: | ---: |
| Half, an even share of each layer | 53-59 t/s | **71-78 t/s** |
| Half, a saved profile | 66-74 t/s | 60-78 t/s |
| A quarter, a saved profile | 56-61 t/s | **60-69 t/s** |

With half the experts in VRAM and no profile, the cache that follows the conversation serves 86% of the routed
experts over the three answers, against 49% when it stays as it started. The tuning barely matters here: slower
forgetting, fewer swaps per round or a round every step all landed within a few percent of the defaults.

**Qwen3.8-Flash-Next Q2_0** (66.5 GB) runs at 19-37 t/s. On this 32 GB machine, the experts outside VRAM don't
fit in free RAM, so they page in from the SSD, and a short prompt takes tens of seconds. Following the conversation
doesn't help there, since its copies read from the SSD too. It needs RAM for the experts outside VRAM: about 20 GB
on a 20 GB card.

**Quality.** The CPU computes its experts in a different order and precision than the GPU would, so greedy answers
match the all-GPU answers for a while and then diverge into different but coherent text. Against the model fully in
VRAM, Gemma 4 with half its experts on the CPU has a KL divergence of about 0.08, the same as `--n-cpu-moe`.

## Using it

```sh
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --expert-cache auto -ub 4096
```

| Flag | What it does |
| --- | --- |
| `--expert-cache N\|auto` | Turns expert handoff on with a VRAM cache of `N` experts (in units of the model's largest expert), or `auto` for the free VRAM less `--vram-reserve-mib`. Keeps the routed experts in RAM. |
| `--moe-cache-mib N` | The same with a cache of `N` MiB. |
| `--vram-reserve-mib N` | VRAM that `auto` leaves free beyond the compute buffers, which are sized first (default 1024). |
| `--pool-workers N` | CPU threads that compute the experts the cache misses (default: one per 4 logical CPUs, at least 2). |
| `--adapt-every N` | Every `N` decode steps, the most used experts outside the cache replace the least used ones in it (default 4; 0 keeps the cache as it started). |
| `--adapt-swaps N` | The most experts swapped per round (default 96). |
| `--adapt-decay F` | How fast old use is forgotten: usage is multiplied by `F` after each round (default 0.7). |
| `--expert-profile FILE` | Starts the cache with the experts this file ranks most used. |
| `--expert-profile-save FILE` | Writes the routing counts of this run (plus those of `--expert-profile`) to `FILE` every 30 s of use and at exit. |

- **A profile helps the first answers only.** Without one, the cache starts with an even share of every layer's
  experts and learns from the first tokens. A saved profile starts it closer to your usual work; pass the same file to
  both profile flags and it keeps learning.
- **Workers.** For 4-bit experts the CPU side is limited by RAM bandwidth, and on the 12-core Ryzen 6 workers beat
  11. 2-bit experts (Flash-Next) need more arithmetic per byte, so 8 to 11 workers get further toward the RAM's speed.
- **Placement.** `--cpu-moe` / `--n-cpu-moe N` still decide which layers keep their experts in RAM, and expert handoff
  splits exactly those. Without them, it splits all of them.
- **MTP drafting** works: the draft layer stays in VRAM whole.
- **Qwen3.8-Flash-Next:** add `-ot per_layer_token_embd=CPU` and `-ub 4096`. Its 28.8 GB per-layer token
  embeddings stay on the SSD and only one row is read per token. Give it RAM for the experts outside VRAM, about
  20 GB on a 20 GB card, or they page in from the SSD.
- **Several GPUs.** With the layer split (the default), each GPU caches the experts of its own layers in its free
  VRAM, and one CPU pool serves them all.

## How it works

1. **Placement.** The routed experts stay in RAM, memory-mapped from the model file. A VRAM cache fills what the rest
   of the model leaves free. Each layer has a slot map on the GPU, which says where each of its experts sits in the
   cache, or that it isn't there.
2. **Each layer, for each small batch.** The GPU picks the experts as usual and looks them up in the slot
   map. If some are missing, it writes the layer input and the list of missing experts into a small block of host
   memory that both sides read and write in place, and bumps a counter there. CPU threads watching that counter
   compute the missing experts from RAM while the GPU computes the cached ones, then hand back their sum, which a GPU
   kernel waits for and adds. The GPU stream never stops for this, so a whole step stays one captured GPU graph.
3. **Layers without a miss** never involve the CPU: the GPU sees it has every expert and goes on.
4. **The cache follows the conversation.** The GPU counts how often each expert is routed. Every few steps, the most
   used experts outside the cache replace the least used ones inside, a layer at a time. A replaced expert leaves the
   slot map at once, so the CPU computes it from then on. A helper thread copies the new expert into its slot beside
   the running steps, and the slot map takes it in once the copy has landed.
5. **The CPU reads each missed expert once per batch**, row by row for all the tokens that miss it, so a batch of
   several tokens costs little more than one.
6. **Prompts** of up to a few hundred tokens go the same way when the CPU is estimated to finish sooner than the GPU
   could upload the experts it lacks, judged from the CPU's measured speed and the share of experts it computes.
   Longer prompts take the usual path: the experts a batch uses are copied to the GPU, those in the cache from VRAM.
7. **Several GPUs.** Each GPU's cache holds experts of its own layers. One block of host memory, mapped into every
   GPU, serves them all: with the layer split, a layer finishes before the next one starts, whichever GPU runs it.

## Limits

- Several GPUs work with the layer split, not with `--split-mode tensor`. The multi-GPU path has been built and
  reviewed but not yet run on two GPUs.
- Experts with Hadamard-rotated weights, and GroveMoE, run without the handoff, as with `--n-cpu-moe`.
- Long prompts are somewhat slower than with `--n-cpu-moe`, since the experts outside the cache are uploaded for every
  batch. A larger batch (`-ub 4096`) spreads that cost.
- The CPU side needs its experts in RAM; whatever doesn't fit is read from the SSD (Flash-Next on 32 GB).
- With the cache following the conversation, answers aren't bit-identical from run to run. `--adapt-every 0` keeps
  the cache fixed.
- Keep the experts memory-mapped (the default load mode). Pinning 18 GB of them fails on Windows.
