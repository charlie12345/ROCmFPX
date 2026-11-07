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

**Qwen3.8-Flash-Next Q2_0** (66.5 GB), measured on a server in daily use on this machine (sampling at temperature
1, other programs holding RAM):

| | |
| --- | ---: |
| Decode, real answers once the cache has learned | 18-20 t/s |
| Decode, short test prompts | 26-31 t/s |
| Decode, the first minute after a load | about 12 t/s |
| 55-token prompt | 1.7 s (7 s right after a load) |
| 700-token prompt | about 19 s |
| Prompts of several thousand tokens | 150-200 t/s |

The VRAM cache holds 36% of the experts. It serves 34% of the routed experts on the first answer after a load, and
63-73% once it has followed a few answers. The experts outside VRAM don't all fit in free RAM next to everything
else on this 32 GB machine, so some page in from the SSD. That makes the first answers after a load slow until those
pages are back, and prompts over a few hundred tokens read every expert once per batch. It needs RAM for the experts
outside VRAM: about 20 GB on a 20 GB card.

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

- **A profile helps the first answers after a start.** Without one, the cache starts with an even share of every
  layer's experts and learns from the first tokens. A saved profile starts it closer to your usual work (see
  [Profiles](#profiles)).
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

## Profiles

A profile lists how often the model routed to each of its experts, layer by layer. At start, the cache fills with the
experts it ranks highest. Without one, a fresh start serves far fewer experts from VRAM until the cache has learned:
Flash-Next on a 20 GB card served 34% of its routed experts from VRAM on the first answer, and 63-73% once the cache
had followed a few answers.

### Making a profile

1. Start the model with a file to write, plus your usual flags:

   ```sh
   llama-server -m model.gguf --expert-cache auto --expert-profile-save my-profile.txt
   ```

2. Use it for the work the profile is meant for: the agent or coding sessions, chats and languages you expect. The
   counts come from the tokens the model writes and from short prompts. Long prompts take the upload path and aren't
   counted. A few thousand written tokens give a usable ranking, and more is better.
3. The file is written every 30 s of use and at exit. After the last answer, wait 30 s or stop the server normally
   before you copy it.

To keep refining one profile, pass the same file to both flags. Each run then adds its counts to the ones it loaded:

```sh
llama-server -m model.gguf --expert-cache auto --expert-profile my-profile.txt --expert-profile-save my-profile.txt
```

The counts only ever add up, so after weeks of one kind of work, a new kind barely moves the ranking. Start a new file
when the work changes.

### Using a profile someone else made

Pass it to `--expert-profile` alone. The file is then only read, never changed.

[models/expert-profiles](../models/expert-profiles) holds the profiles we ship:

| File | Model | Made from |
| --- | --- | --- |
| `Swift-1.5-Flash-Next.txt` | Swift 1.5 Flash-Next (Qwen3.8-Flash-Next), made with its Q2_0 pack | About 2,400 written tokens of everyday agent use, 2026-11-07 |

```sh
llama-server -m Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --expert-cache auto \
  --expert-profile models/expert-profiles/Swift-1.5-Flash-Next.txt
```

- **Hardware doesn't matter.** A profile holds no sizes. Each machine fills as many of the top-ranked experts as fit
  its own VRAM, so the same file works on a 16 GB card, a 32 GB card or several GPUs.
- **The model does.** Expert numbers mean nothing in another model, and a fine-tune may route differently from its
  base. Profiles aren't checked against the model when they load, so use one only with the model it was made for. Its
  first line names the architecture and the number of layers.
- **Other quants of the same model should work.** The routing depends on the model, not on how its experts are
  stored, so the experts a quant picks most often are almost always the same. This hasn't been measured yet.
- **The workload matters most.** A profile from coding sessions starts a chat with experts that suit code. The cache
  still follows the conversation either way, so a poor match only costs the first answers.

### The file

Plain text: a comment line, then one line per layer with the layer number, its number of experts, and one count per
expert:

```text
# expert handoff routing counts, arch qwen4exp, 48 layers
layer 0 512 1843 0 27 ...
layer 1 512 95 412 3 ...
```

To merge profiles from different machines or kinds of work, add their counts layer by layer and expert by expert.
Scale a file first if its share of the mix should be smaller than its raw counts.

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
