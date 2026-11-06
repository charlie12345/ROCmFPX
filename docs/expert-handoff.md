# Expert handoff: MoE models bigger than the card

Mixture-of-experts models such as Qwen3.8-Flash-Next, Qwen3.6-35B-A3B or Gemma 4 26B-A4B need only a few of their
experts for each token. Expert handoff keeps all experts in system RAM and the most used of them in a VRAM cache. For
every token and layer the GPU computes the experts it holds while the CPU computes the others at the same time, from
RAM, and the two results are added. Expert weights never travel over PCIe while the model writes. It works similar
to Strata.

<!-- NUMBERS: filled in from the measurements -->

## Using it

```sh
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --expert-cache auto --expert-profile-save profile.txt
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --expert-cache auto --expert-profile profile.txt --expert-profile-save profile.txt
```

| Flag | What it does |
| --- | --- |
| `--expert-cache N\|auto` | Turns expert handoff on with a VRAM cache of `N` experts (in units of the model's largest expert), or `auto` for the free VRAM less `--vram-reserve-mib`. Keeps the routed experts in RAM. |
| `--vram-reserve-mib N` | VRAM that `auto` leaves free for everything else (default 2048). |
| `--pool-workers N` | CPU threads that compute the experts the cache misses (default: one per physical core but one). |
| `--expert-profile FILE` | Fills the cache with the experts this file ranks most used. |
| `--expert-profile-save FILE` | Writes the routing counts of this run (plus those of `--expert-profile`) to `FILE`, every 30 s of use and at exit. |
| `--moe-cache-mib N` | The same as `--expert-cache` with a cache of `N` MiB. |

- **Learn a profile from your own work.** Without one the cache starts with the same share of every layer's experts.
  A profile from a few minutes of your usual prompts raises the share of experts found in VRAM, and with it the speed.
  Pass the same file to both flags and it keeps learning.
- **Placement.** `--cpu-moe` / `--n-cpu-moe N` still decide which layers keep their experts in RAM; expert handoff
  splits exactly those. Without them, all of them.
- **Qwen3.8-Flash-Next:** add `-ot per_layer_token_embd=CPU`. Its 28.8 GB lookup table stays on the SSD and is read
  a few rows per token.

## How it works

1. **Placement.** The routed experts stay in RAM, memory-mapped from the model file. The most used ones, by the
   profile, are copied into a VRAM cache that fills what the rest of the model leaves free.
2. **Each layer, for each batch of up to 32 tokens.** The GPU picks the experts as usual and looks up which of them are
   in the cache. It writes the layer input and the list of the others into a small block of host memory that both
   sides can read and write in place, and rings it. CPU threads waiting on that block compute those experts from RAM
   while the GPU computes the cached ones, then hand back their sum. A GPU kernel waits for it and adds it. There is
   no stop of the GPU stream per layer, so the whole step stays one captured GPU graph.
3. **Layers without a miss** never involve the CPU: the GPU sees it has every expert and goes on.
4. **Long prompts** take the usual path: the experts a batch uses are copied to the GPU, those in the cache from
   VRAM.

## Limits

- One GPU.
- Experts with biases (gpt-oss) or rotated weights run without the handoff, as with `--n-cpu-moe`.
- Long prompts are slower than with the whole model in VRAM, since the experts outside the cache are uploaded for
  every batch.
- The MTP draft layer computes its experts on the CPU.
