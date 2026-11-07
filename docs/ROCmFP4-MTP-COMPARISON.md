# ROCmFP4 + MTP Comparison

Date: 2026-05-23
Hardware: Framework AMD Strix Halo 395+ desktop, 128 GB unified RAM
Merged tree: `llama.cpp-mtp-rocmfp4`
Branch: `mtp-rocmfp4-strix`
Build: reproducible with `scripts/build-strix-rocmfp4-mtp.sh`

> Historical (measured 2026-05-23 to 05-25 on Strix Halo, gfx1151; not
> re-measured since). The MTP host code tuned here (`llama_get_embeddings_pre_norm_ith()`,
> `common_speculative_batch_add_one_seq()`, the contiguous verify-row copy) was
> replaced by the ROCmFPX upstream MTP code in the 2026-09-15 merge
> (`40c5e4965`), and the rocWMMA FlashAttention build option
> `GGML_HIP_ROCWMMA_FATTN` left the tree in the same merge (removed upstream in
> llama.cpp #26046).

## What Was Tested

This compares Qwen3.6 27B MTP against Qwen3.6 27B MTP. It does not compare Qwen to Gemma.

The same merged binary was used for both model files:

`/path/to/llama.cpp-mtp-rocmfp4/build-strix-rocmfp4/bin/llama-cli`

Models:

- ROCmFP4: `/path/to/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf`
- Baseline Q5: `/path/to/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`

Shared settings:

- Context window: `262144`
- Reasoning: off
- Tools: off
- MTP: on, `draft-mtp`
- KV cache: `q4_0` for K and V
- Prompt tokens/s and generation tokens/s were reported by `llama-cli -st`

Shared flags:

```bash
-ngl 999 -c 262144 -b 512 -ub 512 -fa on \
-ctk q4_0 -ctv q4_0 --no-mmap --jinja -cnv -st \
--reasoning off \
--spec-type draft-mtp \
--spec-draft-ngl all \
--spec-draft-type-k q4_0 \
--spec-draft-type-v q4_0 \
--spec-draft-n-max 4 \
--spec-draft-n-min 0 \
--spec-draft-p-min 0.0 \
--spec-draft-p-split 0.10 \
--seed 123 --temp 0.2 --top-k 20 --top-p 0.9 \
--no-display-prompt -n 160
```

Device-specific flags:

- ROCm: `-dev ROCm0 --spec-draft-device ROCm0`
- Vulkan: `-dev Vulkan0 --spec-draft-device Vulkan0`

## Results

| Model | Backend | Context | MTP | Reasoning | Tools | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | on | off | off | 99.8 | 27.6 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | ROCm0 | 262144 | on | off | off | 47.9 | 15.7 |
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | on | off | off | 123.3 | 24.9 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | Vulkan0 | 262144 | on | off | off | 90.4 | 18.4 |

## Notes

On these controlled 262k-context runs with the same merged binary, ROCmFP4 was faster than the Q5 baseline:

- ROCm0: `27.6 tok/s` vs `15.7 tok/s`, about 76% faster decode.
- Vulkan0: `24.9 tok/s` vs `18.4 tok/s`, about 35% faster decode.

After each run, `rocm-smi --showpids` showed no KFD processes running, so VRAM was released.

## Qwopus3.6 27B v2 MTP ROCmFP4 Native 262k

The Jackrong Qwopus3.6 27B v2 MTP BF16 GGUF was downloaded and converted to
ROCmFP4 STRIX_LEAN on 2026-05-25.

- Source BF16: `/path/to/models/Qwopus3.6-27B-v2-MTP-GGUF/BF16/Qwopus3.6-27B-v2-MTP-BF16.gguf`
- ROCmFP4: `/path/to/models/Qwopus3.6-27B-v2-MTP-GGUF/Qwopus3.6-27B-v2-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf`
- Source size: `52115.19 MiB`, `16.00 BPW`
- ROCmFP4 size: `14120.35 MiB`, `4.34 BPW`
- Native context metadata: `262144`
- MTP metadata: `nextn_predict_layers = 1`

Winning first-pass 262k profile:

```bash
-dev ROCm0 --spec-draft-device ROCm0 \
-ngl 999 -c 262144 -b 512 -ub 512 -fa on \
-ctk q4_0 -ctv q4_0 --no-mmap --jinja -cnv -st \
--reasoning on \
--spec-type draft-mtp \
--spec-draft-ngl all \
--spec-draft-type-k q4_0 \
--spec-draft-type-v q4_0 \
--spec-draft-n-max 4 \
--spec-draft-n-min 0 \
--spec-draft-p-min 0.0 \
--spec-draft-p-split 0.10 \
--seed 123 --temp 0.2 --top-k 20 --top-p 0.9 \
--no-display-prompt -n 160
```

Initial sweep results:

| Model | Backend | Context | KV cache | Draft KV | `n-max` | `p-min` | Reasoning | Tools | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | off | off in CLI guard | 33.8 | 24.2 | stable |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.7 | first-pass best sustained |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 34.9 | 29.6 | retest after rejected sampler tweak was reverted |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 34.9 | 29.9 | `-b 1024 -ub 512`; promoted Qwopus sustained profile |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.8 | `-b 1024 -ub 512`; post MTP embedding-fetch cleanup retest |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.9 | `-b 2048 -ub 512`; tied, heavier batch |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 34.3 | 29.8 | `-b 1024 -ub 1024`; tied/slower |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.8 | `-b 1280 -ub 512`; tied/slower |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 34.9 | 29.9 | `-b 1536 -ub 512`; tied, no promotion over smaller batch |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.8 | `-b 1536 -ub 768`; tied/slower |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.05 | on | off in CLI guard | 35.0 | 29.9 | `-b 1024 -ub 512`; light p-min tied, no promotion |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.10 | on | off in CLI guard | 35.0 | 29.9 | `-b 1024 -ub 512`; light p-min tied, no promotion |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.0 | 29.9 | `--backend-sampling`; sustained tied but prompt throughput fell, no promotion |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.1 | 29.9 | `-t 12 -tb 32 --spec-draft-threads 12 --spec-draft-threads-batch 32`; tied |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.1 | 29.9 | `-t 24 -tb 32 --spec-draft-threads 24 --spec-draft-threads-batch 32`; tied |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 40.0 | 27.7 | Vulkan verified; better burst, lower sustained than ROCm0 |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | q4 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 39.9 | 27.7 | `-b 1024 -ub 512`; same sustained as Vulkan `512/512` |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | q4 K/V | q4 K/V | 3 | 0.00 | on | off in CLI guard | 34.2 | 27.1 | slower than Vulkan n-max 4 |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | q4 K/V | q4 K/V | 5 | 0.00 | on | off in CLI guard | 36.8 | 26.3 | burst-only, sustained regression |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | q8 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 36.5 | 25.7 | rejected; q8 main KV regressed |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q8 K/V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 36.6 | 26.0 | rejected; q8 main KV regressed |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q8 K/V | 4 | 0.00 | on | off in CLI guard | 35.1 | 29.8 | draft-only q8 tied/slower |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q8 K/V | q8 K/V | 4 | 0.00 | on | off in CLI guard | 36.6 | 26.0 | faster burst, sustained regression |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q8 K, q4 V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 35.7 | 22.3 | K-only q8 sustained regression |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K, q8 V | q4 K/V | 4 | 0.00 | on | off in CLI guard | 34.2 | 24.7 | V-only q8 sustained regression |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q8 K/V | q4 K/V | 3 | 0.25 | on | off in CLI guard | 31.9 | 24.5 | rejected; q8 main KV regressed |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 1 | 0.00 | on | off in CLI guard | 21.9 | 19.9 | too conservative |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 2 | 0.00 | on | off in CLI guard | 28.4 | 26.6 | slower sustained |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 3 | 0.00 | on | off in CLI guard | 33.8 | 27.3 | slower sustained |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 3 | 0.25 | on | off in CLI guard | 32.8 | 26.5 | slower sustained |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 5 | 0.00 | on | off in CLI guard | 35.4 | 27.7 | burst-only, slower sustained |
| Qwopus3.6-27B-v2-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | q4 K/V | q4 K/V | 4 | 0.25 | on | off in CLI guard | 34.6 | 27.9 | slower sustained |

This model behaves more like the dense 27B path than the 35B A3B MoE path:
q4 main KV and q4 draft KV are currently best, while the 35B q8-main-KV profile
regresses sustained decode here.

The MTP helper's top-k probability accumulator was briefly narrowed from
`double` to `float` because the stored probabilities are floats. Dense 27B still
passed at `34.0` / `28.1 tok/s`, but the 35B A3B short guard repeated below
floor at `84.1` then `93.4 tok/s`. Restoring the `double` accumulator recovered
the 35B short guard to `104.4 tok/s`, so the float accumulator is rejected.

A later host-path cleanup moved `llama_get_embeddings_pre_norm_ith()` in the
MTP draft loop so it only runs when another draft token will actually be
queued. This avoids fetching a pre-norm embedding pointer for p-min rejects and
for the final accepted draft token at `n-max`. It built cleanly and passed the
focused guards: dense 27B `34.0` / `28.1 tok/s`, 35B A3B `104.4` /
`90.1 tok/s`, and Qwopus best-profile ROCm0 `35.0` / `29.8 tok/s`. The default
serial all-regression gate also passed after the cleanup, ending with dense
27B `34.0` / `28.1 tok/s` and no KFD PIDs running.

A follow-up attempt to skip `common_sampler_accept()` for the final `n-max`
draft token was rejected. It passed the 35B A3B short check at `104.5 tok/s`
but dropped the sustained guard to `81.1 tok/s`, below the `85.0 tok/s` floor.
After reverting only that sampler-accept change and confirming no KFD PIDs were
running, the same 35B guard recovered to `104.3` / `90.0 tok/s`.

A single-sequence MTP `draft()` fast path was also tested and rejected on
2026-05-25. It removed the active-sequence bookkeeping loop for `n_seq == 1`
and passed the dense 27B guard at `33.7` / `28.0 tok/s`, but the 35B A3B
sustained guard collapsed to `25.7 tok/s` despite a passing `103.1 tok/s`
short check. Reverting that path restored the 35B A3B guard to `104.3` /
`90.3 tok/s`, so the shared multi-sequence draft loop remains the promoted
implementation.

## Qwen3.6 35B A3B ROCmFP4 Native 262k

The 35B A3B MTP ROCmFP4 model was also checked on ROCm0 at native `262144`
context with the same STRIX_LEAN ROCmFP4 build. This is a separate MoE model
profile, not a direct replacement for the 27B dense baseline comparison above.
It started from q4 main/draft KV cache, then promoted q8 main KV with q4 draft
KV after the reasoning-on isolation sweep below.

Original q4-KV sweep flags:

```bash
-dev ROCm0 -ngl 999 -c 262144 -b 512 -ub 512 -fa on \
-ctk q4_0 -ctv q4_0 --no-mmap --jinja -cnv -st \
--spec-type draft-mtp \
--spec-draft-device ROCm0 \
--spec-draft-ngl all \
--spec-draft-type-k q4_0 \
--spec-draft-type-v q4_0 \
--spec-draft-n-min 0 \
--spec-draft-p-min 0.0 \
--spec-draft-p-split 0.10
```

Discovery sweep, reasoning off and tools off:

| Model | Backend | Context | `--spec-draft-n-max` | Reasoning | Tools | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 1 | off | off | 77.2 | 72.2 | best sustained in reasoning-off sweep |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 2 | off | off | 91.1 | 70.8 | slower sustained |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 3 | off | off | 102.0 | 68.5 | slower sustained |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 4 | off | off | 85.2 | 67.4 | slower sustained |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 5 | off | off | 107.3 | 64.9 | best burst, sustained regression |

Reasoning-on sweep for the Pi serving profile:

| Model | Backend | Context | `--spec-draft-n-max` | Reasoning | Tools | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 1 | on | off in CLI guard; Pi server exposes `--tools all` | 80.6 | 76.5 | previous sustained default |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 2 | on | off in CLI guard; Pi server exposes `--tools all` | 92.6 | 80.6 | best q4-KV sustained profile |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 3 | on | off in CLI guard; Pi server exposes `--tools all` | 104.3 | 80.1 | close sustained, better burst |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 4 | on | off in CLI guard; Pi server exposes `--tools all` | 101.8 | 77.0 | slower sustained |
| Qwen3.6-35B-A3B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | 5 | on | off in CLI guard; Pi server exposes `--tools all` | 98.6 | 73.3 | burst-only, sustained regression |

The Pi aliases `qwen36-35b-a3b-mtp-rocmfp4-native-262k`,
`qwen36-35b-a3b-mtp-rocmfp4-262k-rocm-best`, and
`qwen36-35b-a3b-mtp-rocmfp4-highest-native-262k` use `n-max 3`, q8 main KV,
and q4 draft KV for sustained interactive serving. A separate burst alias,
`qwen36-35b-a3b-mtp-rocmfp4-262k-burst-nmax4`, keeps the `n-max 4` setting for
short-response experimentation. The older
`qwen36-35b-a3b-mtp-rocmfp4-262k-burst-nmax5` alias is retained for comparison.

Pi server validation for `qwen36-35b-a3b-mtp-rocmfp4-highest-native-262k`
completed on 2026-05-25. The server log showed `n_ctx = 262144`,
`draft-mtp` speculative decoding initialized, built-in tools enabled, and
`thinking = 1`. The server was stopped after validation and ROCm reported no
KFD PIDs running.

The focused guard for this promoted 35B A3B profile is:

```bash
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
```

The focused guard now defaults to `n-max 3`, q8 main KV, q4 draft KV, and
`--spec-draft-p-min 0.25`. The promoted top-k-10 MTP sampler pass measured
`103.9 tok/s` short and `90.0 tok/s` sustained after a first sustained run at
`89.8 tok/s`. A follow-up MTP draft sampler cleanup keeps the same top-10
probability distribution but skips the unused final RNG sampler selection; the
candidate measured `104.6 tok/s` short and `90.2 tok/s` sustained on the
35B A3B guard, while the dense 27B guard held `33.9 tok/s` short and
`28.1 tok/s` sustained. Guard floors are `100.0` and `85.0 tok/s` for 35B,
and `30.0` / `25.5 tok/s` for dense 27B. (The dense 27B floors were lowered to
`12.0` / `11.5 tok/s` on 2026-06-24, `db247885a`.)

Current KV and draft-depth isolation on the promoted reasoning-on 35B profile:

| Setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `n-max 3`, q8 main KV, q4 draft KV, `p-min 0.25` | 104.3 | 90.1 | promoted sustained profile |
| `n-max 3`, q8 main KV, q8 draft KV, `p-min 0.25` | 104.5 | 90.0 | tied, heavier draft KV |
| `n-max 3`, q4 main KV, q8 draft KV, `p-min 0.25` | 104.3 | 82.2 | draft-only q8 is not enough |
| `n-max 3`, q8 K only, q4 V/draft KV, `p-min 0.25` | 89.9 | 70.5 | rejected |
| `n-max 3`, q4 K, q8 V only, q4 draft KV, `p-min 0.25` | 90.2 | 74.6 | rejected |
| `n-max 2`, q8 main KV, q4 draft KV, `p-min 0.25` | 93.4 | 84.8 | slower sustained |
| `n-max 4`, q8 main KV, q4 draft KV, `p-min 0.25` | 109.2 | 80.2 | burst-only, sustained regression |
| `n-max 3`, q8 main KV, q4 draft KV, `p-min 0.0` | 104.4 | 89.4 | close, not promoted |

The updated default guard pass after promoting this shape measured `103.3 tok/s`
short and `90.1 tok/s` sustained, clearing the `100.0` / `85.0 tok/s` floors.

Earlier acceptance-threshold checks on the q4-KV `n-max 2` profile did not
produce a clear sustained improvement. Those checks are retained as background;
the q4-KV profile was later superseded by the q8-main-KV `n-max 3`,
`p-min 0.25` profile above:

| Setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `p-min 0.25` | 93.4 | 80.7 | tied, not promoted |
| `p-min 0.50` | 93.5 | 80.6 | tied, not promoted |
| `p-min 0.75` | 93.3 | 80.5 | tied, not promoted |
| `p-split 0.05` | 93.5 | 80.8 | tied, not promoted |
| `p-split 0.20` | 93.5 | 80.6 | tied, not promoted |
| `n-min 1` | 93.6 | 80.6 | tied, not promoted |

KV-cache isolation found a real sustained gain from q8 main KV. Draft-only q8
did not help, and q8 K-only / V-only main KV regressed, so both accepted K and
V need q8 while the draft KV can remain q4:

| `--spec-draft-n-max` | Main KV | Draft KV | Extra setting | Short decode tok/s | Sustained decode tok/s | Result |
|---:|---|---|---|---:|---:|---|
| 2 | q4 K/V | q4 K/V | default before KV sweep | 93.6 | 80.5 | previous promoted profile |
| 2 | q8 K/V | q8 K/V | none | 93.2 | 85.2 | faster sustained, heavier draft KV |
| 2 | q8 K/V | q4 K/V | none | 93.7 | 85.6 | faster sustained |
| 2 | q4 K/V | q8 K/V | none | 93.7 | 80.5 | draft-only q8 does not help |
| 2 | q8 K, q4 V | q4 K/V | none | 83.9 | 71.8 | rejected |
| 2 | q4 K, q8 V | q4 K/V | none | 83.1 | 73.3 | rejected |
| 2 | q8 K/V | q4 K/V | `p-min 0.25` | 93.7 | 85.4 | tied, not promoted |
| 3 | q8 K/V | q4 K/V | none | 104.3 | 89.3 | promoted sustained profile |
| 4 | q8 K/V | q4 K/V | none | 111.2 | 78.7 | best burst, sustained regression |
| 5 | q8 K/V | q4 K/V | none | 104.9 | 78.9 | slower burst than n-max 4 |

Additional p-min filtering on the promoted `n-max 3`, q8-main/q4-draft
profile originally failed to produce a clear sustained win while the MTP
internal sampler was still hardcoded to `top_k=1`. After changing the internal
MTP sampler to `top_k=10`, the draft loop still selects the top sorted
candidate, but `--spec-draft-p-min` sees a meaningful candidate distribution.
That makes `p-min 0.25` a promoted 35B A3B sustained profile. The dense 27B
profile remains at `p-min 0.0`; the same `p-min 0.25` filter regressed 27B
sustained decode to `24.6 tok/s`.

| Extra setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `top_k=10` internal MTP, `--spec-draft-p-min 0.0` | 104.3 | 89.3 | tied previous promoted path |
| `top_k=10` internal MTP, `--spec-draft-p-min 0.25` | 103.9 | 90.0 | promoted sustained profile |
| pre-change `--spec-draft-p-min 0.25` | 104.2 | 89.1 | near tie, not promoted |
| `--spec-draft-p-min 0.50` | 104.1 | 89.0 | slower sustained |
| `--spec-draft-p-min 0.75` | 104.0 | 89.3 | tied sustained but slower short decode |
| `--spec-draft-p-min 0.90` | 104.2 | 89.1 | near tie, not promoted |
| `--spec-draft-p-split 0.05` | 103.7 | 88.9 | slower |
| `--spec-draft-p-split 0.20` | 103.8 | 89.3 | tied sustained but slower short decode |
| `--spec-draft-n-min 1` | 103.8 | 88.9 | slower |
| `--spec-draft-n-min 2` | 104.2 | 89.2 | near tie, not promoted |

Reasoning-off checks on the final q8-main/q4-draft profile were also measured
because reasoning mode changes the model's generated content path. They do not
replace the reasoning-on promoted profile, but if reasoning is disabled the
best sustained setting in this bracket is `n-max 2`:

| Reasoning | `--spec-draft-n-max` | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---|
| off | 1 | 77.7 | 73.9 | slower short decode |
| off | 2 | 90.3 | 75.5 | best reasoning-off sustained in bracket |
| off | 3 | 100.3 | 71.9 | current reasoning-on n-max, but slower with reasoning off |
| off | 4 | 85.7 | 66.1 | sustained regression |

Batch and thread follow-ups on the promoted `n-max 3`, q8-main/q4-draft profile
did not improve sustained decode, so the Pi/server defaults remain `-b 512`,
`-ub 512`, `-t 16`, `-tb 32`, and matching draft thread counts:

| Setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `-b 512 -ub 512 -t 16 -tb 32` | 104.2 | 89.3 | promoted default guard pass |
| `-b 1024 -ub 512` | 103.1 | 89.1 | no improvement |
| `-b 2048 -ub 512` | 104.4 | 89.1 | no sustained improvement |
| `-b 512 -ub 256` | 104.3 | 89.0 | no improvement |
| `-t 24 -tb 32`, draft threads 24/32 | 104.2 | 89.1 | no improvement |
| `-t 12 -tb 32`, draft threads 12/32 | 104.3 | 88.7 | slower sustained |

Keeping target threads at `-t 16 -tb 32` while changing only draft threads
also failed to improve sustained decode:

| Draft thread setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `--spec-draft-threads 8 --spec-draft-threads-batch 16` | 104.1 | 89.1 | no improvement |
| `--spec-draft-threads 16 --spec-draft-threads-batch 16` | 104.2 | 89.1 | no improvement |
| `--spec-draft-threads 24 --spec-draft-threads-batch 32` | 104.2 | 89.1 | no improvement |

Sampler-chain and backend-sampling follow-ups also failed to beat the promoted
shape. The default sampler path remains active for the guard and Pi profile:

| Setting | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `--samplers top_k;top_p;temperature` | 104.2 | 87.5 | rejected; sustained regression |
| `--samplers penalties;top_k;top_p;min_p;temperature` | 104.1 | 88.9 | rejected; slower sustained |
| `--backend-sampling` | 104.3 | 89.2 | near tie, not promoted |

MoE routing was also checked because the 35B A3B path is much more sensitive to
`MUL_MAT_ID` behavior than the dense 27B comparison. A separate build with
`CMAKE_HIP_FLAGS=-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3` measured only
`95.8 tok/s` short and `74.0 tok/s` sustained, so the promoted build keeps the
default `MMVQ_MAX_BATCH_SIZE` routing threshold. The existing
`GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4` exploratory build was also checked on
the same 35B guard and measured `104.2 tok/s` short and `89.2 tok/s`
sustained. That is a near-tie, but it still does not beat the promoted
`89.3 tok/s` sustained band, so it was not rebuilt from current source or
promoted.

The older rocWMMA FlashAttention build was also checked against the 35B A3B
profile after it had already regressed the dense 27B guard. It measured
`99.7 tok/s` short and `76.1 tok/s` sustained, so rocWMMA remains opt-in only
and is not promoted for the current Strix Halo ROCmFP4 MTP path. (The option
has since been removed; see the note at the top.)

A single-sequence MTP `process()` fast path was also prototyped for the `-np 1`
serving case. It compiled and passed the guard floors, but measured only
`104.1 tok/s` short and `88.5 tok/s` sustained, below the promoted `89.3 tok/s`
sustained band, so the code change was removed.

The all-regression harness can include it with:

```bash
INCLUDE_QWEN35_A3B_GUARD=1 scripts/check-rocmfp4-all-regression.sh
```

## Where The Gains Came From

The largest measured gains so far came from backend-specific ROCmFP4 work:

| Area | Gain |
|---|---|
| ROCmFP4 weight format vs Q5 baseline | controlled Qwen3.6 27B MTP 262k ROCm decode improved from `15.7` to `27.7 tok/s` under the promoted sustained profile; the shorter guard holds `33.5 tok/s` |
| Vulkan ROCmFP4 scale LUT | Vulkan sustained Qwen MTP moved from the older `20.4 tok/s` band to `25.0`-`25.3 tok/s`; q8 main-KV fallback reached `27.0 tok/s` sustained |
| ROCm dual-scale MMVQ vector-dot ratio | sustained Qwen MTP improved from about `26.2` to `27.8 tok/s` |
| ROCm RDNA3.5 MMVQ routing | two-warp ROCmFP4 MMVQ now covers `n=1..2`; focused ROCm `n=2` improved from FAST/dual `68.85` / `60.98` us to `66.56` / `58.40` us while Qwen MTP held `27.7 tok/s` sustained |
| ROCmFP4 FlashAttention thread grouping and Codebook10 decode | focused ROCm FlashAttention dropped from about `122.33` / `115.41` us to `70.89` / `66.10` us for dual-scale / FAST; Qwen-style 128d dropped from `247.06` / `221.19` to `201.53` / `174.23` us after the single-half helper, K/Q block-pair specialization, and `V_ROWS_PER_THREAD=8` |
| ROCmFP4 HIP unaligned quant-byte dword load | promoted after the full gate; the isolated step measured `78.26` / `69.23` us for 64d dual-scale / FAST and `215.82` / `183.77` us for Qwen-style 128d dual-scale / FAST, while Qwen MTP held `33.4 tok/s` short and `27.7 tok/s` sustained |
| ROCmFP4 FAST MMVQ/MMQ packed-byte dword load | extended the ROCmFP4 unaligned packed-byte loader to FAST `MUL_MAT`; focused ROCm FAST moved to `45.17`, `58.38`, `90.54`, and `157.83` us for `n=1/2/4/8`, and Qwen MTP improved to `33.6 tok/s` short / `28.0 tok/s` sustained on the promoted default build |
| MMVQ single-warp reduction bypass | skips the shared-memory reduction storage, barrier, and dead `threadIdx.y > 0` return path when a ROCmFP4 MMVQ specialization launches with one warp; focused ROCm `n=4/8` improved to FAST `87.29` / `156.52` us and dual-scale `82.81` / `141.99` us, while Qwen MTP reached `34.1 tok/s` short / `28.1 tok/s` sustained |
| MTP sampler top-k-10 acceptance filtering | lets `--spec-draft-p-min` operate on a real top-candidate distribution while still drafting the top sorted candidate; the 35B A3B profile with q8 main KV and q4 draft KV moved from the `89.3`-`89.5 tok/s` sustained band to `90.0 tok/s` with `--spec-draft-p-min 0.25` |
| MTP probability-only draft sampler | the MTP draft loop only consumes the sorted top candidate and its probability, so it now fills top-10 probabilities directly and skips the unused final RNG sampler selection; the 35B A3B 262k reasoning-on guard measured `104.6 tok/s` short / `90.2 tok/s` sustained, and dense 27B held `33.9` / `28.1 tok/s` |
| ROCmFP4 CPY/dequant kernels | quant-to-F32 copies moved from the old `~740 us` band to about `182 us` dual-scale and `170 us` FAST |
| ROCmFP4 CPU finite-block quant scoring | normal GGUF quantization avoids per-value guarded decode during scale-MSE scoring after a block-level finite scan; latest guard measured dual-scale / FAST normal quant at `3844.38` / `3582.57` cycles/32 |
| ROCmFP4 weighted/imatrix finite scoring | FAST imatrix GGUF quantization moved from same-session pre-candidate `5258.07` to `4448.73` / `4447.32` cycles/32 on two guarded passes; dual imatrix stayed in the noisy guarded band |
| Speculative host-side cleanup | removed per-draft `std::vector<bool>` allocation from MTP and simple draft paths, changed MTP sequence-index tracking from token-by-sequence scanning to one pass over token seq-ids, skipped full verify-row copies when no draft is pending, moved simple and MTP draft paths to a shared direct one-sequence batch append helper, skipped re-copying `pending_h` when accept lands on the already staged final verify row, stores only the non-final verify rows needed for rollback, copies stored verify rows with one contiguous target-embedding memcpy, reuses the grown verify-row buffer instead of resizing every verification pass, skips idle draft-model decode when no sequence is drafting, skipped debug-candidate loop work unless debug logging is enabled, and hoisted the debug verbosity check out of the per-token draft loop; latest Qwen 27B guard held `33.7 tok/s` short / `27.9 tok/s` sustained and the 35B A3B guard held `104.1 tok/s` short / `89.2 tok/s` sustained |

The GitHub-facing reproduction harness is
`scripts/reproduce-rocmfp4-qwen-mtp-comparison.sh`. On 2026-05-24 it generated
`bench-reports/rocmfp4-qwen-mtp-comparison-20260524-124911.md` with the
promoted sustained profile (`--spec-draft-n-max 4`, 262k context, MTP on,
reasoning/tools off). That run measured:

| Model | Backend | Context | MTP | Reasoning | Tools | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | on | off | off | 99.8 | 27.6 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | ROCm0 | 262144 | on | off | off | 47.9 | 15.7 |
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | on | off | off | 123.3 | 24.9 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | Vulkan0 | 262144 | on | off | off | 90.4 | 18.4 |

The report also recorded `No KFD PIDs currently running` after the run.

The dual-scale ROCmFP4 layout is useful for long-context runs because it protects
coherence better than a single scale per 32 weights while keeping the model
compact. It should not be treated as a major long-context speed feature by
itself. At long context, the limiting costs are KV cache traffic, FlashAttention
over the long KV range, memory bandwidth, and MTP acceptance.

Target/draft "dual-stream" MTP overlap was inspected after the `n=8` guard work.
The current MTP path verifies on the target context, copies pre-norm embeddings
into the MTP context, and then drafts through serial `llama_decode(ctx_dft, ...)`
calls. ggml has async graph execution and pipeline-parallel support underneath,
but pipeline parallelism is only enabled for multi-device layer-split cases, not
this single Strix Halo ROCm0 target-plus-MTP flow. A true dual-stream MTP
prototype would need speculative scheduler changes and new correctness guards;
it is not a low-risk flag flip, and it should not be promoted unless it beats the
Qwen sustained guard.

## Latest Serial Regression Gate

On 2026-05-24, `scripts/check-rocmfp4-all-regression.sh` passed the full
serial ROCmFP4 promoted gate after promoting the ROCmFP4 FAST MMVQ/MMQ
packed-byte dword load on top of the FlashAttention `V_ROWS_PER_THREAD=8`
default and the HIP unaligned quant-byte load path. The all-in-one harness now
uses separate `TEST_BACKEND_OPS_BIN` and `LLAMA_CLI_BIN` values so candidate
builds can run backend-op guards and Qwen CLI guards from the same
`BUILD_DIR`. DeepSeek is no longer printed or scanned in the default gate; it
remains opt-in compatibility smoke only via `INCLUDE_DEEPSEEK_SMOKE=1`.

| Guard | Result |
|---|---|
| CPU quant/dequant/vec-dot | passed; dual quant `4295.14`, FAST quant `3815.74`, dual dequant `33.56`, FAST dequant `33.24`, dual vec-dot `31.11`, FAST vec-dot `27.57`, dual imatrix `5582.40`, FAST imatrix `4836.81` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.61` / `71.37` / `100.42` / `166.09` us and dual-scale `62.74` / `82.68` / `121.32` / `191.97` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `45.17` / `58.38` / `90.54` / `157.83` us and dual-scale `49.18` / `51.40` / `83.54` / `141.75` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `69.34` us, 64d FAST `66.08` us, Qwen-style 128d dual-scale `201.00` us, Qwen-style 128d FAST `171.71` us |
| ROCm CPY | passed; dual source-to-quant `1108.51` / `1010.72` / `1006.60` us, dual-to-F32 `182.28` us, FAST source-to-quant `1047.89` / `951.27` / `952.43` us, FAST-to-F32 `170.54` us |
| DeepSeek ROCmFP4 decode | not run in the promoted gate; optional compatibility smoke only with `INCLUDE_DEEPSEEK_SMOKE=1` |
| Qwen3.6 27B MTP ROCmFP4 decode | passed; `33.6 tok/s` short and `28.0 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-25 MTP Embedding Fetch Cleanup Gate

The current promoted tree passed `scripts/check-rocmfp4-all-regression.sh`
after the MTP draft loop was changed to fetch the draft pre-norm embedding row
only after the p-min and final `n-max` checks confirm another token will be
queued. This keeps sampler behavior unchanged while avoiding unused embedding
row lookups on rejected/final draft tokens.

| Guard | Result |
|---|---|
| CPU quant/dequant/vec-dot | passed; dual quant `4038.91`, FAST quant `3702.28`, dual dequant `33.66`, FAST dequant `33.19`, dual vec-dot `29.91`, FAST vec-dot `27.04`, dual imatrix `5704.54`, FAST imatrix `4698.38` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `53.06` / `72.10` / `104.52` / `166.73` us and dual-scale `64.76` / `78.84` / `118.74` / `192.59` us for `n=1/2/4/8` |
| Vulkan CPY | passed; F32/F16/BF16 source-to-dual `16476.68` / `3739.62` / `3909.14` us, dual-to-F32 `539.79` us, F32/F16/BF16 source-to-FAST `13867.67` / `3271.77` / `3394.37` us, FAST-to-F32 `539.53` us |
| ROCm runtime `MUL_MAT` | passed; FAST `45.66` / `57.81` / `88.27` / `155.05` us and dual-scale `49.16` / `51.58` / `83.34` / `151.42` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `70.86` us, 64d FAST `66.51` us, Qwen-style 128d dual-scale `189.45` us, Qwen-style 128d FAST `172.73` us |
| ROCm CPY | passed; F32/F16/BF16 source-to-dual `1106.89` / `1008.56` / `1006.60` us, dual-to-F32 `182.27` us, F32/F16/BF16 source-to-FAST `1050.49` / `958.98` / `950.50` us, FAST-to-F32 `171.15` us |
| Qwen3.6 27B MTP ROCmFP4 decode | passed; `33.9 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-25 Vulkan Source-To-ROCmFP4 Scale-Pruning Gate

The Vulkan `copy_to_quant.comp` ROCmFP4 scale search now stops scanning lower
scale candidates once clipping the block max alone cannot beat the current best
error. This mirrors the CPU/HIP exact-search bound; it does not change the
candidate set or replace the exhaustive MSE choice with a cheaper max-abs
shortcut. The Vulkan CPY guard script was also changed to stream the perf phase
through `tee` so long RADV runs keep producing output while still being parsed.

| Guard | Result |
|---|---|
| Vulkan CPY correctness | passed; `34/34` cases |
| F32/F16/BF16 source-to-dual | `9525.39` / `2350.54` / `2418.09` us/run |
| dual-to-F32 | `516.65` us/run |
| F32/F16/BF16 source-to-FAST | `10111.85` / `2923.67` / `2949.42` us/run |
| FAST-to-F32 | `509.65` us/run |
| Full promoted gate | passed after rebuild; Qwen3.6 27B MTP `33.9` / `28.0 tok/s`, no KFD PIDs running |
| Tightened Vulkan CPY guard | passed; ceilings now require source-to-dual under `11500` / `3000` / `3100` us for F32/F16/BF16 and source-to-FAST under `12200` / `3600` / `3700` us |
| Vulkan Qwen MTP retest | passed; `34.4 tok/s` short and `24.9 tok/s` sustained, so ROCm0 remains the promoted sustained backend |

### 2026-05-24 Simple Draft Allocation Cleanup

The simple draft path now reuses an object-owned `uint8_t` drafting-state
buffer instead of allocating `std::vector<bool>` on every draft call. MTP
already used the reusable buffer; this keeps both draft implementations on the
same low-overhead shape.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.7 tok/s` short and `28.0 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3984.59`, FAST quant `3621.96`, dual dequant `33.66`, FAST dequant `33.18`, dual vec-dot `31.26`, FAST vec-dot `27.08`, dual imatrix `5441.76`, FAST imatrix `4450.87` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `53.63` / `71.49` / `98.08` / `166.25` us and dual-scale `65.61` / `82.04` / `121.85` / `192.38` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.46` / `57.07` / `87.53` / `156.08` us and dual-scale `50.62` / `51.32` / `82.54` / `142.10` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `70.05` us, 64d FAST `65.74` us, Qwen-style 128d dual-scale `201.37` us, Qwen-style 128d FAST `172.29` us |
| ROCm CPY | passed; dual source-to-quant `1106.98` / `1009.09` / `1007.22` us, dual-to-F32 `183.63` us, FAST source-to-quant `1048.11` / `952.30` / `952.26` us, FAST-to-F32 `170.97` us |
| Qwen3.6 27B MTP in full gate | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-24 Idle Draft Decode Skip

The simple draft and MTP draft paths now return before `llama_decode(ctx_dft,
batch)` when no sequence is actively drafting. This avoids a pointless backend
call on idle/no-draft turns while leaving all active drafting behavior
unchanged.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.8 tok/s` short and `28.0 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3915.72`, FAST quant `3647.68`, dual dequant `33.79`, FAST dequant `33.18`, dual vec-dot `30.23`, FAST vec-dot `27.06`, dual imatrix `5428.11`, FAST imatrix `4427.69` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.68` / `71.43` / `104.49` / `166.14` us and dual-scale `61.90` / `83.45` / `121.43` / `192.32` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.33` / `58.28` / `88.33` / `155.63` us and dual-scale `51.07` / `51.77` / `83.50` / `143.01` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `70.17` us, 64d FAST `66.01` us, Qwen-style 128d dual-scale `201.20` us, Qwen-style 128d FAST `172.31` us |
| ROCm CPY | passed; dual source-to-quant `1109.19` / `1009.65` / `1008.15` us, dual-to-F32 `183.24` us, FAST source-to-quant `1046.87` / `951.34` / `951.50` us, FAST-to-F32 `170.94` us |
| Qwen3.6 27B MTP in full gate | passed; `33.7 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-24 MTP Verify-Row Copy Follow-Up

The MTP `process()` path now avoids storing the final target hidden row in
`verify_h`. That row is needed as the next-call carryover, so it is copied
directly to `pending_h`; `accept()` already skips reading `verify_h` when the
accepted row is the final row. Earlier verify rows are still retained for
partial-accept rollback, so behavior is unchanged.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.9 tok/s` short and `27.9 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3929.65`, FAST quant `3687.42`, dual dequant `33.71`, FAST dequant `33.24`, dual vec-dot `29.90`, FAST vec-dot `27.52`, dual imatrix `5518.54`, FAST imatrix `4423.78` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.32` / `71.52` / `104.81` / `166.35` us and dual-scale `65.21` / `83.47` / `121.34` / `192.54` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.21` / `57.18` / `89.07` / `155.99` us and dual-scale `50.48` / `50.96` / `85.22` / `141.72` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `69.43` us, 64d FAST `66.07` us, Qwen-style 128d dual-scale `202.26` us, Qwen-style 128d FAST `172.24` us |
| ROCm CPY | passed; dual source-to-quant `1106.50` / `1009.75` / `1006.69` us, dual-to-F32 `182.16` us, FAST source-to-quant `1045.70` / `951.60` / `951.16` us, FAST-to-F32 `170.65` us |
| Qwen3.6 27B MTP in full gate | passed; `33.7 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-24 MTP Debug-Guard Follow-Up

The MTP and simple draft hot loops now compute the debug-log enabled state once
per draft call instead of checking `common_log_get_verbosity_thold()` inside
the candidate loop. This is intentionally a small host-side cleanup, not a new
kernel path. It preserves debug output when debug logging is enabled and avoids
extra work in normal decode runs.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3922.91`, FAST quant `3632.90`, dual dequant `33.56`, FAST dequant `33.13`, dual vec-dot `31.23`, FAST vec-dot `27.51`, dual imatrix `5779.38`, FAST imatrix `4607.61` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `53.84` / `71.56` / `104.74` / `156.08` us and dual-scale `65.22` / `83.36` / `121.63` / `181.08` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.13` / `57.55` / `88.45` / `155.17` us and dual-scale `51.85` / `51.79` / `84.17` / `141.91` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `69.34` us, 64d FAST `66.20` us, Qwen-style 128d dual-scale `201.41` us, Qwen-style 128d FAST `172.81` us |
| ROCm CPY | passed; dual source-to-quant `1106.70` / `1008.79` / `1005.43` us, dual-to-F32 `183.70` us, FAST source-to-quant `1045.86` / `951.46` / `951.44` us, FAST-to-F32 `170.72` us |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-24 Shared One-Sequence Batch Append

The simple draft and MTP draft paths now use
`common_speculative_batch_add_one_seq()` instead of calling
`common_batch_add(..., { seq_id }, ...)` for the hot single-sequence case.
This avoids constructing a temporary sequence-id container while keeping
multi-sequence batch semantics untouched outside these speculative paths.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.9 tok/s` short and `27.9 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3970.74`, FAST quant `3618.75`, dual dequant `33.84`, FAST dequant `33.35`, dual vec-dot `30.06`, FAST vec-dot `27.23`, dual imatrix `5517.01`, FAST imatrix `4476.83` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.28` / `71.45` / `99.37` / `166.37` us and dual-scale `65.27` / `83.35` / `121.90` / `192.25` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.19` / `57.33` / `88.27` / `156.92` us and dual-scale `51.47` / `51.48` / `84.29` / `142.12` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `70.10` us, 64d FAST `66.23` us, Qwen-style 128d dual-scale `201.58` us, Qwen-style 128d FAST `172.38` us |
| ROCm CPY | passed; dual source-to-quant `1104.78` / `1008.55` / `1006.28` us, dual-to-F32 `182.14` us, FAST source-to-quant `1045.38` / `950.90` / `951.23` us, FAST-to-F32 `170.59` us |
| Qwen3.6 27B MTP in full gate | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-24 MTP Verify-Buffer Reuse

The MTP verify-row buffer now reserves for the configured draft depth at
construction and only grows when a verification batch needs more storage. It no
longer shrinks/resizes on every `process()` call. The copied rows and
`accept()` rollback behavior are unchanged; this only removes avoidable
host-side vector churn in the promoted MTP path.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| Full all-regression gate | passed |
| CPU quant/dequant/vec-dot | passed; dual quant `3994.32`, FAST quant `3653.92`, dual dequant `33.71`, FAST dequant `33.22`, dual vec-dot `31.27`, FAST vec-dot `27.60`, dual imatrix `5655.31`, FAST imatrix `4542.47` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.68` / `71.54` / `104.78` / `166.54` us and dual-scale `65.39` / `84.72` / `121.89` / `192.94` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.70` / `57.95` / `88.40` / `156.27` us and dual-scale `50.87` / `50.46` / `84.38` / `142.66` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `70.32` us, 64d FAST `66.02` us, Qwen-style 128d dual-scale `204.15` us, Qwen-style 128d FAST `172.47` us |
| ROCm CPY | passed; dual source-to-quant `1108.80` / `1008.77` / `1006.00` us, dual-to-F32 `183.43` us, FAST source-to-quant `1046.98` / `951.20` / `950.79` us, FAST-to-F32 `170.35` us |
| Qwen3.6 27B MTP in full gate | passed; `33.9 tok/s` short and `27.9 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-25 MTP Contiguous Verify-Row Copy

The MTP `process()` path now copies the retained verification hidden rows from
the target embedding buffer in one contiguous block instead of calling
`llama_get_embeddings_pre_norm_ith()` and `memcpy()` once per row. This uses
the same contiguous target embedding layout already used earlier in
`process()` when shifting target embeddings into the draft batch. Rollback
semantics are unchanged: non-final verify rows are still retained for partial
acceptance, and the final row is still staged directly into `pending_h`.

After this change:

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.7 tok/s` short and `27.9 tok/s` sustained |
| Focused Qwen3.6 35B A3B MTP guard | passed; `104.1 tok/s` short and `89.2 tok/s` sustained |

The same build later passed the full all-regression gate with the optional 35B
A3B guard enabled:

| Guard | Result |
|---|---|
| Full all-regression gate | passed with `INCLUDE_QWEN35_A3B_GUARD=1` |
| CPU quant/dequant/vec-dot | passed; dual quant `4064.88`, FAST quant `3626.84`, dual dequant `33.54`, FAST dequant `33.19`, dual vec-dot `30.03`, FAST vec-dot `27.08`, dual imatrix `5710.87`, FAST imatrix `4396.19` cycles / 32 values |
| Vulkan runtime `MUL_MAT` | passed; FAST `54.58` / `71.53` / `104.89` / `167.99` us and dual-scale `64.93` / `83.02` / `122.01` / `182.55` us for `n=1/2/4/8` |
| ROCm runtime `MUL_MAT` | passed; FAST `44.29` / `57.21` / `88.52` / `158.27` us and dual-scale `51.09` / `50.65` / `85.96` / `142.50` us for `n=1/2/4/8` |
| ROCm FlashAttention | passed; 64d dual-scale `69.72` us, 64d FAST `66.26` us, Qwen-style 128d dual-scale `201.39` us, Qwen-style 128d FAST `172.88` us |
| ROCm CPY | passed; dual source-to-quant `1107.35` / `1009.95` / `1007.65` us, dual-to-F32 `181.96` us, FAST source-to-quant `1046.48` / `950.92` / `956.18` us, FAST-to-F32 `170.26` us |
| Qwen3.6 27B MTP in full gate | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| Qwen3.6 35B A3B MTP in full gate | passed; `104.1 tok/s` short and `89.3 tok/s` sustained |
| ROCm cleanup | passed; no KFD PIDs running |

### 2026-05-25 Rejected MTP Draft Hidden-Row Pointer Hoist

The MTP `draft()` loop was tested with a contiguous
`llama_get_embeddings_pre_norm(ctx_dft)` pointer per draft decode iteration,
then direct `h_dft + i_batch*n_embd` row addressing. This mirrors the promoted
target verify-row cleanup, but it did not beat the promoted 35B A3B sustained
profile, so the code change was removed.

| Guard | Result |
|---|---|
| Build | `scripts/build-strix-rocmfp4-mtp.sh` passed |
| Focused Qwen3.6 27B MTP guard | passed; `33.8 tok/s` short and `27.9 tok/s` sustained |
| Focused Qwen3.6 35B A3B MTP guard, run 1 | passed; `104.3 tok/s` short and `88.7 tok/s` sustained |
| Focused Qwen3.6 35B A3B MTP guard, run 2 | passed; `104.3 tok/s` short and `89.2 tok/s` sustained |

Because the best repeat remained below the promoted `89.3 tok/s` sustained
band, MTP `draft()` keeps `llama_get_embeddings_pre_norm_ith()` in the hot loop.

## Follow-Up MTP Sweep

After the guarded quantizer optimization, the Vulkan 262k Qwen3.6 ROCmFP4 MTP
path was rechecked with neighboring `--spec-draft-n-max` settings.

Short one-sentence prompt:

| Backend | Context | `--spec-draft-n-max` | Decode tok/s |
|---|---:|---:|---:|
| Vulkan0 | 262144 | 3 | 24.7 |
| Vulkan0 | 262144 | 4 | 28.3 |
| Vulkan0 | 262144 | 5 | 37.3 |
| Vulkan0 | 262144 | 6 | 33.1 |

Longer forced-output prompt:

| Backend | Context | `--spec-draft-n-max` | Decode tok/s |
|---|---:|---:|---:|
| Vulkan0 | 262144 | 3 | 21.9 |
| Vulkan0 | 262144 | 4 | 20.4 |
| Vulkan0 | 262144 | 5 | 18.0 |
| ROCm0 | 262144 | 3 | 25.6 |
| ROCm0 | 262144 | 4 | 26.3 |
| ROCm0 | 262144 | 5 | 23.9 |

After the Vulkan ROCmFP4 shared UE4M3 scale-LUT optimization, the same Vulkan0
MTP neighborhood was rechecked:

| Backend | Context | `--spec-draft-n-max` | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---:|---|
| Vulkan0 | 262144 | 3 | 28.9 | 25.3 | best Vulkan sustained |
| Vulkan0 | 262144 | 4 | 34.9 | 25.0 | best Vulkan short |
| Vulkan0 | 262144 | 5 | 32.2 | 23.0 | sustained regression |
| Vulkan0 | 262144 | 6 | 29.0 | 19.8 | sustained regression |

The scale-LUT change moved Vulkan sustained output from the older `20.4 tok/s`
at `n-max 4` to `25.0 tok/s`, and `n-max 3` now reaches `25.3 tok/s`
sustained. This is a real end-to-end Vulkan gain, but ROCm0 remains the
promoted backend because its current guard still holds `33.5 tok/s` short and
`27.7 tok/s` sustained with `n-max 4`.

Post-LUT Vulkan0 KV-cache checks found one useful runtime profile:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `n-max 3`, q8 KV / draft KV | 28.8 | 25.3 | ties q4 sustained, slower short |
| `n-max 4`, q8 KV / draft KV | 34.8 | 27.0 | best Vulkan sustained |
| `n-max 4`, q4 KV, q8 draft KV | 34.6 | 25.0 | draft-only q8 does not explain gain |
| `n-max 4`, q8 KV, q4 draft KV | 34.7 | 26.9 | leaner near-tie Vulkan fallback |
| `n-max 4`, q8 KV, q4 draft KV, `p-min 0.25` | 34.8 | 27.0 | best lean Vulkan fallback |
| `n-max 4`, q8 KV, q4 draft KV, `p-min 0.25`, `n-min 1` | 34.7 | 26.9 | no sustained improvement |
| `n-max 4`, q8 KV, q4 draft KV, `p-min 0.75` | 34.7 | 26.9 | no sustained improvement |
| `n-max 4`, q8 KV, q4 draft KV, `p-min 0.25`, `p-split 0.05` | 34.6 | 26.9 | no sustained improvement |
| `n-max 4`, q8 KV, q4 draft KV, `p-min 0.25`, `p-split 0.20` | 34.7 | 26.9 | no sustained improvement |
| `n-max 4`, q8 K only, q4 V/draft KV | 34.8 | 25.4 | K-only q8 is slower sustained |
| `n-max 4`, q8 V only, q4 K/draft KV | 34.6 | 23.7 | V-only q8 is slower sustained |
| `n-max 5`, q8 KV / draft KV | 47.8 | 23.0 | faster burst, worse sustained |
| `n-max 5`, q8 KV, q4 draft KV, `p-min 0.25` | 47.8 | 23.0 | rejected; same sustained regression |
| `n-max 4`, f16 KV / draft KV | 34.9 | 22.5 | slower sustained |
| `n-max 4`, q8 KV, `-b 1024 -ub 512` | 34.9 | 26.9 | no improvement |
| `n-max 4`, q8 KV, `-b 2048 -ub 512` | 34.7 | 26.9 | no improvement |
| `n-max 4`, q8 KV, `-b 512 -ub 256` | 34.9 | 26.9 | no improvement |

Conclusion for Vulkan fallback: use `--spec-draft-n-max 4` with q8 main KV.
Full q8 KV plus q8 draft KV measured `27.0 tok/s` sustained, while q8 main KV
with q4 draft KV plus `--spec-draft-p-min 0.25` tied it at `27.0 tok/s`.
Main K-only q8 and main V-only q8 both regressed sustained output, so the
Vulkan fallback needs both accepted K and V at q8, while the draft KV can
remain q4. Wider or stricter acceptance knobs did not improve sustained output.
This does not beat the promoted ROCm0 profile, but it moves Vulkan0 sustained
output close to the ROCm0 `27.7 tok/s` guard.

ROCm0 also reached `27.7 tok/s` on the short one-sentence guard prompt with
`--spec-draft-n-max 3`, and `33.6 tok/s` with `--spec-draft-n-max 4`,
beating the matching Vulkan0 `24.7` / `28.3 tok/s` runs.

Conclusion: promote ROCm0 with `--spec-draft-n-max 4` for this Qwen MTP ROCmFP4
path. `n-max 4` improved both the short and longer ROCm prompts. `n-max 5`
regressed sustained longer output in this check.

After the ROCmFP4 dual-scale MMVQ vector-dot ratio was raised to full-block
coverage while FAST stayed on the previous half-block ratio, the same promoted
ROCm0 `n-max 4` run measured `33.6 tok/s` on the short prompt and `27.8 tok/s`
on the longer sustained prompt.

Post-MMVQ-tune ROCm0 `--spec-draft-n-max` sweep:

| `--spec-draft-n-max` | Short decode tok/s | Sustained decode tok/s | Result |
|---:|---:|---:|---|
| 3 | 27.8 | 25.6 | slower than promoted |
| 4 | 33.6 | 27.8 | promoted |
| 5 | 45.2 | 24.8 | faster burst, worse sustained |
| 6 | 41.3 | 24.0 | faster burst, worse sustained |

Post-MMVQ-tune ROCm0 acceptance-threshold checks with `n-max 4`:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `--spec-draft-n-min 1` | 33.7 | 27.8 | no sustained improvement |
| `--spec-draft-p-min 0.25` | 33.6 | 27.8 | no sustained improvement |
| `--spec-draft-p-min 0.75` | 33.5 | 27.6 | now enforced in MTP; no sustained improvement |

Follow-up checks on `n-max 5` confirmed that stricter acceptance does not fix
the sustained-output regression:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `n-max 5, --spec-draft-p-min 0.75` | 45.1 | 24.6 | rejected; burst-only gain |
| `n-max 5, --spec-draft-p-min 0.90` | 45.0 | 24.7 | rejected; burst-only gain |
| `n-max 5, --spec-draft-n-min 1` | 45.1 | 24.7 | rejected; burst-only gain |

The `--spec-draft-p-min` cutoff is now applied inside the MTP draft loop. This
fixes the flag behavior for future sweeps, but the tested Strix Qwen3.6 27B
MTP ROCmFP4 runs still promote `n-max 4, p-min 0.0`: stricter cutoffs tied the
promoted path at `n-max 4` and did not rescue the `n-max 5` sustained
regression.

Post-MMVQ-tune ROCm0 batch/KV checks with `n-max 4`:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `-b 1024 -ub 512` | 33.6 | 27.8 | no sustained improvement |
| `-b 1024 -ub 1024` | 33.4 | 27.7 | no improvement |
| `-b 2048 -ub 512` | 33.7 | 27.8 | no sustained improvement |
| `-b 512 -ub 256` | 33.7 | 27.8 | no sustained improvement |
| `q8_0` KV / draft KV | 33.6 | 25.4 | slower sustained |
| `q8_0` KV, q4 draft KV | 33.5 | 25.3 | slower sustained |
| `q8_0` K only, q4 V/draft KV | 31.5 | 21.4 | slower |
| `q8_0` V only, q4 K/draft KV | 31.1 | 21.9 | slower |
| `f16` KV / draft KV | 34.0 | 26.0 | slower sustained |

Post-MMVQ-tune ROCm0 RDNA3.5 launch geometry checks:

| Setting change | Focused FAST `MUL_MAT` | Focused dual `MUL_MAT` | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---:|---|
| `GGML_ROCMFP4_RDNA35_NWARPS=1` | 52.69 us | 54.68 us | 33.6 | 27.7 | previous default |
| `GGML_ROCMFP4_RDNA35_NWARPS=2` | 51.76 us | 52.97 us | 33.6 | 27.7 | promoted; microbench win, no sustained decode regression |
| two-warp launch extended to `n=1..2` | 66.56 us at `n=2` | 58.40 us at `n=2` | 33.5 | 27.7 | promoted; `n=2` microbench win, no sustained decode regression |
| `GGML_ROCMFP4_RDNA35_NWARPS=4` | 52.31 us | 57.87 us | not run | not run | rejected; dual-scale regression |
| `GGML_ROCMFP4_RDNA35_NWARPS=8` | 59.33 us | not run | not run | not run | rejected; FAST n=1 guard regression |
| two-warp launch extended to `n=1..4` | 51.14 us | 53.51 us | 33.4 | 23.6 | rejected; multi-column microbench improved, sustained MTP decode regressed |
| `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=1` | n/a | n/a | 33.4 | 27.7 | tied promoted path; keep default routing |
| `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=2` | n/a | n/a | 33.6 | 27.7 | tied promoted path; keep default routing |
| `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3` | n/a | n/a | 33.8 | 27.9 | tied promoted path; keep default routing |
| `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4` | n/a | n/a | 33.7 | 27.9 | tied promoted path; keep default routing |
| `GGML_ROCMFP4_RDNA35_RPB_WIDE=2`, `n=8` | 131.72 us | 1382.72 us | not run | not run | rejected; full wide rows break dual-scale |
| `GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST=2`, `n=8` | 135.31 us | 145.75 us | 33.3 | 27.6 | rejected for default; useful FAST microbench win but no Qwen MTP gain |
| FAST MMVQ/MMQ ROCmFP4 packed-byte dword load | 45.17 / 58.38 / 90.54 / 157.83 us for `n=1/2/4/8` | unchanged source path; dual guard remained 49.18 / 51.40 / 83.54 / 141.75 us | 33.6 | 28.0 | promoted; direct end-to-end Qwen MTP gain and tighter FAST runtime guard |
| single-warp MMVQ reduction bypass | 44.52 / 57.12 / 87.29 / 156.52 us for `n=1/2/4/8` | 51.01 / 51.19 / 82.81 / 141.99 us for `n=1/2/4/8` | 34.1 | 28.1 | promoted; removes the no-op shared reduction/barrier path from one-warp ROCmFP4 launches and improves the real Qwen MTP guard |
| FAST MMVQ `GGML_ROCMFP4_FAST_Q8_1_MMVQ_VDR=1` after packed-byte load | 60.18 us at `n=1` | not run | not run | not run | rejected; failed the focused FAST ROCm guard ceiling of 50.00 us before Qwen MTP |
| FAST MMVQ `GGML_ROCMFP4_FAST_Q8_1_MMVQ_VDR=4` retest after packed-byte load | 41.37 / 49.29 / 80.91 / 139.58 us for `n=1/2/4/8` | unchanged source path; dual guard stayed inside ceiling at 50.32 / 50.83 / 82.93 / 142.13 us | 34.0 | 24.7 | rejected; better focused ROCm microbench, but sustained Qwen MTP regressed below the 25.5 tok/s floor |
| ROCmFP4 direct `vec_dot_q_cuda_dispatch<type>` call instead of the generic constexpr function pointer | 45.16 / 57.52 / 89.44 / 156.49 us for FAST `n=1/2/4/8` | 50.83 / 51.11 / 84.51 / 143.27 us for dual `n=1/2/4/8` | 33.7 | 27.9 | rejected; compiled and passed floors, but did not improve sustained decode versus the promoted 28.0 tok/s band |

Post-routing ROCm0 MMQ vector-dot ratio checks:

| Setting change | Prompt tok/s | Decode tok/s | Focused ROCm guard note | Result |
|---|---:|---:|---|---|
| default MMQ `vdr=8` | 387.87 | 13.56 | accepted default | baseline |
| FAST MMQ `vdr=4` | 387.79 | 13.56 | not run; full bench tied baseline | rejected; no decode gain |
| FAST MMQ `vdr=16` | 388.24 | 13.56 | not run; full bench tied baseline | rejected; no decode gain |
| dual-scale MMQ `vdr=16` | 385.85 | 13.56 | passed guard; dual `n=4` was `88.39 us/run` vs accepted `87`-`88` us/run band | rejected; lower prompt throughput and no decode gain |

The code keeps compile-time knobs for future controlled checks:
`GGML_ROCMFP4_Q8_1_MMQ_VDR` and `GGML_ROCMFP4_FAST_Q8_1_MMQ_VDR`. The shipped
default remains `vdr=8` for ROCmFP4 MMQ because the alternates either tied or
slightly regressed the real Qwen3.6 27B STRIX_LEAN bench.

Post-routing ROCm0 FlashAttention checks:

| Setting change | Dual-scale FA | FAST FA | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---:|---:|---|
| generic RDNA quantized-K FA `nthreads=2` | 122.33 us | 115.41 us | 33.5 | 27.7 | previous default |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_KQ_NTHREADS=1` | 113.03 us | 109.33 us | 33.5 | 27.6 | promoted; FA microbench win, no material MTP decode regression |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_V_NTHREADS=8` | 103.93 us | 101.52 us | 33.5 | 27.7 | passed; superseded by smaller V grouping |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_V_NTHREADS=4` | 90.21 us | 82.86 us | 33.5 | 27.7 | passed; slower FA than `2` |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_V_NTHREADS=2` | 85.63 us | 80.74 us | 33.4 | 27.6 | promoted; best bracketed V-side FA setting |
| single-half Codebook10 FA decode helper | 82.24 us | 78.15 us | 33.5 | 27.7 | promoted; avoids expanding both nibble streams when FA uses one half |
| tightened FA guard repeat | 82.34 us | 78.18 us | not run | not run | guarded with dual <= 88 us and FAST <= 84 us |
| expanded Qwen-style 128d FA guard, default `KQ=1,V=2` | 246.91 us | 219.23 us | not run | not run | guarded; matches Qwen3.6-style 128d, 8 KV-head, 12x-GQA shape |
| Qwen-style 128d FA after single-half decode helper | 237.12 us | 206.50 us | 33.5 | 27.7 | promoted; original Qwen-style guard tightened to dual <= 250 us and FAST <= 225 us |
| K/Q block-pair decode specialization for `KQ_NTHREADS=1` | 81.62 us | 78.13 us | 33.2 | 27.7 | promoted; Qwen-style 128d improved to 228.58 us dual-scale and 199.32 us FAST, new guard dual <= 240 us and FAST <= 215 us |
| post-K/Q retest with `V_NTHREADS=4` | 86.30 us | 78.97 us | 33.5 | 27.7 | rejected; Qwen-style dual tied at 228.33 us but sustained decode did not improve and 64d/FAST FA regressed |
| ROCmFP4 V dequant `V_ROWS_PER_THREAD=8` after unaligned-load default | 68.82 us | 66.32 us | 33.5 | 27.7 | promoted; Qwen-style 128d improved to 201.13 / 172.22 us and sustained Qwen MTP held the promoted band |
| `V_NTHREADS=4` with promoted `V_ROWS_PER_THREAD=8` | 77.35 us | 77.25 us | 33.5 | 27.7 | rejected for default; Qwen-style dual improved to 187.72 us, but 64d and Qwen-style FAST regressed and Qwen MTP decode did not improve |
| dual-scale-only 128d V-thread specialization `GGML_ROCMFP4_FATTN_V_NTHREADS_D128_DUAL=4` | 70.37 us | 66.31 us | 33.7 | 27.9 | promoted; leaves 64d and FAST defaults intact while Qwen-style 128d dual-scale improves to 188.75 us and FAST stays guarded at 171.89 us |
| `V_NTHREADS=8` with promoted `V_ROWS_PER_THREAD=8` | 81.22 us | 78.36 us | not run | not run | rejected; every focused FA row regressed versus the promoted default, including Qwen-style 128d at 208.94 / 200.45 us |
| `V_ROWS_PER_THREAD=16` | build failed | build failed | not run | not run | rejected; ROCmFP4 V dequant supports only `2/4/8` and the fixed-copy helper rejects the resulting 32-byte move |
| `KQ_NTHREADS=2` with promoted `V_ROWS_PER_THREAD=8` | 80.68 us | 78.27 us | not run | not run | rejected; Qwen-style 128d dual-scale failed the guard at 245.84 us and FAST regressed to 207.04 us |
| Qwen-style 128d FA guard with `V_NTHREADS=4` | 244.01 us | 224.21 us | not run | not run | rejected; tiny dual win, FAST and 64d guard regress |
| Qwen-style 128d FA guard with `KQ_NTHREADS=2,V_NTHREADS=2` | 268.53 us | 226.32 us | not run | not run | rejected; Qwen-style dual and FAST regress |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_V_NTHREADS=1` | build failed | build failed | not run | not run | rejected; gfx1151 HIP local memory exceeded 64 KiB |
| ROCmFP4-only FA `KQ_NTHREADS=2,V_NTHREADS=2` | 91.52 us | 82.90 us | not run | not run | rejected; slower FA than promoted `KQ=1,V=2` |
| ROCmFP4-only FA `KQ_NTHREADS=4,V_NTHREADS=2` | 102.31 us | 92.00 us | not run | not run | rejected; slower FA than promoted `KQ=1,V=2` |
| ROCmFP4-only FA `GGML_ROCMFP4_FATTN_KQ_NTHREADS=4` | 136.79 us | 124.37 us | not run | not run | rejected; FA guard regression |
| `V_ROWS_PER_THREAD=2` on the promoted D128-specialized build | 94.31 us | 88.77 us | not run | not run | rejected; failed the focused FA guard and Qwen-style 128d also regressed to 270.00 / 223.88 us |

The promoted setting affects only `Q4_0_ROCMFP4` and `Q4_0_ROCMFP4_FAST`
vector FlashAttention K/Q and V work on ROCm/HIP. Other quantized K/V-cache FA
paths keep the existing backend RDNA defaults.

The ROCm runtime guard now also checks multi-column ROCmFP4 shapes serially:

| Shape | FAST `MUL_MAT` | Dual-scale `MUL_MAT` | Result |
|---|---:|---:|---|
| `n=1` | 51.87 us | 52.08 us | guarded; ceilings 55 / 56 us |
| `n=2` | 68.01 us | 61.10 us | guarded; ceilings 72 / 65 us |
| `n=4` | 103.07 us | 88.68 us | guarded; ceilings 108 / 94 us |
| `n=8` | 170.12 us | 149.22 us | guarded; ceilings 190 / 170 us |

After promoting the FAST packed-byte dword load, the ROCm runtime guard was
tightened for FAST rows and now protects the improved band:

| Shape | FAST `MUL_MAT` | Dual-scale `MUL_MAT` | Result |
|---|---:|---:|---|
| `n=1` | 45.17 us | 49.18 us | FAST ceiling tightened to 50 us |
| `n=2` | 58.38 us | 51.40 us | FAST ceiling tightened to 64 us |
| `n=4` | 90.54 us | 83.54 us | FAST ceiling tightened to 100 us |
| `n=8` | 157.83 us | 141.75 us | FAST ceiling tightened to 178 us |

The Vulkan runtime guard now checks the matching multi-column shapes:

| Shape | FAST `MUL_MAT` | Dual-scale `MUL_MAT` | Result |
|---|---:|---:|---|
| `n=1` | 54.10 us | 64.57 us | guarded; ceilings 56 / 70 us |
| `n=2` | 71.39 us | 82.79 us | guarded; ceilings 76 / 88 us |
| `n=4` | 104.73 us | 121.57 us | guarded; ceilings 112 / 130 us |
| `n=8` | 163.41 us | 194.27 us | guarded; ceilings 190 / 220 us |

The Vulkan improvement came from adding a shared ROCmFP4 UE4M3 scale lookup
table, matching the existing NVFP4 style but preserving ROCmFP4's half-scale
semantics. The previous serial Vulkan guard pass was FAST `60.85`, `86.66`,
and `128.44` us/run for `n=1/2/4`, and dual-scale `82.86`, `120.77`, and
`181.28` us/run for `n=1/2/4`.

Replacing the shared ROCmFP4 Codebook10 table with inline integer decode was
tested and rejected. It compiled, but regressed the focused Vulkan FAST `n=1`
guard to `99.43` us/run, so the shared codebook table remains on the promoted
Vulkan path.

Replacing the HIP scalar `rocmfp4_decode_i8()` ternaries with a branchless
integer sign/magnitude expression was also tested and rejected. It compiled and
passed the focused ROCm CPY, FlashAttention, and Qwen MTP guards, but did not
produce an end-to-end gain: Qwen3.6 27B MTP measured `33.4 tok/s` short and
`27.7 tok/s` sustained versus the accepted `33.5` / `27.7` band, while the
focused ROCm FlashAttention guard slowed to `86.38` us dual-scale and
`81.13` us FAST versus the prior `85.54` / `80.14` serial pass. The original
decode helper remains the default.

Replacing CPU-side ROCmFP4 decode calls in the quantizer/dequantizer with the
Codebook10 table was tested and rejected. The full table variant failed the CPU
quant guard because dequantization regressed to `49.04` cycles/32 for dual-scale
and `84.23` cycles/32 for FAST. The MSE-loop-only variant passed the ceilings
but still slowed normal quantization to `4183.33` / `4018.68` cycles/32 versus
the restored arithmetic baseline at `4034.33` / `3738.85` cycles/32. The
arithmetic CPU decode helper remains the default.

Replacing the HIP finite UE4M3 scale arithmetic with a constant-memory lookup
table was tested and rejected. It compiled, but the ROCm CPY guard regressed
source-to-quant and FAST-to-F32 paths (`F32->dual 1136.09 us`,
`F32->FAST 1071.33 us`, `FAST->F32 178.40 us`), and the focused ROCm
FlashAttention guard failed with `95.02 us` dual-scale and `90.12 us` FAST.
The arithmetic finite-scale decoder remains the default.

These are regression guards, not new promoted decode numbers. They protect the
ROCmFP4 path against future kernel changes that look fine on single-token
matvec but regress MTP-style multi-column work. The `n=8` rows were added after
the rejected `GGML_ROCMFP4_RDNA35_RPB_WIDE=2` ROCm candidate improved FAST
`n=8` but regressed dual-scale `n=8` to `1382.72 us/run`.

Additional rejected ROCm0 sweeps on the longer forced-output prompt:

| Setting change | Decode tok/s | Result |
|---|---:|---|
| `-b 1024 -ub 512` | 26.2 | no improvement |
| `-b 2048 -ub 512` | 26.2 | no improvement |
| `-b 512 -ub 256` | 26.2 | no improvement |
| `--spec-draft-n-min 1` | 26.2 | no improvement |
| `--spec-draft-p-min 0.25` | 26.2 | no improvement |
| `--spec-draft-p-min 0.75` | 26.2 | no improvement |
| `--spec-draft-p-split 0.05` | 27.7 | no improvement after MMVQ tune |
| `--spec-draft-p-split 0.20` | 27.7 | no improvement after MMVQ tune |
| `q8_0` KV / draft KV | 23.6 | slower |
| `f16` KV / draft KV | 24.6 | slower |
| `--swa-full` | 27.6 | no improvement after MMVQ tune |
| `--no-host` | 26.2 | no improvement |
| `--no-op-offload` | 27.7 | no improvement after MMVQ tune |
| `--no-repack` | 27.6 | no improvement after MMVQ tune |
| `--poll 100 --spec-draft-poll 1` | 26.2 | no improvement |
| `--poll 0 --spec-draft-poll 0` | 26.2 | no improvement |
| `--poll-batch 1 --spec-draft-poll-batch 1` | 27.7 | no improvement after MMVQ tune |
| `-t 8 -tb 16 --spec-draft-threads 8 --spec-draft-threads-batch 16` | 26.2 | no improvement |
| `-t 24 -tb 32 --spec-draft-threads 24 --spec-draft-threads-batch 32` | 27.7 | no improvement after MMVQ tune |
| `--samplers top_k;top_p;temperature` | 26.6 | slower; changed acceptance/output path |
| `--samplers top_k;typ_p;top_p;min_p;temperature` | 27.7 | tied promoted path |
| greedy `--temp 0` | 23.5 | slower |
| `--backend-sampling` | 26.2 | no decode gain; lower prompt speed |
| `--spec-type draft-mtp,ngram-simple` | 19.1 | slower |
| `--spec-type draft-mtp,ngram-map-k` | 19.1 | slower |
| `--spec-type draft-mtp,ngram-map-k4v` | 19.1 | slower |
| `--spec-type draft-mtp,ngram-mod` | 19.1 | slower |
| `--spec-type draft-mtp,ngram-cache` | 21.4 | slower |
| rocWMMA FlashAttention build | 23.3 | slower sustained decode |
| dual-scale and FAST MMVQ `vdr=4` | 24.2 | slower sustained decode |
| RDNA3.5 two-warp launch for `n=1..4` | 23.6 | slower sustained decode |

Rejected Qwen3.6 35B A3B MoE launch-shape checks on the promoted reasoning-on
`n-max 3`, q8-main/q4-draft profile:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| ROCmFP4 MoE `rows_per_block=4` | 103.8 | 89.1 | rejected; close sustained tie, but below the promoted `104.3` / `89.3` band |
| ROCmFP4 MoE `rows_per_block=2` | 104.6 | 90.0 | rejected; tied the promoted sustained band but did not improve it |
| Dedicated MMVQ MoE `rows_per_block=3` | n/a | 86.8 | rejected; slower than promoted sustained band |
| ROCmFP4 MoE `rows_per_block=1` | 103.6 | 88.7 | rejected; slower than default |
| ROCmFP4 MoE `rows_per_block=1` repeat after top-k/p-min promotion | 90.9 | not run | rejected after short-response regression |
| `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS=3` | n/a | 87.5 | rejected; same-session promoted build reached `89.6 tok/s` on identical flags |

Rejected Qwen3.6 35B A3B combined probability-profile checks on the same
promoted profile:

| Setting change | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.05` | 104.4 | 89.3 | tied prior promoted sustained band |
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.30` | 104.6 | 89.5 | repeated same result; same-session default also reached `104.7` / `89.5` |
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.40` | 104.6 | 89.5 | tied same-session default |
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.50` | 104.4 | 89.5 | tied same-session default sustained, lower short decode |
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.70` | 104.8 | 89.5 | tied same-session default sustained; not enough to promote |
| `--spec-draft-p-min 0.25 --spec-draft-p-split 0.90` | 104.7 | 89.5 | tied same-session default |
| `--spec-draft-p-min 0.50 --spec-draft-p-split 0.30` | 104.6 | 89.2 | sustained regression |

Rejected internal MTP sampler candidate-count checks on the same promoted
profile:

| Internal MTP sampler `top_k` | Sustained decode tok/s | Result |
|---:|---:|---|
| 5 | 77.3 | rejected; too little candidate mass for the promoted `p-min 0.25` profile |
| 20 | 69.6 | rejected; more candidates did not improve acceptance enough to offset the path |

Accepted internal MTP sampler cleanup on the same promoted profile:

| Candidate | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| Top-10 probability-only draft sampler | 104.6 | 90.2 | promoted; avoids the unused final RNG sampler while preserving the top-candidate probability distribution used by `p-min 0.25` |

Rejected internal MTP top-k implementation checks on the same promoted profile:

| Candidate | Short decode tok/s | Sustained decode tok/s | Result |
|---|---:|---:|---|
| Fixed insertion top-10 selection instead of `std::partial_sort` | 73.0 | not run | rejected; failed the 35B A3B short guard floor of `100.0 tok/s`, so the promoted helper keeps `std::partial_sort` |
| `std::nth_element` plus top-k slice sort instead of `std::partial_sort` | 71.5 | not run | rejected; failed the same 35B A3B short guard, so the promoted helper keeps `std::partial_sort` |
| `std::partial_sort_copy` into a sampler-owned top-k buffer | 104.4 | 90.2 | not promoted; same-session promoted build measured `104.1` / `90.2`, so this added buffer/clone complexity without a sustained decode gain |
| Fill only top-token probability in the normal MTP path | 92.7 | not run | rejected; preserving only `data[0].p` and leaving full top-k probabilities for debug logging failed the 35B A3B short guard floor |
| Disable internal MTP sampler timing with `sparams.no_perf = true` | 96.2 | not run | rejected; the speculative draft timer already covers this path, but disabling sampler timing still regressed the 35B A3B short guard below the `100.0 tok/s` floor |
| Skip per-draft MTP `common_sampler_reset()` | 68.3 | not run | rejected; the reset is required to preserve the expected sampler/logit state for the top-k probability helper |

Rejected CPU/reference-path checks from the same optimization pass:

| Promoted CPU/reference-path change | Result |
|---|---|
| Packed-byte reuse in CPU fallback vec-dot | promoted; focused dual-scale vec-dot moved from `31.86` to `29.77`-`29.82` cycles / 32 values while FAST stayed in the `27.04`-`27.06` band |
| Finite-block CPU quantizer scoring path | promoted; after a block-level finite scan, normal scale-MSE scoring uses a finite-only nearest-code helper while keeping the safe path for non-finite blocks; latest focused normal quant measured `3844.38` dual-scale and `3582.57` FAST cycles / 32 values |
| Finite-block weighted/imatrix scoring path | promoted for FAST imatrix; same-session pre-candidate FAST imatrix was `5258.07` cycles / 32 values, candidate repeats measured `4448.73` and `4447.32`; dual imatrix stayed guarded but noisy |

| Candidate | Result |
|---|---|
| Packed-byte reuse in CPU row dequantization | rejected; correctness passed, but repeat focused dequant samples had no stable win and dual-scale bounced from `33.57` to `35.21` cycles / 32 values |
| Table Codebook10 decode in CPU row dequantization and scale-MSE scoring | rejected; dequantization slowed from the guarded `33`-cycle band to `51`-`84` cycles / 32 values |
| Direct decoded-value helper for CPU scale-MSE scoring | rejected; correctness passed, but normal quantization regressed in the measured guard |
| Direct decoded-value helper limited to full-block weighted MSE | rejected; correctness passed, but focused FAST imatrix timing regressed to `5007.73` cycles / 32 values |
| Direct finite-scale table helper in CPU quantizer scale search | rejected; dual-scale normal quantization improved, but FAST normal quantization regressed to `4043.60` and `4289.66` cycles / 32 values on repeat guard runs |
| NaN-only and branchless finite-scan variants | rejected for default; the NaN-only scan was noisy and the same-session `isfinite()` scan measured better dual-scale normal quant (`3735.41` vs `3776.53` cycles / 32 values), while branchless boolean-and regressed FAST normal quant to `3773.44` cycles / 32 values |
| Dual-scale-only finite-pack final quantizer loop | rejected; correctness passed, but FAST normal quantization regressed into the `3882`-`4022` cycles / 32 values band versus the clean `3623` cycles / 32 values baseline, even after the shared scale chooser was split back out for FAST |

The CPU quant regression guard now also enforces dequant and vec-dot ceilings.
Latest pass after keeping the `isfinite()` finite-block scoring promotion and
adding the weighted/imatrix finite scorer measured dual-scale quant `3844.38`,
FAST quant `3582.57`, dual dequant `33.59`, FAST dequant `33.13`, dual
vec-dot `29.96`, FAST vec-dot `27.03`, dual imatrix `5587.43`, and FAST
imatrix `4447.32` cycles / 32 values.

Accepted ROCm/HIP CPY optimization from the same pass:

| Candidate | Previous guard band | New guard result | Result |
|---|---:|---:|---|
| Contiguous `Q4_0_ROCMFP4 -> F32` packed-byte CPY kernel | ~740 us/run | 184.48 us/run | promoted |
| Contiguous `Q4_0_ROCMFP4_FAST -> F32` packed-byte CPY kernel | ~740 us/run | 169.99 us/run | promoted |
| Source-to-ROCmFP4-only 128-thread CPY launch | dual source `1116`-`1118 us/run`, FAST source `1230`-`1231 us/run` | clean repeats: dual F32/F16/BF16 `1109.47`/`1009.14`/`1007.26`, FAST F32/F16/BF16 `1221.96`/`1136.73`/`1138.08`; tightened serial guard `1109.40`/`1009.52`/`1006.59`, FAST `1218.69`/`1138.32`/`1138.78` | promoted; source quantization improves while quant-to-F32 stays on the 64-thread path |
| Direct Codebook10 value scoring for FAST source-to-ROCmFP4 CPY | FAST F32/F16/BF16 source `1218.69`/`1138.32`/`1138.78 us/run` | repeat guard: FAST F32/F16/BF16 source `1047.21`/`950.93`/`951.00 us/run`; dual source `1111.73`/`1008.69`/`1006.93`; q-to-F32 unchanged | promoted; FAST scale scoring now avoids index-then-decode while preserving exact output |

Additional ROCm CPY launch-size checks were rejected. A temporary 128-thread
ROCmFP4 CPY launch was roughly tied with the default 64-thread path, with small
source-to-quant wins but slight quant-to-F32 regressions. The promoted split
keeps 128 threads only for source-to-ROCmFP4 quantization and leaves
quant-to-F32 at the accepted 64-thread launch. A 256-thread launch
regressed F16 source-to-quant paths sharply, including FAST F16 source-to-quant
at `1256.84 us/run`, so the default 64-thread ROCmFP4 CPY launch remains
promoted for non-source-quant paths.

After direct Codebook10 value scoring made the FAST source path cheaper, FAST
source-only launch splits were rechecked and rejected. `FAST_CPY_QUANT_BLOCK_SIZE=256`
kept F32 roughly flat but regressed FAST F16 source-to-quant to `1047.24`
us/run. `FAST_CPY_QUANT_BLOCK_SIZE=64` measured FAST F32/F16/BF16 at
`1055.32`/`955.35`/`954.78` us/run, slower than the promoted shared 128-thread
source launch. The code therefore keeps one 128-thread source-to-ROCmFP4 launch
for both dual-scale and FAST layouts.

Bypassing the temporary F32 staging array for source-to-ROCmFP4 CPY was also
rejected. It compiled, but the ROCm CPY guard regressed F32-to-dual from the
accepted `~1.1 ms` band to `9952.29 us/run`, and F32-to-FAST to
`10709.05 us/run`, so the existing staged path remains promoted.

Explicitly unrolling the fixed-size ROCmFP4 source-quant loops was rejected.
The ROCm CPY guard stayed in the same band, with F32-to-dual at
`1119.04 us/run` and F32-to-FAST at `1229.08 us/run`, while HIP emitted
unroll-failed warnings during compilation.

Replacing packed-count multiply/divide expressions in the ROCmFP4 CPY and HIP
helpers with shift/mask arithmetic was also rejected. The focused guard moved
source-to-quant paths only inside noise (`F32/F16/BF16 -> dual` at
`1106.34`/`1008.52`/`1006.51 us/run`, `F32/F16/BF16 -> FAST` at
`1045.16`/`952.41`/`950.70 us/run`) while slightly regressing quant-to-F32
(`dual -> F32` from `182.06` to `183.45 us/run`, FAST from `170.24` to
`170.44 us/run`). The code change was removed.

The promoted Qwen MTP ROCmFP4 path remains the plain `draft-mtp` ROCm0 run
with `--spec-draft-n-max 4` plus the dual-scale-only MMVQ `vdr=4` tune. The
extra ngram speculative modes reduce sustained decode on this prompt, moving
both dual-scale and FAST MMVQ to `vdr=4` regresses sustained decode, and the
rocWMMA FlashAttention build regresses the sustained guard compared with the
default HIP FlashAttention path.

This isolated tree does not currently contain TurboQuant or TriAttention runtime
flags. Adding either would be a source-level merge, not a flag flip, and should
only be promoted after it beats the serial ROCmFP4 regression gate.

Update: the TurboQuant KV kernels (`turbo3`, `turbo4`) were merged on
2026-06-20 (`d859c9e67`), but `-ctk` / `-ctv` stopped accepting them in the
2026-09-15 merge, so there is still no runtime flag. TriAttention is not in the
tree.
