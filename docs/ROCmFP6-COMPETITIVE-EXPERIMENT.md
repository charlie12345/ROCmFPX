# ROCmFP6 Competitive Experiment

Status: experimental. These notes cover the Qwythos 9B Claude Mythos 5 1M MTP BF16 source tested on Strix Halo.

## Presets

Two opt-in ROCmFP6 presets were added without changing existing file types:

- `Q6_0_ROCMFPX_LEAN`: size/speed-biased FP6 routing with no Q8-heavy boosts.
- `Q6_0_ROCMFPX_AGENT_LEAN`: agent-biased FP6 routing that spends limited `Q6_K` on token/output and selected sensitive projections instead of using the older Q8-heavy agent route.

Existing `Q6_0_ROCMFPX` and `Q6_0_ROCMFPX_AGENT` GGUFs continue to load with their previous routing and tensor types.

## Qwythos 9B Size And Speed

Command shape:

```bash
MODEL_SRC=/mnt/seconddrive/models/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16.gguf \
OUT_DIR=/tmp/rocmfp6-qwythos-competitive \
PRESETS='Q6_K Q6_0_ROCMFPX_LEAN Q6_0_ROCMFPX_AGENT_LEAN' \
BACKENDS='ROCm0 Vulkan0' RUN_QUANTIZE=1 RUN_BENCH=1 \
PROMPT_TOKENS=512 GEN_TOKENS=128 REPEAT=2 \
scripts/sweep-rocmfp6-competitive.sh
```

| preset | file MiB | delta vs Q6_K | ROCm pp512 | ROCm tg128 | Vulkan pp512 | Vulkan tg128 |
|---|---:|---:|---:|---:|---:|---:|
| Q6_K | 7208.73 | 0.00 | 772.28 | 29.01 | 951.90 | 31.06 |
| Q6_0_ROCMFPX_LEAN | 7140.21 | -68.52 | 712.06 | 27.72 | 683.90 | 12.72 |
| Q6_0_ROCMFPX_AGENT_LEAN | 7170.50 | -38.23 | 740.33 | 27.93 | 791.12 | 15.53 |

Result: the lean presets beat Q6_K on size, but not decode speed. `Q6_0_ROCMFPX_AGENT_LEAN` is the better ROCmFP6 candidate of the two, but Q6_K remains faster on this model.

## VDR Kernel Probe

An isolated HIP build tested Q6_K-like ROCmFP6 VDR settings:

```bash
BUILD_DIR=build-rocmfp6-vdr1-mmq8 \
CMAKE_HIP_FLAGS='-DVDR_ROCMFP6_Q8_1_MMVQ=1 -DVDR_ROCMFP6_Q8_1_MMQ=8' \
scripts/build-strix-rocmfp4-mtp.sh llama-bench
```

ROCm0 results:

| preset | pp512 | tg128 |
|---|---:|---:|
| Q6_K | 777.98 | 29.08 |
| Q6_0_ROCMFPX_LEAN | 713.91 | 17.57 |
| Q6_0_ROCMFPX_AGENT_LEAN | 736.89 | 19.95 |

Result: do not promote this VDR setting. It regresses ROCmFP6 decode heavily.

## Recommendation

Keep `Q6_0_ROCMFPX_LEAN` and `Q6_0_ROCMFPX_AGENT_LEAN` as opt-in experimental presets for size/coherence testing. Do not claim ROCmFP6 beats Q6_K on speed yet. The next real speed work should target the ROCmFP6 MMVQ decode path or Vulkan Q6_0_ROCMFPX shader path directly, not routing alone.

## 2026-06-27 kernel/shader follow-up

Implementation:

- Added opt-in HIP compile flags for ROCmFP6 MMVQ experiments:
  - `GGML_ROCMFP6_FAST_SIGNMAG_PACK=1`
  - `GGML_ROCMFP6_MMVQ_HALF_BLOCK_SPLIT=1`
- Made the HIP FP6 MMVQ packed-word helper use an explicit zero sentinel word for shifted 24-bit reads at the end of a block.
- Fixed the Vulkan ROCmFP6 packed-bit helper to avoid a 32-bit shift on aligned reads.
- Fixed Vulkan ROCmFP6 mat-vec accumulation so both 4-weight lanes contribute to the partial sum.
- Replaced the Vulkan FP6 mat-vec/FA pack helpers' FP6 table lookup with direct signed-magnitude decode.

Bench target:

- Model: `Qwythos-9B-Claude-Mythos-5-1M-MTP`
- ROCmFP6 model: `/mnt/seconddrive/models/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_0_ROCMFPX_AGENT-00001-of-00001.gguf`
- Q6_K model: `/mnt/seconddrive/models/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf`
- Test: `llama-bench -p 512 -n 128 -r 2 -fa on -ngl 999 -mmp 0`

| Build / model | Backend | pp512 t/s | tg128 t/s | Result |
|---|---:|---:|---:|---|
| Existing Q6_0_ROCMFPX_AGENT | ROCm0 | 731.20 | 25.58 | Baseline |
| HIP `FAST_SIGNMAG + HALF_SPLIT` Q6_0_ROCMFPX_AGENT | ROCm0 | 731.08 | 25.67 | Tie; below 3 tok/s gate |
| Existing Q6_K | ROCm0 | 771.23 | 29.10 | Still faster |
| Existing Q6_0_ROCMFPX_AGENT | Vulkan0 | 834.35 | 16.36 | Baseline |
| Vulkan FP6 shader fixes Q6_0_ROCMFPX_AGENT | Vulkan0 | 854.24 | 16.39 | Prefill +19.89 t/s; decode tie |
| Existing Q6_K | Vulkan0 | 956.96 | 31.07 | Still faster |
| Direct signed-mag shader Q6_0_ROCMFPX_AGENT | Vulkan0 | 828.21 | 16.34 | No decode gain |
| Direct signed-mag shader Q6_0_ROCMFPX_AGENT | ROCm0 | 729.01 | 25.50 | No ROCm regression |
| Vulkan FP6 full-block `K_PER_ITER=32` mat-vec | Vulkan0 | 846.39 | 16.36 | Correct, but no decode gain |

Regression checks:

- Vulkan0 `test-backend-ops test -b Vulkan0 -o MUL_MAT`: `1068/1068 tests passed`.
- ROCm0 `test-backend-ops test -b ROCm0 -o MUL_MAT`: `1241/1241 tests passed`.

Conclusion:

- The HIP opt-in decode tweaks are safe to keep for further sweeps, but they do not beat the baseline on this model.
- The Vulkan shader fixes are worth keeping because they fix FP6 arithmetic edge cases, but the later direct signed-magnitude simplification did not solve ROCmFP6 decode speed.
- A dedicated Vulkan FP6 full-block q8_1 mat-vec specialization (`K_PER_ITER=32`) passed `MUL_MAT` correctness, but decode stayed flat and the experiment was not kept active.
- The next speed attempt should be a real GPU-side/repacked ROCmFP6 layout or a dedicated Vulkan FP6 mat-vec shader that reduces unpack overhead. Small branch/loop tweaks are not enough.
