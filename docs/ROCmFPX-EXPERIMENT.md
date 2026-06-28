# ROCmFPx Experiment

The ROCmFPx experiment is the staging area for possible AMD-native
`ROCmFP3`, `ROCmFP6`, and `ROCmFP8` quant formats.

The implementation lives in:

```text
ggml/rocmfpx/
```

The first stage defines block layouts, finite UE4M3 scale-byte decoding,
pack/unpack, quantize/dequantize, validation, and a deterministic reference
test. `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, and `Q8_0_ROCMFPX` are now promoted to
very experimental GGUF tensor types with CPU reference paths plus ROCm/HIP and
Vulkan acceleration hooks.

ROCm/HIP and Vulkan kernels are wired for the new ROCmFPx family in the same
style as ROCmFP4. They support GPU copy/dequantization (`CPY` to/from
F32/F16/BF16), embedding lookup (`GET_ROWS`), and vector-matrix/matrix-matrix
dot products (`MUL_MAT`/`MUL_MAT_ID` via MMVQ/MMQ and Vulkan DMMV/MMV paths).
As of June 15, 2026, CPU reference checks, CPU backend ops, ROCm backend ops,
Vulkan backend ops, and CPU/ROCm/Vulkan tiny-model offload smokes pass. The
previous wider ROCm backend-op caveat was traced to a generic HIP
`F16 x F16 -> F32` `MUL_MAT` correctness failure that reproduced on the clean
baseline build. The experiment now reports that HIP path as unsupported so graph
placement can use a correct fallback.

The first real BF16-source model test also passed: `unsloth/Qwen3-0.6B-GGUF`
`Qwen3-0.6B-BF16.gguf` was converted to `Q3_0_ROCMFPX` and smoked on CPU,
ROCm0, and Vulkan0.

The staging is structured to ensure that the existing ROCmFP4 path remains stable.

## ROCmFP4 Quant / Kernel / Dequant Contract

The ROCmFPX family should be developed as real GGUF model-weight quant formats,
parallel to `Q4_0_ROCMFP4`, not as a new K/V-cache-only compression layer. K and
V cache flags are runtime cache-type controls in llama.cpp; they are separate
from the model-file quant presets listed below.

ROCmFP4 instructions that carry forward:

- **Block shape:** keep 32 weights per block. This preserves the existing
  Q4/Q8 reduction shape used by CPU dots, HIP MMVQ/MMQ, and Vulkan MMV/MMQ/DMMV
  paths.
- **Scale encoding:** use finite unsigned UE4M3 scale bytes and reject invalid
  scale bytes. ROCmFP4 treats `0x7f` and sign-bit scale bytes as invalid; ROCmFPX
  validation should follow that rule.
- **Scale selection:** use reconstruction-MSE search where low-bit coherency
  depends on local scale quality. ROCmFP4 searches 16-weight half-blocks;
  ROCmFP3 and ROCmFP6 follow that policy. ROCmFP8 is one 32-weight scale today
  because it is the high-quality reference point.
- **Dequantization:** decode integer codes with an explicit codebook/range and
  multiply by the decoded UE4M3 scale. ROCmFP4 uses Codebook10 at half scale;
  ROCmFP3/6/8 keep their own code ranges but must keep the same deterministic
  integer-code-times-scale structure.
- **Kernel coverage:** do not call a ROCmFPX format runtime-complete until CPU
  reference, HIP, and Vulkan paths cover `CPY`, `GET_ROWS`, `SET_ROWS`,
  `MUL_MAT`, and `MUL_MAT_ID`, with backend-op coverage.
- **Feature parity:** MTP, EAGLE3, speculative decoding, RoPE/attention scaling,
  tool-calling grammar paths, and long-context behavior should continue to use
  normal llama.cpp runtime surfaces. The quant format must not require a
  separate inference stack.

ROCmFP4 Codebook10 itself is not copied into FP3/FP6/FP8. The inherited part is
the quantization discipline and kernel contract. The ROCmFPX code ranges are:

| Format | Code range | Scale policy |
|---|---|---|
| `Q3_0_ROCMFPX` | `0, +/-1, +/-2, +/-4` | two UE4M3 scales, one per 16 weights |
| `Q6_0_ROCMFPX` | signed magnitude up to `31` | two UE4M3 scales, one per 16 weights |
| `Q8_0_ROCMFPX` | signed int8 clamped to `[-127, 127]` | one UE4M3 scale per 32 weights |

## Current Layouts

| Format | Block | BPW | Current Role |
|---|---:|---:|---|
| `Q3_0_ROCMFPX` | 32 weights, 12 packed code bytes, 2 scale bytes | 3.50 | Experimental low-bit candidate |
| `Q6_0_ROCMFPX` | 32 weights, 24 packed code bytes, 2 scale bytes | 6.50 | Experimental quality candidate |
| `Q8_0_ROCMFPX` | 32 weights, 32 signed code bytes, 1 scale byte | 8.25 | Experimental high-quality reference |

## Experimental Quantize Names

```bash
llama-quantize model-f16.gguf model-q3-rocmfpx.gguf Q3_0_ROCMFPX
llama-quantize model-f16.gguf model-q6-rocmfpx.gguf Q6_0_ROCMFPX
llama-quantize model-f16.gguf model-q8-rocmfpx.gguf Q8_0_ROCMFPX
llama-quantize model-f16.gguf model-q3-agent-rocmfpx.gguf Q3_0_ROCMFPX_AGENT
llama-quantize model-f16.gguf model-q6-agent-rocmfpx.gguf Q6_0_ROCMFPX_AGENT
llama-quantize model-f16.gguf model-q8-agent-rocmfpx.gguf Q8_0_ROCMFPX_AGENT
```

These formats are integrated into llama.cpp and support experimental hardware
acceleration via the ROCm/HIP and Vulkan backends.

## Check

```bash
scripts/check-rocmfpx-reference.sh
```

This compiles `ggml/rocmfpx/rocmfpx.c` and runs the local reference test in
`build-rocmfpx-reference/`.

The reference test covers finite decode/roundtrip behavior plus weighted
imatrix scale-search checks for ROCmFP3, ROCmFP6, and ROCmFP8. The imatrix
checks verify that calibration weights actually affect scale search by
requiring lower weighted reconstruction error than the plain unweighted path.

The currently used tiny quantized test fixtures live in:

```text
/tmp/rocmfpx-quant-tests/
```

Run focused backend and offload checks from the experiment worktree:

```bash
cd /home/caf/strix-fp4/llama.cpp-mtp-rocmfp4
cmake --build build-strix-rocmfp4 --target llama-quantize llama-cli llama-bench test-backend-ops -j 8
scripts/check-rocmfpx-reference.sh
timeout 120 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b CPU
timeout 180 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b ROCm0
timeout 180 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b Vulkan0
timeout 90 build-strix-rocmfp4/bin/llama-bench -m /tmp/rocmfpx-quant-tests/stories260K-Q3_0_ROCMFPX.gguf,/tmp/rocmfpx-quant-tests/stories260K-Q6_0_ROCMFPX.gguf,/tmp/rocmfpx-quant-tests/stories260K-Q8_0_ROCMFPX.gguf -dev ROCm0 -ngl 99 -p 16 -n 16 -r 1
timeout 90 build-strix-rocmfp4/bin/llama-bench -m /tmp/rocmfpx-quant-tests/stories260K-Q3_0_ROCMFPX.gguf,/tmp/rocmfpx-quant-tests/stories260K-Q6_0_ROCMFPX.gguf,/tmp/rocmfpx-quant-tests/stories260K-Q8_0_ROCMFPX.gguf -dev Vulkan0 -ngl 99 -p 16 -n 16 -r 1
```

Run the current BF16-to-ROCmFP3 model smoke:

```bash
cd /home/caf/strix-fp4/llama.cpp-mtp-rocmfp4
curl -L --fail --continue-at - \
  -o /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-BF16.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf
build-strix-rocmfp4/bin/llama-quantize \
  /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-BF16.gguf \
  /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf \
  Q3_0_ROCMFPX
timeout 120 build-strix-rocmfp4/bin/llama-bench -m /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf -ngl 0 -p 16 -n 16 -r 1
timeout 120 build-strix-rocmfp4/bin/llama-bench -m /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf -dev ROCm0 -ngl 99 -p 16 -n 16 -r 1
timeout 120 build-strix-rocmfp4/bin/llama-bench -m /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf -dev Vulkan0 -ngl 99 -p 16 -n 16 -r 1
```

Observed Qwen3 BF16-to-ROCmFP3 smoke:

```text
BF16 source: /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-BF16.gguf
ROCmFP3 output: /home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf
Quantized size: 262.82 MiB, 3.70 BPW
CPU/no-offload: pp16 7.21 t/s, tg16 6.43 t/s
ROCm0: pp16 676.61 t/s, tg16 237.17 t/s
Vulkan0: pp16 1540.28 t/s, tg16 259.23 t/s
```

## Qwen3 ROCmFP3 Coherency Check

Runtime support does not imply quality. A June 15, 2026 comparison against the
matching `unsloth/Qwen3-0.6B-GGUF` `Q4_K_M` quant showed that the original pure
`Q3_0_ROCMFPX` tensor mix loaded and decoded, but lost too much
instruction-following quality for promotion.

The current fix keeps the ROCmFP3 block format stable but changes quantization
quality in two places:

- `Q3_0_ROCMFPX` now chooses each FP3 half-block scale by reconstruction MSE
  instead of raw `max_abs / 4`.
- The default Q3 preset now uses selective coherence routing: attention Q/O and
  early-layer K/V as `Q5_K`, upper-layer K/V as `Q4_K`, boosted FFN-down as
  `Q5_K`, selective FFN-gate as `Q6_0_ROCMFPX`, bulk FFN-up on `Q3_0_ROCMFPX`,
  and token/output embeddings on `Q4_0_ROCMFP4_FAST`.
- The default Q6 preset now uses lean coherence routing: first-layer K/V and
  early Q/O/down as `Q8_0_ROCMFPX`, embeddings/output on `Q6_0_ROCMFPX`, and
  bulk gate/up on `Q6_0_ROCMFPX` with FP6 MSE scale selection.

Test files:

```text
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q6_0_ROCMFPX_COHERENT-LEAN.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q8_0_ROCMFPX.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-MSE-v3.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-MSE-v4.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q6_0_ROCMFPX_COHERENT.gguf
/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q4_K_M.gguf
```

Observed with ROCm0 offload and deterministic low-temperature prompts:

```text
Coding prompt:
  Q3_0_ROCMFPX: repeated reasoning preamble; did not produce the function.
  Q4_K_M: produced a valid duplicate-finder function.

Three-bullet summary prompt:
  Q3_0_ROCMFPX: repeated the instruction instead of answering.
  Q4_K_M: produced three coherent bullets.

JSON arithmetic prompt:
  Q3_0_ROCMFPX: entered reasoning and did not emit JSON within the token cap.
  Q4_K_M: emitted {"answer": 391, "method": "multiplication"}.

Small fixed-text perplexity:
  Q3_0_ROCMFPX: PPL 32.2383 +/- 12.08643
  Q4_K_M:       PPL 24.9125 +/- 10.10508
```

Follow-up repair test with ROCm0 offload:

```text
Pure Q3 + MSE scales:
  Still drifts into reasoning text on the coding, summary, and JSON probes.

Coherent-MSE v2:
  Protects attention K/V/O and FFN-down as Q5_K.
  Produces the duplicate function and the three bullets, but emits
  {"answer": 17 * 23, ...} on the arithmetic probe.

Coherent-MSE v3:
  Protects attention Q/K/V/O and FFN-down as Q5_K.
  Produces the duplicate function, three coherent bullets, and computes 391.
  For strict machine-readable output, use `llama-completion -no-cnv --strict-json`
  instead of the chat wrapper.

Coherent-LEAN v5 (current default):
  Selective Q3 routing: Q/O and early K/V at `Q5_K`, boosted down at `Q5_K`,
  selective gate at `Q6_0_ROCMFPX`, bulk up on `Q3_0_ROCMFPX`, embeddings on
  `Q4_0_ROCMFP4_FAST`, with FP3 MSE scales.
  Passes the Qwen3 coding, summary, and strict-JSON probes on ROCm0.
  Qwen3-0.6B size: 330.57 MiB / 4.65 BPW (vs Q3_K_M 325.37 MiB / 4.58 BPW).

Coherent-LEAN Q6 preset (current default):
  Lean Q6 routing: early attn and down at `Q8_0_ROCMFPX`, embeddings/output on
  `Q6_0_ROCMFPX`, bulk gate/up on `Q6_0_ROCMFPX`, with FP6 MSE scales.
  Passes the same Qwen3 coherency probes on ROCm0.
  Qwen3-0.6B size: 466.65 MiB / 6.57 BPW (vs Q6_K 466.50 MiB / 6.57 BPW).

Q8_0_ROCMFPX pure preset:
  Qwen3-0.6B size: 586.39 MiB / 8.25 BPW (vs Q8_0 604.15 MiB / 8.50 BPW).
```

Q3 is ~5 MiB over `Q3_K_M` on 0.6B. Agent presets trade size for routing:

| Preset | Size / BPW | vs stock |
|---|---:|---:|
| `Q3_K_M` | 325.37 MiB / 4.58 BPW | baseline |
| `Q3_0_ROCMFPX` | 330.57 MiB / 4.65 BPW | +5.2 MiB |
| `Q3_0_ROCMFPX_AGENT` | 437.62 MiB / 6.16 BPW | +112 MiB vs Q3_K_M |
| `Q6_K` | 466.50 MiB / 6.57 BPW | baseline |
| `Q6_0_ROCMFPX` | 466.65 MiB / 6.57 BPW | +0.15 MiB |
| `Q6_0_ROCMFPX_AGENT` | 541.76 MiB / 7.62 BPW | +75 MiB vs Q6_K |
| `Q8_0` | 604.15 MiB / 8.50 BPW | baseline |
| `Q8_0_ROCMFPX` | 586.39 MiB / 8.25 BPW | −17.8 MiB |
| `Q8_0_ROCMFPX_AGENT` | 598.90 MiB / 8.43 BPW | −5.3 MiB |

Dry-run source: `Qwen3-0.6B-Q4_K_M.gguf` via `scripts/sweep-rocmfpx-agent-size-table.sh`.

## Agent Presets And Harnesses

`Q3_0_ROCMFPX_AGENT`, `Q6_0_ROCMFPX_AGENT`, and `Q8_0_ROCMFPX_AGENT` are opt-in
experimental presets for Hermes/OpenClaw-style tool use. They do not introduce new
tensor block layouts; they keep the ROCmFPx family formats and spend extra bits on
sensitive routing: token/output tensors, attention Q/K/V/O, selected FFN-down,
and gate/up slices. The default `Q3_0_ROCMFPX` / `Q6_0_ROCMFPX` / `Q8_0_ROCMFPX`
LEAN routing is unchanged.

Core validation:

```bash
scripts/check-rocmfpx-qwen-all.sh          # reference + coherency + agent JSON
scripts/check-rocmfpx-all.sh               # qwen-all + optional smokes
scripts/check-rocmfpx-agent-json.sh
scripts/check-rocmfpx-tool-calling.sh
scripts/check-rocmfpx-agent-json-grammar.sh  # llama-server json_schema path
```

Agent fixture build (proxy from Qwen until Hermes/OpenClaw sources are set):

```bash
scripts/build-rocmfpx-agent-fixtures.sh
HERMES_SRC=/path/to/hermes-bf16.gguf scripts/build-rocmfpx-agent-fixtures.sh
```

Agent and large-model smokes:

```bash
MODEL=/path/to/hermes-Q3_0_ROCMFPX_AGENT.gguf scripts/check-rocmfpx-hermes-smoke.sh
MODEL=/path/to/openclaw-Q3_0_ROCMFPX_AGENT.gguf scripts/check-rocmfpx-openclaw-smoke.sh
MODEL=/path/to/qwen-or-agent.gguf scripts/check-rocmfpx-long-context-smoke.sh
MODEL_SRC=/path/to/qwen3-4b-bf16.gguf scripts/check-rocmfpx-qwen-large.sh
MODEL_SRC=/path/to/minimax-or-mixtral-bf16.gguf scripts/check-rocmfpx-moe-routing.sh
MODEL_SRC=/path/to/minimax.gguf scripts/check-rocmfpx-minimax-smoke.sh
```

Sweeps:

```bash
MODEL_SRC=/path/to/model.gguf scripts/sweep-rocmfpx-agent-routing.sh
RUN_AGENT_JSON=1 MODEL_SRC=/path/to/model.gguf scripts/sweep-rocmfpx-agent-routing.sh
MODEL_SRC=/path/to/model.gguf scripts/sweep-rocmfpx-agent-size-table.sh
MODEL_SRC=/path/to/model.gguf scripts/sweep-rocmfpx-perplexity.sh
MODEL_LEAN=/path/to/lean.gguf MODEL_AGENT=/path/to/agent.gguf scripts/sweep-rocmfpx-decode-tune.sh
scripts/check-rocmfpx-spec-decode-all.sh
```

The scripts emit machine-readable JSON. Missing optional model fixtures skip by
default; set `SKIP_MISSING_MODEL=0` when CI should require a local fixture.
`sweep-rocmfpx-agent-routing.sh` dry-runs standard and agent presets, records
size/BPW, and can run the JSON agent harness after quantization with
`RUN_AGENT_JSON=1`.

## Qwen3 Validation Scripts

```bash
scripts/check-rocmfpx-qwen-coherency.sh
scripts/check-rocmfpx-qwen-bench.sh
scripts/check-rocmfpx-qwen-strict-json.sh
scripts/check-rocmfpx-qwen-all.sh
```

## Speculative Decode Smokes

ROCmFPx weight tensors are type-agnostic at the llama speculative layer, but
MTP/EAGLE/draft paths still need backend coverage for the mixed tensor presets.
Use these smoke scripts with local larger-model fixtures:

```bash
MODEL=/path/to/mtp-rocmfpx.gguf BACKENDS="ROCm0 Vulkan0" \
  scripts/check-rocmfpx-mtp-smoke.sh

MODEL=/path/to/target-rocmfpx.gguf DRAFT_MODEL=/path/to/eagle3-draft.gguf \
  scripts/check-rocmfpx-eagle3-smoke.sh

MODEL=/path/to/target-rocmfpx.gguf DRAFT_MODEL=/path/to/draft.gguf \
  scripts/check-rocmfpx-speculative-smoke.sh
```

The scripts skip when default local fixtures are missing. Set
`SKIP_MISSING_MODEL=0` in CI or when a required fixture should be mandatory.
Keep KV cache types on the existing `q4_0`/F16/ROCmFP4 paths for now; ROCmFPx
KV cache experiments need FlashAttention support before promotion.

## Opt-in Decode Tuning

The default build keeps the conservative ROCmFPx MMVQ launch shape. Strix tuning
profiles are opt-in through the existing build wrappers:

```bash
ROCMFPX_DECODE_TUNE=rocmfpx-strix-nwarps2 \
BUILD_DIR=build-strix-rocmfpx-nw2 \
scripts/build-strix-rocmfp4-mtp.sh
```

Available ROCmFPx profiles:

```text
rocmfpx-strix-nwarps1, rocmfpx-strix-nwarps2, rocmfpx-strix-nwarps4
rocmfpx-strix-rpb2
rocmfpx-strix-mmid3, rocmfpx-strix-mmid4
rocmfpx-strix-moe-rpb1, rocmfpx-strix-moe-rpb2, rocmfpx-strix-moe-rpb3, rocmfpx-strix-moe-rpb4
```

These profiles only alter MMVQ launch geometry. They do not change block
layouts, quantized values, FP3/FP6 MSE scale selection, or Q3 LEAN routing.

## Ranked Attention Tensor Policies

`scripts/rocmfpx-ranked-policy.py` generates deterministic
`llama-quantize --tensor-type-file` inputs for ranked ROCmFPX experiments. It
keeps the lowest-error attention tensors in ROCmFPX and restores the remaining
attention plus FFN up/gate/down tensors to `Q6_K`:

```bash
scripts/rocmfpx-ranked-policy.py \
  --rank-csv attention-fp6-residual-rank.csv \
  --leave-count 32 \
  --base-tensor-type-file base.tensor-type.txt \
  --output rankleave32.tensor-type.txt

FORMAT=rocmfp6 PROFILE=straight TENSOR_TYPE_FILE=rankleave32.tensor-type.txt \
SRC=model-BF16.gguf OUT=model-rankleave32.gguf scripts/quantize-rocmfpx-agent.sh
```

The policy is explicit rather than a new ftype because the ranking CSV is
model/evaluation specific. `--leave-count` must be positive; use separate
tensor-type rules if an all-restored policy is needed. When
`--base-tensor-type-file` is set, ranked exact attention rules are emitted
before the base tensor policy because tensor-type overrides are first-match
wins.

For local served-inference experiments, `scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh`
starts a `llama-server` FP3 MTP profile with the observed Strix/Vulkan settings:
`draft-mtp`, `n_max=4`, `p_min=0.75`, F16 target/draft KV, and draft backend
sampling disabled. This is an opt-in speed reproduction helper, not a validation
gate or a universal default.

The helper intentionally accepts environment overrides so the same launch shape
can reproduce the FP4 cap4 rows with a local FP4 MTP GGUF:

```bash
# Baseline FP4 35B cap4 short-prompt run shape.
MODEL=/path/to/Qwen3.6-35B-A3B-MTP-ROCmFP4.gguf \
SPEC_DRAFT_P_MIN=0.0 \
scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh

# ACE/SABER FP4 cap4 short/text winner.
MODEL=/path/to/Qwen3.6-35B-A3B-NSC-ACE-SABER-MTP-ROCmFP4.gguf \
SPEC_DRAFT_P_MIN=0.25 \
scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh
```

Keep the rest of the served profile fixed for comparable rows unless testing a
specific arm: `Vulkan0`, `--parallel 1`, `-b 2048`, `-ub 512`, F16 target and
draft KV, prompt cache disabled, metrics enabled, and
`--no-spec-draft-backend-sampling`.

Local Strix Halo evidence on a Qwen3.6 35B FP3+MTP GGUF showed strong decode
wins when the prompt/generation shape had high MTP acceptance:

| Shape | Profile | Decode tok/s | Draft accepted | Total time vs no MTP |
|---|---:|---:|---:|---:|
| 4.35k prompt / gen512 | no MTP | 77.80 | n/a | baseline |
| 4.35k prompt / gen512 | n4 pmin0.75 | 134.25 | 403 / 408 | 20.9% faster |
| 4.35k prompt / gen1024 | no MTP | 77.53 | n/a | baseline |
| 4.35k prompt / gen1024 | n4 pmin0.75 | 137.00 | 813 / 818 | 30.3% faster |
| 24.0k prompt / gen256 | n4 pmin0.75 | 118.84 | 202 / 202 | 4.9% slower |

The 24k decode row is useful as an acceptance-rate signal, but it was not a full
latency win at only 256 generated tokens because prefill dominated the request.

Longer context ladder, same FP3+MTP served profile, `gen512`, prompt cache
disabled:

| Prompt tokens | Decode tok/s | Prompt tok/s | TTFP s | Draft accepted |
|---:|---:|---:|---:|---:|
| 2361 | 93.81 | 936.37 | 2.53 | 268 / 286 |
| 4708 | 117.96 | 993.96 | 4.74 | 362 / 362 |
| 9392 | 107.06 | 1024.00 | 9.18 | 341 / 349 |
| 18766 | 100.94 | 964.59 | 19.48 | 347 / 351 |
| 37521 | 99.66 | 819.11 | 45.84 | 405 / 405 |
| 75023 | 53.67 | 628.07 | 119.52 | 196 / 220 |
| 128278 | 46.08 | 472.47 | 271.62 | 245 / 287 |

The same `n_max=4`, `p_min=0.75` request profile stays strong through roughly
32k measured prompt tokens, then both prompt processing and draft acceptance
taper sharply. That is a signal to test a separate long-context request policy,
such as lower `n_max` or lower `p_min`, rather than treating the 4k winner as a
universal default.

The request-level speculative controls are above the quant kernels and can be
applied to ROCmFP4 serving as well, but the FP3 bit-packing and shader work
should not be copied to FP4 without separate FP4 evidence. Existing FP4 notes
show model-specific MTP preferences, so FP4 should test these knobs as sweep
arms instead of inheriting the FP3 `p_min=0.75` default.

Historical decode speeds on earlier coherent Qwen3-0.6B presets (ROCm0 / Vulkan0):

```text
Q3 coherent v4: pp16 1296 / 2118 t/s, tg16 228 / 263 t/s
Q6 coherent:    pp16  762 / 1942 t/s, tg16 197 / 225 t/s
Q8 pure:        pp16  773 / 2277 t/s, tg16 212 / 257 t/s
```

Current Q3 LEAN ROCm0 smoke on June 16, 2026:

```text
Q3_0_ROCMFPX_COHERENT-LEAN: pp16 1222.42 t/s, tg16 231.00 t/s
```

## Next Steps

1. Add calibration/perplexity sweeps against representative dense and MoE tensors.
2. Run larger-model coherency and exact-format prompt-following checks for the
   coherent Q3/Q6 presets.
3. Perform end-to-end quality and perplexity tests under GPU offload on MoE models.
4. Gate any upstream PR promotion behind model-level decode stability and performance profiles.
