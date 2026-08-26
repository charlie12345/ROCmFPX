# HaloSpecKV runtime branch

Local long-context inference on AMD Strix Halo was too slow. This branch exists
to make it faster without changing the target model's output.

This branch is the buildable runtime for the independent
[HaloSpecKV](https://github.com/radicalgeek/halospeckv) project. It starts at
ROCmFPX commit `c49ebdbd5c9f01ec242369f9e7f7967855f80cba` and contains the exact
source stack used for the published correctness and performance evidence.

## What changed

Qwen MTP verifies several proposed tokens in one target-model batch. When only
part of a draft was accepted, the target could retain an intermediate recurrent
state produced by that multi-row graph. At 32K context, the next ordinary
decode selected a different token from serial greedy decoding.

Strict-Qwen accepted-prefix replay handles that case as follows:

1. Roll the target sequence back across the bounded draft rows.
2. Replay only the accepted draft-prefix tokens.
3. Decode each replay token through an ordinary one-token target batch.
4. Keep the existing multi-row fast path when the complete draft is accepted.

The branch also contains the state and Vulkan corrections required by the
validated configuration:

- a dedicated immutable recurrent zero-state row;
- cold-prompt MTP side-state reset;
- fp32 query scaling for quantised FlashAttention MMQ;
- scalar Q8-key FlashAttention dispatch on AMD UMA, retaining MMQ;
- recurrent tail-preserving ubatch splitting;
- single-use preservation of pending recurrent rollback state.

## Measured result

All tests generated 128 tokens with greedy sampling, seed 1151, prompt-cache
reuse disabled, Q8_0 K cache, F16 V cache and a frozen real-context corpus.

| Context | Serial tok/s | Strict MTP tok/s | Gain | Exact output |
| --- | ---: | ---: | ---: | --- |
| 512 | 12.31 | 31.92 | 2.59x | yes |
| 4K | 11.80 | 30.08 | 2.55x | yes |
| 16K | 11.39 | 19.34 | 1.70x | yes |
| 32K | 10.87 | 23.79 | 2.19x | yes |

Before accepted-prefix replay, the 32K strict-MTP output differed from serial
at token 113. With replay, all 128 token IDs matched the serial hash
`518a9bfde6489965a6826ff61814b8abcf03a10b156a7413c3e3355bb2f35160`.
The raw requests, complete output arrays and SHA-256 manifest are retained in
the [HaloSpecKV evidence directory](https://github.com/radicalgeek/halospeckv/tree/main/evidence).

## Tested hardware

- CPU: AMD Ryzen AI Max+ 395.
- GPU: integrated AMD Radeon 8060S (`gfx1151`).
- Memory: 128 GB unified system memory; Vulkan exposed 114688 MiB.
- OS: Ubuntu 26.04, kernel 7.0.0-30.
- Graphics: Mesa RADV 26.0.8, Vulkan loader 1.4.341.
- Model: Qwen 3.8 27B ROCmFP4-FAST GGUF.

The replay mechanism is in server orchestration code and has no direct AMD API
dependency. Other platforms have not yet been tested. The AMD UMA
FlashAttention dispatch is deliberately hardware-specific.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --config Release -j
```

The measured server used:

```sh
build/bin/llama-server \
  -m MODEL.gguf \
  --mmproj MMPROJ.gguf \
  -dev Vulkan0 -ngl 99 -fa on -np 1 -kvu \
  -c 200000 -b 512 -ub 256 -t 16 --poll 100 \
  -ctk q8_0 -ctv f16 -ctxcp 0 -cram 0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.60 \
  --spec-mtp-strict-qwen \
  --reasoning-format none --metrics
```

Use the benchmark client and exact-token comparator in the HaloSpecKV
repository to reproduce the gates. The complete build and CTest outcome,
including retained upstream baseline failures, is recorded in the
[runtime validation report](https://github.com/radicalgeek/halospeckv/blob/main/docs/RUNTIME_TESTS_2026-08-26.md).

## Provenance and maintenance

HaloSpecKV is a human-directed agentic engineering project led by Mark Jones at
Radical Geek. OpenAI Codex agents performed substantial implementation and
experimental work under that direction. The code is published independently
and is not presented as conventionally human-authored.

The source project's contribution policy does not accept fully or predominantly
AI-generated pull requests. No upstream pull request has been opened. The full
provenance statement is available in
[PROVENANCE.md](https://github.com/radicalgeek/halospeckv/blob/main/PROVENANCE.md).

ROCmFPX and llama.cpp attribution, licences and third-party notices remain in
force. New work specific to this branch is released under the same MIT licence.
