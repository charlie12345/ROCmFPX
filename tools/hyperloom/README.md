# ROCmFPX Hyperloom adapter

This directory connects ROCmFPX to Hyperloom's `custom` framework mode. It is
development tooling and is not loaded by `llama-cli`, `llama-server`, or the
ROCmFPX plugin ABI.

Hyperloom writes accepted candidates as commits. Always give it a disposable
worktree on an unprotected feature branch. `launch.sh` refuses `main`, `master`,
upstream integration branches, and ROCmFPX upstream maintenance branches.

## Support status

- The benchmark contract supports decode, prefill, and native MTP objectives.
- The quality gate requires exact deterministic output and an unchanged GGUF
  file size.
- The initial gfx1151 lane uses Hyperloom's framework agent with kernel and
  roofline phases disabled.
- Single-node gfx1151 runs default to local execution instead of Ray; set
  `INFERENCE_OPTIMIZER_RAY_EXEC=1` only after qualifying a Ray deployment.
- `HYPERLOOM_EXPERIMENTAL_RDNA_KERNELS=1` opts into the unqualified kernel lane.
  GEAK is currently documented for AMD Instinct CDNA GPUs, so this is not a
  supported default for RDNA.
- Vulkan remains a separate release gate. Hyperloom's kernel tools currently
  target HIP, Triton, and FlyDSL rather than Vulkan shaders.

## Create the immutable quality reference

Build the exact baseline first, then write the reference outside the source
worktree:

```bash
export FRAMEWORK_REPO_PATH=/path/to/disposable/ROCmFPX
export MODEL_PATH=/absolute/path/to/model.gguf
export RESULT_DIR=/tmp/rocmfpx-hyperloom-reference
export ROCMFPX_QUALITY_REFERENCE=/absolute/path/outside/repo/quality-reference.json
export ROCMFPX_CREATE_REFERENCE=1
export ROCMFPX_OBJECTIVE=decode
tools/hyperloom/custom_strixhalo.sh
unset ROCMFPX_CREATE_REFERENCE
```

The reference records the deterministic output hash, model size, backend, and
baseline commit. Do not regenerate it from a candidate that Hyperloom changed.
Use the same no-spec reference for the MTP objective: this makes the exact hash
gate detect speculative-decoding output drift instead of accepting a new MTP
output as its own baseline.

## Launch Hyperloom

Install Hyperloom in a separate workspace and supply credentials through the
agent's credential store or process environment. Never place API keys in this
repository, command arguments, benchmark reports, containers, or checked-in
environment files.

The adapter is tested against Hyperloom commit
`120cb3262087b667de758cb38b51a6a481c7068b` with the companion gfx1151 patch.
Apply it before launch:

```bash
git -C "$HYPERLOOM_ROOT" checkout 120cb3262087b667de758cb38b51a6a481c7068b
git -C "$HYPERLOOM_ROOT" apply --unidiff-zero /path/to/ROCmFPX/tools/hyperloom/patches/hyperloom-gfx1151.patch
```

The launcher fails before starting Hyperloom if this board mapping is absent.
See [the patch notes](patches/README.md) for licensing and provenance.

```bash
export HYPERLOOM_ROOT=/path/to/Hyperloom
export FRAMEWORK_REPO_PATH=/path/to/disposable/ROCmFPX
export MODEL_PATH=/absolute/path/to/model.gguf
export ROCMFPX_QUALITY_REFERENCE=/absolute/path/outside/repo/quality-reference.json
export USER_DATA_PATH=/absolute/path/to/hyperloom-sessions
export ROCMFPX_OBJECTIVE=decode
tools/hyperloom/launch.sh
```

Run separate sessions with `ROCMFPX_OBJECTIVE=prefill` and
`ROCMFPX_OBJECTIVE=mtp`. The gfx1151 defaults are
`ROCMFPX_MTP_N_MAX=6`, `ROCMFPX_MTP_P_MIN=0.6`, and 256 generated tokens;
override them to compare candidates, but require the no-spec output hash to
remain unchanged. A candidate is not ready for promotion until the normal
ROCmFPX CPU, HIP, Vulkan, plugin, MTP, and format-size gates also pass.
