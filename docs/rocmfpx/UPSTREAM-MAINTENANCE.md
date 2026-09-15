# ROCmFPX upstream maintenance

This process is downstream-only. It does not push branches, commits, issues, or
pull requests to `ggml-org/llama.cpp`. The scheduled workflow pushes only to
the repository that owns the workflow and opens its review pull request there.
It explicitly refuses to run in, or push to, official llama.cpp, so the clean
downstream repository may have a new name without weakening that boundary.

ROCmFPX should remain a downstream of `ggml-org/llama.cpp`, not a renamed copy
of it. Keep upstream directory names, targets, executable names, and public APIs
unless a ROCmFPX feature genuinely requires an extension. This minimizes merge
conflicts and makes individual features practical upstream submissions.

## Branch model

- `upstream/master` is the unmodified official llama.cpp history.
- `main` is ROCmFPX and receives complete upstream merges. Do not cherry-pick
  only attractive upstream commits; that recreates an untraceable snapshot.
- `rocmfpx/ancestry-restored` is the exact c49 ROCmFPX reference tree with its
  history attached to the real llama.cpp merge base. Keep it as a permanent
  behavior and performance reference.
- `automation/upstream-*` branches are bot-created merge candidates. They are
  reviewed and tested through pull requests before reaching `main`.

The ROCmFPX changes should be kept in narrow feature areas: GGUF format and CPU
reference code, HIP kernels, Vulkan shaders, quantization recipes, speculative
decoding, and architecture support. Unrelated llama.cpp code follows upstream.

## Build profiles

| Preset | HIP | Vulkan | ROCmI4 IU4/W4A4 | Intended use |
| --- | --- | --- | --- | --- |
| `strix-rocmfpx` | on | on | off | Normal gfx1151 build; exact ROCmI4 and all ROCmFPX formats |
| `strix-rocmfpx-w4a4` | on | off | on | Explicit lossy/native IU4-W4A4 experiment |

Vulkan is not globally disabled. It is disabled only by the specialized W4A4
preset because that path is HIP-only. `GGML_HIP_ROCMI4_W4A4` remains off by
default, so ordinary ROCmI4 execution retains the exact implementation.

## Syncing

Configure the official remote once:

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
```

Then run:

```bash
scripts/rocmfpx/sync-upstream.sh
```

The script refuses dirty worktrees, fetches official master, creates a sync
branch, performs a real merge, and runs the core validation suite. Conflicts are
left visible for human resolution. The scheduled GitHub workflow uses the same
script and opens a pull request only when the merge and portable checks pass.

## Required gates

Every upstream-sync pull request must preserve:

1. Stable GGUF tensor IDs 100-110 and file-type IDs 100-124.
2. CPU reference round trips for ROCmFPX, ROCmFP2, ROCmFP4, and ROCmI4.
3. Successful `llama-quantize` and backend-op builds.
4. Vulkan compilation and gfx1151 HIP compilation.
5. Exact ROCmI4 behavior with W4A4 disabled.
6. Separate opt-in W4A4 compilation with Vulkan disabled.
7. Fixed-model performance checks against the last promoted build. Regressions
   require an explicit explanation and approval; a clean compile is not enough.

Use `ROCMFPX_VALIDATE_VULKAN=1`, `ROCMFPX_VALIDATE_HIP=1`, and
`ROCMFPX_VALIDATE_W4A4=1` with `validate-sync.sh` on capable runners. Run GPU
benchmarks sequentially in detached sessions on the gfx1151 host.

## Path toward upstreaming

This section is future guidance, not current automation or authorization.
Nothing should be submitted upstream until the maintainer explicitly chooses
to begin that separate effort.

Submit independent llama.cpp pull requests in dependency order: format-neutral
AMD fixes, CPU reference/format definitions, Vulkan kernels, HIP kernels, then
optional quantization recipes. Each submission should include focused tests and
before/after measurements. ROCmFPX branding and downstream automation stay in
this repository; generally useful AMD implementation work is what moves
upstream.
