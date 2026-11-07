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

RDNA-FAST (`serrynaimo/RDNA-FAST`) is a downstream of ROCmFPX
(`ROCmFPX/ROCmFPX`, earlier `charlie12345/ROCmFPX`), which tracks llama.cpp.
RDNA-FAST marks its own changes in code with `[TAG_...]` comments;
`git grep TAG_` lists them.

## Branch model

- llama.cpp `master` is the unmodified official history. The sync workflow
  adds it as the remote `upstream`.
- `main` takes llama.cpp changes as curated picks: single upstream commits,
  cherry-picked with their `(#PR)` subject line, plus merges of ROCmFPX/ROCmFPX
  `main`. A full merge of llama.cpp master conflicts here, so the workflow
  reports it instead of merging (see Syncing). This replaces the ROCmFPX rule
  of complete upstream merges only.
- `automation/upstream-*` branches are bot-created merge candidates:
  `automation/upstream-sync` from the workflow, `automation/upstream-<sha>`
  from a local run of the script. They are reviewed and tested through pull
  requests before reaching `main`.
- ROCmFPX's `rocmfpx/ancestry-restored` reference branch (the exact c49 tree
  on the real llama.cpp merge base) is not carried in this repository.

The ROCmFPX changes should be kept in narrow feature areas: GGUF format and CPU
reference code, HIP kernels, Vulkan shaders, quantization recipes, speculative
decoding, and architecture support. RDNA-FAST adds the server (SSD prompt
cache, request priority, several slots on one GPU), the expert handoff, and the
web UI. Unrelated llama.cpp code follows upstream.

## Build profiles

| Preset | HIP | Vulkan | ROCmI4 IU4/W4A4 | Intended use |
| --- | --- | --- | --- | --- |
| `strix-rocmfpx` | on | on | off | Normal gfx1151 build; exact ROCmI4 and all ROCmFPX formats |
| `strix-rocmfpx-w4a4` | on | off | on | Explicit lossy/native IU4-W4A4 experiment |

Vulkan is not globally disabled. It is disabled only by the specialized W4A4
preset because that path is HIP-only. `GGML_HIP_ROCMI4_W4A4` remains off by
default, so ordinary ROCmI4 execution retains the exact implementation.

Both presets target gfx1151 (Strix Halo) with a Linux compiler path, and
`validate-sync.sh` uses them for its HIP and W4A4 steps. RDNA-FAST builds for
the RX 7900 XT (gfx1100) with ROCm 10.1 on Windows without a preset; the
command is in the top-level [Quick start](../../README.md#quick-start).

## Syncing

Configure the official remote once:

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
```

Then run:

```bash
scripts/rocmfpx/sync-upstream.sh
```

The script reads `UPSTREAM_REMOTE` (default `upstream`) and `UPSTREAM_BRANCH`
(default `master`). In a clone where `upstream` already names ROCmFPX/ROCmFPX,
add llama.cpp under another name and set `UPSTREAM_REMOTE` to it.

The script refuses dirty worktrees, fetches official master, creates a sync
branch, performs a real merge, and runs the core validation suite. Conflicts are
left visible for human resolution (exit code 2). The scheduled GitHub workflow
(Mondays 09:23 UTC, or manual dispatch) uses the same script. A conflicting
merge, the normal case here, ends green with a report in the job summary: how
many llama.cpp commits `main` lacks and which paths conflict. A clean merge
that passes the portable checks is pushed as `automation/upstream-sync` with a
pull request.

## Required gates

Every upstream-sync pull request must preserve:

1. Stable GGUF tensor IDs 100-111 (107 stays reserved and rejected) and
   file-type IDs 100-124.
2. CPU reference round trips for ROCmFPX, ROCmFP2, ROCmFP4, and ROCmI4.
3. Successful `llama-quantize` and backend-op builds.
4. Vulkan compilation and gfx1151 HIP compilation.
5. Exact ROCmI4 behavior with W4A4 disabled.
6. Separate opt-in W4A4 compilation with Vulkan disabled.
7. Fixed-model performance checks against the last promoted build. Regressions
   require an explicit explanation and approval; a clean compile is not enough.

Use `ROCMFPX_VALIDATE_VULKAN=1`, `ROCMFPX_VALIDATE_HIP=1`, and
`ROCMFPX_VALIDATE_W4A4=1` with `validate-sync.sh` on capable runners. Run GPU
benchmarks sequentially in detached sessions on the measurement host: gfx1151
for ROCmFPX, the RX 7900 XT (gfx1100) for RDNA-FAST.

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
