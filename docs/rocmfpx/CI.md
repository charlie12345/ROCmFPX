# RDNA-FAST CI

RDNA-FAST (`serrynaimo/RDNA-FAST`, public, default branch `main`) inherited the ROCmFPX and llama.cpp workflow files.
On 2026-11-06 the 48 llama.cpp workflows it does not use, and ROCmFPX's self-hosted gfx1151 qualification, were removed rather than disabled (commit `f45275ab4`).
Many of the llama.cpp ones triggered on pushes to `master` only, so they never ran on `main`.

## Workflows

RDNA-FAST's own:

- `rdna-fast-release.yml` ("RDNA-FAST release"): on `v*.*.*` tags and by hand. Takes the web UI from `rdna-fast-ui.yml` and builds the Linux CPU archive, the OCI image, and ROCm 10.1 archives for Windows and Linux per GPU family; a tag run publishes them as a GitHub prerelease, a run started by hand publishes nothing. See [RELEASES.md](RELEASES.md).
- `rdna-fast-publish.yml` ("RDNA-FAST publish"): by hand. Publishes a successful release run that was started by hand: creates the tag at that run's commit and the prerelease with its archives, and pushes the image.
- `rdna-fast-ui.yml` ("RDNA-FAST UI"): on pushes to `main` that touch `tools/ui`, by hand, and from the release workflow. Keeps the rolling `ui-latest` release current, the prebuilt UI that `-DLLAMA_USE_PREBUILT_UI=ON` (the default) downloads.

From ROCmFPX:

- `check-rocmfpx.yml` ("ROCmFPX reference check"): focused hosted CPU reference, format, plugin ABI, PLE, script, and converter checks, plus a Vulkan ROCmFP4 CM1 build with the option on and off. Runs on every push to `main`, on non-draft pull requests that change a ROCmFPX implementation path, and by hand.
- `rocmfpx-upstream-sync.yml` ("ROCmFPX upstream sync"): weekly (Monday 09:23 UTC) and by hand. Tries to merge llama.cpp `master`. RDNA-FAST takes llama.cpp changes as curated picks, so a merge that conflicts ends green with a report in the job summary; only a clean merge pushes `automation/upstream-sync` and opens a pull request.

From llama.cpp, kept:

- `ui.yml`, with `ui-build.yml` (called by `ui.yml` and `docker.yml`): UI checks and tests on pushes to `main` and on pull requests that touch `tools/ui` or the server tests.
- `python-type-check.yml` and `python-check-requirements.yml`: on pushes and pull requests that change Python or requirements files.
- Scheduled: `docker.yml` (daily, images to ghcr.io), `build-cache.yml` (hourly), `build-cross.yml` and `build-msys.yml` (weekly), `close-issue.yml` (daily). `winget.yml` is scheduled daily, but its job only runs in `ggml-org` repositories.

`bench.yml.disabled` is not a workflow; GitHub ignores the extension.

## The actions policy script

`scripts/rocmfpx/manage-actions-policy.sh` is ROCmFPX's: it enabled an allowlist of workflows and disabled the rest through the GitHub CLI.
It acts only on `ROCmFPX/ROCmFPX` and only while that repository is private, so it refuses RDNA-FAST and, with `ROCmFPX/ROCmFPX` now public, every repository.
RDNA-FAST trims workflows by deleting the files instead.

## Hosted and self-hosted work

Every workflow runs on GitHub-hosted runners; none needs a self-hosted runner since ROCmFPX's manual gfx1151 qualification workflow was removed.
Hosted CI builds the ROCm archives but has no AMD GPU, so it proves compilation, not AMD runtime behavior.

If a self-hosted runner is added again, treat it as a code-execution boundary: use a dedicated machine or isolated runner account, never give its workflow a pull request trigger, and review the selected commit before a manual dispatch.
Do not register the live inference host as a general-purpose runner while protected services are active.
