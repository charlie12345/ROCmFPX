# ROCmFPX CI policy

ROCmFPX keeps the upstream llama.cpp workflow files intact for easier upstream merges, but most inherited workflows are disabled in the private GitHub repository.
Only these workflows remain enabled:

- `check-rocmfpx.yml`: focused hosted CPU reference, format, plugin ABI, PLE, script, and converter checks for ROCmFPX paths.
- `rocmfpx-gfx1151.yml`: manual-only Vulkan and HIP qualification on a dedicated self-hosted gfx1151 runner.
- `rocmfpx-upstream-sync.yml`: weekly and manual upstream merge preparation.
- `rocmfpx-release.yml`: manual or version-tag release packaging.

This policy avoids running the full upstream CPU, Windows, CUDA, server, UI, WebGPU, and cross-platform matrices for every private downstream pull request.
Disabled workflow source files continue to receive normal upstream merges and can be re-enabled temporarily when a cross-platform qualification is required.

## Apply or inspect the policy

The policy script verifies the exact private destination before changing workflow state.
It uses the authenticated GitHub CLI session and never reads or stores a token itself.

```bash
scripts/rocmfpx/manage-actions-policy.sh --dry-run
scripts/rocmfpx/manage-actions-policy.sh --apply
scripts/rocmfpx/manage-actions-policy.sh --check
```

`--dry-run` reports drift and exits successfully, `--apply` reconciles workflow states, and `--check` returns nonzero when the GitHub configuration differs from the allowlist.
Run the policy after the first private publication and after an upstream merge adds a workflow file.

## Hosted and self-hosted work

The focused reference check uses a GitHub-hosted Linux runner only when a pull request or push changes a ROCmFPX implementation path.
The gfx1151 workflow has no push, pull request, or schedule trigger and therefore cannot start automatically.
It requires a runner with the labels `self-hosted`, `linux`, `x64`, `rocmfpx`, and `gfx1151`.

Treat a self-hosted runner as a code-execution boundary.
Use a dedicated machine or isolated runner account, keep the repository private, never add a pull request trigger to the gfx1151 workflow, and review the selected commit before manual dispatch.
Do not register the live inference host as a general-purpose runner while protected services are active.

GitHub billing and budget controls belong to the repository owner account.
Keeping a zero paid-overage budget prevents surprise hosted-runner charges but pauses hosted checks after the included allowance is exhausted.
