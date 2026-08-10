# Standalone UE4M4 Codebook10 `gfx1151` experiment

This directory contains the exact standalone HIP source and source-matched fixture bytes bound by the `ROCmFP4-UE4M4-GFX1151-RESULTS.md` evidence report.

## Files

- `ue4m4fast_deepseek_board.hip` — standalone MMVQ/MMQ and routed-semantics experiment for native `gfx1151`/wave32.
- `ue4m4fast-source-matched-device-oracle-v1.inc` — 64 deterministic S40-derived test blocks containing source values, packed wires, and expected reconstructions.
- `SHA256SUMS.txt` — immutable byte identities used by the report.

The fixture is bounded test data, not model or checkpoint tensors.

## Build

Tested with ROCm/HIP 7.14:

```bash
/opt/rocm/core-7.14/bin/hipcc   --offload-arch=gfx1151   -O2   ue4m4fast_deepseek_board.hip   -o ue4m4fast_deepseek_board
```

The binary refuses a device that is not native `gfx1151` with wave size 32. Its JSON result is a standalone runtime qualification artifact, not a GGML/GGUF integration or a model/server benchmark.

The generated integer dot instructions operate after fused Codebook10 reconstruction; this is not a native FP4-operand claim.

## Licensing status

The human contributor has reviewed and understands the exact staged source, synthetic fixture, provenance, and evidence. The contributor confirms authority to distribute the exact HIP source and synthetic fixture under MIT, approves the parent attribution, will preserve the repository MIT notice, and accepts ownership and maintenance responsibility.

On that basis, the exact two-file bundle identified by `SHA256SUMS.txt` is cleared for source-stage public publication under MIT. The underlying local generation inputs remain recorded by hashes rather than complete public Git history; that is a provenance limitation, not a redistribution blocker under the contributor's attestation.

The evidence-bound source and fixture intentionally carry no inserted header because changing either file would break the published SHA-256 binding. The repository `LICENSE` notice must accompany the distributed copy. See `PROVENANCE.md` for the derivation record and attestation boundary.
