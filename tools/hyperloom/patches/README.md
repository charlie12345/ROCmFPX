# Hyperloom gfx1151 compatibility patch

`hyperloom-gfx1151.patch` targets AMD-AGI/Hyperloom commit
`120cb3262087b667de758cb38b51a6a481c7068b`. It adds the `strixhalo` board
identity, gfx1151 runner mapping, Radeon 8060S product aliases, and focused
tests required by the ROCmFPX custom framework adapter.

Hyperloom remains a separate project under its own Apache-2.0 license. The
patch does not vendor Hyperloom into ROCmFPX. Review Hyperloom's license before
redistributing a patched Hyperloom build.

Apply the patch to the pinned checkout:

```bash
git -C "$HYPERLOOM_ROOT" checkout 120cb3262087b667de758cb38b51a6a481c7068b
git -C "$HYPERLOOM_ROOT" apply --unidiff-zero /path/to/ROCmFPX/tools/hyperloom/patches/hyperloom-gfx1151.patch
```
