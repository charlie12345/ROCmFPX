# Release process

RDNA-FAST releases use SemVer tags (`vMAJOR.MINOR.PATCH`). Both release
workflows publish the GitHub release as a prerelease, and
`rdna-fast-publish.yml` accepts plain `vX.Y.Z` tags only, without a pre-release
suffix. Once the release has been checked, the maintainer promotes the
prerelease to a full release by hand on GitHub.

Release gate:

1. Record upstream llama.cpp and ROCmFPX commits; require a clean tree.
2. Run format reference tests and `test-quantize-fns` on CPU.
3. Build CPU, Vulkan, and HIP variants; run `test-backend-ops` for every native
   ROCmFPX type and record unsupported operations explicitly.
4. Quantize from BF16/F16 and check disk size, row validation, load, fixed-logit
   equivalence, perplexity/KLD, agent tasks, and Qwen MTP on/off.
5. Run the hardware tests on the RX 7900 XT. Do not infer support for other
   cards from compilation alone.
6. Build and smoke-test the OCI image, generate checksums plus provenance/SBOM
   attestations, and attach exact build metadata.
7. Verify `LICENSE`, `NOTICE`, source archive ancestry, and third-party notices.

RDNA-FAST's hardware runs are on one RX 7900 XT (`gfx1100`). Every other target
in the release archives is compiled by CI but not run on a card here.

`.github/workflows/rdna-fast-release.yml` runs on version tags and by hand. It
builds and tests:

- `rdna-fast-linux-x86_64.tar.gz`: a reproducible Linux CPU archive;
- `rdna-fast-windows-rocm-10.1-<family>.zip` and
  `rdna-fast-linux-rocm-10.1-<family>.tar.gz`, HIP builds with ROCm 10.1 for
  `gfx110X` (`gfx1100`-`gfx1103`), `gfx115X` (`gfx1150`-`gfx1152`) and
  `gfx120X` (`gfx1200`, `gfx1201`). The Windows archives carry the HIP runtime
  DLLs; both still need ROCm 10.1 at run time (on Windows its `bin` folder, for
  hipBLAS and rocBLAS);
- `rdna-fast-ui.tar.gz`, the web UI the builds embed;
- the OCI image `ghcr.io/serrynaimo/rdna-fast:<tag>` (a Vulkan build, entrypoint
  `llama-server`), with provenance and SBOM attestations.

Every archive has a `.sha256` file. A tag run pushes the image and creates the
GitHub prerelease with the archives. A run started by hand publishes nothing;
`.github/workflows/rdna-fast-publish.yml` publishes such a run afterwards: it
creates the tag at the run's commit, the prerelease with the run's archives,
and pushes the image. The repository and the image are public. Vulkan/HIP
hardware qualification remains a signed release checklist because hosted CI
does not prove AMD runtime behavior.

The Linux CPU archive includes `include/rocmfpx/rocmfpx-plugin.h` as the
standalone plugin SDK header.

Local image use:

```bash
docker build -f .devops/rocmfpx.Dockerfile -t rocmfpx/llama.cpp:local .
docker run --rm --device /dev/dri -p 8080:8080 \
  -v /path/to/models:/models:ro \
  rocmfpx/llama.cpp:local -m /models/model.gguf -ngl 999 --host 0.0.0.0 --port 8080
```

The image's entrypoint is `llama-server`. Arguments after the image name replace
its default `--host 0.0.0.0 --port 8080`, so pass them again as above.

Plugins can be mounted read-only at `/opt/rocmfpx/plugins`; the image sets
`ROCMFPX_PLUGIN_PATH` to that directory. Device flags vary by Docker/Podman,
host driver, and security policy, so each release must record its tested launch
command rather than claiming one universal GPU invocation.
