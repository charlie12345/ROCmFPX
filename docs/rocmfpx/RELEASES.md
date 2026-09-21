# Release process

ROCmFPX releases use SemVer tags (`vMAJOR.MINOR.PATCH`). Until the FP5/FP7
cross-generation hardware matrix is complete, releases should carry a
pre-release suffix such as `v0.1.0-alpha.1`.

Release gate:

1. Record upstream llama.cpp and ROCmFPX commits; require a clean tree.
2. Run format reference tests and `test-quantize-fns` on CPU.
3. Build CPU, Vulkan, and HIP variants; run `test-backend-ops` for every native
   ROCmFPX type and record unsupported operations explicitly.
4. Quantize from BF16/F16 and check disk size, row validation, load, fixed-logit
   equivalence, perplexity/KLD, agent tasks, and Qwen MTP on/off.
5. Run the RDNA tier matrix on physical hardware. Do not infer old-generation
   support from compilation alone.
6. Build and smoke-test the OCI image, generate checksums plus provenance/SBOM
   attestations, and attach exact build metadata.
7. Verify `LICENSE`, `NOTICE`, source archive ancestry, and third-party notices.

`.github/workflows/rocmfpx-release.yml` builds a reproducible Linux CPU archive,
publishes `ghcr.io/rocmfpx/rocmfpx:<tag>`, and creates a GitHub prerelease on
version tags. The package remains private while this repository is private; its
visibility must be changed deliberately when the project launches. Vulkan/HIP
hardware qualification remains a signed release checklist because hosted CI
does not prove AMD runtime behavior.

The Linux archive includes `include/rocmfpx/rocmfpx-plugin.h` as the standalone
plugin SDK header.

Local image use:

```bash
docker build -f .devops/rocmfpx.Dockerfile -t rocmfpx/llama.cpp:local .
docker run --rm --device /dev/dri -p 8080:8080 \
  -v /path/to/models:/models:ro \
  rocmfpx/llama.cpp:local -m /models/model.gguf -ngl 999
```

Plugins can be mounted read-only at `/opt/rocmfpx/plugins`; the image sets
`ROCMFPX_PLUGIN_PATH` to that directory. Device flags vary by Docker/Podman,
host driver, and security policy, so each release must record its tested launch
command rather than claiming one universal GPU invocation.
