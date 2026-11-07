# Upstream Attribution Notes

This fork contains ROCmFP4-specific work plus selected upstream llama.cpp changes.
When upstream commits are merged or cherry-picked directly, Git preserves the
original `Author` metadata. When a change has to be manually ported because it
conflicts with ROCmFP4, MTP, or local model-graph changes, keep an attribution
record here and cite the source commits in future commit messages.

## Policy for Future Upstream Ports

- Prefer direct merges or `git cherry-pick -x <sha>` when possible.
- Do not rewrite the upstream author's identity on cherry-picked commits.
- If a manual port is required, include `Based on upstream llama.cpp commit(s):`
  in the commit message, with commit SHA, title, and author.
- Keep upstream license and notice files intact.
- Push these adapted commits only to this fork unless an upstream PR is
  explicitly prepared.

## Direct Cherry-Picks / Preserved Author Metadata

The following upstream commits, picked in June 2026, preserve upstream authors
in Git metadata. Later cherry-picks keep their authors the same way and are not
listed here:

- `2998a4d7b` - `chat: fix LFM2/LFM2.5 ignoring json_schema (#24377)` by Tarek Dakhran.
- `b7ec1175d` - `common : relax sampler name matching (#23744)` by ddh0.
- `fab339703` - `server : do not clear slots without unified KV cache (#24190)` by fiesh.
- `bcbb6d0ee` - `server : disable on-device spec checkpoints (#24108)` by Georgi Gerganov.
- `67e8a335a` - `arg: fix double mtp downloads (#24128)` by Xuan-Son Nguyen.
- `7c941cb95` - `spec : fix vocab compatibility check (#24256)` by Sigbjorn Skjaeret.
- `3d669b855` - `server : free draft/MTP resources on sleep to fix VRAM leak (#23461)` by Aman Gupta.
- `500e185e2` - `graph: guard iswa kq_mask on its own buffer (#24294)` by Pascal.
- `ee949e639` - `StepFun 3.5 MTP (#23274)` by Piotr Wilkin.
- `b4d949ab7` - `diffusion: fix Windows build, skip diffusion-gemma in test-llama-archs, drop debug hooks` by danielhanchen.

## Manual Ports / Local Adaptations

Some changes were manually adapted because a direct cherry-pick conflicted with
local ROCmFP4 or MTP graph changes. These commits are local-authored in Git, so
the upstream sources are listed here explicitly.

### Chat Parser Whitespace Fixtures

The promotion CI repair restores the upstream parser expectations from:

- `a6dff7127092a9cd75db81aaef0456598d1d0452` - `chat: fix whitespace problems once and for all (#24624)` by Piotr Wilkin.

Only the affected test fixtures are ported because the corresponding parser
behavior is already present in the experimental branch.

### WebUI CI Provisioning

The adapted `scripts/webui-download.cmake` was removed by the 2026-09-15 merge
of the ROCmFPX org repo; this record is kept as history.

The promotion CI repair manually adapts these upstream fixes:

- `0c3e4fccca8aea028df37d39510e9df11d90c1b3` - `fix: Propagate version tag to WebUI asset download in self-hosted CI (#23051)` by Aleksander Grygier, co-authored by Sigbjorn Skjaeret.
- `1348f67c58f561808136e8a152a9eddec168f221` - `webui: Use lowercase hash for HF checksum check (#23107)` by Omer Ozarslan.

Local adaptations invoke the discovered npm executable for Windows
compatibility, fail when no complete asset source is available, and disable
embedded WebUI in backend-only CI jobs while dedicated WebUI jobs retain
coverage.

### Apple XCFramework Build Script

The promotion CI repair restores the tracked Apple packaging script that was
omitted when the source snapshot imported its calling workflows. The restored
file is byte-for-byte identical to:

- `4d742877b2631bd9094bc7603bc59b65940563e2` - `build : use umbrella Headers directory for XCFramework module map (#23974)` by Gerard Martinez.

### DiffusionGemma Support

The DiffusionGemma model and its `llama-server` completion path were removed by
the 2026-09-15 merge of the ROCmFPX org repo; this record is kept as history.

The upstream DiffusionGemma source branch used locally was
`upstream-diffusiongemma-pr24423`, corresponding to upstream llama.cpp PR
`#24423`.

Local commits:

- `2fc27ded0` - `add upstream diffusion gemma support`
- `1bad3e1d3` - `adapt diffusion gemma support to rocmfp4 tree`
- `2cc69b75c` - `server: add diffusion gemma completion path`
- `a175b26fd` - `server: avoid fixed diffusion token cap`

Upstream/source commits used or referenced during the port:

- `c5fe75b9765965614260f534a89a51ad03244a8e` - `diffusion-visual updates` by danielhanchen.
- `c84e85af61011f9fbfcf41479381d5ed1661a564` - `diffusion: fix Windows build, skip diffusion-gemma in test-llama-archs, drop debug hooks` by danielhanchen.
- `d8794eecd582b35ab5540e748dba057fc48ebe8b` - `examples: refactor diffusion generation (#22590)` by Shakhnazar Sailaukan.
- `6d758839ff741d4966ca92b7f801b7a8b5b96364` - `Add LLaDA-7b-MoE diffusion model (#16003)` by Aman Gupta.
- `8a4a85627702b569d7d2810f2de06a4321656e9d` - `Add LLaDA 8b Diffusion model (#14771)` by Aman Gupta.

ROCmFP4-specific additions in the local commits include adapting the model and
generation support to the ROCmFP4 tree, preserving local build compatibility,
and adding a DiffusionGemma-specific non-streaming `llama-server` completion
path.

### Qwen / Gemma / Step MTP ROCmFP4 Port

Local commit:

- `e766769de` - `port qwen gemma step mtp rocmfp4`

Upstream/source commits used or referenced during the port:

- `eef59a764264efc025be974e0452584f584a3c59` - `llama: add llm_graph_input_mtp (#23643)` by Aman Gupta.
- `2187e003378e4adf2115ee89595bf58d4ecb75fc` - `StepFun 3.5 MTP (#23274)` by Piotr Wilkin.
- `04eb4c446d22b63449d5dc41c038987d4d8cc3a6` - `llama : add Gemma4 MTP (#23398)` by Aman Gupta.
- `166fe29492abb4093ec889b5c6f6fdb4e3b8ba98` - `qwen35: use post-norm hidden state for MTP (#24025)` by Aman Gupta.
- `7d2b45b4f7b663cda74f23fbc3ce6dc3bd4f6545` - `mtp: support for gemma-4 E2B and E4B assistants (#24282)` by Max Krasnyansky.
- `e95dae18d64ae4471d61a9dc87880a64e0e5c86e` - `Remove padding and multiple D2D copies for MTP (#24086)` by Gaurav Garg.

ROCmFP4-specific additions in the local port include preserving local ROCmFP4
quantized tensor behavior, resolving graph conflicts, and keeping the Strix
build/test path working.

### Vulkan FA Dequant Scratch-Size Check

Local commit:

- `84ab95513` - `vulkan : check combined K+V f16 scratch size in FA dequant path (#30487)`

Hand port of:

- `3b83d12b0876f2eaef38c708ac40b15c6e1c7b3f` - `vulkan : check combined K+V f16 scratch size in FA dequant path (#30487)` by Nisbenz.

The local `use_dequant_kv` path has no `kv_raw_strided` branch, so only the
combined-size checks were taken.
