# Third-Party Notices

RDNA-FAST (https://github.com/serrynaimo/RDNA-FAST) is a fork of ROCmFPX
(https://github.com/ROCmFPX/ROCmFPX, earlier https://github.com/charlie12345/ROCmFPX),
which is a fork of `llama.cpp`. The tree keeps the upstream MIT license and the
third-party license files that ship with it. This file lists the third-party code
in the tree, the code taken from other projects, and the third-party files the
binary release packages carry. `NOTICE` holds the ROCmFPX attribution record.

## Main Project

- `llama.cpp` / `ggml`
  - Source: https://github.com/ggml-org/llama.cpp, merged up to `f2cc7282c` (2026-10-11);
    later upstream commits are cherry-picked or ported one by one
    (`docs/UPSTREAM-ATTRIBUTION.md`)
  - License: MIT
  - License file: `LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023-2026 The ggml authors`

- `ROCmFPX`
  - Source: https://github.com/ROCmFPX/ROCmFPX, merged up to `120141f1f` (2026-10-17);
    earlier https://github.com/charlie12345/ROCmFPX
  - License: MIT
  - License file: `LICENSE`; attribution record: `NOTICE`
  - Copyright notice in this checkout:
    `Copyright (c) 2025-2026 Carlo Pasquale (Charlie12345), ROCmFPX additions`

## Bundled Third-Party Components

These come with llama.cpp, except `rocm_wmma_gemm`. `scripts/sync_vendor.py` records
the sources and pins of the llama.cpp ones in `vendor/`.

- `cpp-httplib`
  - Source: https://github.com/yhirose/cpp-httplib, version 0.64.0
  - License: MIT
  - License files: `vendor/cpp-httplib/LICENSE`, `licenses/LICENSE-httplib`
  - Copyright notice in this checkout: `Copyright (c) 2017 yhirose`
  - Used for: the HTTP server and client (`llama-server`, model downloads)

- `nlohmann/json`
  - Source: https://github.com/nlohmann/json, version 3.12.0, with the deep-nesting fixes
    backported in `vendor/nlohmann/json-deep-nesting.patch`
  - License: MIT
  - License file: `licenses/LICENSE-jsonhpp`
  - Copyright notice in this checkout: `Copyright (c) 2013-2025 Niels Lohmann`
  - Used for: JSON parsing and output

- `stb_image`
  - Source: https://github.com/nothings/stb, version 2.30 (`vendor/stb/stb_image.h`)
  - License: MIT or public domain (Unlicense), at the user's choice; the text is at the
    end of the file
  - Copyright notice in this checkout: `Copyright (c) 2017 Sean Barrett`
  - Used for: image decoding for multimodal input (`tools/mtmd`)

- `miniaudio`
  - Source: https://github.com/mackron/miniaudio, version 0.11.26 (commit `877ea4a7c260`,
    `vendor/miniaudio/miniaudio.h`)
  - License: public domain (Unlicense) or MIT No Attribution (MIT-0), at the user's
    choice; the text is at the end of the file
  - Copyright notice in this checkout: `Copyright 2026 David Reid` (MIT-0 alternative)
  - Used for: audio decoding for multimodal input (`tools/mtmd`)

- `subprocess.h`
  - Source: https://github.com/sheredom/subprocess.h, commit `0dccaa9aa176`
    (`vendor/sheredom/subprocess.h`)
  - License: public domain (Unlicense); the text is at the top of the file
  - Used for: starting child processes (`common/subproc.h`, `tools/mtmd`)

- `xxHash`
  - Source: https://github.com/Cyan4973/xxHash, version 0.8.3 (commit `9f465f1ea932`),
    with a local Windows SDK fix listed in `scripts/sync_vendor.py`
  - License: BSD-2-Clause
  - License file: `vendor/hash/xxhash/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2012-2021 Yann Collet`
  - Used for: `examples/gguf-hash`

- `sha1`
  - Source: https://github.com/clibs/sha1, commit `e1e2536fcf6a`, placed in a C++
    namespace locally (`scripts/sync_vendor.py`)
  - License: public domain (stated in `vendor/hash/sha1/sha1.c`)
  - Used for: `examples/gguf-hash`

- `sha256.c`
  - Source: https://github.com/jb55/sha256.c, commit `5e637272c13f`
  - License: public domain
  - License file: `vendor/hash/sha256/LICENSE`
  - Used for: SHA-256 hashes in `tools/mtmd` and `llama-server` (`vendor/hash/hash.h`)

- `rotate-bits.h`
  - Source: https://github.com/jb55/rotate-bits.h, commit `27e784942f67`
  - License: MIT
  - License file: `vendor/hash/rotate-bits/LICENSE.md`
  - Copyright notice in this checkout: `Copyright (c) 2021 William Casarin`
  - Used for: `sha256.c`

- `base64.hpp`
  - Source: not named in the file (`common/base64.hpp`)
  - License: public domain (Unlicense); the text is at the top of the file

- `gguf-py`
  - License: MIT
  - License file: `gguf-py/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023 Georgi Gerganov`

- `rocm_wmma_gemm` (added in `96c837311`, [TAG_WMMA_GEMM])
  - Source: https://github.com/adelj88/rocm_wmma_gemm, commit `ea3aa74`
  - License: MIT
  - License files: `vendor/rocm_wmma_gemm/LICENSE`, `licenses/LICENSE-rocm_wmma_gemm`
  - Copyright notice in this checkout: `Copyright (c) 2024 Adel Johar`
  - Used for: prompt-sized products of quantized weights on gfx11 GPUs, with the weights
    unpacked to f16 (`ggml/src/ggml-cuda/wmma-gemm.cu`), built as the
    `rocm_wmma_gemm_<arch>_accf32` libraries next to the binaries
  - Included parts and local changes: `vendor/rocm_wmma_gemm/README.md` (a Windows path
    fix in `config/generate_config.py`; tuned entries added to
    `config/gemm_config_gfx1100_f32.json`, the upstream entries unchanged)

## Code Taken From Other Projects

ROCmFPX and RDNA-FAST commits that took code from other llama.cpp forks. The RDNA-FAST
commits name the original authors in `Co-authored-by` trailers.

- `AMD-Ecosystem/llama.cpp`
  - Source: https://github.com/AMD-Ecosystem/llama.cpp
  - License: MIT; its `LICENSE` is llama.cpp's (`Copyright (c) 2023-2026 The ggml authors`)
  - Taken:
    - PR #41 (D=256 tile flash-attention configuration for RDNA3.5,
      `ggml/src/ggml-cuda/fattn-tile.cuh`), PR #39 (MoE MMQ tile size from the tokens per
      expert, `GGML_CUDA_MMQ_MOE_TILE_MULT`, `ggml/src/ggml-cuda/mmq.cu`) and PR #32
      (pipelined activation tile loads for RDNA3.5, opt-in `GGML_CUDA_MMQ_Y_PIPELINE`,
      `ggml/src/ggml-cuda/mmq.cuh`): commit `f07a08084`
    - PR #61 by Annie Ren (dequantize-to-float matvec for Q4_K, Q5_K and Q6_K,
      `ggml/src/ggml-cuda/mmvdq.cu`, `mmvdq.cuh`): commit `04b12624f`, from the nasone32
      fork below (its commit `670512936`)

- `nasone32/llama.cpp-RDNA3-7900xtx-opt`
  - Source: https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt
  - License: MIT; its `LICENSE` is llama.cpp's
  - Taken in commit `a6015b580` (multi-GPU; `common/`, `include/llama.h`,
    `src/llama-context.cpp`, `src/llama-model.cpp`, `src/models/dflash.cpp`,
    `ggml/src/ggml-cuda/allreduce.cu`):
    - `84b31d4cd` by wOvAN: a draft context gets the backends of the other model's devices
    - `22ed83e9a` by Denis Artyomov: DFlash2 under tensor split
    - `7c5bb5cb9` and `e06dcf630`: the opt-in direct AllReduce between two HIP devices
      (`GGML_CUDA_AR_P2P`)
  - The mmvdq kernel (`670512936`), see above

- `stew675/llama-cpp-rdna-boosts`
  - Source: https://github.com/stew675/llama-cpp-rdna-boosts
  - License: MIT
  - License file: `licenses/LICENSE-llama-cpp-rdna-boosts`
  - Copyright notice: `Copyright (c) 2026 stew675`
  - Taken:
    - block 04 (RDNA4 WMMA flash attention for heads 320, 512 and 576 with configuration
      rows tuned on gfx1201; Q6_K MMQ row scales on RDNA4): commit `d85a737ce`,
      `ggml/src/ggml-cuda/fattn.cu`, `fattn-mma-f16.cuh`, `mmq-vec-dot.cuh`
      ([TAG_FA_WMMA_RDNA4], [TAG_Q6K_MMQ_RDNA4])
    - block 09 (meta backend compute-container headroom 16 -> 128 views per tensor):
      commit `a6015b580`, `ggml/src/ggml-backend-meta.cpp`
  - The same blocks are in the nasone32 fork as `371270e17` and `91de35039`

## Code From Other Projects In The llama.cpp Sources

These files come with llama.cpp and say where their code comes from. The license is the
one stated in the file or by the source project.

- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`: llamafile (https://github.com/Mozilla-Ocho/llamafile),
  MIT, `Copyright 2024 Mozilla Foundation`
- YaRN RoPE scaling in `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/rope.cu`,
  `ggml/src/ggml-metal/kernels/rope.metal` and `ggml/src/ggml-opencl/kernels/rope.cl`:
  https://github.com/jquesnelle/yarn, MIT, `Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng`
- the BPE tokenizer in `src/llama-vocab.cpp`: adapted from https://github.com/cmp-nct/ggllm.cpp, MIT
- the Z-algorithm of the DRY sampler in `src/llama-sampler.cpp`: adapted from
  https://github.com/ivanyu/string-algorithms, public domain (Unlicense)
- `bytes_to_unicode` and the Tekken vocabulary extraction in `gguf-py/gguf/vocab.py`: from
  Hugging Face Transformers (https://github.com/huggingface/transformers), Apache-2.0
- `tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py`: SigLIP code from
  Hugging Face, Apache-2.0, `Copyright 2024 Google AI and The HuggingFace Team`
- the image resampling in `tools/mtmd/mtmd-image.cpp`: adapted from Pillow
  (https://github.com/python-pillow/Pillow, `src/libImaging/Resample.c`), MIT-CMU
- `exp_ps_sve` in `ggml/src/ggml-cpu/vec.h`: from OpenVINO
  (https://github.com/openvinotoolkit/openvino), Apache-2.0
- the files in `ggml/src/ggml-openvino/openvino/` with an Intel Corporation SPDX header:
  Apache-2.0
- `ggml/src/ggml-sycl/`, `ggml/include/ggml-sycl.h`, `examples/sycl/` and
  `scripts/install-oneapi.bat`: Intel Corporation and Codeplay Software, MIT; the SYCL
  sources also carry LLVM project code under Apache-2.0 WITH LLVM-exception (SPDX headers
  in the files)
- `ggml/src/ggml-cpu/kleidiai/`: Arm Limited, MIT (SPDX headers); the KleidiAI library is
  downloaded at build time only with `GGML_CPU_KLEIDIAI=ON`
- `ggml/src/ggml-hexagon/htp/hmx-utils.h`: ported from https://github.com/haozixu/htp-ops-lib;
  no license stated in the file
- part of `ggml/src/ggml-et/ggml-et.cpp`: adapted from ainekko/et-platform
  (`esperanto-tools-libs/tools/src/bench.cpp`); no license stated in the file
- `tests/gguf-model-data.cpp`: GGUF parser adapted from the Hugging Face `gguf` package
  (https://github.com/huggingface/huggingface.js), MIT
- `examples/llama.android/gradlew`: the Gradle wrapper script, Apache-2.0
- `models/templates/`: chat templates from the model repositories they are named after;
  `HuggingFaceTB-SmolLM3-3B.jinja`, `unsloth-Apriel-1.5.jinja` and
  `unsloth-mistral-Devstral-Small-2507.jinja` carry an Unsloth Apache-2.0 notice

## Web UI (`tools/ui`)

The web UI is built with Vite. The build output is embedded into `llama-server`
(`LLAMA_BUILD_UI`) and published as the `rdna-fast-ui.tar.gz` release asset.

- `big-integer`, `decimal.js` and `nerdamer-prime` (vendored in `tools/ui/src/lib/vendors/`,
  from llama.cpp; used by the JavaScript sandbox tool)
  - `big-integer`: https://github.com/peterolson/BigInteger.js, public domain (Unlicense),
    `tools/ui/src/lib/vendors/big-integer/LICENSE`
  - `decimal.js`: https://github.com/MikeMcl/decimal.js, version 10.6.0, MIT,
    `tools/ui/src/lib/vendors/decimal.js/LICENCE.md`, `Copyright (c) 2025 Michael Mclaughlin`
  - `nerdamer-prime`: https://github.com/together-science/nerdamer-prime (a fork of
    https://github.com/jiggzson/nerdamer), version 1.1.16, MIT,
    `tools/ui/src/lib/vendors/nerdamer-prime/LICENSE`, `Copyright (c) 2023 together-science`

- Exo 2 (added by RDNA-FAST in `f8c3d90c8`)
  - Source: https://github.com/googlefonts/Exo-2.0; Latin subsets of the ExtraBold Italic
    and Black Italic weights (`tools/ui/static/fonts/exo2/*.woff2`)
  - License: SIL Open Font License 1.1
  - License file: `tools/ui/static/fonts/exo2/OFL.txt`, also in the built UI next to the fonts
  - Copyright notice in this checkout: `Copyright 2013 The Exo 2 Project Authors`
  - Used for: the RDNA-FAST wordmark and figures

- `tools/ui/static/recommended-mcp/`: the icons of the recommended MCP servers' providers,
  from llama.cpp

- npm packages
  - `tools/ui/package.json` lists them and `tools/ui/package-lock.json` pins their versions.
    The build bundles the packages the UI imports, with their dependencies.
  - Imported directly, with the license each package states: `svelte`, `@sveltejs/kit`,
    `bits-ui`, `svelte-sonner`, `vaul-svelte`, `mode-watcher`, `tailwind-merge`,
    `tailwind-variants`, `clsx`, `@modelcontextprotocol/sdk`, `unified`, `remark`,
    `remark-gfm`, `remark-math`, `remark-breaks`, `remark-rehype`, `rehype-katex`,
    `rehype-highlight`, `rehype-stringify`, `unist-util-visit`, `unist-util-visit-parents`,
    `katex` (with its fonts), `mermaid`, `lowlight`, `fflate`, `@vite-pwa/sveltekit`, `vite-plugin-pwa`
    (`workbox-window`) and `modern-screenshot` (added by RDNA-FAST for the speed card
    export): MIT; `@lucide/svelte`: ISC; `highlight.js`: BSD-3-Clause; `pdfjs-dist` and
    `dexie`: Apache-2.0; `dompurify`: MPL-2.0 OR Apache-2.0
  - Their dependencies add packages under ISC, BSD-2-Clause, BSD-3-Clause, Apache-2.0,
    0BSD and Unlicense (for example `mermaid`'s d3 modules).
  - No per-package license list is generated for the built UI, and the minified bundle
    does not keep the packages' license comments.

## Binary Release Packages

- The binary archives (Linux CPU, Linux and Windows ROCm) built by
  `.github/workflows/rdna-fast-release.yml` carry `LICENSE`, `NOTICE` and `licenses/`.
  The container image and `rdna-fast-ui.tar.gz` do not.
- The Windows ROCm archives also carry the HIP runtime of the ROCm 10.1 SDK they are built
  with, copied next to the binaries ([TAG_BUNDLE_HIP_RUNTIME],
  `ggml/src/ggml-hip/CMakeLists.txt`). Their license texts are in `licenses/hip-runtime/`,
  outside the `licenses/LICENSE-*` files that `app/CMakeLists.txt` embeds into the `llama`
  binary:
  - `amdhip64_7.dll`: the HIP runtime (ROCm CLR, https://github.com/ROCm/rocm-systems,
    `projects/clr`), MIT, `Copyright (C) Advanced Micro Devices, Inc.`,
    `licenses/hip-runtime/LICENSE-amdhip64`
  - `amd_comgr.dll`: the Code Object Manager (https://github.com/ROCm/llvm-project,
    `amd/comgr`), Apache-2.0 WITH LLVM-exception, with the legacy University of
    Illinois/NCSA license, `licenses/hip-runtime/LICENSE-amd_comgr` (from the SDK's
    `share/doc/amd_comgr/LICENSE.txt`)
  - `rocm_kpack.dll`: kpack (https://github.com/ROCm/rocm-systems, `shared/kpack`), MIT,
    `Copyright Advanced Micro Devices, Inc.`, `licenses/hip-runtime/LICENSE-rocm_kpack`
- The `rocm_wmma_gemm_<arch>_accf32` libraries: see `rocm_wmma_gemm` above.
- Optional downloads at build time are off by default and not used by the release
  workflows; they come with their own licenses. Examples: BoringSSL or LibreSSL
  (`LLAMA_BUILD_BORINGSSL`, `LLAMA_BUILD_LIBRESSL`), LLVM OpenMP (`GGML_OPENMP_FETCH`),
  KleidiAI (`GGML_CPU_KLEIDIAI`), NVIDIA CCCL (`GGML_CUDA_CCCL_VERSION`), llguidance
  (`LLAMA_LLGUIDANCE`), ZenDNN and the prebuilt `llama-connect` binary (`LLAMA_CONNECT`).

## Generated and Ignored Artifacts

Build directories, generated benchmark reports, logs, and GGUF model files are
ignored by `.gitignore` and are not intended to be published in this source
repository. The exceptions are the vocabulary-only test files `models/ggml-vocab-*.gguf`
from llama.cpp, which hold the tokenizer data of the models they are named after.

Model weights are not included. Any model downloaded or quantized for testing
remains subject to the original model publisher's license and terms.

## ROCmFPX and RDNA-FAST Additions

The ROCmFPX and RDNA-FAST source files, scripts, and documentation added on top of
llama.cpp are provided under the same MIT license as the rest of this source tree
(`LICENSE`) unless a file states otherwise.
