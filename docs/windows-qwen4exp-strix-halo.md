# Windows Qwen4Exp / Strix Halo

This branch is a Windows HIP build of the ROCmFPX fork with the Qwen4Exp
architecture, embedded MTP support, and a Windows mmap-backed PLE pager.
It targets AMD Strix Halo / `gfx1151` and the ROCmFP4 Qwen3.8 Flash-Next GGUF.

## Build

Use a Visual Studio Developer PowerShell with the Windows SDK, CMake, and a
ROCm installation that provides HIP for the target device:

```powershell
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-fit-params llama-bench -j 4
```

The resulting executables are under `build/bin/`.

## Runtime profile

For the Qwen3.8 Flash-Next ROCmFP4 STRIX model, the tested profile is:

```text
--ctx-size 262144 --spec-type draft-mtp --flash-attn on --spec-draft-n-max 3
--fit off --spec-draft-model C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf
--ctx-checkpoints 32 --batch-size 2048 --ubatch-size 512 --parallel 1
--no-webui --cache-ram 0 --mmap --tensor-read-lazy on
```

Use `--device ROCm0` when selecting a device explicitly. `--tensor-read-lazy`
requires `--mmap` and activates the mmap-backed PLE path in this branch.

## Windows memory behavior

`--tensor-read-lazy on` delays tensor access and avoids eagerly prefetching the
large PLE tensor. Windows can still retain file-mapped pages in the process
working set or standby cache after they are touched. Therefore the tested
Windows deployment also applies a hard working-set limit after the server is
healthy:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\set-windows-working-set.ps1 -Port 8080 -MaxGiB 8
```

The limit is per process and must be applied again after a server restart. It
reduces resident process pages; it does not reduce GPU/UMA committed memory and
does not provide a strict SSD-only guarantee. A lower limit can increase PLE
page faults and reduce prompt-processing speed.

## Scope and limitations

- This is a hardware- and driver-specific experimental build for Windows HIP /
  `gfx1151`.
- The embedded MTP path is intended for text generation. Keep vision disabled
  unless the multimodal path has been tested separately with the exact build.
- The repository does not contain model files, MTP files, ROCm DLLs, or local
  benchmark logs. Supply those separately.
