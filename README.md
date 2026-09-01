# win-Qwen4exp-rocm-Strix-Halo

Windows 专用的 `llama.cpp` 分支，用于在 AMD Strix Halo / `gfx1151` 上运行
Qwen3.8-Flash-Next ROCmFP4 模型。

本项目基于 [kingjones30/ROCmFPX](https://github.com/kingjones30/ROCmFPX)，
保留 ROCmFPX 的模型格式和 HIP 内核，并加入了针对本机大模型部署的 Windows
Qwen4Exp、MTP 和 PLE mmap 支持。

## 适用环境

- Windows 10/11
- AMD Ryzen AI MAX+ / Strix Halo
- GPU 架构：`gfx1151`
- HIP/ROCm 环境（本机使用 TheRock HIP）
- ROCmFP4 Qwen3.8-Flash-Next GGUF

这是硬件和驱动相关的实验性分支，不是通用的跨平台发行版。

## 本项目增加的内容

- Qwen4Exp 架构支持
- ROCmFPX 权重格式的 HIP 加速
- 嵌入式 Qwen MTP 推测解码
- Windows PLE mmap pager
- `--tensor-read-lazy off|auto|on`
- 有界 PLE 读取和 mmap 延迟读取测试
- Windows 进程工作集限制脚本

## 构建

在 Visual Studio Developer PowerShell 中执行：

```powershell
cmake -S . -B build `
  -DGGML_HIP=ON `
  -DGPU_TARGETS=gfx1151 `
  -DGGML_NATIVE=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server llama-fit-params llama-bench -j 4
```

输出文件位于 `build/bin/`。

如果 HIP 未被 CMake 自动找到，请先确保 `hipcc`、HIP headers、hipBLAS 和
ROCm DLL 已加入当前开发环境。

## Qwen3.8 + MTP 启动示例

将目标模型和 MTP 模型路径替换为本机实际路径：

```powershell
./build/bin/llama-server.exe `
  -m "C:/models/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf" `
  --host 127.0.0.1 `
  --port 8080 `
  --device ROCm0 `
  --mmap `
  --tensor-read-lazy on `
  --ctx-size 262144 `
  --flash-attn on `
  --fit off `
  --spec-type draft-mtp `
  --spec-draft-n-max 3 `
  --spec-draft-model "C:/models/MTP/Qwen3.8-Flash-Next-MTP-Q8_0.gguf" `
  --ctx-checkpoints 32 `
  --batch-size 2048 `
  --ubatch-size 512 `
  --parallel 1 `
  --cache-ram 0 `
  --no-mmproj `
  --no-webui
```

在 llama-hub 中对应的核心参数是：

```text
--ctx-size 262144 --spec-type draft-mtp --flash-attn on
--spec-draft-n-max 3 --fit off
--spec-draft-model C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf
--ctx-checkpoints 32 --batch-size 2048 --ubatch-size 512
--parallel 1 --cache-ram 0 --mmap --tensor-read-lazy on
```

## Windows 内存限制

`--tensor-read-lazy on` 会让大 PLE 张量使用 mmap 延迟读取。当前构建在模型加载
完成后还会自动执行一次 `EmptyWorkingSet`，清理加载阶段进入进程工作集的文件页，
日志中会出现 `PLE pager: loading-page trim completed`。

这只是一次性清理，不是持续的 8 GiB 硬上限；Windows 仍可能把之后访问的文件页
留在进程工作集或系统缓存中，也不等于严格的“PLE 只放 SSD”。只有在自动清理后
内存仍然紧张，或需要持续限制工作集时，才运行下面的可选脚本：

模型健康启动后，在管理员 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ./scripts/set-windows-working-set.ps1 `
  -Port 8080 `
  -MaxGiB 8
```

脚本设置的硬上限只对当前 `llama-server.exe` PID 生效，重启模型后需要重新执行。
限制过低可能增加 PLE 缺页读取并降低预填充速度；它也不会降低 GPU/UMA 提交量。

## DFlash

本分支保留上游的通用 DFlash 支持，启动类型为：

```text
--spec-type draft-dflash
--model-draft <DFlash GGUF>
```

名称为 DFlash2 的 GGUF 通常使用 `general.architecture=dflash` 元数据，因此不使用
`draft-dflash2`。DFlash 草稿模型必须与对应的目标模型配套。

## 验证状态

本分支已在 Windows + AMD Strix Halo / `gfx1151` 上完成：

- HIP Release 编译
- `llama-server` 链接
- `llama-fit-params` 链接
- `llama-bench` 链接
- Qwen4Exp 模型加载和文本 MTP 运行
- PLE mmap pager 激活验证

当前仓库的 GitHub Actions 只执行 ROCmFPX 参考检查。GitHub 托管 runner 没有
本机 Strix Halo 硬件，因此不能替代真实的 Windows HIP、PLE 和 MTP 验收。

## 已知限制

- 仅针对 Windows HIP / `gfx1151` 优先维护。
- 不包含模型文件、MTP 文件、mmproj、ROCm DLL 或本地日志。
- 当前推荐的 MTP 配置是 `n-max 3`；实际收益取决于输入内容。
- PLE mmap 和工作集限制是 Windows 下的近似方案，不保证所有文件页始终只存在 SSD。
- 多模态请求应使用单独的、经过验证的配置；当前文本 MTP 配置默认关闭 mmproj。

## 上游来源

- [kingjones30/ROCmFPX](https://github.com/kingjones30/ROCmFPX)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

上游项目的许可证和第三方声明仍以本仓库中的 `LICENSE`、`AUTHORS` 和
`THIRD_PARTY_NOTICES.md` 为准。
