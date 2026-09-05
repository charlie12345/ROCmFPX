param(
    [string] $Runtime = "C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-ple-mmap-gfx1151",
    [string] $Model = "C:\models\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf",
    [string] $DraftModel = "C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf",
    [int] $Port = 8230,
    [int] $ContextSize = 8192
)

$ErrorActionPreference = "Stop"
$server = Join-Path $Runtime "llama-server.exe"
$log = Join-Path $env:TEMP "llama-ple-mmap-acceptance-$Port.log"
$err = Join-Path $env:TEMP "llama-ple-mmap-acceptance-$Port.err.log"
$process = $null

try {
    $arguments = @(
        "--model", $Model,
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--mmap",
        "--tensor-read-lazy", "on",
        "--ctx-size", "$ContextSize",
        "--flash-attn", "on",
        "--fit", "off",
        "--gpu-layers", "999",
        "--device", "rocm0",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3",
        "--spec-draft-model", $DraftModel,
        "--ctx-checkpoints", "32",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--parallel", "1",
        "--no-mmproj",
        "--no-webui",
        "--metrics"
    )

    $process = Start-Process -FilePath $server -ArgumentList $arguments `
        -WorkingDirectory $Runtime -RedirectStandardOutput $log `
        -RedirectStandardError $err -WindowStyle Hidden -PassThru

    $baseUrl = "http://127.0.0.1:$Port"
    $ready = $false
    for ($attempt = 0; $attempt -lt 240; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($process.HasExited) {
            throw "llama-server exited during model loading (code $($process.ExitCode)); see $err"
        }
        try {
            $health = Invoke-RestMethod "$baseUrl/health" -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
            # Model loading is still in progress.
        }
    }
    if (-not $ready) {
        throw "llama-server did not become healthy; see $err"
    }

    $combinedLog = (Get-Content -Path @($log, $err) -Raw) -join "`n"
    if ($combinedLog -notmatch "PLE mmap pager active") {
        throw "PLE mmap pager was not activated"
    }

    $slots = Invoke-RestMethod "$baseUrl/slots" -TimeoutSec 10
    if (-not ($slots | Where-Object { $_.speculative -eq $true })) {
        throw "speculative decoding is not active"
    }

    $body = @{
        prompt = "请用中文说明为什么真实服务验收必须同时检查输出、进程存活和推测解码接受率。"
        n_predict = 128
        temperature = 0
        stream = $false
    } | ConvertTo-Json
    $response = Invoke-RestMethod "$baseUrl/completion" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 300
    if ([string]::IsNullOrWhiteSpace($response.content)) {
        throw "completion returned empty content"
    }

    $drafted = [int64] $response.timings.draft_n
    $accepted = [int64] $response.timings.draft_n_accepted
    if ($drafted -le 0 -or $accepted -le 0) {
        throw "MTP produced no accepted draft tokens (drafted=$drafted, accepted=$accepted)"
    }
    if ($process.HasExited) {
        throw "llama-server exited after completion"
    }

    $liveProcess = Get-Process -Id $process.Id
    [pscustomobject]@{
        Status = "passed"
        ProcessId = $process.Id
        PromptTokensPerSecond = $response.timings.prompt_per_second
        PredictedTokensPerSecond = $response.timings.predicted_per_second
        DraftedTokens = $drafted
        AcceptedTokens = $accepted
        AcceptancePercent = [math]::Round(100 * $accepted / $drafted, 2)
        WorkingSetGiB = [math]::Round($liveProcess.WorkingSet64 / 1GB, 2)
        Log = $log
        ErrorLog = $err
    } | Format-List
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
}
