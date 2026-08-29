param(
    [string]$Runtime = 'C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-mtp-gfx1151\llama-server.exe',
    [int]$Port = 18131,
    [string]$OutputDirectory = 'C:\Users\james\OneDrive\文档\llamacpp\test-logs\qwen4exp-mtp-accept-regression'
)

$ErrorActionPreference = 'Stop'
$mainModel = 'C:\models\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf'
$draftModel = 'C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf'
$mmproj = 'C:\models\mmproj\mmproj-Qwen3.8-Flash-Next-BF16.gguf'
$template = 'C:\llama.cpp-hub\cache\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.jinja'
$workingSetHelper = 'C:\Users\james\OneDrive\文档\llamacpp\set-llama-working-set-8g.ps1'

foreach ($path in @($Runtime, $mainModel, $draftModel, $mmproj, $template, $workingSetHelper)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Test port $Port is already in use."
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$stderr = Join-Path $OutputDirectory 'stderr.log'
$stdout = Join-Path $OutputDirectory 'stdout.log'
$process = $null

function Wait-Health([System.Diagnostics.Process]$Process, [int]$TimeoutSeconds = 1200) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) { throw "Server exited before health, code=$($Process.ExitCode)." }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
            if ($health.status -eq 'ok') { return }
        } catch { }
        Start-Sleep -Seconds 2
    }
    throw 'Health timeout.'
}

try {
    $args = @(
        '--model', $mainModel, '--host', '127.0.0.1', '--port', "$Port",
        '--device', 'rocm0', '--ctx-size', '262144', '--flash-attn', 'on',
        '--spec-type', 'draft-mtp', '--spec-draft-model', $draftModel, '--spec-draft-n-max', '3',
        '--fit', 'on', '--ctx-checkpoints', '32', '--batch-size', '2048', '--ubatch-size', '512',
        '--parallel', '1', '--no-webui', '--mmproj', $mmproj,
        '--reasoning-budget', '512', '--chat-template-file', $template,
        '--metrics', '--alias', 'Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX', '--timeout', '36000'
    )
    $process = Start-Process -FilePath $Runtime -ArgumentList $args -WorkingDirectory (Split-Path -Parent $Runtime) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Wait-Health $process
    & $workingSetHelper -Port $Port -WaitSeconds 60 -MaxGiB 8 | Out-File -LiteralPath (Join-Path $OutputDirectory 'working-set.txt') -Encoding utf8

    $requests = @(
        '请给这段新对话生成一个不超过十个字的中文标题：一堆话',
        '用户正在测试本地大模型。请回复：MTP第一次请求正常。',
        '继续测试同一服务，请用一句话说明为什么真实日志比猜测更可靠。',
        '这是第四次连续请求，请确认推测解码服务仍然正常。',
        '这是第五次连续请求，请只回答：服务仍然存活。'
    )
    $responses = @()
    for ($i = 0; $i -lt $requests.Count; $i++) {
        $body = [ordered]@{
            model = 'Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX'
            messages = @(@{ role = 'user'; content = $requests[$i] })
            max_tokens = 128
            temperature = 0.7
            top_p = 0.8
            top_k = 20
            stream = $false
            cache_prompt = $false
        }
        $reply = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/v1/chat/completions" -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 8 -Compress) -TimeoutSec 600
        if ($process.HasExited) { throw "Server crashed after request $($i + 1), code=$($process.ExitCode)." }
        $message = $reply.choices[0].message
        if ([string]::IsNullOrWhiteSpace([string]$message.content) -and
            [string]::IsNullOrWhiteSpace([string]$message.reasoning_content)) {
            throw "Request $($i + 1) returned neither content nor reasoning content."
        }
        $responses += $reply
    }
    $healthAfter = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
    if ($healthAfter.status -ne 'ok' -or $process.HasExited) { throw 'Server was not healthy after three MTP requests.' }
    $responses | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'responses.json') -Encoding utf8
    Write-Host 'QWEN4EXP_MTP_ACCEPT_REGRESSION_PASSED'
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)" -ErrorAction SilentlyContinue
        if ($cim -and $cim.ExecutablePath -eq $Runtime -and $cim.CommandLine -match "--port\s+$Port(\s|$)") {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
            try { $process.WaitForExit(30000) } catch { }
        }
    }
}
