[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Binary,
    [Parameter(Mandatory = $true)] [string] $Model,
    [int] $Port = 18219,
    [int] $Context = 4096,
    [int] $MaxTokens = 32,
    [double] $MaxWorkingSetGiB = 6
)

$ErrorActionPreference = 'Stop'
$logRoot = Join-Path $env:TEMP ("qwen4exp-ple-direct-smoke-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $logRoot | Out-Null
$stdoutLog = Join-Path $logRoot 'stdout.log'
$stderrLog = Join-Path $logRoot 'stderr.log'
$server = $null

try {
    $arguments = @(
        '-m', $Model,
        '--host', '127.0.0.1',
        '--port', $Port,
        '--ctx-size', $Context,
        '--batch-size', '512',
        '--ubatch-size', '128',
        '--parallel', '1',
        '--no-webui',
        '--no-mmproj',
        '-sm', 'none',
        '--device', 'rocm0',
        '--ple-ssd', 'direct',
        '--ple-io-depth', '32',
        '--ple-buffer-mib', '32',
        '--metrics'
    )
    $server = Start-Process -FilePath $Binary -ArgumentList $arguments -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    $health = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($server.HasExited) {
            throw "llama-server exited during startup with code $($server.ExitCode)"
        }
        try {
            $healthResponse = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($healthResponse.status -eq 'ok') {
                $health = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 1000
        }
    }
    if (-not $health) {
        throw "server did not become healthy"
    }

    $workingSetGiB = (Get-Process -Id $server.Id).WorkingSet64 / 1GB
    if ($workingSetGiB -gt $MaxWorkingSetGiB) {
        throw "working set after direct-PLE load is $([math]::Round($workingSetGiB, 2)) GiB, above $MaxWorkingSetGiB GiB"
    }

    $request = @{
        messages = @(@{ role = 'user'; content = '请用一句话说明 SSD 分页读取的目的。' })
        max_tokens = $MaxTokens
        temperature = 0
        seed = 42
        stream = $false
    } | ConvertTo-Json -Depth 8
    $response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/v1/chat/completions" -ContentType 'application/json' -Body $request
    $content = [string] $response.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($content)) {
        $content = [string] $response.choices[0].message.reasoning_content
    }
    if ([string]::IsNullOrWhiteSpace($content)) {
        throw 'completion returned empty content'
    }

    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        $server.WaitForExit()
    }
    $log = ((Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue) + "`n" +
            (Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue))
    if ($log -notmatch 'PLE direct pager active') {
        throw 'missing PLE direct pager active log line'
    }
    if ($log -notmatch 'PLE direct pager stats:.*failures=0') {
        throw 'missing successful PLE direct pager stats log line'
    }
    [pscustomobject]@{
        Port = $Port
        Content = $content
        WorkingSetGiB = [math]::Round($workingSetGiB, 3)
        LogDirectory = $logRoot
        ServerPid = $server.Id
    } | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
}
