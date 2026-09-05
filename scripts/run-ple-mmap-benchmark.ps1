param(
    [int] $Port = 8232,
    [int[]] $Targets = @(1024, 2048, 4096, 8192, 16384, 32768, 65536),
    [int] $OutputTokens = 128,
    [ValidateRange(1, 20)] [int] $Repetitions = 1,
    [string] $Label = 'mmap-lazy',
    [string] $OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'

function Get-TokenCount([string] $Text) {
    $body = @{ content = $Text; add_special = $false; parse_special = $true } | ConvertTo-Json -Compress
    $bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post `
        -ContentType 'application/json; charset=utf-8' -Body $bodyBytes -TimeoutSec 120
    return @($result.tokens).Count
}

function New-ChatPrompt([string] $Corpus, [int] $Characters) {
    $system = @'
<|im_start|>system
You are a senior systems engineer reviewing a mixed documentation and C++ source snapshot. Produce a concrete, evidence-based technical assessment.<|im_end|>
<|im_start|>user
'@
    $task = @'

Analyze the material above as one evolving local inference-server project. Identify architectural risks, performance bottlenecks, unsafe assumptions, and inconsistencies between documentation and implementation. Then propose a prioritized validation plan in Chinese. Cite concrete identifiers from the supplied material.<|im_end|>
<|im_start|>assistant
<think>
'@
    return $system + $Corpus.Substring(0, $Characters) + $task
}

function New-CalibratedPrompt([string] $Corpus, [int] $Target) {
    $low = 1
    $high = $Corpus.Length
    $best = $null
    $bestTokens = 0
    $bestDistance = [int]::MaxValue
    while ($low -le $high) {
        $mid = [int](($low + $high) / 2)
        $candidate = New-ChatPrompt $Corpus $mid
        $tokens = Get-TokenCount $candidate
        $distance = [math]::Abs($tokens - $Target)
        if ($distance -lt $bestDistance) {
            $best = $candidate
            $bestTokens = $tokens
            $bestDistance = $distance
        }
        if ($tokens -lt $Target) { $low = $mid + 1 }
        elseif ($tokens -gt $Target) { $high = $mid - 1 }
        else { break }
    }
    return [pscustomobject]@{ Prompt = $best; Tokens = $bestTokens }
}

function Get-Metric([string] $Metrics, [string] $Name) {
    $match = [regex]::Match($Metrics, "(?m)^$([regex]::Escape($Name))\s+([0-9.eE+-]+)\s*$")
    if (-not $match.Success) { return 0.0 }
    return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Get-SpecMetrics {
    $metrics = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/metrics" -TimeoutSec 15).Content
    return [pscustomobject]@{
        Draft = Get-Metric $metrics 'llamacpp:spec_decode_num_draft_tokens_total'
        Accepted = Get-Metric $metrics 'llamacpp:spec_decode_num_accepted_tokens_total'
    }
}

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $root 'benchmark-results'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$runTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$safeLabel = $Label -replace '[^A-Za-z0-9_.-]', '_'
$resultBase = Join-Path $OutputDirectory "ple-$safeLabel-$runTimestamp"
$corpusFiles = @(
    (Join-Path $root 'README.md'),
    (Join-Path $root 'tools\server\README.md'),
    (Join-Path $root 'common\arg.cpp'),
    (Join-Path $root 'tools\server\server.cpp'),
    (Join-Path $root 'src\llama.cpp'),
    (Join-Path $root 'src\llama-model.cpp')
)
$builder = [Text.StringBuilder]::new()
foreach ($file in $corpusFiles) {
    [void] $builder.AppendLine("`n===== FILE: $file =====`n")
    [void] $builder.AppendLine((Get-Content -Raw -Encoding UTF8 -LiteralPath $file))
}
$corpus = $builder.ToString()

$rows = [Collections.Generic.List[object]]::new()
foreach ($target in $Targets) {
    $calibrated = New-CalibratedPrompt $corpus $target
    for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
        Write-Output "START label=$Label target=$target actual=$($calibrated.Tokens) repetition=$repetition"
        $before = Get-SpecMetrics
        $body = [ordered]@{
            prompt = $calibrated.Prompt
            n_predict = $OutputTokens
            temperature = 1.0
            top_p = 0.95
            top_k = 20
            min_p = 0.0
            repeat_penalty = 1.0
            seed = 42
            cache_prompt = $false
            stream = $false
            ignore_eos = $true
        } | ConvertTo-Json -Depth 6 -Compress
        $bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $bodyBytes -TimeoutSec 1800
        $after = Get-SpecMetrics
        $process = Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort $Port).OwningProcess
        $os = Get-CimInstance Win32_OperatingSystem
        $drafted = if ($response.timings.PSObject.Properties.Name -contains 'draft_n') {
            [double] $response.timings.draft_n
        } else {
            $after.Draft - $before.Draft
        }
        $accepted = if ($response.timings.PSObject.Properties.Name -contains 'draft_n_accepted') {
            [double] $response.timings.draft_n_accepted
        } else {
            $after.Accepted - $before.Accepted
        }
        $row = [pscustomobject][ordered]@{
            label = $Label
            repetition = $repetition
            target = $target
            prompt_tokens = [int] $response.timings.prompt_n
            pp_tps = [math]::Round([double] $response.timings.prompt_per_second, 3)
            output_tokens = [int] $response.timings.predicted_n
            tg_tps = [math]::Round([double] $response.timings.predicted_per_second, 3)
            drafted = [int] $drafted
            accepted = [int] $accepted
            acceptance = if ($drafted -gt 0) { [math]::Round($accepted / $drafted, 4) } else { 0.0 }
            working_set_gib = [math]::Round($process.WorkingSet64 / 1GB, 3)
            private_gib = [math]::Round($process.PrivateMemorySize64 / 1GB, 3)
            free_physical_gib = [math]::Round($os.FreePhysicalMemory / 1MB, 3)
        }
        $rows.Add($row)
        Write-Output ("RESULT " + ($row | ConvertTo-Json -Compress))
    }
}

$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$resultBase.json" -Encoding UTF8
$rows | Export-Csv -LiteralPath "$resultBase.csv" -NoTypeInformation -Encoding UTF8
Write-Output "SAVED json=$resultBase.json csv=$resultBase.csv"
$rows | ConvertTo-Json -Depth 4
