param(
    [int[]]$PromptTargets = @(1024, 2048, 4096, 8192, 16384, 32768, 65536),
    [int]$OutputTokens = 512,
    [int]$Port = 18120,
    [string[]]$Modes = @('existing-no-mtp', 'new-no-mtp', 'new-mtp-q8-n3'),
    [string]$ExistingRuntime = 'C:\llama.cpp-hub\llamacpp\rocmfpx-qwen4exp-hip-gfx1151\llama-server.exe',
    [string]$NewRuntime = 'C:\Users\james\OneDrive\文档\llamacpp\build-kingjones30-qwen4exp-hip-mtp\bin\llama-server.exe',
    [string]$MainModel = 'C:\models\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX\Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf',
    [string]$DraftModel = 'C:\models\MTP\Qwen3.8-Flash-Next-MTP-Q8_0.gguf',
    [string]$WorkingSetHelper = 'C:\Users\james\OneDrive\文档\llamacpp\set-llama-working-set-8g.ps1',
    [string]$OutputDirectory = '',
    [switch]$ValidateOnly,
    [switch]$SkipRegressionReruns
)

$ErrorActionPreference = 'Stop'
$validModes = @('existing-no-mtp', 'new-no-mtp', 'new-mtp-q8-n3')
$sourceRoot = Split-Path -Parent $PSScriptRoot
$corpusGuide = Join-Path $PSScriptRoot 'data\qwen4exp-conversation-corpus-zh.txt'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputDirectory = "C:\Users\james\OneDrive\文档\llamacpp\test-logs\qwen4exp-hip-mtp-benchmark-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Save-Json([string]$Path, $Value, [int]$Depth = 12) {
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $Path -Encoding utf8
}

foreach ($mode in $Modes) {
    if ($mode -notin $validModes) { throw "Unknown benchmark mode: $mode" }
}
if ($PromptTargets.Count -eq 0 -or ($PromptTargets | Where-Object { $_ -lt 128 -or $_ -gt 65536 })) {
    throw 'Prompt targets must be between 128 and 65536 tokens.'
}
if ($OutputTokens -lt 1 -or $OutputTokens -gt 4096) { throw 'OutputTokens must be between 1 and 4096.' }

$requiredPaths = @($ExistingRuntime, $NewRuntime, $MainModel, $DraftModel, $WorkingSetHelper, $corpusGuide)
$corpusFiles = @(
    (Join-Path $sourceRoot 'README.md'),
    (Join-Path $sourceRoot 'tools\server\README.md'),
    (Join-Path $sourceRoot 'common\arg.cpp'),
    (Join-Path $sourceRoot 'tools\server\server.cpp'),
    (Join-Path $sourceRoot 'src\llama.cpp')
)
$requiredPaths += $corpusFiles
$missing = @($requiredPaths | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($missing.Count -gt 0) { throw "Required paths are missing:`n$($missing -join "`n")" }

$validation = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    required_paths_ok = $true
    prompt_targets = @($PromptTargets)
    output_tokens = $OutputTokens
    modes = @($Modes)
    port = $Port
    existing_runtime = $ExistingRuntime
    new_runtime = $NewRuntime
    main_model = $MainModel
    draft_model = $DraftModel
    working_set_helper = $WorkingSetHelper
    output_directory = $OutputDirectory
}
Save-Json (Join-Path $OutputDirectory 'validation.json') $validation
if ($ValidateOnly) {
    Write-Host "BENCHMARK_VALIDATION_OK output=$OutputDirectory"
    exit 0
}

function Get-ListenerEvidence([int]$LocalPort) {
    @(Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue | ForEach-Object {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" -ErrorAction SilentlyContinue
        [pscustomobject]@{
            address = $_.LocalAddress
            port = $_.LocalPort
            pid = $_.OwningProcess
            name = $owner.Name
            executable = $owner.ExecutablePath
            command_line = $owner.CommandLine
        }
    })
}

function Wait-Health([System.Diagnostics.Process]$Process, [int]$LocalPort, [int]$TimeoutSeconds = 1200) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) { throw "Server PID $($Process.Id) exited with code $($Process.ExitCode)." }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" -TimeoutSec 5
            if ($health.status -eq 'ok') { return }
        } catch { }
        Start-Sleep -Seconds 2
    }
    throw "Health timeout on port $LocalPort."
}

function Wait-PortReleased([int]$LocalPort, [int]$TimeoutSeconds = 60) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-ListenerEvidence $LocalPort).Count -eq 0) { return }
        Start-Sleep -Seconds 1
    }
    throw "Port $LocalPort was not released."
}

function Stop-OwnedServer([System.Diagnostics.Process]$Process, [string]$ExpectedExecutable, [int]$LocalPort) {
    if ($null -eq $Process -or $Process.HasExited) {
        Wait-PortReleased $LocalPort
        return
    }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$($Process.Id)" -ErrorAction SilentlyContinue
    if ($null -eq $cim -or $cim.Name -ne 'llama-server.exe' -or
        $cim.ExecutablePath -ne $ExpectedExecutable -or $cim.CommandLine -notmatch "--port\s+$LocalPort(\s|$)") {
        throw "Refusing to stop unexpected PID $($Process.Id)."
    }
    Stop-Process -Id $Process.Id -ErrorAction Stop
    try { $Process.WaitForExit(30000) } catch { }
    Wait-PortReleased $LocalPort
}

function Get-CounterValue([string]$Metrics, [string]$Name) {
    $match = [regex]::Match($Metrics, "(?m)^$([regex]::Escape($Name))\s+([0-9.eE+-]+)\s*$")
    if (-not $match.Success) { return 0.0 }
    [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Get-SpecCounters([int]$LocalPort) {
    $metrics = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$LocalPort/metrics" -TimeoutSec 15).Content
    [pscustomobject]@{
        draft = Get-CounterValue $metrics 'llamacpp:spec_decode_num_draft_tokens_total'
        accepted = Get-CounterValue $metrics 'llamacpp:spec_decode_num_accepted_tokens_total'
        steps = Get-CounterValue $metrics 'llamacpp:spec_decode_num_drafts_total'
    }
}

function Get-TokenCount([int]$LocalPort, [string]$Content) {
    $body = @{ content = $Content; add_special = $false; parse_special = $true } | ConvertTo-Json -Compress
    $reply = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$LocalPort/tokenize" -ContentType 'application/json' -Body $body -TimeoutSec 180
    @($reply.tokens).Count
}

function New-ConversationPrompt([string]$Corpus, [int]$CharacterCount) {
    $material = $Corpus.Substring(0, $CharacterCount)
    $turnCount = 6
    $chunkSize = [math]::Ceiling($material.Length / $turnCount)
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append("<|im_start|>system`n你是一名资深系统工程师。请基于完整多轮上下文进行严格、可验证的技术分析，并用中文回答。<|im_end|>`n")
    for ($turn = 0; $turn -lt $turnCount; $turn++) {
        $offset = $turn * $chunkSize
        if ($offset -ge $material.Length) { break }
        $length = [math]::Min($chunkSize, $material.Length - $offset)
        $chunk = $material.Substring($offset, $length)
        [void]$builder.Append("<|im_start|>user`n这是工程快照第 $($turn + 1) 部分。请先记录其中的具体实现和约束，稍后统一审查：`n$chunk<|im_end|>`n")
        [void]$builder.Append("<|im_start|>assistant`n已记录第 $($turn + 1) 部分。我会保留其中的文件、符号、参数和平台约束，等待后续材料后统一分析。<|im_end|>`n")
    }
    [void]$builder.Append("<|im_start|>user`n现在请把全部材料视为同一个持续演进的本地推理服务。识别架构风险、性能瓶颈、不安全假设以及文档与实现的不一致；引用具体标识符，并给出按优先级排序的验证方案。写成详细中文技术报告，不要只做摘要。<|im_end|>`n<|im_start|>assistant`n")
    $builder.ToString()
}

function New-CalibratedPrompt([int]$LocalPort, [string]$Corpus, [int]$TargetTokens) {
    $low = 1
    $high = $Corpus.Length
    $best = $null
    $bestTokens = 0
    $bestDistance = [int]::MaxValue
    while ($low -le $high) {
        $mid = [int][math]::Floor(($low + $high) / 2)
        $candidate = New-ConversationPrompt $Corpus $mid
        $tokens = Get-TokenCount $LocalPort $candidate
        $distance = [math]::Abs($tokens - $TargetTokens)
        if ($distance -lt $bestDistance) {
            $best = $candidate
            $bestTokens = $tokens
            $bestDistance = $distance
        }
        if ($tokens -lt $TargetTokens) { $low = $mid + 1 }
        elseif ($tokens -gt $TargetTokens) { $high = $mid - 1 }
        else { break }
    }
    if ($bestDistance -gt [math]::Ceiling($TargetTokens * 0.01)) {
        throw "Unable to calibrate target $TargetTokens within 1%; best=$bestTokens."
    }
    [pscustomobject]@{ prompt = $best; tokens = $bestTokens; target = $TargetTokens }
}

function Invoke-StreamingCompletion([int]$LocalPort, [string]$Prompt, [int]$MaxTokens) {
    $payload = [ordered]@{
        prompt = $Prompt
        n_predict = $MaxTokens
        temperature = 0.0
        top_p = 1.0
        top_k = 0
        min_p = 0.0
        repeat_penalty = 1.0
        seed = 42
        cache_prompt = $false
        stream = $true
        ignore_eos = $true
        timings_per_token = $true
    }
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(3600)
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "http://127.0.0.1:$LocalPort/completion")
    $request.Content = [Net.Http.StringContent]::new(($payload | ConvertTo-Json -Depth 5 -Compress), [Text.Encoding]::UTF8, 'application/json')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $reply = $client.Send($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
    [void]$reply.EnsureSuccessStatusCode()
    $reader = [IO.StreamReader]::new($reply.Content.ReadAsStream())
    $content = [Text.StringBuilder]::new()
    $firstTokenMs = $null
    $last = $null
    try {
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if (-not $line.StartsWith('data:')) { continue }
            $json = $line.Substring(5).Trim()
            if ([string]::IsNullOrWhiteSpace($json) -or $json -eq '[DONE]') { continue }
            $chunk = $json | ConvertFrom-Json
            if ($null -ne $chunk.content -and [string]$chunk.content -ne '') {
                if ($null -eq $firstTokenMs) { $firstTokenMs = $watch.Elapsed.TotalMilliseconds }
                [void]$content.Append([string]$chunk.content)
            }
            if ($null -ne $chunk.timings) { $last = $chunk }
        }
    } finally {
        $watch.Stop()
        $reader.Dispose()
        $reply.Dispose()
        $request.Dispose()
        $client.Dispose()
    }
    if ($null -eq $last -or $null -eq $last.timings) { throw 'Streaming completion ended without final timings.' }
    [pscustomobject]@{
        content = $content.ToString()
        timings = $last.timings
        ttft_ms = $firstTokenMs
        wall_ms = $watch.Elapsed.TotalMilliseconds
        stopped_eos = $last.stopped_eos
        stopped_limit = $last.stopped_limit
        stopped_word = $last.stopped_word
        stop = $last.stop
    }
}

$corpusBuilder = [Text.StringBuilder]::new()
[void]$corpusBuilder.AppendLine((Get-Content -LiteralPath $corpusGuide -Raw))
foreach ($file in $corpusFiles) {
    [void]$corpusBuilder.AppendLine("`n===== FILE: $file =====`n")
    [void]$corpusBuilder.AppendLine((Get-Content -LiteralPath $file -Raw))
}
$corpus = $corpusBuilder.ToString()

$preflight = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    listeners = @(Get-ListenerEvidence $Port)
    llama_processes = @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Select-Object ProcessId, ExecutablePath, CommandLine)
    available_memory_mib = [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue, 0)
}
Save-Json (Join-Path $OutputDirectory 'preflight.json') $preflight
if ($preflight.listeners.Count -ne 0) { throw "Benchmark port $Port is already in use." }

$promptDir = Join-Path $OutputDirectory 'prompts'
New-Item -ItemType Directory -Force -Path $promptDir | Out-Null
$prompts = @{}
$rows = [System.Collections.Generic.List[object]]::new()
$resultsJson = Join-Path $OutputDirectory 'results.json'
$resultsCsv = Join-Path $OutputDirectory 'results.csv'

function Save-Results {
    Save-Json $resultsJson ([ordered]@{ generated_at=(Get-Date).ToString('o'); rows=@($rows) })
    @($rows) | Export-Csv -LiteralPath $resultsCsv -NoTypeInformation -Encoding utf8
}

function Invoke-BenchmarkMode([string]$Mode, [string]$Trial, [int[]]$Targets) {
    $runtime = if ($Mode -eq 'existing-no-mtp') { $ExistingRuntime } else { $NewRuntime }
    $tag = if ($Trial -eq 'primary') { $Mode } else { "$Mode-$Trial" }
    $modeDir = Join-Path $OutputDirectory $tag
    New-Item -ItemType Directory -Force -Path $modeDir | Out-Null
    $args = @(
        '--model', $MainModel,
        '--host', '127.0.0.1', '--port', "$Port",
        '--device', 'rocm0', '--ctx-size', '131072', '--flash-attn', 'on',
        '--batch-size', '2048', '--ubatch-size', '512', '--parallel', '1',
        '--fit', 'off', '--no-webui', '--metrics', '--props', '--timeout', '36000',
        '--alias', $tag
    )
    if ($Mode -eq 'new-mtp-q8-n3') {
        $args += @('--spec-type', 'draft-mtp', '--spec-draft-model', $DraftModel, '--spec-draft-n-max', '3')
    }
    Save-Json (Join-Path $modeDir 'launch.json') ([ordered]@{ executable=$runtime; arguments=$args; timestamp=(Get-Date).ToString('o') })
    $process = $null
    try {
        if (@(Get-ListenerEvidence $Port).Count -ne 0) { throw "Port $Port is not free before $tag." }
        $process = Start-Process -FilePath $runtime -ArgumentList $args -WorkingDirectory (Split-Path -Parent $runtime) -RedirectStandardOutput (Join-Path $modeDir 'stdout.log') -RedirectStandardError (Join-Path $modeDir 'stderr.log') -WindowStyle Hidden -PassThru
        Wait-Health $process $Port
        $workingSetText = (& $WorkingSetHelper -Port $Port -WaitSeconds 60 -MaxGiB 8 2>&1 | Out-String)
        $workingSetText | Set-Content -LiteralPath (Join-Path $modeDir 'working-set.txt') -Encoding utf8
        if ($workingSetText -notmatch 'hard_max_enabled\s+:\s+True') { throw "8 GiB hard working-set verification failed for $tag." }
        $process.Refresh()
        Save-Json (Join-Path $modeDir 'process.json') ([ordered]@{
            pid=$process.Id; executable=$runtime; working_set_bytes=$process.WorkingSet64; private_bytes=$process.PrivateMemorySize64
        })

        if ($prompts.Count -eq 0) {
            foreach ($target in $PromptTargets) {
                $calibrated = New-CalibratedPrompt $Port $corpus $target
                $promptPath = Join-Path $promptDir "$target.txt"
                $calibrated.prompt | Set-Content -LiteralPath $promptPath -Encoding utf8NoBOM
                $hash = (Get-FileHash -LiteralPath $promptPath -Algorithm SHA256).Hash
                $prompts[$target] = [pscustomobject]@{ prompt=$calibrated.prompt; tokens=$calibrated.tokens; path=$promptPath; sha256=$hash }
                Write-Host "CALIBRATED target=$target actual=$($calibrated.tokens) hash=$hash"
            }
            Save-Json (Join-Path $promptDir 'manifest.json') @($PromptTargets | ForEach-Object {
                [ordered]@{ target=$_; tokens=$prompts[$_].tokens; path=$prompts[$_].path; sha256=$prompts[$_].sha256 }
            })
        }

        $warmPrompt = New-ConversationPrompt $corpus ([math]::Min(6000, $corpus.Length))
        $null = Invoke-StreamingCompletion $Port $warmPrompt ([math]::Min(32, $OutputTokens))
        Write-Host "WARM mode=$tag complete"

        foreach ($target in $Targets) {
            $promptInfo = $prompts[$target]
            if ($null -eq $promptInfo) { throw "No saved prompt for target $target." }
            $before = Get-SpecCounters $Port
            $started = (Get-Date).ToString('o')
            Write-Host "RUN mode=$tag target=$target prompt=$($promptInfo.tokens) output=$OutputTokens"
            $response = Invoke-StreamingCompletion $Port $promptInfo.prompt $OutputTokens
            $after = Get-SpecCounters $Port
            $draftDelta = [int]([double]$after.draft - [double]$before.draft)
            $acceptedDelta = [int]([double]$after.accepted - [double]$before.accepted)
            $draft = if ($null -ne $response.timings.draft_n) { [int]$response.timings.draft_n } else { $draftDelta }
            $accepted = if ($null -ne $response.timings.draft_n_accepted) { [int]$response.timings.draft_n_accepted } else { $acceptedDelta }
            $steps = [int]([double]$after.steps - [double]$before.steps)
            $textHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($response.content)))
            $finishReason = if ($response.stopped_limit -or [int]$response.timings.predicted_n -ge $OutputTokens) { 'length' } elseif ($response.stopped_eos) { 'eos' } elseif ($response.stopped_word) { 'stop' } else { 'unknown' }
            $row = [pscustomobject][ordered]@{
                timestamp = $started
                mode = $Mode
                trial = $Trial
                prompt_target = $target
                prompt_tokens = [int]$response.timings.prompt_n
                prompt_ms = [math]::Round([double]$response.timings.prompt_ms, 3)
                prompt_tps = [math]::Round([double]$response.timings.prompt_per_second, 3)
                output_requested = $OutputTokens
                output_tokens = [int]$response.timings.predicted_n
                output_ms = [math]::Round([double]$response.timings.predicted_ms, 3)
                output_tps = [math]::Round([double]$response.timings.predicted_per_second, 3)
                ttft_ms = [math]::Round([double]$response.ttft_ms, 3)
                wall_ms = [math]::Round([double]$response.wall_ms, 3)
                draft_tokens = $draft
                accepted_tokens = $accepted
                acceptance = if ($draft -gt 0) { [math]::Round($accepted / $draft, 6) } else { 0.0 }
                draft_steps = $steps
                finish_reason = $finishReason
                prompt_sha256 = $promptInfo.sha256
                response_sha256 = $textHash
                status = 'ok'
            }
            $rows.Add($row)
            Save-Json (Join-Path $modeDir "$target-response.json") ([ordered]@{ row=$row; content=$response.content; timings=$response.timings })
            Save-Results
            Write-Host ("RESULT mode={0} target={1} pp={2} tg={3} accept={4:P2}" -f $tag,$target,$row.prompt_tps,$row.output_tps,$row.acceptance)
        }
    } finally {
        if ($null -ne $process) { Stop-OwnedServer $process $runtime $Port }
        Save-Json (Join-Path $modeDir 'listener-after-stop.json') @(Get-ListenerEvidence $Port)
    }
}

foreach ($mode in $Modes) {
    Invoke-BenchmarkMode $mode 'primary' $PromptTargets
}

function Get-AdjudicatedRow([string]$Mode, [int]$Target) {
    $rerun = @($rows | Where-Object { $_.mode -eq $Mode -and $_.prompt_target -eq $Target -and $_.trial -eq 'rerun' }) | Select-Object -Last 1
    if ($null -ne $rerun) { return $rerun }
    @($rows | Where-Object { $_.mode -eq $Mode -and $_.prompt_target -eq $Target -and $_.trial -eq 'primary' }) | Select-Object -Last 1
}

$flagged = @()
if ('existing-no-mtp' -in $Modes -and 'new-no-mtp' -in $Modes) {
    foreach ($target in $PromptTargets) {
        $old = Get-AdjudicatedRow 'existing-no-mtp' $target
        $new = Get-AdjudicatedRow 'new-no-mtp' $target
        if ($null -ne $old -and $null -ne $new -and (([double]$new.output_tps / [double]$old.output_tps) - 1.0) * 100.0 -lt -3.0) {
            $flagged += $target
        }
    }
}
if ($flagged.Count -gt 0 -and -not $SkipRegressionReruns) {
    Write-Host "RERUN_REGRESSIONS targets=$($flagged -join ',')"
    Invoke-BenchmarkMode 'existing-no-mtp' 'rerun' $flagged
    Invoke-BenchmarkMode 'new-no-mtp' 'rerun' $flagged
}

$comparisons = @()
foreach ($target in $PromptTargets) {
    $old = Get-AdjudicatedRow 'existing-no-mtp' $target
    $new = Get-AdjudicatedRow 'new-no-mtp' $target
    $mtp = Get-AdjudicatedRow 'new-mtp-q8-n3' $target
    $comparisons += [pscustomobject][ordered]@{
        prompt_target = $target
        existing_tg = $old.output_tps
        new_tg = $new.output_tps
        mtp_tg = $mtp.output_tps
        existing_pp = $old.prompt_tps
        new_pp = $new.prompt_tps
        mtp_pp = $mtp.prompt_tps
        new_vs_existing_tg_percent = if ($old) { [math]::Round((([double]$new.output_tps/[double]$old.output_tps)-1)*100, 3) } else { $null }
        mtp_vs_new_tg_percent = if ($new -and $mtp) { [math]::Round((([double]$mtp.output_tps/[double]$new.output_tps)-1)*100, 3) } else { $null }
        mtp_acceptance = $mtp.acceptance
        actual_prompt_tokens = $new.prompt_tokens
        actual_output_tokens = $new.output_tokens
        finish_reason = $new.finish_reason
    }
}

function Get-WeightedRate([object[]]$SelectedRows, [string]$TokenField, [string]$MillisecondField) {
    if ($SelectedRows.Count -eq 0) { return $null }
    $tokens = ($SelectedRows | Measure-Object -Property $TokenField -Sum).Sum
    $milliseconds = ($SelectedRows | Measure-Object -Property $MillisecondField -Sum).Sum
    if ($milliseconds -le 0) { return $null }
    [math]::Round(1000.0 * $tokens / $milliseconds, 3)
}

$adjudicatedByMode = @{}
foreach ($mode in $validModes) {
    $adjudicatedByMode[$mode] = @($PromptTargets | ForEach-Object { Get-AdjudicatedRow $mode $_ } | Where-Object { $null -ne $_ })
}
$aggregate = [ordered]@{}
foreach ($mode in $validModes) {
    $selected = $adjudicatedByMode[$mode]
    $aggregate[$mode] = [ordered]@{
        weighted_tg = Get-WeightedRate $selected 'output_tokens' 'output_ms'
        weighted_pp = Get-WeightedRate $selected 'prompt_tokens' 'prompt_ms'
        rows = $selected.Count
        draft_tokens = ($selected | Measure-Object -Property draft_tokens -Sum).Sum
        accepted_tokens = ($selected | Measure-Object -Property accepted_tokens -Sum).Sum
    }
}
$oldWeighted = $aggregate['existing-no-mtp'].weighted_tg
$newWeighted = $aggregate['new-no-mtp'].weighted_tg
$mtpWeighted = $aggregate['new-mtp-q8-n3'].weighted_tg
$verdict = [ordered]@{
    new_no_mtp_regression_percent = if ($oldWeighted -and $newWeighted) { [math]::Round((($newWeighted/$oldWeighted)-1)*100, 3) } else { $null }
    mtp_speedup_percent = if ($newWeighted -and $mtpWeighted) { [math]::Round((($mtpWeighted/$newWeighted)-1)*100, 3) } else { $null }
    unresolved_regression = if ($oldWeighted -and $newWeighted) { $newWeighted -lt $oldWeighted } else { $null }
}
Save-Json (Join-Path $OutputDirectory 'summary.json') ([ordered]@{
    generated_at=(Get-Date).ToString('o'); comparisons=$comparisons; aggregate=$aggregate; verdict=$verdict
})
$comparisons | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'comparison.csv') -NoTypeInformation -Encoding utf8
Write-Host "BENCHMARK_COMPLETE output=$OutputDirectory"
