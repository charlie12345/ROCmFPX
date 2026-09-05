$ErrorActionPreference = 'Stop'

$scriptPath = Join-Path $PSScriptRoot '..\benchmark-qwen4exp-hip-mtp.ps1'
$testOutput = Join-Path $env:TEMP 'qwen4exp-hip-mtp-benchmark-validation-test'

if (Test-Path -LiteralPath $testOutput) {
    Remove-Item -LiteralPath $testOutput -Recurse -Force
}

& $scriptPath -ValidateOnly -OutputDirectory $testOutput

$validation = Get-Content -LiteralPath (Join-Path $testOutput 'validation.json') -Raw | ConvertFrom-Json
$expectedTargets = @(1024, 2048, 4096, 8192, 16384, 32768, 65536)
$expectedModes = @('existing-no-mtp', 'new-no-mtp', 'new-mtp-q8-n3')

if (@($validation.prompt_targets).Count -ne $expectedTargets.Count -or
    (Compare-Object @($validation.prompt_targets) $expectedTargets)) {
    throw 'Validation did not report the required seven prompt targets.'
}
if (@($validation.modes).Count -ne $expectedModes.Count -or
    (Compare-Object @($validation.modes) $expectedModes)) {
    throw 'Validation did not report the required three benchmark modes.'
}
if (-not $validation.required_paths_ok) {
    throw 'Validation did not verify all required runtime, model, corpus, and helper paths.'
}

Write-Host 'BENCHMARK_VALIDATION_TEST_PASSED'
