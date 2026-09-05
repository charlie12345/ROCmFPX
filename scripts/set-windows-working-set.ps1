param(
    [Parameter(Mandatory = $true)] [int] $Port,
    [ValidateRange(1, 64)] [double] $MaxGiB = 8,
    [ValidateRange(1, 1800)] [int] $WaitSeconds = 300
)

$ErrorActionPreference = 'Stop'

if (-not ('WinWorkingSetControl' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class WinWorkingSetControl {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetProcessWorkingSetSizeEx(
        IntPtr hProcess, UIntPtr minSize, UIntPtr maxSize, uint flags);

    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetProcessWorkingSetSizeEx(
        IntPtr hProcess, out UIntPtr minSize, out UIntPtr maxSize, out uint flags);

    [DllImport("psapi.dll", SetLastError=true)]
    public static extern bool EmptyWorkingSet(IntPtr hProcess);
}
'@
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$connection = $null
$health = $null

Write-Host "Waiting for a healthy llama-server on port $Port..."
while ((Get-Date) -lt $deadline) {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($connection) {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
        } catch {
            $health = $null
        }
        if ($health.status -eq 'ok') { break }
    }
    Start-Sleep -Seconds 2
}

if (-not $connection -or $health.status -ne 'ok') {
    throw "llama-server on port $Port did not become healthy within $WaitSeconds seconds."
}

$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)"
if ($processInfo.Name -ne 'llama-server.exe') {
    throw "Port $Port belongs to $($processInfo.Name), not llama-server.exe."
}

$process = Get-Process -Id $connection.OwningProcess
$maxBytes = [uint64]($MaxGiB * 1GB)
# QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x4
$ok = [WinWorkingSetControl]::SetProcessWorkingSetSizeEx(
    $process.Handle, [UIntPtr]::new(64MB), [UIntPtr]::new($maxBytes), 0x4)
if (-not $ok) {
    $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    throw "SetProcessWorkingSetSizeEx failed with Win32 error $errorCode."
}

if (-not [WinWorkingSetControl]::EmptyWorkingSet($process.Handle)) {
    $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    throw "EmptyWorkingSet failed with Win32 error $errorCode."
}

$min = [UIntPtr]::Zero
$max = [UIntPtr]::Zero
$flags = 0
$readBack = [WinWorkingSetControl]::GetProcessWorkingSetSizeEx(
    $process.Handle, [ref]$min, [ref]$max, [ref]$flags)
$process.Refresh()

[pscustomobject]@{
    pid = $process.Id
    port = $Port
    executable = $processInfo.ExecutablePath
    healthy = $health.status -eq 'ok'
    hard_max_enabled = $readBack -and (($flags -band 0x4) -ne 0)
    configured_max_gib = if ($readBack) { [math]::Round($max.ToUInt64() / 1GB, 2) } else { $null }
    current_working_set_gib = [math]::Round($process.WorkingSet64 / 1GB, 3)
} | Format-List

Write-Warning 'This limits the Windows process working set only. It does not lower committed GPU/UMA memory or prove that every PLE page remains exclusively on SSD.'
