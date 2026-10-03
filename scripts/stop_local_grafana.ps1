$ErrorActionPreference = "Stop"

$runtimeRoot = Join-Path $env:LOCALAPPDATA "YATA_MPC"
$pidPath = Join-Path $runtimeRoot "pids"
$projectRoot = Split-Path -Parent $PSScriptRoot
$apiScript = Join-Path $projectRoot "scripts\grafana_local_api.py"
$grafanaExe = Join-Path $runtimeRoot "grafana-13.2.3\bin\grafana.exe"

foreach ($name in @("grafana", "replay-api")) {
    $pidFile = Join-Path $pidPath "$name.pid"
    if (-not (Test-Path $pidFile)) { continue }
    $processId = [int](Get-Content -LiteralPath $pidFile -Raw)
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($processInfo) {
        $expected = if ($name -eq "grafana") {
            [string]$processInfo.ExecutablePath -eq $grafanaExe
        } else {
            ([string]$processInfo.CommandLine).Contains($apiScript, [System.StringComparison]::OrdinalIgnoreCase)
        }
        if ($expected) {
            Stop-Process -Id $processId -Force
            Write-Host "Stopped $name (PID $processId)"
        } else {
            Write-Host "Skipped PID $processId; it no longer matches the YATA $name process."
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
