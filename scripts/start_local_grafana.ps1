$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA "YATA_MPC"
$archivePath = Join-Path $runtimeRoot "grafana-13.2.3.windows-amd64.zip"
$installRoot = Join-Path $runtimeRoot "grafana-13.2.3"
$grafanaExe = Join-Path $installRoot "bin\grafana.exe"
$pluginsPath = Join-Path $runtimeRoot "plugins"
$statePath = Join-Path $runtimeRoot "grafana-data"
$logPath = Join-Path $runtimeRoot "logs"
$pidPath = Join-Path $runtimeRoot "pids"

New-Item -ItemType Directory -Force $runtimeRoot, $pluginsPath, $statePath, $logPath, $pidPath | Out-Null

if (-not (Test-Path $grafanaExe)) {
    if (-not (Test-Path $archivePath)) {
        throw "Grafana archive is missing: $archivePath"
    }
    Write-Host "Extracting Grafana into $runtimeRoot ..."
    Expand-Archive -LiteralPath $archivePath -DestinationPath $runtimeRoot -Force
}

if (-not (Test-Path $grafanaExe)) {
    throw "Grafana executable was not found at $grafanaExe"
}

python (Join-Path $projectRoot "scripts\generate_grafana_dashboard.py")
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard JSON generation failed."
}

$listenPorts = @(3000, 8123)
$occupied = Get-NetTCPConnection -State Listen -LocalPort $listenPorts -ErrorAction SilentlyContinue
if ($occupied) {
    $ports = ($occupied.LocalPort | Sort-Object -Unique) -join ", "
    throw "Local dashboard port(s) already in use: $ports. Stop the existing service or change its port."
}

$pluginsInstalled = Get-ChildItem -LiteralPath $pluginsPath -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "yesoreyeram-infinity-datasource" }
if (-not $pluginsInstalled) {
    Write-Host "Installing the signed Infinity JSON datasource plugin ..."
    & $grafanaExe cli --homepath $installRoot --pluginsDir $pluginsPath plugins install yesoreyeram-infinity-datasource
    if ($LASTEXITCODE -ne 0) {
        throw "Grafana could not install the Infinity datasource plugin. See Grafana CLI output above."
    }
}

$apiStdout = Join-Path $logPath "replay-api.stdout.log"
$apiStderr = Join-Path $logPath "replay-api.stderr.log"
$apiArgs = '-u "' + (Join-Path $projectRoot "scripts\grafana_local_api.py") + '" --host 127.0.0.1 --port 8123 --rate 1 --start-index 159 --refresh-seconds 5'
$apiProc = Start-Process -FilePath (Get-Command python.exe).Source -ArgumentList $apiArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $apiStdout -RedirectStandardError $apiStderr
Set-Content -LiteralPath (Join-Path $pidPath "replay-api.pid") -Value $apiProc.Id -Encoding ascii

$grafanaEnv = @{
    GF_SERVER_HTTP_ADDR = "127.0.0.1"
    GF_SERVER_HTTP_PORT = "3000"
    GF_PATHS_DATA = $statePath
    GF_PATHS_LOGS = $logPath
    GF_PATHS_PLUGINS = $pluginsPath
    GF_PATHS_PROVISIONING = (Join-Path $projectRoot "grafana\provisioning")
    GF_AUTH_ANONYMOUS_ENABLED = "true"
    GF_AUTH_ANONYMOUS_ORG_ROLE = "Viewer"
    GF_USERS_ALLOW_SIGN_UP = "false"
    GF_USERS_DEFAULT_THEME = "dark"
}
$grafanaStdout = Join-Path $logPath "grafana.stdout.log"
$grafanaStderr = Join-Path $logPath "grafana.stderr.log"
$grafanaProc = Start-Process -FilePath $grafanaExe -ArgumentList @("server", "--homepath", $installRoot) -WorkingDirectory $installRoot -WindowStyle Hidden -PassThru -Environment $grafanaEnv -RedirectStandardOutput $grafanaStdout -RedirectStandardError $grafanaStderr
Set-Content -LiteralPath (Join-Path $pidPath "grafana.pid") -Value $grafanaProc.Id -Encoding ascii

$ready = $false
$api = $null
for ($attempt = 0; $attempt -lt 300; $attempt++) {
    $httpListener = Get-NetTCPConnection -State Listen -LocalPort 3000 -ErrorAction SilentlyContinue
    if ($httpListener) {
        try {
            $api = Invoke-RestMethod -Uri "http://127.0.0.1:8123/api/health" -TimeoutSec 3
            $grafana = Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/health" -TimeoutSec 3
            if ($api.status -eq "ok" -and $grafana.database -eq "ok") {
                $ready = $true
                break
            }
        } catch {
            if ($apiProc.HasExited -or $grafanaProc.HasExited) {
                break
            }
        }
    } elseif ($apiProc.HasExited -or $grafanaProc.HasExited) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Host "Local services did not become ready. Logs are in $logPath"
    if (-not $apiProc.HasExited) { Stop-Process -Id $apiProc.Id -Force }
    if (-not $grafanaProc.HasExited) { Stop-Process -Id $grafanaProc.Id -Force }
    throw "Local Grafana preview failed to start."
}

$dashboardUrl = "http://127.0.0.1:3000/d/yata-bwsc2025-local/yata-mpc-bwsc2025-shadow-operations"
Write-Host "Replay API: http://127.0.0.1:8123/api/health"
Write-Host "Grafana:    $dashboardUrl"
Write-Host "Rows:       $($api.rows)"
Write-Host "Replay:     Day 1, index $($api.replay_index), 1x"
