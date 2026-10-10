# Windows Task Scheduler entry point. Start the local API if needed, then run the batch.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ApiUrl = 'http://127.0.0.1:5000'
$LogsDir = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$NightlyLog = Join-Path $LogsDir 'nightly.log'

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -Path $NightlyLog -Value $line
    Write-Output $line
}

if (-not (Test-Path $PythonExe)) {
    Write-Log "ERROR: Virtual-environment Python not found at $PythonExe. Follow README setup steps."
    exit 2
}

$healthy = $false
try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 3
    if ($health.model_loaded -eq $true) { $healthy = $true }
} catch { $healthy = $false }

if (-not $healthy) {
    Write-Log 'API is not ready; starting app.py in the background.'
    Start-Process -FilePath $PythonExe -ArgumentList 'app.py' -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogsDir 'api-stdout.log') `
        -RedirectStandardError (Join-Path $LogsDir 'api-stderr.log')
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 2
            if ($health.model_loaded -eq $true) { $healthy = $true; break }
        } catch { }
    }
}

if (-not $healthy) {
    Write-Log 'ERROR: API did not become healthy within 30 seconds. Batch skipped.'
    exit 2
}

Write-Log 'Starting batch_predict.py.'
Push-Location $ProjectRoot
try {
    & $PythonExe 'batch_predict.py' 2>&1 | ForEach-Object { Write-Log ([string]$_) }
    $batchExit = $LASTEXITCODE
} finally {
    Pop-Location
}
Write-Log "Batch finished with exit code $batchExit."
exit $batchExit
