$ErrorActionPreference='Stop'
$repo = Split-Path $PSScriptRoot
$state = Join-Path $repo 'data/explorer'
New-Item -ItemType Directory -Force -Path $state | Out-Null
if (Test-Path "$state/service.pid") {
    $savedPid = [int](Get-Content "$state/service.pid")
    if (Get-Process -Id $savedPid -ErrorAction SilentlyContinue) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid"
        if ($owner.CommandLine -like "*$PSScriptRoot\serve.py*") { return }
    }
}
$python = Join-Path $repo 'data/medium/.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = Join-Path $repo '.venv/Scripts/python.exe' }
$env:HF_HUB_OFFLINE='1'
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:PYTHONPATH = Join-Path $repo 'src'
if (Test-Path "$state/relay.token") { $env:SIGNAL_RELAY_TOKEN = (Get-Content "$state/relay.token" -Raw).Trim() }
$process = Start-Process $python -ArgumentList @('-u',('"{0}\serve.py"' -f $PSScriptRoot)) -WindowStyle Hidden -WorkingDirectory $repo -PassThru -RedirectStandardOutput "$state/stdout.log" -RedirectStandardError "$state/stderr.log"
Set-Content -LiteralPath "$state/service.pid" -Value $process.Id

