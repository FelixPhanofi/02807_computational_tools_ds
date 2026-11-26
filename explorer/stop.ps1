$pidFile = Join-Path (Split-Path $PSScriptRoot) 'data/explorer/service.pid'
if (Test-Path $pidFile) {
    $savedPid = [int](Get-Content $pidFile)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid"
    if ($proc -and $proc.CommandLine -like "*$PSScriptRoot\serve.py*") {
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$savedPid" | Where-Object CommandLine -Like "*$PSScriptRoot\serve.py*" | ForEach-Object { Stop-Process -Id $_.ProcessId }
        Stop-Process -Id $savedPid -ErrorAction SilentlyContinue
    }
}

