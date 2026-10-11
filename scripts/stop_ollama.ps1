# Stop only the Ollama executable recorded by this project's launcher.
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskExecutable = Join-Path $taskProjectRoot '.cache\ollama\ollama.exe'
$taskPidFile = Join-Path $taskProjectRoot '.cache\ollama-server.pid'
if (!(Test-Path -LiteralPath $taskPidFile)) {
    Write-Output 'No project Ollama PID is recorded.'
    exit 0
}
$taskProcess = Get-Process -Id ([int](Get-Content -LiteralPath $taskPidFile -Raw)) -ErrorAction SilentlyContinue
if ($taskProcess) {
    if ($taskProcess.Path -ne $taskExecutable) { throw 'Recorded PID does not belong to this project Ollama executable.' }
    $taskBody = '{"model":"qwen3:1.7b","keep_alive":0}'
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/generate' -Method Post -ContentType 'application/json' -Body $taskBody -TimeoutSec 30 | Out-Null
    } catch {
        Write-Warning 'Model unload was unavailable; stopping the verified project process tree.'
    } finally {
        # Terminate this verified runtime and its model runner children together.
        if (!$taskProcess.HasExited) {
            taskkill.exe /PID $taskProcess.Id /T /F | Out-Null
            if ($LASTEXITCODE -ne 0) {
                # The model runner can exit during unload, before taskkill reaches
                # it. Only fail if this verified server is actually still running.
                $taskRemainingProcess = Get-Process -Id $taskProcess.Id -ErrorAction SilentlyContinue
                if ($taskRemainingProcess -and $taskRemainingProcess.Path -eq $taskExecutable) {
                    throw 'Could not stop the project Ollama process tree.'
                }
            }
        }
    }
}
Remove-Item -LiteralPath $taskPidFile
Write-Output 'Project Ollama process tree stopped.'
