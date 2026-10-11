# Start this project's local-only Ollama process, without registering a service.
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskExecutable = Join-Path $taskProjectRoot '.cache\ollama\ollama.exe'
$taskPidFile = Join-Path $taskProjectRoot '.cache\ollama-server.pid'
if (!(Test-Path -LiteralPath $taskExecutable)) {
    throw 'Run scripts/setup_ollama.ps1 first.'
}
if (Test-Path -LiteralPath $taskPidFile) {
    $taskExistingProcess = Get-Process -Id ([int](Get-Content -LiteralPath $taskPidFile -Raw)) -ErrorAction SilentlyContinue
    if ($taskExistingProcess -and $taskExistingProcess.Path -eq $taskExecutable) {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2 | Out-Null
        Write-Output "This project's Ollama process is ready: PID $($taskExistingProcess.Id)."
        exit 0
    }
}
$taskSocket = New-Object System.Net.Sockets.TcpClient
try {
    $taskSocket.Connect('127.0.0.1', 11435)
    throw 'Port 11435 is already in use; the existing listener was not changed.'
} catch [System.Net.Sockets.SocketException] {
    # Connection refused is expected when this project's listener is not running.
} finally {
    $taskSocket.Dispose()
}

$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $taskProjectRoot 'models\ollama'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_MAX_QUEUE = '1'
$env:OLLAMA_CONTEXT_LENGTH = '2048'
$env:OLLAMA_KEEP_ALIVE = '5m'
New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
$taskProcess = Start-Process -FilePath $taskExecutable -ArgumentList 'serve' -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $taskPidFile -Value $taskProcess.Id -Encoding ASCII
try {
    for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
        if ($taskProcess.HasExited) { throw 'Ollama stopped during startup.' }
        $taskVersionResponse = $null
        try {
            $taskVersionResponse = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 1
        } catch {
            Start-Sleep -Milliseconds 250
        }
        if ($taskVersionResponse) {
            Write-Output "Ollama $($taskVersionResponse.version) ready on 127.0.0.1:11435; PID $($taskProcess.Id)."
            exit 0
        }
    }
    throw 'Ollama did not become ready.'
} catch {
    if (!$taskProcess.HasExited) { Stop-Process -Id $taskProcess.Id -Force }
    Remove-Item -LiteralPath $taskPidFile -ErrorAction SilentlyContinue
    throw
}
