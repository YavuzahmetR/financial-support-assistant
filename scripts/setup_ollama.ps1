# Install the portable Windows runtime used by this demo.
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskVersion = '0.40.2'
$taskRuntimePath = Join-Path $taskProjectRoot '.cache\ollama'
$taskArchivePath = Join-Path $taskProjectRoot '.cache\ollama-install.zip'
$taskExecutable = Join-Path $taskRuntimePath 'ollama.exe'

if (Test-Path -LiteralPath $taskExecutable) {
    Write-Output 'Portable Ollama is already installed.'
    exit 0
}

New-Item -ItemType Directory -Path $taskRuntimePath -Force | Out-Null
try {
    curl.exe --fail --location --output $taskArchivePath "https://github.com/ollama/ollama/releases/download/v$taskVersion/ollama-windows-amd64.zip"
    if ($LASTEXITCODE -ne 0) { throw 'Ollama download failed.' }
    Expand-Archive -LiteralPath $taskArchivePath -DestinationPath $taskRuntimePath -Force
    if (!(Test-Path -LiteralPath $taskExecutable)) { throw 'Ollama installation failed.' }
} finally {
    Remove-Item -LiteralPath $taskArchivePath -ErrorAction SilentlyContinue
}
Write-Output "Portable Ollama $taskVersion installed."
