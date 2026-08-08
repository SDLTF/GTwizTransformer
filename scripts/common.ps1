$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = 'D:\Code\GSDD\GSDD-Bench-v1.0.0-DShield-Integration\GSDD-Bench-v1.0.0\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "CUDA Python runtime not found: $PythonExe"
}
Set-Location -LiteralPath $ProjectRoot

