[CmdletBinding()]
param([switch]$Rebuild)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot
$toolRoot = Join-Path $repoRoot 'dataset-tools\collector-v3.2'
Require-Command python.exe | Out-Null
$venv = Join-Path $toolRoot '.venv'
if ($Rebuild -and (Test-Path -LiteralPath $venv)) { Remove-Item -LiteralPath $venv -Recurse -Force }
if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) { & python -m venv $venv }
$python = Join-Path $venv 'Scripts\python.exe'
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $toolRoot 'requirements-collector.txt') -r (Join-Path $toolRoot 'requirements-cleaner.txt')
& $python (Join-Path $toolRoot 'test_pipeline.py')
if ($LASTEXITCODE -ne 0) { throw "Dataset tool self-test failed with exit code $LASTEXITCODE" }
Write-Host 'Dataset tools are ready.' -ForegroundColor Green
