[CmdletBinding()]
param(
    [ValidateSet('cu128', 'cu130', 'cpu')]
    [string]$TorchBuild = 'cu128',
    [switch]$Rebuild,
    [switch]$SkipSmokeTest
)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot
$trainingRoot = Join-Path $repoRoot 'training\current'
Require-Command python.exe | Out-Null

Push-Location $trainingRoot
try {
    & .\setup.ps1 -TorchBuild $TorchBuild -Rebuild:$Rebuild
    if ($LASTEXITCODE -ne 0) { throw "Training setup failed with exit code $LASTEXITCODE" }
    if (-not $SkipSmokeTest) {
        & .\.venv\Scripts\python.exe .\smoke_test.py
        if ($LASTEXITCODE -ne 0) { throw "Training smoke test failed with exit code $LASTEXITCODE" }
    }
} finally { Pop-Location }

Write-Host 'Training environment is ready.' -ForegroundColor Green
