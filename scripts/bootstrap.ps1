[CmdletBinding()]
param(
    [ValidateSet('All', 'Training', 'Dataset', 'Magpie', 'Benchmark')]
    [string]$Component = 'All',
    [string]$ArtifactRoot,
    [switch]$VerifyArtifacts
)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot

Write-Host "SilverWolfsGameSR bootstrap: $Component" -ForegroundColor Cyan
Require-Command git.exe | Out-Null

if ($Component -in @('All', 'Training', 'Dataset')) {
    Require-Command python.exe | Out-Null
    & python --version
}
if ($Component -in @('All', 'Magpie')) {
    Require-Command cmake.exe | Out-Null
    Require-Command conan.exe | Out-Null
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Build Tools / vswhere is missing.' }
}

if ($VerifyArtifacts) {
    $artifactPath = Get-GameSRArtifactRoot $ArtifactRoot
    $ids = switch ($Component) {
        'Training' { @('training-v1.8.2-patch') }
        'Dataset' { @('collector-v3.2', 'cleaner-v3.2.1') }
        'Magpie' { @('magpie-v4.1.1') }
        'Benchmark' { @('benchmark-v1.3', 'sweep-models-20260905', 'sweep-results-20260905') }
        default { @('training-v1.8.2-patch', 'collector-v3.2', 'cleaner-v3.2.1', 'benchmark-v1.3', 'sweep-models-20260905', 'sweep-results-20260905', 'magpie-v4.1.1') }
    }
    foreach ($id in $ids) { Test-GameSRArtifact $artifactPath $id | Out-Null }
}

Write-Host 'Bootstrap checks passed.' -ForegroundColor Green
