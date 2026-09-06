[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ArtifactRoot,
    [switch]$StageRuntime
)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot
$artifactRoot = Get-GameSRArtifactRoot $ArtifactRoot
Test-GameSRArtifact $artifactRoot 'magpie-v4.1.1' | Out-Null
$magpieRoot = Join-Path $repoRoot 'magpie\current\full-source'
$payload = Join-Path $artifactRoot 'magpie-v4.1.1-runtime'

if ($StageRuntime) {
    Copy-GameSRArtifactTree (Join-Path $payload 'third_party\sesr\model') (Join-Path $magpieRoot 'third_party\sesr\model')
    Copy-GameSRArtifactTree (Join-Path $payload 'third_party\sesr\onnx') (Join-Path $magpieRoot 'third_party\sesr\onnx')
}

Push-Location $magpieRoot
try { & .\Build-SESR.ps1 } finally { Pop-Location }
