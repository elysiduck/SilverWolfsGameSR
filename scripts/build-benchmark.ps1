[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ArtifactRoot,
    [int]$Warmup = 20,
    [int]$Runs = 600,
    [switch]$StageRuntime,
    [switch]$NoTaskManager
)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot
$artifactRoot = Get-GameSRArtifactRoot $ArtifactRoot
Test-GameSRArtifact $artifactRoot 'benchmark-v1.3' | Out-Null
$benchmarkRoot = Join-Path $repoRoot 'benchmark\xdna-v1.3'
$payload = Join-Path $artifactRoot 'benchmark-v1.3-runtime'

if ($StageRuntime) {
    foreach ($name in @('runtime', 'models', 'control')) { Copy-GameSRArtifactTree (Join-Path $payload $name) (Join-Path $benchmarkRoot $name) }
}

Push-Location $benchmarkRoot
try { & .\Run-XDNAArchitectureBenchmark.ps1 -Warmup $Warmup -Runs $Runs -NoTaskManager:$NoTaskManager } finally { Pop-Location }
