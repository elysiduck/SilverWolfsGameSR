Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-GameSRRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Get-GameSRArtifactRoot([string]$ArtifactRoot) {
    if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
        $ArtifactRoot = $env:GAMESR_ARTIFACTS_ROOT
    }
    if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
        throw 'ArtifactRoot is required. Pass -ArtifactRoot or set GAMESR_ARTIFACTS_ROOT.'
    }
    if (-not (Test-Path -LiteralPath $ArtifactRoot -PathType Container)) {
        throw "ArtifactRoot does not exist: $ArtifactRoot"
    }
    return (Resolve-Path -LiteralPath $ArtifactRoot).Path
}

function Test-GameSRArtifact([string]$ArtifactRoot, [string]$Id) {
    $repoRoot = Get-GameSRRoot
    $lock = Get-Content -Raw -LiteralPath (Join-Path $repoRoot 'artifacts.lock.json') | ConvertFrom-Json
    $artifact = @($lock.artifacts | Where-Object { $_.id -eq $Id }) | Select-Object -First 1
    if (-not $artifact) { throw "Unknown artifact id: $Id" }
    $path = Join-Path $ArtifactRoot $artifact.file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing artifact: $path" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
    if ($actual -ne $artifact.sha256) { throw "SHA256 mismatch for $($artifact.file). Expected $($artifact.sha256), got $actual" }
    Write-Host "[PASS] $($artifact.id): $($artifact.file)" -ForegroundColor Green
    return $path
}

function Require-Command([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) { throw "Required command not found: $Name" }
    return $command.Source
}

function Copy-GameSRArtifactTree([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Artifact payload is missing: $Source" }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source '*') -Destination $Destination -Recurse -Force
}
