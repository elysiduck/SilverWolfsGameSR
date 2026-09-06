[CmdletBinding()]
param(
    [string]$Outputs = '3,4,6,8',
    [switch]$SkipExisting,
    [string]$PythonExe
)

. (Join-Path $PSScriptRoot 'GameSR.Build.Common.ps1')
$repoRoot = Get-GameSRRoot
$trainingRoot = Join-Path $repoRoot 'training\current'

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    foreach ($candidate in @(
        (Join-Path $trainingRoot '.venv-quark\Scripts\python.exe'),
        (Join-Path $trainingRoot '.venv\Scripts\python.exe')
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $PythonExe = $candidate; break }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) { throw 'No GameSR virtual-environment Python found. Pass -PythonExe explicitly.' }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python executable not found: $PythonExe" }

$builder = Join-Path $trainingRoot 'tools\generate_output_channel_probes.py'
$args = @($builder, '--root', $trainingRoot, '--outputs', $Outputs)
if ($SkipExisting) { $args += '--skip-existing' }
& $PythonExe @args
if ($LASTEXITCODE -ne 0) { throw "Output-channel probe builder failed with exit code $LASTEXITCODE" }

Write-Host "Probe payload generated under $trainingRoot\output_channel_build" -ForegroundColor Green
