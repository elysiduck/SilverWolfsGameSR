[CmdletBinding()]
param(
    [ValidateSet("Training", "Quark")]
    [string]$Kind = "Training",
    [string]$StartDirectory
)

$ErrorActionPreference = "Stop"
if (-not $StartDirectory) {
    $StartDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$overrideName = if ($Kind -eq "Quark") { "GAMESR_QUARK_PYTHON" } else { "GAMESR_TRAIN_PYTHON" }
$override = [Environment]::GetEnvironmentVariable($overrideName)
if ($override) {
    $overridePath = [IO.Path]::GetFullPath($override.Trim('"'))
    if (Test-Path -LiteralPath $overridePath -PathType Leaf) {
        Write-Output $overridePath
        exit 0
    }
    [Console]::Error.WriteLine("[FAIL] $overrideName points to a missing file: $overridePath")
    exit 2
}

$relativeCandidates = if ($Kind -eq "Quark") {
    @(".venv-quark\Scripts\python.exe", "venv-quark\Scripts\python.exe")
} else {
    @(".venv\Scripts\python.exe", "venv\Scripts\python.exe")
}

$directory = [IO.DirectoryInfo]([IO.Path]::GetFullPath($StartDirectory))
while ($null -ne $directory) {
    foreach ($relative in $relativeCandidates) {
        $candidate = Join-Path $directory.FullName $relative
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            Write-Output ([IO.Path]::GetFullPath($candidate))
            exit 0
        }
    }
    $directory = $directory.Parent
}

[Console]::Error.WriteLine("[FAIL] No $Kind environment was found in this folder or any parent folder.")
[Console]::Error.WriteLine("       Expected .venv for training or .venv-quark for INT8 export.")
exit 2
