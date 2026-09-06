[CmdletBinding()]
param(
    [ValidateSet("cu128", "cu130", "cpu")]
    [string]$TorchBuild = "cu128",
    [string]$EnvironmentRoot,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:UV_LINK_MODE = "copy"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentRoot = if ($EnvironmentRoot) {
    [IO.Path]::GetFullPath($EnvironmentRoot)
} else {
    [IO.Path]::GetFullPath((Split-Path -Parent $ProjectRoot))
}
$ToolsDir = Join-Path $EnvironmentRoot ".tools"
$UvExe = Join-Path $ToolsDir "uv.exe"
$VenvDir = Join-Path $EnvironmentRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

Write-Host "GameSR isolated training environment" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Shared environment root: $EnvironmentRoot"

if ($env:CONDA_PREFIX -or $env:VIRTUAL_ENV) {
    Write-Warning "An environment is currently active. It will not be reused."
}

if (-not (Test-Path $UvExe)) {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $Archive = Join-Path $env:TEMP "gamesr-uv.zip"
    $Release = Invoke-RestMethod "https://api.github.com/repos/astral-sh/uv/releases/latest"
    $Asset = $Release.assets | Where-Object { $_.name -eq "uv-x86_64-pc-windows-msvc.zip" } | Select-Object -First 1
    if (-not $Asset) { throw "Could not locate the official Windows uv package." }
    Invoke-WebRequest $Asset.browser_download_url -OutFile $Archive
    $ExtractDir = Join-Path $env:TEMP "gamesr-uv-extract"
    if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
    Expand-Archive $Archive $ExtractDir -Force
    Copy-Item (Join-Path $ExtractDir "uv.exe") $UvExe -Force
}

if ($Rebuild -and (Test-Path $VenvDir)) {
    $Expected = [System.IO.Path]::GetFullPath((Join-Path $EnvironmentRoot ".venv"))
    $Actual = [System.IO.Path]::GetFullPath($VenvDir)
    if ($Actual -ne $Expected -or (Split-Path $Actual -Leaf) -ne ".venv") {
        throw "Safety check refused to remove unexpected path: $Actual"
    }
    Remove-Item $Actual -Recurse -Force
}

& $UvExe python install 3.11
if (-not (Test-Path $PythonExe)) {
    & $UvExe venv $VenvDir --python 3.11 --seed
}

# Install general packages first. Some packages (notably lpips) declare torch as a
# dependency and PyPI may select a newer CPU-only build. Installing PyTorch last
# from its dedicated index guarantees that the requested CUDA build wins.
& $UvExe pip install --python $PythonExe --upgrade -r $Requirements

if ($TorchBuild -eq "cpu") {
    & $UvExe pip install --python $PythonExe --reinstall torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
} else {
    $TorchIndex = "https://download.pytorch.org/whl/$TorchBuild"
    & $UvExe pip install --python $PythonExe --reinstall torch torchvision --index-url $TorchIndex
}

& $PythonExe (Join-Path $ProjectRoot "check_env.py")
if ($LASTEXITCODE -ne 0 -and $TorchBuild -ne "cpu") {
    throw "Environment installed, but CUDA verification failed. See the report above."
}

& $UvExe pip freeze --python $PythonExe | Set-Content (Join-Path $ProjectRoot "installed-packages.txt") -Encoding utf8
Write-Host "Shared setup complete. Every child code folder can reuse this environment." -ForegroundColor Green
