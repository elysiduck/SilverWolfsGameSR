$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "Build-SESR.log"
Start-Transcript -Path $log -Force | Out-Null
Set-Location $root

function Stop-Build([string]$message) {
    Write-Host ""
    Write-Host "BUILD STOPPED: $message" -ForegroundColor Red
    Write-Host "Run 0-first-check-environment.cmd and send the result file back to me."
    exit 1
}

if (-not (Test-Path "third_party\sesr\model\sesr_nhwc_int8_256x256.onnx") -or
    -not (Test-Path "third_party\sesr\model\AMD_AIE2P_4x4_Overlay.xclbin") -or
    -not (Test-Path "third_party\sesr\model\vaip_config_stride.json") -or
    -not (Test-Path "third_party\sesr\model\modelcachekey_sesr_nhwc_int8_256x256\context.json") -or
    -not (Test-Path "third_party\sesr\model\modelcachekey_sesr_nhwc_int8_256x256\compiled.0xa000205001fdb46.xmodel") -or
    -not (Test-Path "third_party\sesr\onnx\bin\onnxruntime_vitisai_ep.dll")) {
    Stop-Build "The SESR ONNX model or VitisAI runtime is missing. Extract the complete package first."
}

$programFilesX86 = ${env:ProgramFiles(x86)}
$vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Stop-Build "Visual Studio 2026 or Build Tools 2026 is missing." }
$vsRoot = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
if (-not $vsRoot) { Stop-Build "MSBuild is missing from Visual Studio." }
$winUiRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools -property installationPath 2>$null
if (-not $winUiRoot) { Stop-Build "The C++ WinUI application build tools are missing." }
$appxTargets = Join-Path $vsRoot "MSBuild\Microsoft\VisualStudio\v18.0\AppxPackage\Microsoft.AppXPackage.Targets"
if (-not (Test-Path $appxTargets)) { Stop-Build "The AppX packaging build targets are missing. Run 3-Appx installer first." }

$v145 = Get-ChildItem (Join-Path $vsRoot "VC\Tools\MSVC") -Directory -Filter "14.5*" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "bin\Hostx64\x64\cl.exe") } |
    Select-Object -First 1
if (-not $v145) { Stop-Build "The MSVC v145 C++ compiler is missing." }

$sdk = Join-Path $programFilesX86 "Windows Kits\10\Lib\10.0.26100.0"
if (-not (Test-Path $sdk)) { Stop-Build "Windows 11 SDK 10.0.26100.0 is missing." }

$pythonExe = $null
$pythonPrefix = @()
if (Get-Command py.exe -ErrorAction SilentlyContinue) {
    foreach ($candidate in @("3.13", "3.12", "3.11")) {
        try {
            & py "-$candidate" --version *> $null
            if ($LASTEXITCODE -eq 0) {
                $pythonExe = "py"
                $pythonPrefix = @("-$candidate")
                break
            }
        } catch { }
    }
}
if (-not $pythonExe -and (Get-Command python.exe -ErrorAction SilentlyContinue)) {
    try {
        & python --version *> $null
        if ($LASTEXITCODE -eq 0) { $pythonExe = "python" }
    } catch { }
}
if (-not $pythonExe) { Stop-Build "Python 3.11 or newer is missing." }

$pythonScripts = & $pythonExe @pythonPrefix -c "import sysconfig; print(sysconfig.get_path('scripts'))"
$userScripts = & $pythonExe @pythonPrefix -c "import os,sysconfig; print(sysconfig.get_path('scripts', scheme=os.name+'_user'))"
if ($pythonScripts) { $env:PATH = "$pythonScripts;$env:PATH" }
if ($userScripts) { $env:PATH = "$userScripts;$env:PATH" }

$conanExe = $null
if ($userScripts -and (Test-Path (Join-Path $userScripts "conan.exe"))) {
    $conanExe = Join-Path $userScripts "conan.exe"
} elseif (Get-Command conan.exe -ErrorAction SilentlyContinue) {
    $conanExe = (Get-Command conan.exe).Source
}
if (-not $conanExe) { Stop-Build "Conan 2 is missing." }
$conanVersion = & $conanExe --version 2>$null
if ($LASTEXITCODE -ne 0 -or -not $conanVersion) { Stop-Build "Conan 2 could not be started." }

if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) {
    $bundledCmakeDir = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if (Test-Path (Join-Path $bundledCmakeDir "cmake.exe")) {
        $env:PATH = "$bundledCmakeDir;$env:PATH"
    } else {
        Stop-Build "CMake is missing."
    }
}

Write-Host "Starting the Release x64 build..." -ForegroundColor Cyan
& $pythonExe @pythonPrefix "scripts\publish.py" --compiler MSVC --platform x64 --version-string SESR-M7-ONNX-NPU-v2.1
if ($LASTEXITCODE -ne 0) { Stop-Build "Compilation failed. Send a screenshot of the last error lines to me." }

$magpie = Join-Path $root "publish\x64\Magpie.exe"
if (-not (Test-Path $magpie)) { Stop-Build "The build ended without producing Magpie.exe." }

$publishedRuntime = Join-Path $root "publish\x64\onnxruntime_vitisai_ep.dll"
$publishedContext = Join-Path $root "publish\x64\sesr\modelcachekey_sesr_nhwc_int8_256x256\context.json"
$publishedXmodel = Join-Path $root "publish\x64\sesr\modelcachekey_sesr_nhwc_int8_256x256\compiled.0xa000205001fdb46.xmodel"
$publishedExactEffect = Join-Path $root "publish\x64\effects\SESR-M7 NPU Exact 1080p.hlsl"
if (-not (Test-Path $publishedRuntime) -or
    -not (Test-Path $publishedContext) -or
    -not (Test-Path $publishedXmodel) -or
    -not (Test-Path $publishedExactEffect)) {
    Stop-Build "Build output is missing the Ryzen AI runtime, X1 cache, or Exact 1080p effect."
}

Write-Host ""
Write-Host "BUILD COMPLETED" -ForegroundColor Green
Write-Host "Ryzen AI runtime, X1 cache, and Exact 1080p effect: VERIFIED" -ForegroundColor Green
Write-Host "Output folder: $(Join-Path $root 'publish\x64')"
Stop-Transcript | Out-Null
Start-Process explorer.exe -ArgumentList "/select,`"$magpie`""
