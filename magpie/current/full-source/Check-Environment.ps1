$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$report = Join-Path $root "Environment-Check-Result.txt"
$lines = [System.Collections.Generic.List[string]]::new()
$missing = 0

function Add-Pass([string]$name) {
    $script:lines.Add("[PASS] $name")
    Write-Host "[PASS] $name" -ForegroundColor Green
}

function Add-Missing([string]$name) {
    $script:lines.Add("[MISSING] $name")
    $script:missing++
    Write-Host "[MISSING] $name" -ForegroundColor Yellow
}

function Add-Info([string]$name) {
    $script:lines.Add("[INFO] $name")
    Write-Host "[INFO] $name" -ForegroundColor Cyan
}

try {
    Set-Location $root
    Write-Host "Checking the build environment..." -ForegroundColor Cyan
    Write-Host ""

    if ((Test-Path "$env:WINDIR\System32\xrt_coreutil.dll") -or
        (Test-Path "$env:WINDIR\System32\AMD\xrt_coreutil.dll")) {
        Add-Pass "AMD Ryzen AI NPU driver"
    } else {
        Add-Info "AMD NPU runtime DLL was not found in the two common locations. This does not block compilation."
    }

    $modelFiles = @(
        "third_party\sesr\model\sesr_nhwc_int8_256x256.onnx",
        "third_party\sesr\model\AMD_AIE2P_4x4_Overlay.xclbin",
        "third_party\sesr\model\vaip_config_stride.json",
        "third_party\sesr\model\modelcachekey_sesr_nhwc_int8_256x256\context.json",
        "third_party\sesr\model\modelcachekey_sesr_nhwc_int8_256x256\compiled.0xa000205001fdb46.xmodel",
        "third_party\sesr\onnx\bin\onnxruntime.dll",
		"third_party\sesr\onnx\bin\onnxruntime_vitisai_ep.dll",
		"third_party\sesr\onnx\bin\dyn_dispatch_core.dll",
		"third_party\sesr\onnx\bin\aiecompiler_client.dll"
    )
    $modelReady = $true
    foreach ($file in $modelFiles) {
        if (-not (Test-Path (Join-Path $root $file))) { $modelReady = $false }
    }
    if ($modelReady) {
        $ortHash = (Get-FileHash (Join-Path $root "third_party\sesr\onnx\bin\onnxruntime.dll") -Algorithm SHA256).Hash
        if ($ortHash -eq "919AC18B9AD61588C1D65EDBA775B012DA626FACD01623FEAB3A470042EAF4E3") {
            Add-Pass "SESR-M7 model, precompiled X1 cache, and matching Ryzen AI 1.7.1 runtime"
        } else {
            Add-Missing "Matching Ryzen AI 1.7.1 ONNX Runtime (runtime file was changed or is from an older build)"
        }
    } else {
        Add-Missing "SESR-M7 ONNX model, X1 config, NPU overlay, or VitisAI runtime files"
    }

    $vsRoot = $null
    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
    }
    if ($vsRoot) {
        Add-Pass "Visual Studio or Build Tools with MSBuild"
    } else {
        Add-Missing "Visual Studio or Build Tools with MSBuild"
    }

    $winUiRoot = $null
    if (Test-Path $vswhere) {
        $winUiRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools -property installationPath 2>$null
    }
    if ($winUiRoot) {
        Add-Pass "C++ WinUI application build tools"
    } else {
        Add-Missing "C++ WinUI application build tools"
    }

    $appxTargets = $null
    if ($vsRoot) {
        $appxTargets = Join-Path $vsRoot "MSBuild\Microsoft\VisualStudio\v18.0\AppxPackage\Microsoft.AppXPackage.Targets"
    }
    if ($appxTargets -and (Test-Path $appxTargets)) {
        Add-Pass "AppX packaging build targets"
    } else {
        Add-Missing "AppX packaging build targets"
    }

    $v145 = $null
    if ($vsRoot) {
        $v145 = Get-ChildItem (Join-Path $vsRoot "VC\Tools\MSVC") -Directory -Filter "14.5*" -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "bin\Hostx64\x64\cl.exe") } |
            Select-Object -First 1
    }
    if ($v145) {
        Add-Pass "MSVC C++ compiler v145"
    } else {
        Add-Missing "MSVC C++ compiler v145"
    }

    $sdk = Join-Path $programFilesX86 "Windows Kits\10\Lib\10.0.26100.0"
    if (Test-Path $sdk) {
        Add-Pass "Windows 11 SDK 10.0.26100.0"
    } else {
        Add-Missing "Windows 11 SDK 10.0.26100.0"
    }

    $pythonKind = $null
    $pythonPrefix = @()
    $pythonVersion = $null
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($candidate in @("3.13", "3.12", "3.11")) {
            try {
                $pythonVersion = & py "-$candidate" --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $pythonKind = "py"
                    $pythonPrefix = @("-$candidate")
                    break
                }
            } catch { }
        }
    }
    if (-not $pythonKind -and (Get-Command python.exe -ErrorAction SilentlyContinue)) {
        try {
            $pythonVersion = & python --version 2>&1
            if ($LASTEXITCODE -eq 0) { $pythonKind = "python" }
        } catch { }
    }
    if ($pythonKind) {
        Add-Pass "$pythonVersion"
    } else {
        Add-Missing "Python 3.11 or newer"
    }

    $conanVersion = $null
    $conanExe = $null
    try {
        if ($pythonKind -eq "py") {
            $userScripts = & py @pythonPrefix -c "import os,sysconfig; print(sysconfig.get_path('scripts', scheme=os.name+'_user'))"
        } elseif ($pythonKind -eq "python") {
            $userScripts = & python -c "import os,sysconfig; print(sysconfig.get_path('scripts', scheme=os.name+'_user'))"
        }
        if ($userScripts -and (Test-Path (Join-Path $userScripts "conan.exe"))) {
            $conanExe = Join-Path $userScripts "conan.exe"
        } elseif (Get-Command conan.exe -ErrorAction SilentlyContinue) {
            $conanExe = (Get-Command conan.exe).Source
        }
        if ($conanExe) { $conanVersion = & $conanExe --version 2>$null }
    } catch { }
    if ($LASTEXITCODE -eq 0 -and $conanVersion) {
        Add-Pass "$conanVersion"
    } else {
        Add-Missing "Conan 2"
    }

    $cmakeReady = [bool](Get-Command cmake.exe -ErrorAction SilentlyContinue)
    if (-not $cmakeReady -and $vsRoot) {
        $bundledCmake = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        $cmakeReady = Test-Path $bundledCmake
    }
    if ($cmakeReady) {
        Add-Pass "CMake"
    } else {
        Add-Missing "CMake"
    }
} catch {
    $lines.Add("[ERROR] $($_.Exception.Message)")
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
} finally {
    $output = [System.Collections.Generic.List[string]]::new()
    $output.Add("Magpie SESR-M7 NPU Environment Check")
    $output.Add("====================================")
    foreach ($line in $lines) { $output.Add($line) }
    $output.Add("")
    if ($missing -eq 0 -and -not ($lines | Where-Object { $_.StartsWith("[ERROR]") })) {
        $output.Add("RESULT: READY TO BUILD")
    } else {
        $output.Add("RESULT: $missing ITEM(S) MISSING OR AN ERROR OCCURRED")
    }
    $output | Set-Content -Path $report -Encoding UTF8
    Write-Host ""
    Write-Host "Result saved to:" -ForegroundColor Cyan
    Write-Host $report
    Start-Process notepad.exe -ArgumentList $report
}
