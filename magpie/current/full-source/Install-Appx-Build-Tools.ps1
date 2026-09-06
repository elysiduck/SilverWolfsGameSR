$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "Install-Appx-Build-Tools.log"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting administrator permission..." -ForegroundColor Cyan
    $arguments = "-NoExit -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
    exit
}

Start-Transcript -Path $log -Force | Out-Null
try {
    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    $setup = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\setup.exe"
    if (-not (Test-Path $vswhere) -or -not (Test-Path $setup)) {
        throw "Visual Studio Installer was not found."
    }

    $vsRoot = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
    if (-not $vsRoot) { throw "Visual Studio Build Tools was not found." }

    Write-Host "Adding the WinUI and AppX packaging build tools..." -ForegroundColor Cyan
    $argumentLine = "modify --installPath `"$vsRoot`" --add Microsoft.VisualStudio.Workload.UniversalBuildTools --add Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools --includeRecommended --passive --norestart"
    $process = Start-Process -FilePath $setup -ArgumentList $argumentLine -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Visual Studio Installer returned code $($process.ExitCode)." }

    $appxTargets = Join-Path $vsRoot "MSBuild\Microsoft\VisualStudio\v18.0\AppxPackage\Microsoft.AppXPackage.Targets"
    if (-not (Test-Path $appxTargets)) {
        throw "Microsoft.AppXPackage.Targets is still missing after installation."
    }

    Write-Host ""
    Write-Host "APPX BUILD TOOLS INSTALLED" -ForegroundColor Green
    Write-Host "Restart Windows, then run the environment check again."
} catch {
    Write-Host ""
    Write-Host "INSTALLATION ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Send Install-Appx-Build-Tools.log back to me."
} finally {
    Stop-Transcript | Out-Null
    Write-Host ""
    Read-Host "Press Enter after you have read the result"
}
