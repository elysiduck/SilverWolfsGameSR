$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "Repair-Build-Environment.log"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting administrator permission..." -ForegroundColor Cyan
    $arguments = "-NoExit -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
    exit
}

Start-Transcript -Path $log -Force | Out-Null
try {
    Write-Host "Repairing the two remaining build components..." -ForegroundColor Cyan
    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    $setup = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\setup.exe"
    if (-not (Test-Path $vswhere) -or -not (Test-Path $setup)) {
        throw "Visual Studio Installer was not found."
    }

    $vsRoot = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
    if (-not $vsRoot) { throw "The existing Visual Studio Build Tools installation was not found." }

    $winUiRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools -property installationPath 2>$null
    if (-not $winUiRoot) {
        Write-Host ""
        Write-Host "[1/3] Adding the C++ WinUI build tools..." -ForegroundColor Cyan
        $argumentLine = "modify --installPath `"$vsRoot`" --add Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools --passive --norestart"
        $process = Start-Process -FilePath $setup -ArgumentList $argumentLine -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Visual Studio Installer returned code $($process.ExitCode)." }
    } else {
        Write-Host "[1/3] C++ WinUI build tools are already installed." -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "[2/3] Checking Python 3.13..." -ForegroundColor Cyan
    $pythonReady = $false
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        try {
            & py -3.13 --version *> $null
            if ($LASTEXITCODE -eq 0) { $pythonReady = $true }
        } catch { }
    }
    if (-not $pythonReady) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) { throw "Windows Package Manager (winget) was not found." }
        & winget install --id Python.Python.3.13 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Python 3.13 installation returned code $LASTEXITCODE." }
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")
    }

    & py -3.13 --version
    if ($LASTEXITCODE -ne 0) { throw "Python 3.13 is installed but could not be started. Restart Windows and run this repair again." }

    Write-Host ""
    Write-Host "[3/3] Installing Conan 2 with Python 3.13..." -ForegroundColor Cyan
    & py -3.13 -m pip install --user --upgrade "conan>=2,<3"
    if ($LASTEXITCODE -ne 0) { throw "Conan installation returned code $LASTEXITCODE." }
    $userScripts = & py -3.13 -c "import os,sysconfig; print(sysconfig.get_path('scripts', scheme=os.name+'_user'))"
    $conanExe = Join-Path $userScripts "conan.exe"
    if (-not (Test-Path $conanExe)) { throw "Conan was installed but conan.exe could not be located." }
    & $conanExe --version
    if ($LASTEXITCODE -ne 0) { throw "Conan was installed but could not be started." }

    Write-Host ""
    Write-Host "REPAIR COMPLETED" -ForegroundColor Green
    Write-Host "Restart Windows and run the environment check again."
} catch {
    Write-Host ""
    Write-Host "REPAIR ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Send Repair-Build-Environment.log back to me."
} finally {
    Stop-Transcript | Out-Null
    Write-Host ""
    Read-Host "Press Enter after you have read the result"
}
