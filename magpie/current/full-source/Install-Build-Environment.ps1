$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "Install-Build-Environment.log"

function Stop-Install([string]$message) {
    Write-Host ""
    Write-Host "INSTALLATION STOPPED: $message" -ForegroundColor Red
    exit 1
}

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting administrator permission..." -ForegroundColor Cyan
    $arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
    exit
}

Start-Transcript -Path $log -Force | Out-Null
try {
    Write-Host "Magpie SESR build environment installer" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will install:" 
    Write-Host "  - Visual Studio Build Tools 2026"
    Write-Host "  - MSVC C++ x64/x86 tools and CMake"
    Write-Host "  - WinUI C++ build tools and Windows 11 SDK 26100"
    Write-Host "  - Python 3.12 and Conan 2"
    Write-Host ""
    Write-Host "Allow about 20 GB of free disk space. Download and installation can take 30-90 minutes." -ForegroundColor Yellow
    $answer = Read-Host "Type YES and press Enter to continue"
    if ($answer -cne "YES") { Stop-Install "Cancelled by user." }

    $systemDrive = Get-PSDrive -Name (($env:SystemDrive).TrimEnd(':'))
    if ($systemDrive.Free -lt 20GB) {
        Stop-Install "Less than 20 GB is free on the system drive."
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { Stop-Install "Windows Package Manager (winget) is missing." }

    Write-Host ""
    Write-Host "[1/3] Installing Visual Studio Build Tools 2026..." -ForegroundColor Cyan
    $vsArguments = @(
        "install", "--id", "Microsoft.VisualStudio.BuildTools", "--exact",
        "--source", "winget", "--accept-package-agreements", "--accept-source-agreements",
        "--override",
        "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.UniversalBuildTools --add Microsoft.VisualStudio.ComponentGroup.WindowsAppDevelopment.VC.BuildTools --includeRecommended"
    )
    & winget @vsArguments
    if ($LASTEXITCODE -ne 0) { Stop-Install "Visual Studio Build Tools installation failed with code $LASTEXITCODE." }

    Write-Host ""
    Write-Host "[2/3] Installing Python 3.12..." -ForegroundColor Cyan
    & winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Stop-Install "Python installation failed with code $LASTEXITCODE." }

    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
        [Environment]::GetEnvironmentVariable("Path", "User")

    $python = $null
    $pythonPrefix = @()
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        try {
            & py -3.12 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                $python = "py"
                $pythonPrefix = @("-3.12")
            }
        } catch { }
    }
    if (-not $python) {
        $pythonCandidate = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
        if (Test-Path $pythonCandidate) { $python = $pythonCandidate }
    }
    if (-not $python) { Stop-Install "Python was installed but could not be located. Restart Windows and run this installer again." }

    Write-Host ""
    Write-Host "[3/3] Installing Conan 2..." -ForegroundColor Cyan
    & $python @pythonPrefix -m pip install --user --upgrade "conan>=2,<3"
    if ($LASTEXITCODE -ne 0) { Stop-Install "Conan installation failed with code $LASTEXITCODE." }

    Write-Host ""
    Write-Host "INSTALLATION COMPLETED" -ForegroundColor Green
    Write-Host "Restart Windows, then run 0-first-check-environment.cmd again."
} catch {
    Write-Host ""
    Write-Host "INSTALLATION ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Send Install-Build-Environment.log back to me."
} finally {
    Stop-Transcript | Out-Null
}
