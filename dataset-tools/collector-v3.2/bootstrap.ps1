$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
Write-Host '=== GameSR Dataset Collector V3.2 ===' -ForegroundColor Cyan

function Get-PythonCommand {
    $candidates = @(
        @{ cmd = 'py'; args = @('-3.14') },
        @{ cmd = 'py'; args = @('-3.13') },
        @{ cmd = 'py'; args = @('-3.12') },
        @{ cmd = 'py'; args = @('-3.11') },
        @{ cmd = 'py'; args = @('-3.10') },
        @{ cmd = 'python'; args = @() }
    )
    foreach ($c in $candidates) {
        try {
            $out = & $c.cmd @($c.args) -c "import sys; print(sys.version_info[0]); print(sys.version_info[1])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out.Count -ge 2) {
                $ma = [int]$out[0]; $mi = [int]$out[1]
                if ($ma -eq 3 -and $mi -ge 10 -and $mi -le 14) {
                    return $c
                }
            }
        } catch {}
    }
    return $null
}

$Py = Get-PythonCommand
if (-not $Py) {
    Write-Host 'No supported Python found. Attempting automatic Python 3.14 installation via winget...' -ForegroundColor Yellow
    $wingetCmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $wingetCmd) {
        Write-Host 'winget is not available. Please install Python 3.10-3.14 x64, then run install.bat again.' -ForegroundColor Red
        exit 1
    }
    & $wingetCmd.Source install --id Python.Python.3.14 --scope user --accept-source-agreements --accept-package-agreements
    Start-Sleep -Seconds 2
    $Py = Get-PythonCommand
}
if (-not $Py) {
    Write-Host 'Python installation was not detected yet. Close this window, open a new terminal, and run install.bat again.' -ForegroundColor Red
    exit 1
}

$Vpy = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Vpy)) {
    Write-Host 'Creating local virtual environment...' -ForegroundColor Green
    & $Py.cmd @($Py.args) -m venv (Join-Path $Root '.venv')
}
if (-not (Test-Path $Vpy)) {
    throw 'Virtual environment creation failed.'
}

Write-Host 'Upgrading pip from official PyPI...' -ForegroundColor Green
& $Vpy -m pip install --upgrade pip --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }

Write-Host 'Installing collector dependencies...' -ForegroundColor Green
& $Vpy -m pip install -r (Join-Path $Root 'requirements-collector.txt') --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Collector dependency installation failed.' }

Write-Host 'Installing cleaner dependencies...' -ForegroundColor Green
& $Vpy -m pip install -r (Join-Path $Root 'requirements-cleaner.txt') --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Cleaner dependency installation failed.' }

Write-Host 'Running environment self-test...' -ForegroundColor Green
& $Vpy -c "import sys, dxcam, psutil, PIL, numpy; print('Python:', sys.version); print('dxcam:', getattr(dxcam,'__version__','installed')); print('Pillow:', PIL.__version__); print('numpy:', numpy.__version__); print('psutil:', psutil.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Environment self-test failed.' }

New-Item -ItemType Directory -Force (Join-Path $Root 'GameSR_Dataset\raw') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root 'GameSR_Dataset\clean') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root 'GameSR_Dataset\reports') | Out-Null

Write-Host ''
Write-Host 'INSTALL OK' -ForegroundColor Green
Write-Host 'Run start_collector.bat to collect raw screenshots.' -ForegroundColor Cyan
Write-Host 'Run run_cleaner.bat later to clean and deduplicate the raw dataset.' -ForegroundColor Cyan
