@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo [FAIL] GameSR .venv not found.
    pause
    exit /b 2
)
set "PYTHON=.venv\Scripts\python.exe"
if "%~1"=="" (
    "%PYTHON%" pipeline.py --benchmark-only
) else (
    "%PYTHON%" pipeline.py --benchmark-only --config "%~1"
)
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [FAIL] Benchmark failed or no valid result was produced.
) else (
    echo [OK] Benchmark completed and cached.
)
echo.
pause
exit /b %CODE%
