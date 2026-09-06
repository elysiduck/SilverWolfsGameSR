@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [FAIL] GameSR .venv was not found in this folder.
    echo Install GameSR-local-training-env first.
    echo.
    pause
    exit /b 2
)
set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" pipeline.py %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [FAIL] GameSR pipeline stopped with exit code %CODE%.
) else (
    echo [OK] GameSR pipeline finished.
)
echo.
pause
exit /b %CODE%
