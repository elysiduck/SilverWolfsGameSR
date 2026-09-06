@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PYTHON="
if exist "%~dp0Resolve-GameSR-Environment.cmd" (
  call "%~dp0Resolve-GameSR-Environment.cmd" Training >nul 2>&1
  if not errorlevel 1 set "PYTHON=%GAMESR_RESOLVED_PYTHON%"
)
if not defined PYTHON set "PYTHON=python"

"%PYTHON%" "%~dp0verify_v186_patch.py"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
  echo [OK] v1.8.6 stable-training patch is installed correctly.
) else (
  echo [FAIL] Patch verification failed.
)
pause
exit /b %CODE%
