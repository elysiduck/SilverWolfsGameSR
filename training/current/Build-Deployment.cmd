@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  echo [FAIL] .venv not found.
  pause
  exit /b 2
)
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=tiny-npu-fast"
".venv\Scripts\python.exe" build_deployment.py --model "%MODEL%" --root "%CD%"
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
