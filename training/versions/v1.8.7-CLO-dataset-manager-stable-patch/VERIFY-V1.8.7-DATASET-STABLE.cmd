@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" verify_v187_patch.py
) else (
  python verify_v187_patch.py
)

if errorlevel 1 (
  echo.
  echo [FAIL] v1.8.7 verification failed.
  pause
  exit /b 1
)

echo.
echo [OK] v1.8.7 verification passed.
pause
