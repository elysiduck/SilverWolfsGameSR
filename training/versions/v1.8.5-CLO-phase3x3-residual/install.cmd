@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Installation failed. Keep this window open and send the error output for diagnosis.
  pause
  exit /b 1
)
echo.
echo Installation and GPU verification completed.
pause

