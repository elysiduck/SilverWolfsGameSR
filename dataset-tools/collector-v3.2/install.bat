@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
if errorlevel 1 (
  echo.
  echo INSTALL FAILED. See the error above.
  pause
  exit /b 1
)
echo.
echo Install completed successfully.
pause
endlocal
