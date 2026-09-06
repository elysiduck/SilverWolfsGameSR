@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Environment not found. Run install.bat first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0cleaner.py" %*
pause
endlocal
