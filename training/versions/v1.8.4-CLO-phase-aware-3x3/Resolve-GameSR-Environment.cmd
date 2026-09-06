@echo off
set "GAMESR_RESOLVED_PYTHON="
for /f "usebackq delims=" %%P in (`powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Find-GameSR-Python.ps1" -Kind "%~1"`) do set "GAMESR_RESOLVED_PYTHON=%%P"
if not defined GAMESR_RESOLVED_PYTHON exit /b 2
exit /b 0
