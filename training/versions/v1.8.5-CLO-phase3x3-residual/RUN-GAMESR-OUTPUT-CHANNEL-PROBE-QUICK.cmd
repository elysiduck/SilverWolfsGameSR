@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo GameSR NPU Output Channel Probe v1.0 - QUICK
echo C16-L4 fixed, O3/O4/O6/O8, Warmup=5 Runs=120
echo ============================================================
set "PAYLOAD=%~1"
if defined PAYLOAD (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-OutputChannels.ps1" "%PAYLOAD%" -Warmup 5 -Runs 120 -CooldownSeconds 1
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-OutputChannels.ps1" -Warmup 5 -Runs 120 -CooldownSeconds 1
)
set ERR=%ERRORLEVEL%
echo.
if "%ERR%"=="0" echo [OK] Quick probe complete.
if not "%ERR%"=="0" echo [WARN] Quick probe exited with code %ERR%.
pause
exit /b %ERR%
