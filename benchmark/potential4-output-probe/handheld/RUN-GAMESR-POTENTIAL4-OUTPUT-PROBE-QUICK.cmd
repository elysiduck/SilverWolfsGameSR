@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo GameSR Potential-4 Output Probe QUICK v1.1.0 - HANDHELD ONLY
echo 4 backbones x O3/O4/O6/O8 = 16 models
echo Warmup=5 Runs=120 Cooldown=1s
echo ============================================================
echo.
set "PAYLOAD=%~1"
if defined PAYLOAD (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-Potential4-Outputs.ps1" "%PAYLOAD%" -Warmup 5 -Runs 120 -CooldownSeconds 1
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-Potential4-Outputs.ps1" -Warmup 5 -Runs 120 -CooldownSeconds 1
)
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" echo [WARN] Quick probe exited with code %ERR%.
pause
exit /b %ERR%
