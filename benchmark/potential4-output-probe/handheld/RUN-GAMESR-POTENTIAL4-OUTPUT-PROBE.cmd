@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo GameSR Potential-4 Output Probe FORMAL v1.1.0 - HANDHELD ONLY
echo 4 backbones x O3/O4/O6/O8 = 16 models
echo Warmup=20 Runs=600 Cooldown=2s
echo ============================================================
echo.
set "PAYLOAD=%~1"
if defined PAYLOAD (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-Potential4-Outputs.ps1" "%PAYLOAD%" -Warmup 20 -Runs 600 -CooldownSeconds 2
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-Potential4-Outputs.ps1" -Warmup 20 -Runs 600 -CooldownSeconds 2
)
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" echo [WARN] Formal probe exited with code %ERR%.
pause
exit /b %ERR%
