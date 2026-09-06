@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo GameSR NPU Output Channel Probe v1.0 - FORMAL
echo C16-L4 fixed, O3/O4/O6/O8, Warmup=20 Runs=600
echo ============================================================
set "PAYLOAD=%~1"
if defined PAYLOAD (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-OutputChannels.ps1" "%PAYLOAD%" -Warmup 20 -Runs 600 -CooldownSeconds 2
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-OutputChannels.ps1" -Warmup 20 -Runs 600 -CooldownSeconds 2
)
set ERR=%ERRORLEVEL%
echo.
if "%ERR%"=="0" echo [OK] Probe complete. Send the newest results\GameSR-NPU-Output-Channel-Probe-*-COMPACT.zip back to ChatGPT.
if not "%ERR%"=="0" echo [WARN] Probe exited with code %ERR%. Check the newest output-channel logs/results.
pause
exit /b %ERR%
