@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo GameSR NPU Architecture Sweep v1.0
echo Default grid: C4,6,8,10,12,14,16,20,24,28,32 x L2,3,4,5,6,8
 echo ============================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Sweep-GameSR-XDNA.ps1"
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" echo Sweep failed with code %ERR%.
if "%ERR%"=="0" echo Sweep complete. Open the newest results\GameSR-NPU-Sweep-*-RANKING.csv
pause
exit /b %ERR%
