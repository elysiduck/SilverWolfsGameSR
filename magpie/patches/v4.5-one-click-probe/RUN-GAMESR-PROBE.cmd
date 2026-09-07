@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo GameSR One-Click Deployment Probe v4.5
echo ============================================================
echo.
if not exist "Magpie.exe" (
  echo [ERROR] Magpie.exe not found. Extract this patch into the Magpie runtime root.
  pause
  exit /b 1
)

if exist "GameSR-Probe.request" del /q "GameSR-Probe.request" >nul 2>nul
>"GameSR-Probe.request" echo ONE_CLICK_PROBE_V1

echo [ARMED] Probe request created.
echo.
echo Please keep the game at 1280x720 and use GameSR NPU O4.
echo Start Magpie scaling normally. The first probe frame can pause for several seconds
 echo while the FP32 CPU reference is generated. Then keep scaling for about 180 fresh GameSR results.
echo.

tasklist /FI "IMAGENAME eq Magpie.exe" 2>NUL | find /I "Magpie.exe" >NUL
if errorlevel 1 start "" "Magpie.exe"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0GameSR-Probe\Finalize-Probe.ps1" -Root "%CD%"
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" echo [WARN] Probe finalizer returned %ERR%.
pause
exit /b %ERR%
