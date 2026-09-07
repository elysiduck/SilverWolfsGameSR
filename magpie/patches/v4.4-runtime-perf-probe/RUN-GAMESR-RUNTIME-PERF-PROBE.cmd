@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo GameSR Runtime Performance Probe (no rebuild required)
echo ============================================================
echo This immediate runtime probe measures the existing v4.4 pipeline.
echo It does NOT perform FP32-vs-INT8 same-frame A/B; that requires the source Probe build.
echo.
if not exist "Magpie.exe" (
  echo [ERROR] Magpie.exe not found. Extract this patch into the Magpie runtime root.
  pause
  exit /b 1
)

tasklist /FI "IMAGENAME eq Magpie.exe" 2>NUL | find /I "Magpie.exe" >NUL
if errorlevel 1 start "" "Magpie.exe"

echo Keep the game at 1280x720, select GameSR NPU O4, then start scaling normally.
echo The script will detect GameSR automatically and collect 10 pipeline samples.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0GameSR-Probe\Runtime-Perf-Probe.ps1" -Root "%~dp0"
set ERR=%ERRORLEVEL%
echo.
pause
exit /b %ERR%
