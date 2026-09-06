@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo GameSR Architecture Sweep Runner v1.0.2 - QUICK pass
echo Warmup 5 / Runs 120 / NPU only
echo ============================================================
echo.

if "%~1"=="" goto RUN_DEFAULT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-XDNA.ps1" -PayloadPath "%~1" -Warmup 5 -Runs 120 -CooldownSeconds 0
goto DONE

:RUN_DEFAULT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-XDNA.ps1" -Warmup 5 -Runs 120 -CooldownSeconds 0

:DONE
set ERR=%ERRORLEVEL%
echo.
if "%ERR%"=="0" (
  echo [OK] Quick Architecture Sweep finished.
) else (
  echo [FAIL] Quick Architecture Sweep exited with code %ERR%.
)
echo.
pause
exit /b %ERR%
