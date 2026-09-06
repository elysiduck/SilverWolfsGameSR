@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo GameSR Architecture Sweep Runner v1.0.2 - Handheld / NPU only
echo No Python / No PyTorch / No Quark required
echo ============================================================
echo.

if "%~1"=="" goto RUN_DEFAULT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-XDNA.ps1" -PayloadPath "%~1" -Warmup 20 -Runs 600 -CooldownSeconds 1
goto DONE

:RUN_DEFAULT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Runner-GameSR-XDNA.ps1" -Warmup 20 -Runs 600 -CooldownSeconds 1

:DONE
set ERR=%ERRORLEVEL%
echo.
if "%ERR%"=="0" (
  echo [OK] Architecture Sweep finished.
) else (
  echo [FAIL] Architecture Sweep exited with code %ERR%.
)
echo.
pause
exit /b %ERR%
