@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
echo Installing or updating the shared GameSR training environment...
echo The environment will be stored in the parent folder.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" echo [OK] Shared training environment is ready.
if not "%CODE%"=="0" echo [FAIL] Setup stopped with exit code %CODE%.
pause
exit /b %CODE%
