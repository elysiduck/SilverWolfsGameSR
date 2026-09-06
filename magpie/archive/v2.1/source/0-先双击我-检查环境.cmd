@echo off
setlocal
cd /d "%~dp0"
title Magpie SESR Environment Check
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Check-Environment.ps1"
echo.
echo This window will stay open. You can close it manually.
echo Please send Environment-Check-Result.txt back to me.
echo.
pause
cmd.exe /k
