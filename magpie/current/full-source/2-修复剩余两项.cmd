@echo off
setlocal
cd /d "%~dp0"
title Repair Magpie SESR Build Environment
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Repair-Build-Environment.ps1"
echo.
echo The repair window has finished. This window will stay open.
echo.
pause
cmd.exe /k
