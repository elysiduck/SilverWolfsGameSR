@echo off
setlocal
cd /d "%~dp0"
title Install AppX and WinUI Build Tools
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Appx-Build-Tools.ps1"
echo.
echo This window will stay open. You can close it manually.
echo.
pause
cmd.exe /k
