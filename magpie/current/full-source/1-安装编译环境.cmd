@echo off
setlocal
cd /d "%~dp0"
title Install Magpie SESR Build Environment
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Build-Environment.ps1"
echo.
echo This window will stay open. You can close it manually.
echo.
pause
cmd.exe /k
