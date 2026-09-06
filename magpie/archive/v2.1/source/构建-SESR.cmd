@echo off
setlocal
cd /d "%~dp0"
title Build Magpie SESR-M7 NPU
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Build-SESR.ps1"
echo.
echo This window will stay open. You can close it manually.
echo.
pause
cmd.exe /k
