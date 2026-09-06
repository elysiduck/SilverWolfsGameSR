@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo GameSR XDNA Architecture Benchmark v1.3
echo ========================================
echo This version restores the original GameSR Benchmark v2.4.1 runner.
echo It first runs two diagnostics, then benchmarks five 720p XINT8 probes.
echo First-time NPU compilation can take several minutes per model.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-XDNAArchitectureBenchmark.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
echo Benchmark exit code: %CODE%
echo Results are in results\
echo Please zip the entire results folder and send it back.
echo.
pause
exit /b %CODE%
