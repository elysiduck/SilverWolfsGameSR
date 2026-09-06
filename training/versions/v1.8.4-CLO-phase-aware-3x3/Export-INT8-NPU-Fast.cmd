@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call GameSR_Export.cmd tiny-npu-fast
if errorlevel 1 exit /b %ERRORLEVEL%
call Build-INT8.cmd tiny-npu-fast
exit /b %ERRORLEVEL%
