@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call GameSR_Export.cmd tiny-npu-quality
if errorlevel 1 exit /b %ERRORLEVEL%
call Build-INT8.cmd tiny-npu-quality
exit /b %ERRORLEVEL%
