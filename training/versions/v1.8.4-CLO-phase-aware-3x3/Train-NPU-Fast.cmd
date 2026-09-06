@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call GameSR_Train.cmd --model fast
exit /b %ERRORLEVEL%
