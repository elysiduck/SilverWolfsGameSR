@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call GameSR_Train.cmd --model quality
exit /b %ERRORLEVEL%
