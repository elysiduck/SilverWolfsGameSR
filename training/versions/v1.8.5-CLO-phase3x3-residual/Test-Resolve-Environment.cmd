@echo off
call "%~dp0Resolve-GameSR-Environment.cmd" Training
if errorlevel 1 exit /b %errorlevel%
echo TRAIN=[%GAMESR_RESOLVED_PYTHON%]
call "%~dp0Resolve-GameSR-Environment.cmd" Quark
if errorlevel 1 exit /b %errorlevel%
echo QUARK=[%GAMESR_RESOLVED_PYTHON%]
exit /b 0
