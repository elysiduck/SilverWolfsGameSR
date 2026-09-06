@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe smoke_test.py
set CODE=%ERRORLEVEL%
echo.
pause
exit /b %CODE%
