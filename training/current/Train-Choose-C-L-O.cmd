@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" goto :NO_VENV

echo ============================================================
echo GameSR one-click training - custom C / L / O
echo C = network width
echo L = total convolution layers
echo O = NPU output channels
echo Recommended first test: C16, L4, O4
echo Use O3 for the original compatible path
echo ============================================================

".venv\Scripts\python.exe" "pipeline.py" --model custom %*
set "CODE=%ERRORLEVEL%"
goto :RESULT

:NO_VENV
echo.
echo [FAIL] Training environment .venv was not found.
echo Copy this package over the original gamesr-local-training folder.
echo.
pause
exit /b 2

:RESULT
echo.
if "%CODE%"=="0" goto :SUCCESS
echo [FAIL] Training stopped. Exit code: %CODE%
goto :END

:SUCCESS
echo [OK] Training, export and deployment packaging completed.

:END
echo.
pause
exit /b %CODE%
