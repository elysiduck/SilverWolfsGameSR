@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

call "%~dp0Resolve-GameSR-Environment.cmd" Training
if errorlevel 1 goto :NO_VENV
set "PYTHON=%GAMESR_RESOLVED_PYTHON%"

echo ============================================================
echo GameSR one-click training - custom C / L / O
echo C = network width
echo L = total convolution layers
echo O = NPU output channels
echo Recommended first test: C16, L4, O4
echo Use O3 for the original compatible path
echo ============================================================

echo Shared training Python: %PYTHON%
"%PYTHON%" "pipeline.py" --model custom %*
set "CODE=%ERRORLEVEL%"
goto :RESULT

:NO_VENV
echo.
echo [FAIL] Training environment was not found here or in a parent folder.
echo Put .venv beside this code folder, or run setup.ps1 once.
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
