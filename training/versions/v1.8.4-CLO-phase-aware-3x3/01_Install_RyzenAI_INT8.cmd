@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ================================================================
echo  GameSR v1.7.1 - Install AMD Quark Ryzen AI INT8 environment
echo ================================================================
echo.
echo This creates or reuses .venv-quark in the parent folder.
echo Multiple GameSR code folders can share it.
echo The download can take several minutes.
echo.

call "%~dp0Resolve-GameSR-Environment.cmd" Training
if errorlevel 1 (
  echo [FAIL] Training environment was not found here or in a parent folder.
  echo Run setup.ps1 once first.
  pause
  exit /b 2
)
set "TRAIN_PYTHON=%GAMESR_RESOLVED_PYTHON%"
for %%D in ("%~dp0..") do set "SHARED_ROOT=%%~fD"
set "QUARK_DIR=%SHARED_ROOT%\.venv-quark"
set "QUARK_PYTHON=%QUARK_DIR%\Scripts\python.exe"

call "%~dp0Resolve-GameSR-Environment.cmd" Quark >nul 2>nul
if not errorlevel 1 (
  set "QUARK_PYTHON=%GAMESR_RESOLVED_PYTHON%"
  for %%D in ("%GAMESR_RESOLVED_PYTHON%\..\..") do set "QUARK_DIR=%%~fD"
)

if not exist "%QUARK_PYTHON%" (
  echo [1/5] Creating isolated AMD Quark environment...
  "%TRAIN_PYTHON%" -m venv "%QUARK_DIR%"
  if errorlevel 1 goto :fail
) else (
  echo [1/5] Existing .venv-quark found.
)

echo [2/5] Updating pip...
"%QUARK_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/5] Installing PyTorch 2.11 CPU required by the AMD Quark wheel...
"%QUARK_PYTHON%" -m pip install --upgrade "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [4/5] Installing AMD Quark 0.12.post1 CPU wheel and compatible ONNX tools...
"%QUARK_PYTHON%" -m pip install --upgrade --extra-index-url https://pypi.amd.com/quark/cpu/simple -r quark_requirements.txt
if errorlevel 1 goto :fail

echo [5/5] Verifying Ryzen AI quantizer...
"%QUARK_PYTHON%" check_quark_env.py
if errorlevel 1 goto :fail

echo.
echo [OK] AMD Quark INT8 environment is ready.
echo [NEXT] Double-click Build-INT8.cmd.
echo.
pause
exit /b 0

:fail
echo.
echo [FAIL] AMD Quark installation failed.
echo Keep this window open and send a screenshot of the last error.
echo.
pause
exit /b 1
