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
echo This creates a separate .venv-quark folder.
echo It does not modify the CUDA training environment in .venv.
echo The download can take several minutes.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [FAIL] Training environment .venv was not found.
  echo Copy this v1.7.1 package into the installed GameSR project first.
  pause
  exit /b 2
)

if not exist ".venv-quark\Scripts\python.exe" (
  echo [1/5] Creating isolated AMD Quark environment...
  ".venv\Scripts\python.exe" -m venv ".venv-quark"
  if errorlevel 1 goto :fail
) else (
  echo [1/5] Existing .venv-quark found.
)

echo [2/5] Updating pip...
".venv-quark\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/5] Installing PyTorch 2.11 CPU required by the AMD Quark wheel...
".venv-quark\Scripts\python.exe" -m pip install --upgrade "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [4/5] Installing AMD Quark 0.12.post1 CPU wheel and compatible ONNX tools...
".venv-quark\Scripts\python.exe" -m pip install --upgrade --extra-index-url https://pypi.amd.com/quark/cpu/simple -r quark_requirements.txt
if errorlevel 1 goto :fail

echo [5/5] Verifying Ryzen AI quantizer...
".venv-quark\Scripts\python.exe" check_quark_env.py
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
