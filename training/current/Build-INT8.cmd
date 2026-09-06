@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv-quark\Scripts\python.exe" (
  echo [FAIL] AMD Quark environment .venv-quark was not found.
  echo Double-click 01_Install_RyzenAI_INT8.cmd first.
  pause
  exit /b 2
)
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=tiny-npu-fast"
set "CAL=%~2"
if "%CAL%"=="" set "CAL=dataset\train.csv"
set "IN=models\%MODEL%-720p-to-1080p.onnx"
set "OUT=models\%MODEL%-720p-to-1080p-int8-qdq.onnx"
set "CORE_FLAG="
if /I "%MODEL%"=="tiny-npu-fast" set "CORE_FLAG=--require-core-only"
if /I "%MODEL%"=="tiny-npu-quality" set "CORE_FLAG=--require-core-only"
if not exist "%IN%" (
  echo [FAIL] Float ONNX not found: %IN%
  pause
  exit /b 2
)
if not exist "%CAL%" (
  echo [FAIL] Calibration source not found: %CAL%
  echo        Use: Build-INT8.cmd %MODEL% "D:\path\to\clean"
  echo        or prepare dataset\train.csv first.
  pause
  exit /b 2
)
echo [INT8] Building AMD Ryzen AI XINT8/QDQ model (v1.7.1)...
echo [INT8] This uses AMD Quark and can take several minutes.
".venv-quark\Scripts\python.exe" int8_ptq.py --model "%IN%" --output "%OUT%" --calibration "%CAL%" --count 128 --config XINT8 %CORE_FLAG%
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" echo [OK] Ryzen AI XINT8/QDQ build completed: %OUT%
if not "%CODE%"=="0" echo [FAIL] INT8 build failed. Exit code %CODE%
echo.
pause
exit /b %CODE%
