@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  echo [FAIL] .venv not found. This command is PC-side only.
  pause
  exit /b 2
)
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=tiny-npu-fast"
set "IN=models\%MODEL%-720p-to-1080p.onnx"
set "OUT=models\%MODEL%-720p-to-1080p-fp16.onnx"
".venv\Scripts\python.exe" fp16_convert.py --model "%IN%" --output "%OUT%"
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" echo [OK] FP16 candidate created: %OUT%
pause
exit /b %CODE%
