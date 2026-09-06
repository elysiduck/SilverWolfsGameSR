@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo [FAIL] GameSR .venv not found.
    pause
    exit /b 2
)
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=tiny-c8"
set "FILE=models\%MODEL%-720p-to-1080p.onnx"
if not exist "%FILE%" (
    echo [FAIL] ONNX not found: %FILE%
    pause
    exit /b 3
)
.venv\Scripts\python.exe validate_onnx.py "%FILE%"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" echo [OK] ONNX validation passed.
if not "%CODE%"=="0" echo [FAIL] ONNX validation failed.
pause
exit /b %CODE%
