@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo [FAIL] GameSR .venv not found.
    pause
    exit /b 2
)
set "PYTHON=.venv\Scripts\python.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=tiny-npu-fast"
if /I "%MODEL%"=="tiny-npu-fast" set "CONFIG=configs\tiny-npu-fast.yaml"
if /I "%MODEL%"=="tiny-npu-quality" set "CONFIG=configs\tiny-npu-quality.yaml"
if /I "%MODEL%"=="tiny-c8" set "CONFIG=configs\tiny-c8.yaml"
if /I "%MODEL%"=="balanced-c12" set "CONFIG=configs\balanced-c12.yaml"
if /I "%MODEL%"=="quality-c16" set "CONFIG=configs\quality-c16.yaml"
if not defined CONFIG (
    echo Usage: GameSR_Export.cmd [tiny-npu-fast^|tiny-npu-quality^|tiny-c8^|balanced-c12^|quality-c16]
    pause
    exit /b 2
)
set "CKPT=outputs\%MODEL%\best.pth"
set "OUT=models\%MODEL%-720p-to-1080p.onnx"
if not exist "%CKPT%" (
    echo [FAIL] Checkpoint not found: %CKPT%
    pause
    exit /b 3
)
echo ========================================================================
echo Exporting %MODEL% to ONNX
echo ========================================================================
"%PYTHON%" export_onnx.py --config "%CONFIG%" --checkpoint "%CKPT%" --output "%OUT%" --use-ema
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
    echo [OK] ONNX export completed: %OUT%
) else (
    echo [FAIL] ONNX export failed. Your checkpoint is unchanged; retry this script.
)
echo.
pause
exit /b %CODE%
