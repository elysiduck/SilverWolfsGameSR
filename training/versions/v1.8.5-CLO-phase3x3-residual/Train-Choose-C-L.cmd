@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [FAIL] GameSR .venv was not found in this folder.
    echo Install/use the existing GameSR local training environment first.
    echo.
    pause
    exit /b 2
)
set "PYTHON=.venv\Scripts\python.exe"
echo ============================================================
echo GameSR v1.8.2 Late-Polish / Information-Aware Training
echo Choose C (width), L (Conv layers), and O (output channels)
echo Recommended first test: C16-L4-O4. Original compatible mode: O3
echo v1.8.2: Late Polish + Information-aware L1 + Detail Hard Mining + Masked HF
echo Remote dataset: train cache + validation tensor cache, auto RAM sizing
echo ============================================================
"%PYTHON%" pipeline.py --model custom %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [FAIL] GameSR custom training stopped with exit code %CODE%.
) else (
    echo [OK] GameSR custom training finished.
)
echo.
pause
exit /b %CODE%
