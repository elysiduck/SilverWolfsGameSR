@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo [FAIL] .venv\Scripts\python.exe not found.
  pause
  exit /b 2
)
".venv\Scripts\python.exe" -c "from gamesr.streaming import PersistentValidationLoader; from gamesr.data import information_weight_per_image; import train; print('[OK] GameSR v1.8.2 imports verified'); print('[OK] PersistentValidationLoader + Information-Aware L1 available')"
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" echo [FAIL] v1.8.2 verification failed.
pause
exit /b %CODE%
