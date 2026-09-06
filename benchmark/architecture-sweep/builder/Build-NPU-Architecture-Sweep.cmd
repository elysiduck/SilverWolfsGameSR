@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ================================================================
echo  GameSR NPU Architecture Sweep Builder v1.0
echo  Runs on the TRAINING LAPTOP - no Python is needed on handheld
echo ================================================================
echo.

if not exist ".venv-quark\Scripts\python.exe" (
  echo [FAIL] Existing GameSR AMD Quark environment was not found:
  echo        %CD%\.venv-quark\Scripts\python.exe
  echo.
  echo This Builder is designed to plug into GameSR-training-code-v1.7.1.
  echo If needed, run 01_Install_RyzenAI_INT8.cmd once in this folder.
  echo.
  pause
  exit /b 2
)
if not exist "int8_ptq.py" (
  echo [FAIL] int8_ptq.py was not found. Extract this patch into the
  echo        GameSR-training-code-v1.7.1 project root.
  pause
  exit /b 2
)
if not exist "gamesr\model.py" (
  echo [FAIL] gamesr\model.py was not found. Wrong folder.
  pause
  exit /b 2
)

echo [ENV] Reusing existing .venv-quark environment...
".venv-quark\Scripts\python.exe" check_quark_env.py
if errorlevel 1 (
  echo.
  echo [FAIL] The existing AMD Quark environment did not pass verification.
  echo        Run 01_Install_RyzenAI_INT8.cmd to repair it.
  pause
  exit /b 2
)

echo.
echo Select sweep grid:
echo   1. FULL     - C4..C32 step 2 x L2..L8 = 105 models  [recommended]
echo   2. STANDARD - C4,6,8,10,12,14,16,20,24,28,32 x L2,3,4,5,6,8 = 66 models
echo   3. CUSTOM   - enter your own channel/layer lists
echo.
set /p "MODE=Choice [1]: "
if "%MODE%"=="" set "MODE=1"

if "%MODE%"=="1" (
  set "CHANNELS=4,6,8,10,12,14,16,18,20,22,24,26,28,30,32"
  set "LAYERS=2,3,4,5,6,7,8"
) else if "%MODE%"=="2" (
  set "CHANNELS=4,6,8,10,12,14,16,20,24,28,32"
  set "LAYERS=2,3,4,5,6,8"
) else if "%MODE%"=="3" (
  set /p "CHANNELS=Channels comma list [8,16,24,32]: "
  if "!CHANNELS!"=="" set "CHANNELS=8,16,24,32"
  set /p "LAYERS=Layers comma list [2,3,4,5,6,7,8]: "
  if "!LAYERS!"=="" set "LAYERS=2,3,4,5,6,7,8"
) else (
  echo [FAIL] Invalid choice.
  pause
  exit /b 2
)

echo.
echo Calibration count controls only XINT8 probe calibration, not model quality.
echo 8 is recommended for architecture benchmarking.
set /p "CALCOUNT=Calibration images [8]: "
if "%CALCOUNT%"=="" set "CALCOUNT=8"

echo.
echo [BUILD] Channels: %CHANNELS%
echo [BUILD] Layers  : %LAYERS%
echo [BUILD] Cal count: %CALCOUNT%
echo.

".venv-quark\Scripts\python.exe" tools\architecture_sweep_builder.py ^
  --root "%CD%" ^
  --channels "%CHANNELS%" ^
  --layers "%LAYERS%" ^
  --cal-count %CALCOUNT% ^
  --skip-existing ^
  --package
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo ================================================================
  echo [OK] Architecture Sweep model build completed.
  echo [OUT] architecture_sweep\handheld_payload\
  echo [ZIP] architecture_sweep\exports\GameSR-NPU-Architecture-Sweep-Models-*.zip
  echo.
  echo The generated ZIP is what you will copy to the handheld later.
  echo The handheld does NOT need Python / PyTorch / AMD Quark.
  echo ================================================================
) else (
  echo [FAIL] Builder stopped with exit code %CODE%.
  echo Existing successfully built models are kept; run again to resume.
)
echo.
pause
exit /b %CODE%
