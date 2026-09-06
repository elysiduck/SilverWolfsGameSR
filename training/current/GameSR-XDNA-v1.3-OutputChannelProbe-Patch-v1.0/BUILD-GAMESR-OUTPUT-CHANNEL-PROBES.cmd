@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo GameSR Output Channel Probe Builder v1.0
echo Fixed body: C16-L4   Outputs: O3,O4,O6,O8
echo ============================================================
echo.
set "PYEXE="
if defined GAMESR_QUARK_PYTHON if exist "%GAMESR_QUARK_PYTHON%" set "PYEXE=%GAMESR_QUARK_PYTHON%"
if not defined PYEXE if exist "%~dp0.venv-quark\Scripts\python.exe" set "PYEXE=%~dp0.venv-quark\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"
echo Python: %PYEXE%
echo.
"%PYEXE%" "%~dp0tools\generate_output_channel_probes.py" --root "%~dp0" --outputs 3,4,6,8 --skip-existing
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" (
  echo [FAIL] Builder failed with code %ERR%.
  echo Use the same Python environment that successfully built the 105-model XINT8 sweep ^(torch + onnx + amd-quark^).
  echo You can set GAMESR_QUARK_PYTHON to the full path of that python.exe.
) else (
  echo [OK] Builder finished. Copy the new GameSR-NPU-Output-Channel-Probe-Models-*.zip to the handheld benchmark folder.
)
pause
exit /b %ERR%
