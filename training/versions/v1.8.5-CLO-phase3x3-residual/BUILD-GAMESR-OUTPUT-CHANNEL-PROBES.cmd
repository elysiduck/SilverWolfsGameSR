@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem IMPORTANT: %~dp0 ends with a backslash. Passing it directly as a quoted
rem Python argument can make Windows' argv parser treat the closing quote as
rem escaped. Use %CD% instead, which has no trailing backslash.
set "ROOT=%CD%"

echo ============================================================
echo GameSR Output Channel Probe Builder v1.0.1 FIX
echo Fixed body: C16-L4   Outputs: O3,O4,O6,O8
echo ============================================================
 echo.
echo Builder root: %ROOT%

set "PYEXE="
if defined GAMESR_QUARK_PYTHON if exist "%GAMESR_QUARK_PYTHON%" set "PYEXE=%GAMESR_QUARK_PYTHON%"
if not defined PYEXE if exist "%ROOT%\.venv-quark\Scripts\python.exe" set "PYEXE=%ROOT%\.venv-quark\Scripts\python.exe"
if not defined PYEXE if exist "%ROOT%\.venv\Scripts\python.exe" set "PYEXE=%ROOT%\.venv\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"

echo Python: %PYEXE%
echo.

if not exist "%ROOT%\tools\generate_output_channel_probes.py" (
  echo [FAIL] Missing builder script: %ROOT%\tools\generate_output_channel_probes.py
  pause
  exit /b 2
)
if not exist "%ROOT%\tools\int8_ptq.py" (
  echo [FAIL] Missing bundled quantizer: %ROOT%\tools\int8_ptq.py
  echo Re-extract the v1.0.1 FIX patch; this file is included in the patch.
  pause
  exit /b 2
)

"%PYEXE%" "%ROOT%\tools\generate_output_channel_probes.py" --root "%ROOT%" --outputs 3,4,6,8 --skip-existing
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo [FAIL] Builder failed with code %ERR%.
  echo Use the same Python environment that successfully built the 105-model XINT8 sweep ^(torch + onnx + amd-quark^).
  echo You can set GAMESR_QUARK_PYTHON to the full path of that python.exe.
) else (
  echo [OK] Builder finished.
  echo Copy the new GameSR-NPU-Output-Channel-Probe-Models-*.zip to the handheld benchmark folder.
)
pause
exit /b %ERR%
