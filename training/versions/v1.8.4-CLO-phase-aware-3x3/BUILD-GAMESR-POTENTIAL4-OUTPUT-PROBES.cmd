@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"

echo ============================================================
echo GameSR Potential-4 Output Probe Builder v1.1.0 - PC ONLY
echo Backbones: C8-L5 / C16-L4 / C24-L3 / C32-L3
echo Outputs  : O3 / O4 / O6 / O8  ^(16 models^)
echo ============================================================
echo.

set "PYEXE="
if defined GAMESR_QUARK_PYTHON if exist "%GAMESR_QUARK_PYTHON%" set "PYEXE=%GAMESR_QUARK_PYTHON%"
if not defined PYEXE if exist "%ROOT%\.venv-quark\Scripts\python.exe" set "PYEXE=%ROOT%\.venv-quark\Scripts\python.exe"
if not defined PYEXE if exist "%ROOT%\.venv\Scripts\python.exe" set "PYEXE=%ROOT%\.venv\Scripts\python.exe"
if not defined PYEXE set "PYEXE=python"

echo Root  : %ROOT%
echo Python: %PYEXE%
echo.

if not exist "%ROOT%\tools\generate_potential4_output_probes.py" (
  echo [FAIL] Missing generator.
  pause
  exit /b 2
)
if not exist "%ROOT%\tools\int8_ptq.py" (
  echo [FAIL] Missing bundled GameSR v1.7.1 quantizer.
  pause
  exit /b 2
)
if not exist "%ROOT%\config\potential_backbones.json" (
  echo [FAIL] Missing candidate config.
  pause
  exit /b 2
)

"%PYEXE%" "%ROOT%\tools\generate_potential4_output_probes.py" --root "%ROOT%" --outputs 3,4,6,8 --skip-existing
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo [FAIL] Builder failed with code %ERR%.
  echo Use the same Quark Python environment that successfully built the 105-model XINT8 sweep.
  echo You can set GAMESR_QUARK_PYTHON to the full path of that python.exe.
) else (
  echo [OK] Builder finished.
  echo The generated GameSR-NPU-Potential4-Output-Probe-Models-16-*.zip is the ONLY generated file to copy to the handheld.
)
pause
exit /b %ERR%
