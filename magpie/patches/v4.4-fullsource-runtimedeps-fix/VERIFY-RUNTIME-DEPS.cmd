@echo off
setlocal
cd /d "%~dp0"
set "BASE=%CD%"
set "DIR=%BASE%\third_party\sesr\onnx\bin"
echo Checking Magpie GameSR source runtime dependencies...
set ERR=0
for %%F in (onnxruntime.dll onnxruntime_vitisai_ep.dll dyn_dispatch_core.dll aiecompiler_client.dll) do (
  if exist "%DIR%\%%F" (
    echo [PASS] %%F
  ) else (
    echo [MISSING] %%F
    set ERR=1
  )
)
if "%ERR%"=="0" (
  echo.
  echo [PASS] Required runtime files are present.
) else (
  echo.
  echo [FAIL] One or more files are missing.
)
pause
exit /b %ERR%
