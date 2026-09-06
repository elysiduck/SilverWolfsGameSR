@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [FAIL] 没有找到训练环境 .venv
    echo 请把本补丁覆盖到原来的 gamesr-local-training 文件夹后再运行。
    echo.
    pause
    exit /b 2
)
echo ============================================================
echo GameSR 一键训练 - 自定义 C / L / O
echo C = 中间通道数，L = 卷积层数，O = NPU 输出通道数
echo 首次建议：C16、L4、O4；兼容旧方案请选择 O3
echo ============================================================
".venv\Scripts\python.exe" pipeline.py --model custom %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [FAIL] 训练流程停止，错误代码 %CODE%。
) else (
    echo [OK] 训练、导出与打包全部完成。
)
echo.
pause
exit /b %CODE%
