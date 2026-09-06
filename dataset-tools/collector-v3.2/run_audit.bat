@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call install.bat
".venv\Scripts\python.exe" dataset_audit.py
pause
