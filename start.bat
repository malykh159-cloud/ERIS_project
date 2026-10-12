@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "src\main.py"
if errorlevel 1 pause
