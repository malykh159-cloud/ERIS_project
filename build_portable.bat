@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" exit /b 1
powershell.exe -NoProfile -Command "$taskTarget = [IO.Path]::GetFullPath((Join-Path (Get-Location) 'dist\EISArcFit\EISArcFit.exe')); $taskActive = Get-Process EISArcFit -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $taskTarget }; if ($taskActive) { Write-Host 'Close EISArcFit before building. The running application locks portable files.'; exit 1 }; exit 0"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --windowed --onedir --name EISArcFit --paths src --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets --exclude-module PySide6.QtWebEngineQuick --exclude-module PySide6.QtPdf --exclude-module PySide6.QtPdfWidgets --exclude-module pandas --exclude-module pytest --exclude-module tkinter --exclude-module matplotlib src\main.py
if errorlevel 1 exit /b 1
if not exist "dist\EISArcFit\EISArcFit.exe" exit /b 1
echo Portable build: dist\EISArcFit\EISArcFit.exe
