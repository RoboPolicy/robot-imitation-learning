@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  echo Please run from PowerShell with your correct Python path:
  echo ^& "C:\path\to\python.exe" ".\scripts\capture_label_yolo.py"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\capture_label_yolo.py"
if errorlevel 1 echo [ERROR] Capture failed. Close other camera apps and check camera indexes.
pause
