@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\capture_label_yolo.py" --relabel %*
if errorlevel 1 echo [ERROR] Relabel failed.
pause
