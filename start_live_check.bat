@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\live_check_yolo.py" %*
if errorlevel 1 echo [ERROR] Live check failed. Close other camera apps and check camera indexes.
pause
