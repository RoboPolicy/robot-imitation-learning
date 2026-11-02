@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\align_top_camera.py"
if errorlevel 1 echo [ERROR] Alignment tool failed. Close other camera apps and check the message above.
pause
