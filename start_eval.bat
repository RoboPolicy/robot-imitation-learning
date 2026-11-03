@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
set "HF_HUB_OFFLINE=1"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\eval_act.py" %*
if errorlevel 1 echo [ERROR] Evaluation stopped with an error. Support the arm and read the message above.
pause
