@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\lerobot-env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Cannot find Python: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" ".\scripts\record_episode.py" %*
if errorlevel 1 echo [ERROR] Recording stopped with an error. Support the arm and read the message above.
pause
