@echo off
rem Red at the bottom (black-blue-red, blue-black-red) up to 20 trials each, in the full protocol folder. Re-running resumes.
set "RUN=redbox_pool_v3"
set /p "RUN=Training run (Enter = redbox_pool_v3, or type redbox_pool_v2): "
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (e.g. 110000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run %RUN% --checkpoint %CKPT% --replan 2 --protocol --conditions black-blue-red blue-black-red --dataset redbox_30fps redbox_redonly_30fps redbox_redtop_30fps
