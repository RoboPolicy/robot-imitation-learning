@echo off
rem Full evaluation: 7 conditions x 20 trials at 50%% speed, replan 2. Re-running resumes.
set "RUN=redbox_pool_v3"
set /p "RUN=Training run (Enter = redbox_pool_v3, or type redbox_pool_v2): "
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (e.g. 110000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run %RUN% --checkpoint %CKPT% --replan 2 --protocol --dataset redbox_30fps redbox_redonly_30fps redbox_redtop_30fps
