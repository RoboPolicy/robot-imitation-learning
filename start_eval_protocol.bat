@echo off
rem Full evaluation: 7 conditions x 20 trials at 50%% speed, replan 2. Re-running resumes.
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (e.g. 100000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run redbox_pool_v2 --checkpoint %CKPT% --replan 2 --protocol --dataset redbox_30fps redbox_redonly_30fps
