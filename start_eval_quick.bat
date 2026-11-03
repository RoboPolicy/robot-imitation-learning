@echo off
rem Step 1, checkpoint comparison: 7 conditions x 5 trials at 50%% speed, replan 2.
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (e.g. 100000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run redbox_pool_v2 --checkpoint %CKPT% --replan 2 --protocol --trials 5 --tag quick5 --dataset redbox_30fps redbox_redonly_30fps
