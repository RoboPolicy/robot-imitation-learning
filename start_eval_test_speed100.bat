@echo off
rem Step 2a: the two worst layouts at 100%% speed (demo speed), 5 trials each.
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (best from step 1, e.g. 100000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run redbox_pool_v2 --checkpoint %CKPT% --replan 2 --speed 100 --protocol --trials 5 --tag speed100 --conditions red-black-blue black-blue-red --dataset redbox_30fps redbox_redonly_30fps
