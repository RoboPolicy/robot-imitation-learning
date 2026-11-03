@echo off
rem Step 2b: the two worst layouts with a new prediction every step (0.2 s at 100%%), 5 trials each.
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (best from step 1, e.g. 100000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run redbox_pool_v2 --checkpoint %CKPT% --replan 1 --protocol --trials 5 --tag replan1 --conditions red-black-blue black-blue-red --dataset redbox_30fps redbox_redonly_30fps
