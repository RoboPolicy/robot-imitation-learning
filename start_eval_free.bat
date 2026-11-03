@echo off
rem Free test: no layout guide and no verdict table. Place the boxes anywhere in the trained area.
set "RUN=redbox_pool_v3"
set /p "RUN=Training run (Enter = redbox_pool_v3, or type redbox_pool_v2): "
set "CKPT=%~1"
if "%CKPT%"=="" set /p "CKPT=Checkpoint number (e.g. 110000): "
if "%CKPT%"=="" exit /b 1
call "%~dp0start_eval.bat" --run %RUN% --checkpoint %CKPT% --replan 2 --tag free --dataset redbox_30fps redbox_redonly_30fps redbox_redtop_30fps
