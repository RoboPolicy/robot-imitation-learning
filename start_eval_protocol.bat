@echo off
call "%~dp0start_eval.bat" --run redbox_pool_v2 --checkpoint 065000 --replan 2 --protocol %*
