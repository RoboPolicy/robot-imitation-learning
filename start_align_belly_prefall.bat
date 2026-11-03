@echo off
rem Align belly to the pose used for the red-only/red-top demos and the 100k evaluation (last trial before the camera fell).
rem Place the boxes as red top, blue middle, black bottom first, like that trial.
call "%~dp0start_align_camera.bat" --camera belly --reference-trial "%~dp0logs\eval\redbox_pool_v3_110000_quick5\trial_175446_50pct.mp4" %*
