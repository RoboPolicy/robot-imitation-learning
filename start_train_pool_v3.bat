@echo off
setlocal
cd /d "%~dp0"
set "TRAIN_EXE=%USERPROFILE%\lerobot-env\Scripts\lerobot-train.exe"
set "RUN=redbox_pool_v3"
set "STEPS=200000"
set "HF_DATASETS_CACHE=%TEMP%\h33"
set "HF_HUB_OFFLINE=1"
set "PYTHONIOENCODING=utf-8"
if exist ".\outputs\act\%RUN%\checkpoints\last\pretrained_model\train_config.json" (
  echo [RESUME] Continuing %RUN% from its last checkpoint.
  "%TRAIN_EXE%" --config_path=".\outputs\act\%RUN%\checkpoints\last\pretrained_model\train_config.json" --resume=true --steps=%STEPS% >> ".\logs\train_%RUN%.log" 2>&1
) else (
  echo [START] Continuing redbox_pool_v2 step 100000 with the red-top demos added.
  "%TRAIN_EXE%" --config_path=".\outputs\act\redbox_pool_v2\checkpoints\100000\pretrained_model\train_config.json" --resume=true --dataset.repo_id=local/%RUN% --dataset.root=./datasets/lerobot/%RUN% --steps=%STEPS% --save_freq=5000 --job_name=%RUN% --output_dir=./outputs/act/%RUN% >> ".\logs\train_%RUN%.log" 2>&1
)
if errorlevel 1 (
  echo [ERROR] Training stopped. See logs\train_%RUN%.log
) else (
  echo [DONE] Training finished. See outputs\act\%RUN%\checkpoints
)
pause
