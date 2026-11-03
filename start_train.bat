@echo off
setlocal
cd /d "%~dp0"
set "TRAIN_EXE=%USERPROFILE%\lerobot-env\Scripts\lerobot-train.exe"
set "RUN=redbox_rgb_v1"
set "HF_DATASETS_CACHE=%TEMP%\h33"
set "HF_HUB_OFFLINE=1"
set "PYTHONIOENCODING=utf-8"
if exist ".\outputs\act\%RUN%\checkpoints\last\pretrained_model\train_config.json" (
  echo [RESUME] Continuing %RUN% from its last checkpoint.
  "%TRAIN_EXE%" --config_path=".\outputs\act\%RUN%\checkpoints\last\pretrained_model\train_config.json" --resume=true >> ".\logs\train_%RUN%.log" 2>&1
) else (
  "%TRAIN_EXE%" --dataset.repo_id=local/redbox_5fps_split6 --dataset.root=./datasets/lerobot/redbox_5fps_split6 --dataset.video_backend=pyav --policy.type=act --policy.chunk_size=10 --policy.n_action_steps=10 --policy.device=cuda --policy.push_to_hub=false --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=200 --num_workers=4 --seed=1000 --job_name=%RUN% --wandb.enable=false --output_dir=./outputs/act/%RUN% >> ".\logs\train_%RUN%.log" 2>&1
)
if errorlevel 1 (
  echo [ERROR] Training stopped. See logs\train_%RUN%.log
) else (
  echo [DONE] Training finished. See outputs\act\%RUN%\checkpoints
)
pause
