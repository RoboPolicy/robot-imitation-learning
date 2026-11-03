"""SO-ARM101 ACT 평가: 저장된 체크포인트로 팔로워를 움직입니다. 학습과 데이터는 바꾸지 않습니다.
배치 (토크 꺼짐) -> Enter 시작 자세로 천천히 이동 -> Space 속도 변경, Enter 실행 -> Space 시도 끝, Esc 즉시 멈춤.
시도 후 Enter로 시작 자세로 돌아갑니다. 시작 자세에서 Esc 두 번이면 토크를 끄고 끝납니다. 비상시에는 전원을 차단하세요.
모델 출력은 시연의 관절 범위와 속도 안으로 제한하고, 지연이 시연보다 크게 벌어지면 멈춥니다.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import permutations
import json
from pathlib import Path
import queue
import tempfile
import threading
import time

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.utils.robot_utils import precise_sleep

from record_episode import (CONTROL_FPS, DATASET_ROOT, FINISH_CONFIRM_SECONDS, JOINTS, LOG_DIR, PROJECT_DIR,
                            TASK, VK_N, VK_R, VK_Y, CommGuard, FakeFollower, Keys, Preview, Status, build_robot,
                            connect_without_torque, enable_holding, esc_pressed, hold_here, positions, press_when,
                            red_only_guide, shutdown)
from replay_episode import GREEN, MODEL, ORANGE, RED, SPEEDS, judge_top, move_to, sample_action, to_action

POLICY_FPS = 5
DEFAULT_SPEED = 1  # SPEEDS의 번호. 처음에는 50%로 시작합니다.
MAX_POLICY_SECONDS = 25.0  # 100% 속도 기준입니다. 시연은 15초 이내입니다.
RANGE_MARGIN = 3.0
LAG_FACTOR, LAG_MARGIN = 1.5, 5.0
MAX_STEP_SECONDS = 0.05
GPU_MEMORY_FRACTION = 0.12  # 학습 중에도 돌릴 수 있도록 이 프로세스의 GPU 메모리를 제한합니다.
HW_LOW, HW_HIGH = np.array([-100.0] * 5 + [0.0]), np.full(6, 100.0)
TRIAL_FRAME_SECONDS = 0.1
TILE_SIZE = (320, 240)
CAMERA_TOLERANCE = 8.0
FINGER_RATIO = 0.15  # 닫힌 그리퍼 0.60~0.67, 벌린 그리퍼 0.29, 다른 카메라 0.00
BASKET_WIDE = 1.2
TRAIN_DATASET = "redbox_5fps_split6"
NO_WRIST_CAMERAS = {"top_cam": 0, "belly_cam": 1}  # 손목 카메라가 빠지면 belly가 1번으로 당겨집니다.
RUNS_DIR = PROJECT_DIR / "outputs" / "act"
CONFIG_DIR = PROJECT_DIR / "config"
SLOTS = {"TOP": (450, 105), "MID": (455, 210), "BOT": (450, 345)}  # top 화면 좌표. 긴 변을 화면 세로로 놓습니다.
PROTOCOL_CONDITIONS = [("red_only", None)] + [("-".join(p), dict(zip(SLOTS, p))) for p in permutations(("red", "black", "blue"))]
PROTOCOL_TRIALS = 20
PROTOCOL_FIELDS = ["time", "condition", "trial", "layout", "outcome", "reason", "seconds", "speed", "replan", "checkpoint", "trial_file"]


def demo_limits(datasets):
    """시연에서 시작 자세, 관절 범위, 최대 속도, 최대 지연을 잽니다. 여러 시연 묶음을 합쳐서 잴 수 있습니다."""
    datasets = [datasets] if isinstance(datasets, str) else datasets
    folders = [f for dataset in datasets for f in sorted((DATASET_ROOT / dataset).glob("episode_*"))]
    if not folders:
        raise RuntimeError(f"시연이 없습니다: {datasets}")
    names = json.loads((folders[0] / "meta" / "info.json").read_text(encoding="utf-8"))["features"]["action"]["names"]
    if list(names) != [f"{j}.pos" for j in JOINTS]:
        raise RuntimeError(f"관절 순서가 다릅니다: {names}")
    actions, speeds, lags, starts = [], [], [], []
    for folder in folders:
        table = pq.read_table(next((folder / "data").rglob("*.parquet")), columns=["action", "observation.state"]).to_pydict()
        action, state = np.array(table["action"], dtype=np.float64), np.array(table["observation.state"], dtype=np.float64)
        actions.append(action)
        speeds.append(np.abs(np.diff(action, axis=0)) * 30)
        lags.append(np.abs(action - state))
        starts.append(state[0])
    actions = np.concatenate(actions)
    return {"start": np.median(starts, axis=0),
            "low": np.maximum(actions.min(axis=0) - RANGE_MARGIN, HW_LOW),
            "high": np.minimum(actions.max(axis=0) + RANGE_MARGIN, HW_HIGH),
            "max_speed": np.concatenate(speeds).max(axis=0),
            "lag": np.concatenate(lags).max(axis=0) * LAG_FACTOR + LAG_MARGIN}


def mean_image(camera):
    """학습 데이터의 평균 색으로 채운 화면입니다. 정규화하면 0에 가까워 모델에 정보를 주지 않습니다."""
    stats = json.loads((DATASET_ROOT / TRAIN_DATASET / "meta" / "stats.json").read_text(encoding="utf-8"))
    mean = np.array(stats[f"observation.images.{camera}"]["mean"], dtype=np.float64).ravel()
    return np.full((480, 640, 3), np.round(mean * 255), dtype=np.uint8)


def latest_checkpoint(run):
    """저장이 끝난 체크포인트 중 가장 최근 것을 고릅니다."""
    root = RUNS_DIR / run / "checkpoints"
    done = [p for p in root.iterdir() if p.name.isdigit() and (p / "training_state").is_dir()
            and (p / "pretrained_model" / "train_config.json").exists()] if root.exists() else []
    if not done:
        raise RuntimeError(f"저장된 체크포인트가 없습니다: {root}")
    return max(done, key=lambda p: int(p.name))


class ActPredictor:
    """관측 하나로 chunk 전체를 실제 관절값으로 예측합니다."""

    def __init__(self, checkpoint, device, substitutes=None):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.substitutes = substitutes or {}
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        path = str(checkpoint / "pretrained_model")
        self.policy = ACTPolicy.from_pretrained(path).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path,
                                                       preprocessor_overrides={"device_processor": {"device": device}})

    def __call__(self, snapshot):
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(snapshot["state"]).float(), "task": TASK}
        for name, image in {**snapshot["images"], **self.substitutes}.items():
            batch[f"observation.images.{name}"] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch))
            return self.post(chunk)[0].cpu().numpy().astype(np.float64)


class Planner(threading.Thread):
    """제어 루프를 막지 않도록 별도 스레드에서 추론합니다. 모터 통신은 제어 루프에서만 합니다."""

    def __init__(self, predict):
        super().__init__(daemon=True)
        self.predict = predict
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.busy = False
        self.error = None

    def request(self, snapshot, observed_at):
        self.busy = True
        self.jobs.put((snapshot, observed_at))

    def take(self):
        """끝난 예측이 있으면 (chunk, 관측 시각, 걸린 시간)을 돌려줍니다."""
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def drain(self):
        """이전 시도에서 남은 예측을 버립니다."""
        while self.busy and not self.error:
            if self.take() is None:
                time.sleep(0.005)

    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            snapshot, observed_at = job
            try:
                chunk = self.predict(snapshot)
                if not np.isfinite(chunk).all():
                    raise ValueError("모델 출력에 NaN 또는 무한대가 있습니다.")
            except Exception as exc:  # 제어 스레드에서 확인해 멈춥니다.
                self.error = exc
                return
            self.results.put((chunk, observed_at, time.perf_counter() - observed_at))

    def close(self):
        self.jobs.put(None)


class CameraCheck:
    """top, belly의 바구니 위치를 맞춘 기준과 비교합니다. 멈추지는 않고 화면에 알립니다."""

    def __init__(self, model, cameras):
        self.model, self.checks = model, []
        for short in ("top", "belly"):
            path = CONFIG_DIR / f"{short}_camera_reference.json"
            if model and f"{short}_cam" in cameras and path.exists():
                self.checks.append((short, cameras[f"{short}_cam"], json.loads(path.read_text(encoding="utf-8"))["target_basket_box"]))
        self.text, self.checked = "", 0.0

    def line(self):
        if not self.checks or time.monotonic() - self.checked < 1.0:
            return self.text
        self.checked = time.monotonic()
        parts = []
        for short, camera, target in self.checks:
            frame = cv2.cvtColor(camera.read_latest(), cv2.COLOR_RGB2BGR)
            result = self.model.predict(frame, conf=0.5, imgsz=640, device=0, verbose=False)[0]
            boxes = [(float(b.conf), b.xyxy[0].tolist()) for b in result.boxes if int(b.cls) == 1]
            if boxes:
                offset = max(abs(a - b) for a, b in zip(max(boxes)[1], target))
                parts.append(f"{short} {offset:.0f}px " + ("OK" if offset <= CAMERA_TOLERANCE else "CHECK"))
            else:
                parts.append(f"{short} no basket")
        self.text = "  ".join(parts)
        return self.text


def basket_shape(model, frame):
    """가장 확실한 바구니 박스의 가로/세로 비를 돌려줍니다. 없으면 None입니다."""
    result = model.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=0.5, imgsz=640, device=0, verbose=False)[0]
    boxes = [(float(b.conf), b.xyxy[0].tolist()) for b in result.boxes if int(b.cls) == 1]
    if not boxes:
        return None
    x0, y0, x1, y1 = max(boxes)[1]
    return (x1 - x0) / (y1 - y0)


def finger_ratio(frame):
    """화면 아래 가운데에서 보라색 그리퍼가 차지하는 비율입니다. 손목 카메라에서만 큽니다."""
    return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV), (120, 60, 40), (150, 255, 255))[300:480, 160:480].mean() / 255


def view_mismatches(live, shape):
    """카메라 번호가 바뀌어도 열리기는 하므로 화면 내용으로 확인합니다. 기준은 시연 50개에서 쟀습니다."""
    problems = []
    for name, frame in live.items():
        ratio = finger_ratio(frame)
        if (name == "wrist_cam") != (ratio >= FINGER_RATIO):
            problems.append(f"{name}: 그리퍼 비율 {ratio:.2f}")
    for name, wide in (("top_cam", False), ("belly_cam", True)):
        value = shape(live[name])
        if value is None:
            problems.append(f"{name}: 바구니를 찾지 못했습니다")
        elif (value >= BASKET_WIDE) != wide:
            problems.append(f"{name}: 바구니 가로세로 비 {value:.2f}")
    return problems

class Protocol:
    """조건별 반복 평가의 진행과 결과를 CSV에 남깁니다. 다시 실행하면 이어서 진행합니다."""

    def __init__(self, path, per_condition=PROTOCOL_TRIALS, only=None):
        self.path, self.per_condition = Path(path), per_condition
        self.conditions = [c for c in PROTOCOL_CONDITIONS if not only or c[0] in only]
        self.rows = []
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as f:
                self.rows = list(csv.DictReader(f))

    def done(self, condition):
        return sum(r["condition"] == condition and r["outcome"] != "discard" for r in self.rows)

    def current(self):
        """다음 시도의 조건, 번호, 배치를 돌려줍니다. 모두 끝났으면 None입니다."""
        for name, layout in self.conditions:
            done = self.done(name)
            if done < self.per_condition:
                # 빨간 상자만 있을 때는 세 자리를 돌아가며 놓습니다.
                return {"condition": name, "trial": done + 1,
                        "layout": layout or {list(SLOTS)[done % len(SLOTS)]: "red"}}
        return None

    def guide(self, item):
        return [(*SLOTS[slot], "V", color) for slot, color in item["layout"].items()]

    def record(self, item, outcome, summary, stem):
        row = {"time": summary["time"], "condition": item["condition"], "trial": item["trial"],
               "layout": " ".join(f"{slot}={color}" for slot, color in item["layout"].items()), "outcome": outcome,
               "reason": summary["reason"], "seconds": summary["seconds"], "speed": summary["speed"],
               "replan": summary["replan"], "checkpoint": summary["checkpoint"], "trial_file": stem}
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, PROTOCOL_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})

    def totals(self):
        parts = []
        for name, _ in self.conditions:
            done = self.done(name)
            wins = sum(r["condition"] == name and r["outcome"] == "success" for r in self.rows)
            parts.append(f"{name} {wins}/{done}")
        return ", ".join(parts)


def ask_verdict(keys, status):
    """사람이 본 결과를 받습니다. 기록의 자동 판정은 참고용입니다."""
    status.state = "VERDICT"
    status.set("Box in basket?  Y success   N fail   R discard and redo", ORANGE)
    keys.clear()
    while True:
        if keys.yes.is_set():
            return "success"
        if keys.no.is_set():
            return "fail"
        if keys.redo.is_set():
            return "discard"
        time.sleep(0.05)


def read_snapshot(follower):
    now = positions(follower.bus)
    return {"state": np.array([now[f"{j}.pos"] for j in JOINTS], dtype=np.float32),
            "images": {name: camera.read_latest() for name, camera in follower.cameras.items()}}


def read_tile(follower):
    tiles = [cv2.resize(cv2.cvtColor(camera.read_latest(), cv2.COLOR_RGB2BGR), TILE_SIZE) for camera in follower.cameras.values()]
    return cv2.hconcat(tiles)


def run_policy(cfg, speed, follower, keys, status):
    """모델 명령을 제한해서 보냅니다. 돌려주는 값은 (요약, 궤적, 3대 영상 프레임)입니다."""
    limits, planner, replan = cfg["limits"], cfg["planner"], cfg["replan"]
    period = 1 / CONTROL_FPS
    rate = POLICY_FPS * speed
    max_speed = limits["max_speed"] * speed
    total = cfg.get("max_seconds", MAX_POLICY_SECONDS) / speed
    guard = CommGuard()
    planner.drain()
    goal = np.array([positions(follower.bus)[f"{j}.pos"] for j in JOINTS])
    plan, plan_time, lead = None, 0.0, 0.0
    latencies, lags, trace, frames = [], {j: [] for j in JOINTS}, [], []
    clipped, limited, over_budget, commands = 0, 0, 0, 0
    reason = None
    keys.clear()
    started = last = time.perf_counter()
    last_read, last_frame = 0.0, 0.0
    while reason is None:
        loop = time.perf_counter()
        elapsed = loop - started
        if keys.esc.is_set():
            reason = "operator_stop"
        elif keys.space.is_set():
            reason = "operator_done"
        elif planner.error:
            reason = f"planner_error: {planner.error!r}"
        elif elapsed >= total:
            reason = "time_limit"
        if reason:
            break
        finished = planner.take()
        if finished:
            plan, plan_time, latency = finished
            latencies.append(latency)
            lead = max(latencies[-3:]) * rate
        position = (loop - plan_time) * rate if plan is not None else 0.0
        if not planner.busy and (plan is None or position >= replan - lead):
            observed_at = time.perf_counter()
            snapshot = guard.run(read_snapshot, follower)
            if snapshot:
                planner.request(snapshot, observed_at)
        if plan is not None:
            wanted = sample_action(plan, position)
            target = np.clip(wanted, limits["low"], limits["high"])
            clipped += not np.array_equal(wanted, target)
            step = max_speed * min(loop - last, MAX_STEP_SECONDS)
            limited += bool((np.abs(target - goal) > step).any())
            goal = goal + np.clip(target - goal, -step, step)
            guard.run(follower.send_action, to_action(goal))
            commands += 1
        last = loop
        if loop - last_read >= 0.1:
            last_read = loop
            now = guard.run(positions, follower.bus)
            if now:
                present = np.array([now[f"{j}.pos"] for j in JOINTS])
                errors = np.abs(goal - present)
                for j, value in zip(JOINTS, errors):
                    lags[j].append(float(value))
                trace.append(np.concatenate([[elapsed], goal, present]))
                over = errors - limits["lag"]
                if plan is not None and over.max() > 0:
                    worst = int(np.argmax(over))
                    reason = f"lag_limit: {JOINTS[worst]} {errors[worst]:.1f}"
        if loop - last_frame >= TRIAL_FRAME_SECONDS:
            last_frame = loop
            tile = guard.run(read_tile, follower)
            if tile is not None:
                frames.append(tile)
        status.set(f"ACT {speed:.0%}  {elapsed:4.1f}/{total:.0f}s  plans {len(latencies)}  SPACE done  ESC stop", RED)
        used = time.perf_counter() - loop
        over_budget += used > period
        precise_sleep(max(0.0, period - used))
    seconds = time.perf_counter() - started
    summary = {"reason": reason, "speed": speed, "replan": replan, "seconds": round(seconds, 2), "plans": len(latencies),
               "commands": commands, "clipped_loops": clipped, "speed_limited_loops": limited,
               "over_budget_loops": over_budget, "comm_misses": guard.total,
               "latency_ms": {"median": round(float(np.median(latencies)) * 1000), "max": round(max(latencies) * 1000)} if latencies else None,
               "lag": {j: {"median": round(float(np.median(v)), 2), "max": round(float(np.max(v)), 2)} if v else None
                       for j, v in lags.items()}}
    return summary, trace, frames


def save_trial(cfg, summary, trace, frames, model):
    """시도별 요약, 궤적, 3대 영상, 마지막 top 화면을 남깁니다. 실패 원인을 볼 때 씁니다."""
    stem = f"trial_{datetime.now():%H%M%S}_{summary['speed'] * 100:.0f}pct"
    cfg["log_dir"].mkdir(parents=True, exist_ok=True)
    final = cfg.get("camera")
    frame = cv2.cvtColor(final.read_latest(), cv2.COLOR_RGB2BGR) if final else None
    summary["time"] = datetime.now().isoformat(timespec="seconds")
    summary["checkpoint"] = cfg["name"]
    summary["yolo_top"] = judge_top(model, frame) if frame is not None else None
    (cfg["log_dir"] / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if trace:
        np.save(cfg["log_dir"] / f"{stem}_trace.npy", np.array(trace))
    if frame is not None:
        cv2.imwrite(str(cfg["log_dir"] / f"{stem}_final.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if frames:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(cfg["log_dir"] / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 1 / TRIAL_FRAME_SECONDS, (width, height))
        for tile in frames:
            writer.write(tile)
        writer.release()
    print(f"[RESULT] {summary['reason']}, {summary['seconds']}초, 예측 {summary['plans']}회, 추론 {summary['latency_ms']}ms, "
          f"범위 제한 {summary['clipped_loops']}, 속도 제한 {summary['speed_limited_loops']}, YOLO {summary['yolo_top']}", flush=True)
    print(f"[LOG] {cfg['log_dir'] / stem}", flush=True)
    return stem


def run_session(cfg, follower, keys, status, model=None):
    start = cfg["limits"]["start"]
    check = CameraCheck(model, follower.cameras)
    protocol = cfg.get("protocol")
    item = protocol.current() if protocol else None
    status.guide = protocol.guide(item) if item else None
    prefix = f"{item['condition']} trial {item['trial']}: " if item else ""
    status.title = f"{prefix}{cfg['name']}: place the boxes | ENTER move to start | ESC quit"
    status.state = "PLACE"
    keys.clear()
    while not keys.enter.is_set():
        if keys.esc.is_set():
            print("[EXIT] 토크를 켜지 않고 종료합니다.", flush=True)
            return "cancelled"
        status.set(f"PLACE: torque OFF  {check.line()}", GREEN)
        time.sleep(0.05)
    enable_holding(follower)
    print("[MOVE] 시작 자세로 천천히 이동합니다.", flush=True)
    status.state = "MOVE"
    at_start = move_to(follower, start, keys, status, "MOVE to start")
    speed_index = cfg.get("speed_index", DEFAULT_SPEED)
    results = []
    finish_until = 0.0
    esc_ignore = 0.0
    keys.clear()
    while True:
        status.state = "READY" if at_start else "STOPPED"
        speed = SPEEDS[speed_index]
        if protocol:
            label = f"{item['condition']} trial {item['trial']}/{protocol.per_condition}" if item else "PROTOCOL DONE"
            status.title = f"{label}  SPEED {speed:.0%} | ENTER " + ("run ACT" if at_start else "move to start") + "  ESC twice finish"
        else:
            status.title = (f"{cfg['name']}  SPEED {speed:.0%} | SPACE change speed  ENTER "
                            + ("run ACT" if at_start else "move to start") + "  ESC twice finish")
        armed = time.monotonic() < finish_until
        status.set("Press ESC again: torque OFF, arm may drop" if armed else
                   (f"READY: set the boxes  {check.line()}" if at_start else "STOPPED: arm holding. ENTER returns to start"),
                   ORANGE if armed else GREEN)
        if keys.space.is_set():
            keys.space.clear()
            if not protocol:  # 조건끼리 비교할 수 있게 평가 중에는 속도를 고정합니다.
                speed_index = (speed_index + 1) % len(SPEEDS)
                print(f"[SPEED] {SPEEDS[speed_index]:.0%}", flush=True)
        if esc_pressed(keys, esc_ignore):
            keys.esc.clear()
            if armed:
                follower.bus.disable_torque()
                status.set("Torque OFF", GREEN)
                print("[TORQUE OFF] 팔로워 토크를 껐습니다.", flush=True)
                return results
            finish_until = time.monotonic() + FINISH_CONFIRM_SECONDS
            esc_ignore = time.monotonic() + 0.7
            print(f"[CONFIRM] 토크를 끄려면 {FINISH_CONFIRM_SECONDS:.0f}초 안에 Esc를 한 번 더 누르세요. 팔이 처질 수 있습니다.", flush=True)
        if keys.enter.is_set():
            keys.enter.clear()
            if not at_start:
                status.state = "MOVE"
                print("[MOVE] 시작 자세로 돌아갑니다.", flush=True)
                at_start = move_to(follower, start, keys, status, "MOVE to start")
            elif protocol and item is None:
                print(f"[PROTOCOL] 모두 끝났습니다. {protocol.totals()}", flush=True)
            else:
                status.state = "RUN"
                print(f"[RUN] {speed:.0%} 실행 시작. Space: 시도 끝 / Esc: 즉시 멈춤", flush=True)
                summary, trace, frames = run_policy(cfg, speed, follower, keys, status)
                CommGuard().run(hold_here, follower)
                time.sleep(0.5)
                stem = save_trial(cfg, summary, trace, frames, model)
                results.append(summary)
                at_start = False
                if protocol:
                    outcome = ask_verdict(keys, status)
                    protocol.record(item, outcome, summary, stem)
                    print(f"[PROTOCOL] {item['condition']} {item['trial']}: {outcome} | {protocol.totals()}", flush=True)
                    item = protocol.current()
                    status.guide = protocol.guide(item) if item else None
            keys.clear()
            esc_ignore = time.monotonic() + 0.7
            finish_until = 0.0
        time.sleep(0.05)


def dry_run(follower, predict, limits, seconds):
    """토크를 켜지 않고 실제 카메라와 관절값으로 추론만 합니다. 팔은 움직이지 않습니다."""
    started = time.perf_counter()
    latencies, jumps, outside = [], [], 0
    while time.perf_counter() - started < seconds:
        begin = time.perf_counter()
        snapshot = read_snapshot(follower)
        chunk = predict(snapshot)
        latencies.append(time.perf_counter() - begin)
        jumps.append(np.abs(chunk[0] - snapshot["state"]))
        outside += int(((chunk < limits["low"]) | (chunk > limits["high"])).any())
    jumps = np.array(jumps)
    print(f"[DRY RUN] 추론 {len(latencies)}회, 관측+추론 중앙값 {np.median(latencies) * 1000:.0f}ms, 최대 {np.max(latencies) * 1000:.0f}ms", flush=True)
    print(f"[DRY RUN] 현재 자세와 첫 명령의 차이(최대): {dict(zip(JOINTS, np.round(jumps.max(axis=0), 1).tolist()))}", flush=True)
    print(f"[DRY RUN] 시연 범위를 벗어난 예측: {outside}회. 모터 명령은 보내지 않았습니다.", flush=True)


def self_test():
    """모터와 모델 없이 제한, 멈춤, 재예측, 오류 처리를 검사합니다."""
    limits = {"start": np.full(6, 1.0), "low": np.full(6, -50.0), "high": np.full(6, 50.0),
              "max_speed": np.full(6, 100.0), "lag": np.full(6, 1000.0)}

    def steady(snapshot):
        return snapshot["state"][None, :] + np.arange(10)[:, None] * 2.0

    def session(tmp, predict, *steps, **extra):
        planner = Planner(predict)
        planner.start()
        cfg = {"limits": limits, "planner": planner, "replan": 10, "name": "test", "log_dir": Path(tmp), "camera": None, **extra}
        keys, status, follower = Keys(), Status(""), FakeFollower()
        press_when(status, keys, *steps)
        return run_session(cfg, follower, keys, status), follower, keys

    finish = (("STOPPED", 0.9, "esc"), ("STOPPED", 0.9, "esc"))
    with tempfile.TemporaryDirectory() as tmp:
        result, follower, _ = session(tmp, steady, ("PLACE", 0.1, "esc"))
        assert result == "cancelled" and follower.bus.torque == 0 and not follower.bus.goals

        # 기본 50%에서 실행 후 Space로 끝냅니다. 100%로 바꿔 Esc로도 멈춥니다.
        results, follower, _ = session(tmp, steady, ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"),
                                       ("RUN", 1.0, "space"), ("STOPPED", 0.2, "enter"),
                                       ("READY", 0.1, "space"), ("READY", 0.1, "enter"),
                                       ("RUN", 0.5, "esc"), *finish)
        assert follower.bus.torque == 0, "must end with torque off"
        assert follower.bus.goals[0] == {j: 0.0 for j in JOINTS}, "goal must be primed before torque"
        assert [r["reason"] for r in results] == ["operator_done", "operator_stop"], results
        assert [r["speed"] for r in results] == [0.5, 1.0], results
        assert results[0]["plans"] >= 1 and results[1]["seconds"] < 1.0, results
        assert len(list(Path(tmp).glob("trial_*.json"))) == 2 and len(list(Path(tmp).glob("trial_*.mp4"))) == 2

        # 범위 밖 명령은 범위 안으로, 큰 점프는 속도 제한 안으로 들어와야 합니다.
        results, follower, _ = session(tmp, lambda s: np.full((10, 6), 500.0), ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"),
                                       ("RUN", 1.0, "esc"), *finish)
        sent = np.array([[g[j] for j in JOINTS] for g in follower.bus.goals if len(g) == 6])
        assert sent.max() <= 50.0 and results[0]["clipped_loops"] > 0 and results[0]["speed_limited_loops"] > 0, results
        moved = np.abs(np.diff(sent, axis=0)).max(axis=0)
        assert moved.max() <= 100.0 * 0.5 * 0.05 + 1e-6, moved

        def fail(snapshot):
            raise RuntimeError("simulated inference failure")
        results, *_ = session(tmp, fail, ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"), *finish)
        assert results[0]["reason"].startswith("planner_error") and results[0]["commands"] == 0, results

        results, *_ = session(tmp, lambda s: np.full((10, 6), np.nan), ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"), *finish)
        assert results[0]["reason"].startswith("planner_error") and results[0]["commands"] == 0, results

        results, *_ = session(tmp, steady, ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"), *finish, max_seconds=0.5)
        assert results[0]["reason"] == "time_limit" and results[0]["seconds"] < 1.5, results

        saved = dict(limits)
        limits["lag"] = np.full(6, -1.0)
        results, *_ = session(tmp, steady, ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"), *finish)
        assert results[0]["reason"].startswith("lag_limit"), results
        limits.update(saved)

        # 재예측 주기: 추론이 0.1초 걸려도 짧은 주기일수록 더 자주 예측합니다.
        def slow(snapshot):
            time.sleep(0.1)
            return steady(snapshot)
        counts = []
        for replan in (10, 2):
            results, *_ = session(tmp, slow, ("PLACE", 0.1, "enter"), ("READY", 0.1, "space"), ("READY", 0.1, "enter"),
                                  ("RUN", 2.5, "esc"), *finish, replan=replan)
            counts.append(results[0]["plans"])
        assert counts[0] <= 3 and counts[1] >= 2 * counts[0], counts
    # 조건별 평가: 사람 판정을 남기고, R은 세지 않으며, 다시 실행하면 이어집니다.
    from pynput.keyboard import KeyCode
    keys = Keys()
    keys.press(KeyCode.from_vk(VK_Y))
    keys.press(KeyCode.from_vk(VK_N))
    keys.press(KeyCode.from_vk(VK_R))
    assert keys.yes.is_set() and keys.no.is_set() and keys.redo.is_set()
    assert len(PROTOCOL_CONDITIONS) == 7 and len({name for name, _ in PROTOCOL_CONDITIONS}) == 7
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.csv"
        protocol = Protocol(path, per_condition=2)
        one_trial = (("READY", 0.1, "enter"), ("RUN", 0.3, "space"))
        results, follower, _ = session(tmp, steady, ("PLACE", 0.1, "enter"),
                                       *one_trial, ("VERDICT", 0.1, "yes"), ("STOPPED", 0.1, "enter"),
                                       *one_trial, ("VERDICT", 0.1, "redo"), ("STOPPED", 0.1, "enter"),
                                       *one_trial, ("VERDICT", 0.1, "no"), ("STOPPED", 0.1, "enter"),
                                       *one_trial, ("VERDICT", 0.1, "yes"), *finish, protocol=protocol)
        assert follower.bus.torque == 0 and len(results) == 4 and all(r["speed"] == 0.5 for r in results), results
        again = Protocol(path, per_condition=2)
        assert [r["outcome"] for r in again.rows] == ["success", "discard", "fail", "success"], again.rows
        assert again.done("red_only") == 2 and again.done(PROTOCOL_CONDITIONS[1][0]) == 1
        # 빨간 상자만 있을 때 번호에 따라 자리가 바뀝니다.
        fresh = Protocol(Path(tmp) / "none.csv", per_condition=3)
        assert fresh.current() == {"condition": "red_only", "trial": 1, "layout": {"TOP": "red"}}
        assert again.current()["condition"] == PROTOCOL_CONDITIONS[1][0] and again.current()["trial"] == 2
        assert len(again.current()["layout"]) == 3
        assert again.guide(again.current())[0][2] == "V"
        assert "red_only 1/2" in again.totals()
        picked = Protocol(path, per_condition=2, only=[PROTOCOL_CONDITIONS[4][0]])
        assert picked.current()["condition"] == PROTOCOL_CONDITIONS[4][0] and "red_only" not in picked.totals()
        speedy, follower, _ = session(tmp, steady, ("PLACE", 0.1, "enter"), *one_trial, ("VERDICT", 0.1, "no"), *finish,
                                      protocol=Protocol(Path(tmp) / "speed.csv", per_condition=1), speed_index=2)
        assert speedy[0]["speed"] == 1.0, speedy
    grid = red_only_guide()
    assert len(grid) == 30 and len(set(grid)) == 30

    # 카메라 번호 확인: 정상 순서는 통과하고, 바뀐 순서는 잡아야 합니다.
    plain = np.full((480, 640, 3), 200, dtype=np.uint8)
    wide = plain.copy()
    purple = plain.copy()
    purple[300:480, 160:480] = cv2.cvtColor(np.uint8([[[135, 200, 200]]]), cv2.COLOR_HSV2RGB)[0, 0]
    shapes = {id(plain): 0.9, id(wide): 1.7, id(purple): None}
    shape = lambda frame: shapes[id(frame)]
    assert not view_mismatches({"top_cam": plain, "wrist_cam": purple, "belly_cam": wide}, shape)
    assert not view_mismatches({"top_cam": plain, "belly_cam": wide}, shape)
    assert view_mismatches({"top_cam": wide, "wrist_cam": purple, "belly_cam": plain}, shape)
    assert view_mismatches({"top_cam": plain, "wrist_cam": wide, "belly_cam": purple}, shape)
    print("[SELF-TEST] 범위/속도 제한, 즉시 멈춤, 지연 중단, 시간 제한, 모델 오류, 재예측 주기, 조건별 평가, 카메라 번호, 토크 해제 검사 통과. "
          "모터는 사용하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="redbox_rgb_v1")
    parser.add_argument("--checkpoint", default=None, help="예: 060000. 없으면 저장이 끝난 가장 최근 체크포인트")
    parser.add_argument("--dataset", nargs="+", default=["redbox_30fps"], help="안전 한계와 시작 자세를 잴 시연 (여러 개 가능)")
    parser.add_argument("--replan", type=int, default=10, help="몇 스텝마다 새로 예측할지 (1~10). 1 스텝은 100%% 속도에서 0.2초")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wrist", action="store_true", help="손목 영상 대신 평균 화면을 넣는 임시 시험입니다. 잡기 성능은 믿을 수 없습니다.")
    parser.add_argument("--dry-run", type=float, default=0.0, help="지정한 초 동안 추론만 합니다. 토크를 켜지 않습니다.")
    parser.add_argument("--protocol", action="store_true", help="빨간 상자만 + 3개 배치 6가지를 조건별 20회 평가합니다.")
    parser.add_argument("--trials", type=int, default=PROTOCOL_TRIALS, help="--protocol의 조건별 시도 수")
    parser.add_argument("--tag", default="", help="결과 폴더 이름 끝에 붙입니다. 설정을 바꾼 평가를 따로 모을 때 씁니다.")
    parser.add_argument("--conditions", nargs="+", choices=[name for name, _ in PROTOCOL_CONDITIONS], default=None,
                        help="--protocol에서 이 조건만 평가합니다.")
    parser.add_argument("--speed", type=int, choices=[round(s * 100) for s in SPEEDS], default=None,
                        help="시작 속도(%%). --protocol에서는 이 속도로 고정됩니다. 기본 50")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.replan <= 10:
        parser.error("--replan은 1~10이어야 합니다.")
    from ultralytics import YOLO

    if args.checkpoint and args.checkpoint.isdigit():
        args.checkpoint = f"{int(args.checkpoint):06d}"  # 80000 -> 080000
    checkpoint = RUNS_DIR / args.run / "checkpoints" / args.checkpoint if args.checkpoint else latest_checkpoint(args.run)
    if not checkpoint.is_dir():
        saved = sorted(p.name for p in checkpoint.parent.glob("[0-9]*") if p.is_dir())
        raise RuntimeError(f"체크포인트가 없습니다: {checkpoint.name}. 저장된 것: {', '.join(saved)}")
    if not (checkpoint / "training_state").is_dir():
        raise RuntimeError(f"아직 저장이 끝나지 않은 체크포인트입니다: {checkpoint}")
    limits = demo_limits(args.dataset)
    print(f"[CHECKPOINT] {checkpoint}", flush=True)
    print(f"[LIMITS] 시작 자세 {np.round(limits['start'], 1).tolist()}", flush=True)
    suffix = ("_nowrist" if args.no_wrist else "") + (f"_{args.tag}" if args.tag else "")
    predict = ActPredictor(checkpoint, args.device, {"wrist_cam": mean_image("wrist_cam")} if args.no_wrist else None)
    model = YOLO(str(MODEL))
    if args.no_wrist:
        print("[NO WRIST] 손목 영상 없이 평균 화면을 넣습니다. 임시 시험이며 결과는 따로 저장합니다.", flush=True)
    follower, leader = build_robot(NO_WRIST_CAMERAS) if args.no_wrist else build_robot()
    keys, status, preview, planner = Keys(), Status(""), None, Planner(predict)
    try:
        connect_without_torque(follower, leader)
        time.sleep(1.0)
        problems = view_mismatches({name: camera.read_latest() for name, camera in follower.cameras.items()},
                                   lambda frame: basket_shape(model, frame))
        if problems:
            raise RuntimeError("카메라 번호가 바뀐 것 같습니다. 토크를 켜지 않고 중단합니다.\n" + "\n".join(problems))
        print(f"[CAMERAS] {', '.join(follower.cameras)} 화면 확인 완료.", flush=True)
        if args.dry_run:
            dry_run(follower, predict, limits, args.dry_run)
            return
        planner.start()
        print("[READY] 토크 꺼짐. 상자를 놓고 Enter를 누르세요. 키는 어느 창에서든 입력됩니다.", flush=True)
        keys.start()
        preview = Preview(follower.cameras, status)
        preview.start()
        cfg = {"limits": limits, "planner": planner, "replan": args.replan, "name": f"{args.run}_{checkpoint.name}{suffix}",
               "log_dir": LOG_DIR / "eval" / f"{args.run}_{checkpoint.name}{suffix}", "camera": follower.cameras["top_cam"]}
        if args.speed:
            cfg["speed_index"] = [round(s * 100) for s in SPEEDS].index(args.speed)
        if args.protocol:
            cfg["protocol"] = Protocol(cfg["log_dir"] / "protocol_results.csv", per_condition=args.trials, only=args.conditions)
            print(f"[PROTOCOL] 지금까지: {cfg['protocol'].totals()}", flush=True)
        run_session(cfg, follower, keys, status, model)
    finally:
        keys.stop()
        planner.close()
        if preview:
            preview.close()
        shutdown(follower, leader)
        print("[END] 평가를 마쳤습니다.", flush=True)


if __name__ == "__main__":
    main()
