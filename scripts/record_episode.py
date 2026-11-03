"""SO-ARM101 에피소드 수집: 끄지 않고 이어서 찍고, 에피소드는 폴더별로 따로 저장합니다.
준비 -> Enter -> 자세 맞추기 (기록 안 함) -> Enter -> 3초 대기 -> 최대 15초 기록 -> Space 저장 -> 준비.
대기나 기록 중 Esc: 기록을 버리고 같은 번호의 자세 맞추기로 돌아가며, Enter 한 번이면 다시 시작합니다.
자세 맞추기에서 Esc: 준비로 돌아갑니다. 준비에서 Esc: 종료 절차로 갑니다.
15초를 넘기면 저장하지 않습니다. 비상시에는 전원을 차단하세요.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import faulthandler
import json
from pathlib import Path
import queue
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR / "datasets" / "lerobot"
LOG_DIR = PROJECT_DIR / "logs"
CAMERAS = {"top_cam": 0, "wrist_cam": 1, "belly_cam": 2}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Pick up only the red box and place it inside the gray basket"
CONTROL_FPS = 60
DATASET_FPS = 30
COUNTDOWN_SECONDS = 3.0
MAX_EPISODE_SECONDS = 15.0
MIN_EPISODE_SECONDS = 2.0
RAMP_SECONDS = 1.5
START_MAX_ERROR = 20.0
TORQUE_MOVE_LIMIT = 3.0
RED_POSITIONS = 5
COMM_RETRIES = 3
POSITION_READ_TRIES = 5
COMM_MISS_LIMIT = 30
ESC_GUARD_SECONDS = 0.7
FINISH_CONFIRM_SECONDS = 3.0
WARN_FPS = 29.0
WARN_GAP_MS = 100.0
LOG_FIELDS = ["time", "status", "episode", "red_position", "frames", "seconds", "fps", "gap_p95_ms",
              "gap_max_ms", "late_samples", "control_over_budget"]
GREEN, RED, YELLOW, ORANGE = (80, 230, 80), (60, 60, 255), (0, 255, 255), (0, 200, 255)
VK_Y, VK_N, VK_R = 89, 78, 82
GUIDE_COLORS = {"red": (40, 40, 245), "black": (20, 20, 20), "blue": (230, 90, 30)}
GUIDE_HIDDEN_STATES = ("COUNTDOWN", "RECORD", "RUN")
RED_GRID_X = (410, 455, 505)
RED_GRID_Y = (90, 165, 240, 315, 390)
SLOTS = {"TOP": (450, 105), "MID": (455, 210), "BOT": (450, 345)}  # 평가 자리(top 화면 좌표). 긴 변을 화면 세로로 놓습니다.
RED_TOP_SHIFTS = [(0, 0), (-20, 0), (20, 0), (0, -15), (0, 15), (-20, -15), (20, 15), (-20, 15), (20, -15), (0, 0)]


def red_only_guide():
    """빨간 상자 전용 시연의 배치입니다. 위치 15곳 x 방향 2개를 섞어 시간에 따른 치우침을 줄입니다."""
    items = [[(x, y, orient, "red")] for x in RED_GRID_X for y in RED_GRID_Y for orient in "VH"]
    random.Random(0).shuffle(items)
    return items


def red_top_guide():
    """빨강 위, 검정 가운데, 파랑 아래 배치입니다. 평가 자리를 그대로 외우지 않도록 빨강은 상하좌우로, 나머지는 조금 옮깁니다."""
    items = []
    for i, (dx, dy) in enumerate(RED_TOP_SHIFTS):
        side = 10 if i % 2 else -10
        items.append([(SLOTS["TOP"][0] + dx, SLOTS["TOP"][1] + dy, "V", "red"),
                      (SLOTS["MID"][0] + side, SLOTS["MID"][1], "V", "black"),
                      (SLOTS["BOT"][0] - side, SLOTS["BOT"][1], "V", "blue")])
    return items


GUIDES = {"red_only": red_only_guide, "red_top": red_top_guide}


def draw_guide(frame, guide):
    """top 화면에 놓을 위치를 그립니다. V는 긴 변이 화면 세로, H는 화면 가로입니다. 저장 데이터에는 들어가지 않습니다."""
    for x, y, orient, color in guide:
        long_half = int(28 + 0.07 * y)
        short_half = int(long_half * 0.65)
        half_w, half_h = (short_half, long_half) if orient == "V" else (long_half, short_half)
        cv2.rectangle(frame, (x - half_w, y - half_h), (x + half_w, y + half_h), GUIDE_COLORS[color], 3)
        cv2.rectangle(frame, (x - half_w - 2, y - half_h - 2), (x + half_w + 2, y + half_h + 2), (255, 255, 255), 1)


class Keys:
    """창 포커스 없이 Enter, Space, Esc를 받습니다."""

    def __init__(self):
        self.enter, self.space, self.esc = threading.Event(), threading.Event(), threading.Event()
        self.yes, self.no, self.redo = threading.Event(), threading.Event(), threading.Event()
        self.listener = None

    def press(self, key):
        from pynput import keyboard

        if key == keyboard.Key.enter:
            self.enter.set()
        elif key == keyboard.Key.space:
            self.space.set()
        elif key == keyboard.Key.esc:
            self.esc.set()
        else:
            # 한글 입력 상태에서도 받도록 글자 대신 키 코드로 봅니다.
            event = {VK_Y: self.yes, VK_N: self.no, VK_R: self.redo}.get(getattr(key, "vk", None))
            if event:
                event.set()

    def start(self):
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self.press)
        self.listener.start()

    def clear(self):
        for event in (self.enter, self.space, self.esc, self.yes, self.no, self.redo):
            event.clear()

    def stop(self):
        if self.listener:
            self.listener.stop()


class Status:
    def __init__(self, title):
        self.lock = threading.Lock()
        self.title = title
        self.line = ""
        self.color = YELLOW
        self.state = ""
        self.guide = None

    def set(self, line, color=YELLOW):
        with self.lock:
            self.line, self.color = line, color

    def get(self):
        with self.lock:
            return self.title, self.line, self.color


class Preview(threading.Thread):
    """제어 루프를 막지 않도록 별도 스레드에서 화면만 그립니다."""

    def __init__(self, cameras, status):
        super().__init__(daemon=True)
        self.cameras, self.status = cameras, status
        self.stop_event = threading.Event()

    def run(self):
        window = "Episode recorder"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while not self.stop_event.is_set():
            tiles = []
            for name, camera in self.cameras.items():
                try:
                    frame = cv2.cvtColor(camera.read_latest(), cv2.COLOR_RGB2BGR)
                except Exception:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if name == "top_cam" and self.status.guide and self.status.state not in GUIDE_HIDDEN_STATES:
                    draw_guide(frame, self.status.guide)
                tile = cv2.resize(frame, (480, 360))
                cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                tiles.append(tile)
            title, line, color = self.status.get()
            canvas = cv2.copyMakeBorder(cv2.hconcat(tiles), 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.putText(canvas, title, (8, 386), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
            cv2.putText(canvas, line, (8, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(window, canvas)
            cv2.waitKey(66)
        cv2.destroyWindow(window)

    def close(self):
        self.stop_event.set()
        self.join(timeout=3)


class DatasetWriter(threading.Thread):
    """영상 저장이 제어 루프를 막지 않게 합니다. 파일이 깨지지 않도록 끝날 때 항상 finalize합니다."""

    def __init__(self, dataset):
        super().__init__(daemon=True)
        self.dataset = dataset
        self.jobs = queue.Queue(maxsize=256)
        self.error = None

    def add(self, frame):
        if self.error:
            raise RuntimeError("데이터 저장 실패") from self.error
        self.jobs.put_nowait(("frame", frame))

    def finish(self, save):
        self.jobs.put(("save" if save else "discard", None))

    def run(self):
        try:
            while True:
                command, frame = self.jobs.get()
                if command == "frame":
                    self.dataset.add_frame(frame)
                    continue
                if command == "save":
                    self.dataset.save_episode()
                else:
                    self.dataset.clear_episode_buffer()
                break
        except BaseException as exc:  # 제어 스레드에서 확인해 보고합니다.
            self.error = exc
        finally:
            try:
                self.dataset.finalize()
            except BaseException as exc:
                self.error = self.error or exc


class PendingEpisode:
    """저장 또는 폐기가 끝나기를 기다리는 에피소드입니다."""

    def __init__(self, writer, root, outcome, frames, row):
        self.writer, self.root, self.outcome, self.frames, self.row = writer, root, outcome, frames, row


class CommGuard:
    """모터나 카메라의 일시적 오류는 건너뛰고, 계속되면 중단합니다."""

    def __init__(self, limit=COMM_MISS_LIMIT):
        self.limit, self.misses, self.total = limit, 0, 0

    def run(self, function, *args):
        try:
            result = function(*args)
        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            self.misses += 1
            self.total += 1
            if self.total == 1:
                print(f"[COMM] 일시적 통신 오류, 재시도합니다: {exc}", flush=True)
            if self.misses > self.limit:
                raise
            return None
        self.misses = 0
        return result


def positions(arm_bus):
    """일시적 통신 오류로 프로그램이 끝나 토크가 꺼지지 않도록 몇 번 더 읽습니다."""
    for attempt in range(POSITION_READ_TRIES):
        try:
            return {f"{j}.pos": float(v) for j, v in arm_bus.sync_read("Present_Position", num_retry=COMM_RETRIES).items()}
        except ConnectionError:
            if attempt == POSITION_READ_TRIES - 1:
                raise
            time.sleep(0.02)


def esc_pressed(keys, ignore_until):
    """단계가 바뀐 직후의 Esc는 연타로 보고 무시합니다."""
    if keys.esc.is_set() and time.monotonic() < ignore_until:
        keys.esc.clear()
    return keys.esc.is_set()


def pose_error(leader_pose, follower_pose):
    errors = {j: abs(leader_pose[f"{j}.pos"] - follower_pose[f"{j}.pos"]) for j in JOINTS}
    joint = max(errors, key=errors.get)
    return joint, errors[joint]


def episode_dir(root, index):
    return root / f"episode_{index:04d}"


def episode_frames(path):
    """제대로 저장된 에피소드 폴더의 프레임 수를 돌려줍니다. 아니면 None입니다."""
    info = path / "meta" / "info.json"
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
        rows = sum(pq.read_metadata(p).num_rows for p in (path / "data").rglob("*.parquet"))
    except Exception:
        return None
    if int(data["total_episodes"]) != 1 or rows != int(data["total_frames"]) or rows < 1:
        return None
    return rows


def set_aside_incomplete(root):
    """중단되어 불완전한 폴더는 지우지 않고 옮겨 둡니다. 그 번호는 다시 찍습니다."""
    if not root.exists():
        return
    for path in sorted(root.glob("episode_*")):
        if episode_frames(path) is None:
            target = root / "_incomplete" / f"{path.name}_{datetime.now():%Y%m%d_%H%M%S}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            print(f"[INCOMPLETE] 불완전한 폴더를 옮겼습니다: {target}", flush=True)


def saved_episodes(root):
    return sorted(int(p.name.removeprefix("episode_")) for p in root.glob("episode_*") if p.is_dir()) if root.exists() else []


def next_episode(root, target):
    """비어 있는 가장 작은 번호를 돌려줍니다. 지운 에피소드는 같은 위치로 다시 찍게 됩니다."""
    saved = set(saved_episodes(root))
    return next((i for i in range(target) if i not in saved), None)


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def enable_holding(follower):
    """현재 자세를 목표값으로 먼저 쓴 뒤 토크를 켭니다. 켜는 순간 팔이 튀지 않게 합니다."""
    if any(int(v) != 0 for v in follower.bus.sync_read("Torque_Enable").values()):
        raise RuntimeError("토크가 이미 켜져 있습니다. 다른 프로그램을 확인하세요.")
    start = positions(follower.bus)
    follower.bus.sync_write("Goal_Position", {j: start[f"{j}.pos"] for j in JOINTS})
    follower.configure()
    follower.bus.enable_torque()
    time.sleep(0.15)
    now = positions(follower.bus)
    moved = max(abs(now[k] - start[k]) for k in start)
    if moved > TORQUE_MOVE_LIMIT:
        raise RuntimeError(f"토크를 켤 때 {moved:.1f}단위 움직였습니다. 중단합니다.")
    return now


def hold_here(follower):
    """지금 자세를 유지합니다. 리더에서 손을 떼도 팔로워는 움직이지 않습니다."""
    now = positions(follower.bus)
    follower.bus.sync_write("Goal_Position", {j: now[f"{j}.pos"] for j in JOINTS})
    return now


def teleop_step(leader, follower, start_pose, ramp_started):
    """시작 직후에는 리더 자세까지 서서히 이동하고, 이후에는 리더를 그대로 따라갑니다."""
    action = leader.get_action()
    blend = min(1.0, (time.perf_counter() - ramp_started) / RAMP_SECONDS)
    if blend < 1.0:
        action = {k: start_pose[k] + (action[k] - start_pose[k]) * blend for k in action}
    return follower.send_action(action)


def follow_until(follower, leader, keys, status, start_pose, ramp_started, line, color, stop):
    """stop()이 참이 될 때까지 팔로워가 리더를 따라갑니다."""
    period = 1 / CONTROL_FPS
    guard = CommGuard()
    while not stop():
        loop = time.perf_counter()
        guard.run(teleop_step, leader, follower, start_pose, ramp_started)
        status.set(line(), color)
        precise_sleep(max(0.0, period - (time.perf_counter() - loop)))


def complete_pending(pending, cfg):
    """저장이나 폐기가 끝난 에피소드를 확인하고 기록합니다. 저장 실패면 False를 돌려줍니다."""
    pending.writer.join()
    outcome, ok = pending.outcome, True
    if outcome == "saved":
        frames = episode_frames(pending.root)
        if pending.writer.error or frames != pending.frames:
            print(f"[ERROR] 저장 확인 실패: {pending.root} ({pending.writer.error!r}, 기록 {pending.frames}, 파일 {frames})", flush=True)
            set_aside_incomplete(cfg["root"])
            outcome, ok = "save_failed", False
    elif pending.root.exists():
        shutil.rmtree(pending.root)
    pending.row["status"] = outcome
    append_log(cfg["log"], pending.row)
    r = pending.row
    print(f"[RESULT] {outcome}: {r['frames']}프레임, {r['seconds']}초, {r['fps']} FPS, 최대 간격 {r['gap_max_ms']}ms, "
          f"지연 샘플 {r['late_samples']}", flush=True)
    if outcome == "saved":
        print(f"[SAVED] {pending.root.name} 저장 확인 ({len(saved_episodes(cfg['root']))}/{cfg['target']})", flush=True)
        if r["fps"] < WARN_FPS or r["gap_max_ms"] > WARN_GAP_MS or r["late_samples"]:
            print(f"[WARNING] 기록이 고르지 않습니다. 기준: {WARN_FPS:.0f} FPS 이상, 간격 {WARN_GAP_MS:.0f}ms 이하.", flush=True)
    return ok


def record_attempt(cfg, follower, leader, keys, status, index, position):
    """3초 대기 후 기록합니다. 끝나면 PendingEpisode를 돌려줍니다."""
    root = episode_dir(cfg["root"], index)
    features = {**hw_to_dataset_features(follower.observation_features, OBS_STR),
                **hw_to_dataset_features(follower.action_features, ACTION)}
    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=f"local/{cfg['name']}_{index:04d}", fps=DATASET_FPS, features=features,
                                    root=root, robot_type=follower.name, use_videos=True,
                                    image_writer_processes=0, image_writer_threads=4)
    writer = DatasetWriter(dataset)
    writer.start()
    start_pose = positions(follower.bus)
    ramp_started = phase_started = time.perf_counter()
    keys.clear()
    status.state = "COUNTDOWN"
    follow_until(follower, leader, keys, status, start_pose, ramp_started,
                 lambda: f"GET READY {cfg['countdown'] - (time.perf_counter() - phase_started):.1f}s  (not recording)  ESC back",
                 ORANGE, lambda: time.perf_counter() - phase_started >= cfg["countdown"] or keys.esc.is_set())

    sample_times, late, over_budget = [], 0, 0
    guard = CommGuard()
    outcome = "discarded" if keys.esc.is_set() else None
    period, sample_period = 1 / CONTROL_FPS, 1 / DATASET_FPS
    keys.clear()
    started = time.perf_counter()
    next_sample = 0.0
    if outcome is None:
        status.state = "RECORD"
        print(f"[REC] episode_{index:04d} 기록 시작. Space: 저장 / Esc: 버리고 처음으로", flush=True)
    while outcome is None:
        loop = time.perf_counter()
        elapsed = loop - started
        if keys.esc.is_set():
            outcome = "discarded"
            break
        if keys.space.is_set():
            keys.space.clear()
            if elapsed >= MIN_EPISODE_SECONDS:
                outcome = "saved"
                break
            print(f"[IGNORED] {MIN_EPISODE_SECONDS:.0f}초 전의 Space는 무시합니다.", flush=True)
        if elapsed >= cfg["max_seconds"]:
            outcome = "timeout"
            break
        sample = elapsed >= next_sample
        observation = None
        if sample:
            observation = guard.run(follower.get_observation)
            next_sample += sample_period
            if elapsed - next_sample > sample_period:
                late += 1
                next_sample = elapsed + sample_period
        sent = guard.run(teleop_step, leader, follower, start_pose, ramp_started)
        if sample and (observation is None or sent is None):
            late += 1
        elif sample:
            writer.add({**build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                        **build_dataset_frame(dataset.features, sent, prefix=ACTION), "task": TASK})
            sample_times.append(time.perf_counter())
        status.set(f"REC {elapsed:4.1f}/{cfg['max_seconds']:.0f}s  frames {len(sample_times)}  "
                   "SPACE save  ESC discard", RED)
        used = time.perf_counter() - loop
        over_budget += used > period
        precise_sleep(max(0.0, period - used))

    writer.finish(outcome == "saved")
    gaps = np.diff(sample_times) * 1000 if len(sample_times) > 1 else np.array([0.0])
    seconds = sample_times[-1] - sample_times[0] if len(sample_times) > 1 else 0.0
    fps = (len(sample_times) - 1) / seconds if seconds > 0 else 0.0
    row = {"time": datetime.now().isoformat(timespec="seconds"), "status": outcome,
           "episode": index if outcome == "saved" else "", "red_position": position,
           "frames": len(sample_times), "seconds": round(seconds, 2), "fps": round(fps, 2),
           "gap_p95_ms": round(float(np.percentile(gaps, 95)), 1), "gap_max_ms": round(float(gaps.max()), 1),
           "late_samples": late, "control_over_budget": over_budget}
    return PendingEpisode(writer, root, outcome, len(sample_times), row)


def run_session(cfg, follower, leader, keys, status):
    """준비 -> 자세 맞추기 -> 3초 대기 -> 기록을 반복하고, 끝날 때 팔을 내려놓고 토크를 끕니다."""
    set_aside_incomplete(cfg["root"])
    if next_episode(cfg["root"], cfg["target"]) is None:
        print(f"[DONE] 목표 {cfg['target']}개를 이미 모두 저장했습니다: {cfg['root']}", flush=True)
        return "target_reached"
    torque_on = False
    pending = None
    save_failed = False
    preflight_started = time.monotonic()
    esc_ignore_until = finish_armed_until = 0.0
    guard = CommGuard(limit=100)
    leader_pose = follower_pose = None
    keys.clear()

    while True:
        # 준비: 팔로워는 움직이지 않습니다. 상자를 옮기고 리더를 팔로워 자세에 맞춥니다.
        status.state = "BUSY" if pending else "PREPARE"
        if pending and not pending.writer.is_alive():
            save_failed = not complete_pending(pending, cfg) or save_failed
            pending = None
        index = next_episode(cfg["root"], cfg["target"])
        done = index is None and not pending
        guide = cfg.get("guide")
        if guide and index is not None:
            status.guide = guide[index % len(guide)]
            item = next(box for box in status.guide if box[3] == "red")
            position = f"x{item[0]} y{item[1]} {item[2]}"
            others = " + BLACK, BLUE" if len(status.guide) > 1 else ""
            label = f"RED BOX {'LONG SIDE VERTICAL' if item[2] == 'V' else 'LONG SIDE HORIZONTAL'}{others} (see top view)"
        else:
            position = (index % RED_POSITIONS + 1) if index is not None else 0
            label = f"RED POSITION {position}/{RED_POSITIONS}"
        count = len(saved_episodes(cfg["root"]))
        status.title = (f"{cfg['name']}  saved {count}/{cfg['target']}  "
                        + (f"NEXT episode_{index:04d}  {label}" if index is not None else "ALL DONE")
                        + "  | ENTER next  ESC finish")
        leader_pose = guard.run(positions, leader.bus) or leader_pose
        follower_pose = guard.run(positions, follower.bus) or follower_pose
        if leader_pose is None or follower_pose is None:
            time.sleep(0.1)
            continue
        joint, error = pose_error(leader_pose, follower_pose)
        ok = error <= cfg["start_max_error"] and guard.misses == 0
        hold = "arm HOLDING" if torque_on else "torque OFF"
        finish_armed = time.monotonic() < finish_armed_until
        if pending:
            status.set(f"SAVING... {hold}", ORANGE)
        elif finish_armed:
            status.set("Press ESC again to finish and lower the arm", ORANGE)
        elif save_failed:
            status.set("SAVE FAILED: see console. ESC to finish", RED)
        elif done:
            status.set(f"ALL DONE. {joint} diff {error:.0f}. Align leader, then ESC to finish", GREEN)
        else:
            status.set(f"PREPARE: {hold}. {joint} diff {error:.0f} (limit {cfg['start_max_error']:.0f}) "
                       + ("READY" if ok else "ALIGN LEADER"), GREEN if ok else RED)

        if cfg.get("preflight_seconds"):
            if time.monotonic() - preflight_started >= cfg["preflight_seconds"]:
                print(f"[PREFLIGHT] {joint} 차이 {error:.1f}. 토크는 켜지 않았습니다.", flush=True)
                for j in JOINTS:
                    print(f"  {j:14s} leader={leader_pose[f'{j}.pos']:8.2f} follower={follower_pose[f'{j}.pos']:8.2f}", flush=True)
                return "preflight"
            keys.clear()
            time.sleep(0.1)
            continue

        if esc_pressed(keys, esc_ignore_until):
            keys.esc.clear()
            if not torque_on:
                print("[EXIT] 토크를 켜지 않은 상태로 종료합니다.", flush=True)
                return "finished"
            if pending:
                print("[WAIT] 저장이 끝난 뒤 다시 Esc를 누르세요.", flush=True)
            elif not ok:
                print(f"[WAIT] {joint} 차이가 {error:.0f}입니다. 리더를 맞춘 뒤 다시 Esc를 누르세요.", flush=True)
            elif not finish_armed:
                # 실수로 종료하지 않도록 한 번 더 확인합니다.
                finish_armed_until = time.monotonic() + FINISH_CONFIRM_SECONDS
                esc_ignore_until = time.monotonic() + ESC_GUARD_SECONDS
                print(f"[CONFIRM] 종료하려면 {FINISH_CONFIRM_SECONDS:.0f}초 안에 Esc를 한 번 더 누르세요.", flush=True)
            else:
                finish_armed_until = 0.0
                # 종료: 리더로 팔을 내려놓은 뒤 토크를 끕니다. 끄는 순간 팔이 떨어지기 때문입니다.
                keys.clear()
                status.state = "FINISH"
                print("[FINISH] 리더로 팔을 내려놓고 Enter: 토크 끄기 / Esc: 준비로 돌아가기", flush=True)
                start_pose, ramp = positions(follower.bus), time.perf_counter()
                follow_until(follower, leader, keys, status, start_pose, ramp,
                             lambda: "FINISH: lower arm with leader, ENTER = torque OFF, ESC = back", YELLOW,
                             lambda: keys.enter.is_set() or keys.esc.is_set())
                if keys.enter.is_set():
                    follower.bus.disable_torque()
                    status.set("Torque OFF", GREEN)
                    print("[TORQUE OFF] 팔로워 토크를 껐습니다.", flush=True)
                    return "finished"
                guard.run(hold_here, follower)
                keys.clear()
                esc_ignore_until = time.monotonic() + ESC_GUARD_SECONDS
                continue

        if keys.enter.is_set():
            keys.enter.clear()
            if pending:
                print("[WAIT] 저장 중입니다. 잠시 후 Enter를 누르세요.", flush=True)
            elif done:
                print("[DONE] 목표를 모두 저장했습니다. 리더를 맞춘 뒤 Esc로 종료하세요.", flush=True)
            elif save_failed:
                print("[STOP] 저장 실패 후에는 이어 찍지 않습니다. Esc로 종료하고 메시지를 확인하세요.", flush=True)
            elif not ok:
                print(f"[WAIT] {joint} 차이가 {error:.0f}입니다. 리더를 팔로워 자세에 맞춰 주세요.", flush=True)
            else:
                if not torque_on:
                    enable_holding(follower)
                    torque_on = True
                # 자세 맞추기: 팔이 리더를 따라가지만 기록하지 않습니다.
                keys.clear()
                status.state = "POSITION"
                print(f"[POSITION] episode_{index:04d}, 빨간 상자 위치 {position}번. 시작 자세를 잡고 Enter.", flush=True)
                start_pose, ramp = positions(follower.bus), time.perf_counter()
                position_esc_ignore = time.monotonic() + ESC_GUARD_SECONDS
                while True:
                    follow_until(follower, leader, keys, status, start_pose, ramp,
                                 lambda: f"POSITION episode_{index:04d}: start pose (not recording)  ENTER start  ESC prepare",
                                 YELLOW, lambda: keys.enter.is_set() or esc_pressed(keys, position_esc_ignore))
                    if keys.esc.is_set():
                        print("[BACK] 준비로 돌아갑니다. 팔은 멈춰 있습니다.", flush=True)
                        break
                    try:
                        pending = record_attempt(cfg, follower, leader, keys, status, index, position)
                    except (ConnectionError, TimeoutError):
                        raise
                    except Exception:
                        # 저장 문제로 팔을 떨어뜨리지 않습니다. 멈춘 뒤 Esc로 안전하게 끝낼 수 있습니다.
                        traceback.print_exc()
                        print("[ERROR] 이번 시도를 시작하지 못했습니다. Esc로 팔을 내려놓고 메시지를 확인하세요.", flush=True)
                        save_failed = True
                        break
                    if pending.outcome == "saved":
                        print("[SAVING] 저장 중입니다.", flush=True)
                        break
                    # 버린 시도는 같은 번호의 시작 지점으로 돌아갑니다. Enter 한 번이면 다시 시작합니다.
                    print({"discarded": "[RETRY] 기록을 버렸습니다.",
                           "timeout": f"[TIMEOUT] {cfg['max_seconds']:.0f}초를 넘겨 저장하지 않습니다."}[pending.outcome]
                          + f" episode_{index:04d}의 시작 지점으로 돌아갑니다.", flush=True)
                    status.state = "BUSY"
                    writer_done = pending.writer
                    start_pose, ramp = positions(follower.bus), time.perf_counter()
                    follow_until(follower, leader, keys, status, start_pose, ramp,
                                 lambda: "Discarding...", ORANGE, lambda: not writer_done.is_alive())
                    save_failed = not complete_pending(pending, cfg) or save_failed
                    pending = None
                    keys.clear()
                    position_esc_ignore = time.monotonic() + ESC_GUARD_SECONDS
                    status.state = "POSITION"
                guard.run(hold_here, follower)
                keys.clear()
                esc_ignore_until = time.monotonic() + ESC_GUARD_SECONDS
                continue
        time.sleep(0.1)


def build_robot(camera_map=CAMERAS):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.opencv.configuration_opencv import Cv2Backends
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    cameras = {n: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=DATASET_FPS, fourcc="MJPG",
                                     backend=Cv2Backends.DSHOW) for n, i in camera_map.items()}
    follower = SO101Follower(SO101FollowerConfig(port="COM10", id="my_so101", use_degrees=False, cameras=cameras))
    leader = SO101Leader(SO101LeaderConfig(port="COM3", id="my_so101_leader", use_degrees=False))
    return follower, leader


def connect_without_torque(follower, leader):
    """캘리브레이션은 바꾸지 않습니다. 일치하지 않으면 중단합니다."""
    follower.bus.connect()
    leader.bus.connect()
    if not follower.bus.is_calibrated or not leader.bus.is_calibrated:
        raise RuntimeError("캘리브레이션이 파일과 다릅니다. 이 프로그램은 캘리브레이션을 바꾸지 않습니다.")
    for camera in follower.cameras.values():
        camera.connect()


def shutdown(follower, leader):
    try:
        if follower.bus.is_connected:
            follower.bus.disable_torque()
            follower.bus.disconnect(disable_torque=False)
    finally:
        if leader.bus.is_connected:
            leader.bus.disconnect(disable_torque=False)
        for camera in follower.cameras.values():
            if camera.is_connected:
                camera.disconnect()


class FakeBus:
    def __init__(self, pose):
        self.pose = dict(pose)
        self.torque = 0
        self.goals = []
        self.is_connected = True
        self.is_calibrated = True
        self.reads = 0
        self.fail_every = 0

    def sync_read(self, field, num_retry=0):
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            raise ConnectionError("simulated missing status packet")
        return {j: self.torque for j in JOINTS} if field == "Torque_Enable" else dict(self.pose)

    def sync_write(self, field, values):
        self.goals.append(dict(values))
        self.pose.update(values)

    def enable_torque(self):
        self.torque = 1

    def disable_torque(self):
        self.torque = 0


class FakeCamera:
    def __init__(self):
        self.count = 0

    def read_latest(self, max_age_ms=500):
        self.count += 1
        return np.full((480, 640, 3), self.count % 255, dtype=np.uint8)


class FakeFollower:
    name = "so_follower"

    def __init__(self):
        self.bus = FakeBus({j: 0.0 for j in JOINTS})
        self.cameras = {n: FakeCamera() for n in CAMERAS}
        self.action_features = {f"{j}.pos": float for j in JOINTS}
        self.observation_features = {**self.action_features, **{n: (480, 640, 3) for n in CAMERAS}}

    def configure(self):
        pass

    def get_observation(self):
        return {**{f"{j}.pos": v for j, v in self.bus.pose.items()},
                **{n: c.read_latest() for n, c in self.cameras.items()}}

    def send_action(self, action):
        self.bus.sync_write("Goal_Position", {k.removesuffix(".pos"): v for k, v in action.items()})
        return dict(action)


class FakeLeader:
    def __init__(self, offset=0.0):
        self.bus = FakeBus({j: offset for j in JOINTS})

    def get_action(self):
        return {f"{j}.pos": float(v) for j, v in self.bus.sync_read("Present_Position").items()}


def press_after(*steps):
    def run():
        for delay, event in steps:
            time.sleep(delay)
            event.set()
    threading.Thread(target=run, daemon=True).start()


def press_when(status, keys, *steps):
    """지정한 상태가 되면 잠시 뒤 키를 누릅니다. (상태, 대기 초, 키 이름)"""
    def run():
        for state, delay, name in steps:
            while status.state != state:
                time.sleep(0.01)
            time.sleep(delay)
            getattr(keys, name).set()
            time.sleep(0.25)
    threading.Thread(target=run, daemon=True).start()


def self_test():
    """모터와 카메라 없이 실제 LeRobot 저장 경로로 연속 수집 규칙을 검사합니다."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = {"root": tmp / "data", "name": "selftest", "target": 3, "start_max_error": START_MAX_ERROR,
               "countdown": 0.6, "max_seconds": 3.2, "log": tmp / "episodes.csv"}

        def session(*steps, leader_offset=0.0):
            keys, status, follower = Keys(), Status(""), FakeFollower()
            press_when(status, keys, *steps)
            return run_session(cfg, follower, FakeLeader(leader_offset), keys, status), follower

        result, follower = session(("PREPARE", 0.3, "enter"), ("PREPARE", 0.3, "esc"), leader_offset=50.0)
        assert result == "finished" and follower.bus.torque == 0 and not follower.bus.goals, "mismatch must not enable torque"
        assert saved_episodes(cfg["root"]) == []

        cfg["preflight_seconds"] = 0.6
        result, follower = session(("PREPARE", 0.1, "enter"))
        assert result == "preflight" and follower.bus.torque == 0 and not follower.bus.goals, "preflight must ignore Enter"
        cfg["preflight_seconds"] = 0.0

        # 한 번 실행에서: 폐기 -> 자세에서 복귀 -> 시간 초과 -> 저장 -> 저장 -> 목표 도달 후 종료.
        result, follower = session(
            ("PREPARE", 0.1, "enter"), ("POSITION", 0.1, "enter"), ("RECORD", 0.5, "esc"),
            ("POSITION", 0.1, "esc"),  # 복귀 직후의 Esc 연타는 무시합니다.
            ("POSITION", 0.1, "enter"), ("COUNTDOWN", 0.05, "esc"),
            ("POSITION", 0.9, "esc"),
            ("PREPARE", 0.9, "enter"), ("POSITION", 0.1, "enter"), ("RECORD", 0.3, "space"), ("RECORD", 2.2, "space"),
            ("PREPARE", 0.9, "enter"), ("POSITION", 0.1, "enter"), ("RECORD", 2.3, "space"),
            ("PREPARE", 0.9, "esc"), ("PREPARE", 0.8, "esc"), ("FINISH", 0.1, "esc"),
            ("PREPARE", 0.9, "esc"), ("PREPARE", 0.8, "esc"), ("FINISH", 0.1, "enter"))
        assert result == "finished" and follower.bus.torque == 0
        assert follower.bus.goals[0] == {j: 0.0 for j in JOINTS}, "goal must be primed before torque"
        assert saved_episodes(cfg["root"]) == [0, 1], saved_episodes(cfg["root"])
        for index in (0, 1):
            root = episode_dir(cfg["root"], index)
            frames = episode_frames(root)
            assert frames and frames >= 50, (index, frames)
            loaded = LeRobotDataset(f"local/selftest_{index:04d}", root=root)
            assert loaded.num_episodes == 1 and loaded.fps == DATASET_FPS and loaded.num_frames == frames
            assert loaded[frames - 1]["observation.images.top_cam"].shape == (3, 480, 640)

        # 다시 실행: 불완전한 폴더는 옮기고, 지운 번호부터 찍고, 시간 초과는 저장하지 않습니다.
        shutil.rmtree(episode_dir(cfg["root"], 0))
        broken = episode_dir(cfg["root"], 2)
        broken.mkdir()
        (broken / "partial.txt").write_text("x", encoding="utf-8")
        assert next_episode(cfg["root"], cfg["target"]) == 0, "must refill the deleted number"
        result, _ = session(("PREPARE", 0.1, "enter"), ("POSITION", 0.1, "enter"), ("RECORD", 0.1, "enter"),
                            ("POSITION", 0.9, "esc"), ("PREPARE", 0.9, "esc"), ("PREPARE", 0.8, "esc"), ("FINISH", 0.1, "enter"))
        assert result == "finished" and saved_episodes(cfg["root"]) == [1], "timeout must not save"

        # 일시적 통신 오류가 있어도 멈추지 않고 저장합니다.
        keys, status, follower, leader = Keys(), Status(""), FakeFollower(), FakeLeader()
        follower.bus.fail_every, leader.bus.fail_every = 13, 17
        press_when(status, keys, ("PREPARE", 0.1, "enter"), ("POSITION", 0.1, "enter"), ("RECORD", 2.3, "space"),
                   ("PREPARE", 0.9, "esc"), ("PREPARE", 0.8, "esc"), ("FINISH", 0.1, "enter"))
        assert run_session(cfg, follower, leader, keys, status) == "finished" and follower.bus.torque == 0
        assert saved_episodes(cfg["root"]) == [0, 1], "transient errors must not stop recording"
        shutil.rmtree(episode_dir(cfg["root"], 0))
        assert (cfg["root"] / "_incomplete").exists(), "incomplete folder must be set aside, not deleted"
        with cfg["log"].open(encoding="utf-8", newline="") as f:
            statuses = [row["status"] for row in csv.DictReader(f)]
        assert statuses == ["discarded", "discarded", "saved", "saved", "timeout", "saved"], statuses

        cfg["target"] = 1
        shutil.copytree(episode_dir(cfg["root"], 1), episode_dir(cfg["root"], 0))
        assert run_session(cfg, FakeFollower(), FakeLeader(), Keys(), Status(""))  == "target_reached"
    print("[SELF-TEST] 연속 수집, 폐기 후 복귀, 시간 초과, 폴더별 저장, 번호 채우기, 자세 불일치 검사 통과. 모터와 카메라는 사용하지 않았습니다.")


class Tee:
    """콘솔 출력을 파일에도 남깁니다. 창을 닫아도 종료 이유를 볼 수 있습니다."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_console_log(dataset):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"record_{dataset}_{datetime.now():%Y%m%d_%H%M%S}.log"
    file = path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=file, all_threads=True)
    sys.stdout, sys.stderr = Tee(sys.stdout, file), Tee(sys.stderr, file)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="redbox_30fps")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--guide", choices=tuple(GUIDES), default=None, help="top 화면에 놓을 위치를 표시합니다.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-seconds", type=float, default=0.0,
                        help="토크를 켜지 않고 카메라와 자세만 확인한 뒤 종료합니다.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    cfg = {"root": DATASET_ROOT / args.dataset, "name": args.dataset, "target": args.target,
           "start_max_error": START_MAX_ERROR, "countdown": COUNTDOWN_SECONDS, "max_seconds": MAX_EPISODE_SECONDS,
           "log": LOG_DIR / f"episodes_{args.dataset}.csv", "preflight_seconds": args.preflight_seconds,
           "guide": GUIDES[args.guide]() if args.guide else None}
    log_file = start_console_log(args.dataset)
    follower, leader = build_robot()
    keys, status, preview = Keys(), Status(""), None
    try:
        connect_without_torque(follower, leader)
        print("[READY] 토크 꺼짐. 상자와 팔을 준비한 뒤 Enter를 누르세요. 시간 제한은 없습니다.", flush=True)
        keys.start()
        preview = Preview(follower.cameras, status)
        preview.start()
        result = run_session(cfg, follower, leader, keys, status)
        print(f"[EXIT REASON] 정상 종료: {result}", flush=True)
    except BaseException:
        print("[EXIT REASON] 오류로 종료했습니다. 토크가 꺼지니 팔을 받쳐 주세요.", flush=True)
        traceback.print_exc()
        raise
    finally:
        print(f"[LOG] {log_file}", flush=True)
        keys.stop()
        if preview:
            preview.close()
        shutdown(follower, leader)
        print(f"[END] 저장된 에피소드 {len(saved_episodes(cfg['root']))}/{cfg['target']}개: {cfg['root']}", flush=True)


if __name__ == "__main__":
    main()
