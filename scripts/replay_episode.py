"""SO-ARM101 재생 시험: 저장된 시연의 명령을 그대로 보내 로봇이 재현할 수 있는지 확인합니다. 데이터는 바꾸지 않습니다.
배치 (토크 꺼짐) -> Enter 시작 자세로 천천히 이동 -> Space 속도 변경, Enter 재생 -> Esc 즉시 멈춤.
재생 후 Enter로 시작 자세로 돌아갑니다. 시작 자세에서 Esc 두 번이면 토크를 끄고 끝납니다. 비상시에는 전원을 차단하세요.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile
import threading
import time

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.utils.robot_utils import precise_sleep

from record_episode import (CONTROL_FPS, DATASET_FPS, DATASET_ROOT, FINISH_CONFIRM_SECONDS, JOINTS, LOG_DIR, CommGuard,
                            FakeFollower, FakeLeader, Keys, Status, build_robot, connect_without_torque,
                            enable_holding, esc_pressed, hold_here, positions, press_when, shutdown)

SPEEDS = (0.25, 0.5, 1.0)
MOVE_SPEED = 25.0
LAG_LIMIT = 60.0
MODEL = Path(__file__).resolve().parents[1] / "models" / "yolo" / "graybasket_v2" / "weights" / "best.pt"
GREEN, RED, YELLOW, ORANGE = (80, 230, 80), (60, 60, 255), (0, 255, 255), (0, 200, 255)


def load_episode(dataset, index):
    folder = DATASET_ROOT / dataset / f"episode_{index:04d}"
    table = pq.read_table(next((folder / "data").rglob("*.parquet")), columns=["action"]).to_pydict()
    names = json.loads((folder / "meta" / "info.json").read_text(encoding="utf-8"))["features"]["action"]["names"]
    if list(names) != [f"{j}.pos" for j in JOINTS]:
        raise RuntimeError(f"관절 순서가 다릅니다: {names}")
    video = next((folder / "videos" / "observation.images.top_cam").rglob("*.mp4"))
    with av.open(str(video)) as container:
        first = next(container.decode(video=0)).to_ndarray(format="bgr24")
    return np.array(table["action"], dtype=np.float64), first


def to_action(values):
    return {f"{j}.pos": float(v) for j, v in zip(JOINTS, values)}


def sample_action(actions, position):
    """저장된 30 FPS 명령 사이를 직선으로 잇습니다."""
    low = min(int(position), len(actions) - 1)
    high = min(low + 1, len(actions) - 1)
    return actions[low] + (actions[high] - actions[low]) * (position - low)


class ReplayPreview(threading.Thread):
    """실시간 top 화면에 기준 시연의 첫 장면을 반투명으로 겹칩니다."""

    def __init__(self, camera, guide, status):
        super().__init__(daemon=True)
        self.camera, self.guide, self.status = camera, guide, status
        self.stop_event = threading.Event()
        self.show_guide = True

    def run(self):
        window = "Replay test"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while not self.stop_event.is_set():
            try:
                frame = cv2.cvtColor(self.camera.read_latest(), cv2.COLOR_RGB2BGR)
            except Exception:
                frame = np.zeros_like(self.guide)
            if self.show_guide:
                frame = cv2.addWeighted(frame, 0.5, self.guide, 0.5, 0)
            canvas = cv2.resize(frame, None, fx=1.5, fy=1.5)
            canvas = cv2.copyMakeBorder(canvas, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            title, line, color = self.status.get()
            y = canvas.shape[0] - 52
            cv2.putText(canvas, title, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(canvas, line, (8, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(window, canvas)
            cv2.waitKey(50)
        cv2.destroyWindow(window)

    def close(self):
        self.stop_event.set()
        self.join(timeout=3)


def move_to(follower, target, keys, status, label, speed=MOVE_SPEED):
    """모든 관절을 제한 속도로 목표 자세까지 옮깁니다. Esc를 누르면 그 자리에 멈춥니다."""
    period = 1 / CONTROL_FPS
    goal = np.array([positions(follower.bus)[f"{j}.pos"] for j in JOINTS])
    guard = CommGuard()
    keys.clear()
    last = time.perf_counter()
    while not keys.esc.is_set():
        loop = time.perf_counter()
        step = speed * (loop - last)
        last = loop
        goal = goal + np.clip(target - goal, -step, step)
        guard.run(follower.send_action, to_action(goal))
        left = float(np.abs(target - goal).max())
        status.set(f"{label}  remaining {left:.0f}  (ESC stop)", ORANGE)
        if left < 0.01:
            return True
        precise_sleep(max(0.0, period - (time.perf_counter() - loop)))
    hold_here(follower)
    print("[STOP] 이동을 멈췄습니다.", flush=True)
    return False


def play(follower, actions, speed, keys, status, lag_limit):
    """명령을 그대로 재생하며 지연을 기록합니다. 돌려주는 값은 (끝난 이유, 기록)입니다."""
    period = 1 / CONTROL_FPS
    total = (len(actions) - 1) / DATASET_FPS / speed
    lags = {j: [] for j in JOINTS}
    guard = CommGuard()
    keys.clear()
    started = time.perf_counter()
    last_read = 0.0
    commands, over_budget = 0, 0
    reason = "completed"
    while True:
        loop = time.perf_counter()
        elapsed = loop - started
        if keys.esc.is_set():
            reason = "operator_stop"
            break
        position = elapsed * DATASET_FPS * speed
        goal = sample_action(actions, position)
        guard.run(follower.send_action, to_action(goal))
        commands += 1
        if loop - last_read >= 0.1:
            last_read = loop
            now = guard.run(positions, follower.bus)
            if now:
                errors = [abs(goal[i] - now[f"{j}.pos"]) for i, j in enumerate(JOINTS)]
                for j, value in zip(JOINTS, errors):
                    lags[j].append(value)
                if max(errors) > lag_limit:
                    reason = f"lag_limit: {JOINTS[int(np.argmax(errors))]} {max(errors):.1f}"
                    break
        status.set(f"REPLAY {speed:.0%}  {elapsed:4.1f}/{total:.1f}s  (ESC stop)", RED)
        if position >= len(actions) - 1:
            break
        used = time.perf_counter() - loop
        over_budget += used > period
        precise_sleep(max(0.0, period - used))
    seconds = time.perf_counter() - started
    summary = {"reason": reason, "speed": speed, "planned_seconds": round(total, 2), "seconds": round(seconds, 2),
               "commands": commands, "over_budget_loops": over_budget, "comm_misses": guard.total,
               "lag": {j: {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                           "max": round(float(np.max(v)), 2)} if v else None for j, v in lags.items()}}
    return reason, summary


def judge_top(model, frame):
    """마지막 장면에서 빨간 상자 중심이 바구니 박스 안인지 봅니다."""
    if model is None:
        return None
    result = model.predict(frame, conf=0.5, imgsz=640, device=0, verbose=False)[0]
    found = {}
    for box in result.boxes:
        cls, conf = int(box.cls), float(box.conf)
        if cls not in found or conf > found[cls][0]:
            found[cls] = (conf, box.xyxy[0].tolist())
    if 0 not in found or 1 not in found:
        return {"red_conf": found.get(0, (None,))[0], "basket_conf": found.get(1, (None,))[0], "inside": None}
    (red_conf, r), (basket_conf, b) = found[0], found[1]
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    return {"red_conf": round(red_conf, 2), "basket_conf": round(basket_conf, 2),
            "inside": bool(b[0] <= cx <= b[2] and b[1] <= cy <= b[3])}


def run_replay(cfg, follower, keys, status, preview, model=None):
    actions = cfg["actions"]
    start = actions[0]
    label = f"episode_{cfg['episode']:04d}"
    status.title = f"{label}: place objects on the overlay | ENTER move to start | ESC quit"
    status.state = "PLACE"
    status.set("PLACE: torque OFF. Match boxes to the overlay", GREEN)
    keys.clear()
    while not keys.enter.is_set():
        if keys.esc.is_set():
            print("[EXIT] 토크를 켜지 않고 종료합니다.", flush=True)
            return "cancelled"
        time.sleep(0.05)
    enable_holding(follower)
    print("[MOVE] 시작 자세로 천천히 이동합니다.", flush=True)
    status.state = "MOVE"
    at_start = move_to(follower, start, keys, status, "MOVE to start")
    speed_index = 0
    results = []
    finish_until = 0.0
    esc_ignore = 0.0
    keys.clear()
    while True:
        status.state = "READY" if at_start else "STOPPED"
        speed = SPEEDS[speed_index]
        status.title = (f"{label}  SPEED {speed:.0%}  | SPACE change speed  ENTER "
                        + ("replay" if at_start else "move to start") + "  ESC twice finish")
        armed = time.monotonic() < finish_until
        status.set("Press ESC again: torque OFF, arm may drop" if armed else
                   ("READY: check the boxes" if at_start else "STOPPED: arm holding. ENTER returns to start"),
                   ORANGE if armed else GREEN)
        if keys.space.is_set():
            keys.space.clear()
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
            else:
                if preview:
                    preview.show_guide = False
                status.state = "REPLAY"
                print(f"[REPLAY] {speed:.0%} 재생 시작", flush=True)
                reason, summary = play(follower, actions, speed, keys, status, cfg.get("lag_limit", LAG_LIMIT))
                hold_here(follower)
                time.sleep(0.5)
                frame = None
                if cfg.get("camera"):
                    frame = cv2.cvtColor(cfg["camera"].read_latest(), cv2.COLOR_RGB2BGR)
                summary["time"] = datetime.now().isoformat(timespec="seconds")
                summary["episode"] = cfg["episode"]
                summary["yolo_top"] = judge_top(model, frame) if frame is not None else None
                stem = f"replay_ep{cfg['episode']:04d}_{speed * 100:.0f}pct_{datetime.now():%H%M%S}"
                cfg["log_dir"].mkdir(parents=True, exist_ok=True)
                (cfg["log_dir"] / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                if frame is not None:
                    cv2.imwrite(str(cfg["log_dir"] / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                results.append(summary)
                worst = {j: v["max"] for j, v in summary["lag"].items() if v}
                print(f"[RESULT] {speed:.0%}: {reason}, {summary['seconds']}초, 관절 최대 지연 {worst}, YOLO {summary['yolo_top']}", flush=True)
                print(f"[LOG] {cfg['log_dir'] / stem}", flush=True)
                if preview:
                    preview.show_guide = True
                at_start = False
            keys.clear()
            esc_ignore = time.monotonic() + 0.7
            finish_until = 0.0
        time.sleep(0.05)


def self_test():
    """모터 없이 재생 흐름과 속도, 멈춤, 지연 중단을 검사합니다."""
    steps = np.linspace(0.0, 1.0, 31)[:, None] * np.array([30.0, -20.0, 10.0, 5.0, -5.0, 40.0])
    assert np.allclose(sample_action(steps, 1.5), (steps[1] + steps[2]) / 2)
    assert np.allclose(sample_action(steps, 99.0), steps[-1])
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"actions": steps, "episode": 7, "log_dir": Path(tmp)}
        keys, status, follower = Keys(), Status(""), FakeFollower()
        press_when(status, keys, ("PLACE", 0.1, "esc"))
        assert run_replay(cfg, follower, keys, status, None) == "cancelled" and follower.bus.torque == 0 and not follower.bus.goals

        keys, status, follower = Keys(), Status(""), FakeFollower()
        press_when(status, keys,
                   ("PLACE", 0.1, "enter"),
                   ("READY", 0.1, "space"), ("READY", 0.1, "space"),  # 25% -> 50% -> 100%
                   ("READY", 0.1, "enter"),            # 100%, 1초
                   ("STOPPED", 0.2, "enter"),          # 시작 자세로 복귀
                   ("READY", 0.1, "space"), ("READY", 0.1, "enter"),  # 25%, 4초 중 멈춤
                   ("REPLAY", 0.5, "esc"),
                   ("STOPPED", 0.9, "esc"), ("STOPPED", 0.9, "esc"))
        results = run_replay(cfg, follower, keys, status, None)
        assert follower.bus.torque == 0, "must end with torque off"
        assert follower.bus.goals[0] == {j: 0.0 for j in JOINTS}, "goal must be primed before torque"
        assert [r["speed"] for r in results] == [1.0, 0.25], results
        assert results[0]["reason"] == "completed" and 0.9 <= results[0]["seconds"] <= 1.3, results[0]
        assert results[1]["reason"] == "operator_stop" and results[1]["seconds"] < 2.0, results[1]
        assert len(list(Path(tmp).glob("replay_*.json"))) == 2

        cfg["lag_limit"] = -1.0
        keys, status, follower = Keys(), Status(""), FakeFollower()
        press_when(status, keys, ("PLACE", 0.1, "enter"), ("READY", 0.1, "enter"),
                   ("STOPPED", 0.9, "esc"), ("STOPPED", 0.9, "esc"))
        results = run_replay(cfg, follower, keys, status, None)
        assert results[0]["reason"].startswith("lag_limit"), results
    print("[SELF-TEST] 재생 속도, 복귀, 즉시 멈춤, 지연 중단, 토크 해제 검사 통과. 모터는 사용하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="redbox_30fps")
    parser.add_argument("--episode", type=int, default=25)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    from ultralytics import YOLO

    actions, guide = load_episode(args.dataset, args.episode)
    print(f"[EPISODE] episode_{args.episode:04d}: {len(actions)}프레임, {len(actions) / DATASET_FPS:.1f}초", flush=True)
    model = YOLO(str(MODEL))
    follower, leader = build_robot()
    keys, status, preview = Keys(), Status(""), None
    try:
        connect_without_torque(follower, leader)
        print("[READY] 토크 꺼짐. 화면에 겹친 시연 첫 장면에 맞춰 상자를 놓고 Enter를 누르세요.", flush=True)
        keys.start()
        preview = ReplayPreview(follower.cameras["top_cam"], guide, status)
        preview.start()
        cfg = {"actions": actions, "episode": args.episode, "log_dir": LOG_DIR / "replay", "camera": follower.cameras["top_cam"]}
        run_replay(cfg, follower, keys, status, preview, model)
    finally:
        keys.stop()
        if preview:
            preview.close()
        shutdown(follower, leader)
        print("[END] 재생 시험을 마쳤습니다.", flush=True)


if __name__ == "__main__":
    main()
