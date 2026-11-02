"""SO-ARM101 텔레오퍼레이션 확인: 리더를 움직여 팔로워가 따라오는지 봅니다. 데이터는 저장하지 않습니다.
준비 (토크 꺼짐) -> Enter 조종 시작 -> 리더로 팔을 내려놓고 Enter 토크 끄기. 비상시에는 전원을 차단하세요.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import time

import numpy as np

from lerobot.utils.robot_utils import precise_sleep

from record_episode import (CONTROL_FPS, JOINTS, LOG_DIR, START_MAX_ERROR, FakeFollower, FakeLeader, Keys, Preview,
                            Status, build_robot, connect_without_torque, enable_holding, pose_error, positions,
                            press_after, shutdown, teleop_step)


def run_check(follower, leader, keys, status, log_path, start_max_error=START_MAX_ERROR):
    status.title = "TELEOP CHECK (not recording)  | ENTER start  ESC quit"
    keys.clear()
    while True:
        joint, error = pose_error(positions(leader.bus), positions(follower.bus))
        ok = error <= start_max_error
        status.set(f"PREPARE: torque OFF. {joint} diff {error:.0f} (limit {start_max_error:.0f}) "
                   + ("READY" if ok else "ALIGN ARMS"), (80, 230, 80) if ok else (60, 60, 255))
        if keys.esc.is_set():
            print("[EXIT] 토크를 켜지 않고 종료합니다.", flush=True)
            return None
        if keys.enter.is_set():
            keys.enter.clear()
            if ok:
                break
            print(f"[WAIT] {joint} 차이가 {error:.0f}입니다. 리더와 팔로워 자세를 맞춰 주세요.", flush=True)
        time.sleep(0.1)

    start_pose = enable_holding(follower)
    print("[TELEOP] 토크 켜짐. 리더를 천천히 움직이세요. 끝나면 팔을 내려놓고 Enter.", flush=True)
    keys.clear()
    period = 1 / CONTROL_FPS
    started = time.perf_counter()
    last_read = 0.0
    loops, over_budget = 0, 0
    errors = {j: [] for j in JOINTS}
    motion = {j: [start_pose[f"{j}.pos"], start_pose[f"{j}.pos"]] for j in JOINTS}
    worst_text = ""
    while not keys.enter.is_set() and not keys.esc.is_set():
        loop = time.perf_counter()
        sent = teleop_step(leader, follower, start_pose, started)
        loops += 1
        if loop - last_read >= 0.1:
            last_read = loop
            now = positions(follower.bus)
            for j in JOINTS:
                value = now[f"{j}.pos"]
                errors[j].append(abs(sent[f"{j}.pos"] - value))
                motion[j] = [min(motion[j][0], value), max(motion[j][1], value)]
            worst = max(JOINTS, key=lambda j: errors[j][-1])
            worst_text = f"largest lag {worst} {errors[worst][-1]:.1f}"
        status.set(f"TELEOP {loop - started:5.1f}s  {worst_text}  | lower arm, ENTER = torque OFF", (0, 255, 255))
        used = time.perf_counter() - loop
        over_budget += used > period
        precise_sleep(max(0.0, period - used))
    seconds = time.perf_counter() - started
    follower.bus.disable_torque()
    print("[TORQUE OFF] 팔로워 토크를 껐습니다.", flush=True)

    result = {"time": datetime.now().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
              "control_hz": round(loops / seconds, 1) if seconds > 0 else 0.0, "over_budget_loops": over_budget,
              "joints": {j: {"moved_range": round(motion[j][1] - motion[j][0], 1),
                             "lag_median": round(float(np.median(errors[j])), 2) if errors[j] else None,
                             "lag_p95": round(float(np.percentile(errors[j], 95)), 2) if errors[j] else None,
                             "lag_max": round(float(np.max(errors[j])), 2) if errors[j] else None}
                         for j in JOINTS}}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"[SAVED] {log_path}", flush=True)
    return result


def self_test():
    """모터 없이 흐름만 검사합니다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "teleop.json"
        keys, follower = Keys(), FakeFollower()
        press_after((0.2, keys.esc))
        assert run_check(follower, FakeLeader(50.0), keys, Status(""), log) is None
        assert follower.bus.torque == 0 and not follower.bus.goals and not log.exists()

        keys, follower = Keys(), FakeFollower()
        press_after((0.2, keys.enter), (1.0, keys.enter))
        result = run_check(follower, FakeLeader(10.0), keys, Status(""), log)
        assert follower.bus.torque == 0 and log.exists() and result["control_hz"] > 30
        assert follower.bus.goals[0] == {j: 0.0 for j in JOINTS}, "goal must be primed before torque"
        steps = [g["elbow_flex"] for g in follower.bus.goals]
        assert max(abs(b - a) for a, b in zip(steps, steps[1:])) < 1.0, "start must ramp, not jump"
    print("[SELF-TEST] 자세 불일치 차단, 시작 램프, 토크 해제 검사 통과. 모터는 사용하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    follower, leader = build_robot()
    keys, status, preview = Keys(), Status(""), None
    try:
        connect_without_torque(follower, leader)
        print("[READY] 토크 꺼짐. 리더와 팔로워 자세를 맞춘 뒤 Enter를 누르세요.", flush=True)
        keys.start()
        preview = Preview(follower.cameras, status)
        preview.start()
        run_check(follower, leader, keys, status, LOG_DIR / f"teleop_check_{datetime.now():%Y%m%d_%H%M%S}.json")
    finally:
        keys.stop()
        if preview:
            preview.close()
        shutdown(follower, leader)
        print("[END] 확인을 마쳤습니다.", flush=True)


if __name__ == "__main__":
    main()
