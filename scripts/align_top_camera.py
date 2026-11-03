"""SO-ARM101 카메라 맞추기 (top, belly): 기준 에피소드 화면을 실시간 영상에 겹쳐 보여 주고 바구니 위치 차이를 표시합니다.
모터는 사용하지 않습니다. T 겹침 방식 전환, S 현재 화면을 기록하고 종료, Q/Esc 기록 없이 종료.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
from ultralytics import YOLO

from capture_label_yolo import CAMERAS, Camera

PROJECT_DIR = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_DIR / "datasets" / "lerobot" / "redbox_30fps"
MODEL = PROJECT_DIR / "models" / "yolo" / "graybasket_v2" / "weights" / "best.pt"
CONFIG = PROJECT_DIR / "config"
WINDOW = "Top camera alignment"
TOLERANCE = 8.0
MODES = ("edges", "blend", "reference", "live")


def first_frame(index, camera="top"):
    video = next((EPISODES / f"episode_{index:04d}" / "videos" / f"observation.images.{camera}_cam").rglob("*.mp4"))
    with av.open(str(video)) as container:
        return next(container.decode(video=0)).to_ndarray(format="bgr24")


def basket_box(model, frame):
    result = model.predict(frame, conf=0.5, imgsz=640, device=0, verbose=False)[0]
    boxes = [(float(b.conf), [float(v) for v in b.xyxy[0].tolist()]) for b in result.boxes if int(b.cls) == 1]
    return max(boxes)[1] if boxes else None


def target_box(model, center, camera="top", spread=2):
    """기준 에피소드와 앞뒤 시연의 중간값으로 목표 위치를 정합니다."""
    boxes = [basket_box(model, first_frame(i, camera)) for i in range(center - spread, center + spread + 1)]
    boxes = [b for b in boxes if b]
    if not boxes:
        raise RuntimeError("기준 에피소드에서 바구니를 찾지 못했습니다.")
    return np.median(np.array(boxes), axis=0).tolist()


def compose(live, reference, edges, mode, target, box, line, color):
    if mode == "blend":
        canvas = cv2.addWeighted(live, 0.5, reference, 0.5, 0)
    elif mode == "reference":
        canvas = reference.copy()
    else:
        canvas = live.copy()
    if mode == "edges":
        canvas[edges > 0] = (0, 255, 0)
    cv2.rectangle(canvas, tuple(int(v) for v in target[:2]), tuple(int(v) for v in target[2:]), (255, 200, 0), 2)
    if box:
        cv2.rectangle(canvas, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (0, 255, 255), 2)
    canvas = cv2.resize(canvas, None, fx=1.5, fy=1.5)
    canvas = cv2.copyMakeBorder(canvas, 0, 74, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    y = canvas.shape[0] - 46
    cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(canvas, f"view={mode}  T switch view | S save and quit | Q quit   blue=target  yellow=basket now",
                (8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=25)
    parser.add_argument("--camera", choices=("top", "belly"), default="top")
    parser.add_argument("--seconds", type=float, default=0.0, help="지정하면 그 시간 뒤 결과만 출력하고 종료합니다.")
    args = parser.parse_args()
    model = YOLO(str(MODEL))
    reference = first_frame(args.episode, args.camera)
    target = target_box(model, args.episode, args.camera)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    print(f"[TARGET] episode_{args.episode:04d} 주변의 바구니 박스: {[round(v) for v in target]}", flush=True)
    camera = Camera(args.camera, CAMERAS[args.camera])
    camera.start()
    time.sleep(1.0)
    mode, box, frame, diffs = 0, None, None, None
    started = time.monotonic()
    try:
        while True:
            live, _ = camera.latest()
            if live is None:
                time.sleep(0.02)
                continue
            frame = live.copy()
            box = basket_box(model, frame)
            if box:
                diffs = [box[i] - target[i] for i in range(4)]
                worst = max(abs(d) for d in diffs)
                ok = worst <= TOLERANCE
                line = (f"left {diffs[0]:+.0f}  top {diffs[1]:+.0f}  right {diffs[2]:+.0f}  bottom {diffs[3]:+.0f}  "
                        + ("OK" if ok else "ADJUST"))
                color = (80, 230, 80) if ok else (60, 60, 255)
            else:
                diffs, line, color = None, "Basket not found", (60, 60, 255)
            cv2.imshow(WINDOW, compose(frame, reference, edges, MODES[mode], target, box, line, color))
            key = cv2.waitKey(1) & 0xFF
            if args.seconds and time.monotonic() - started >= args.seconds:
                print(f"[CHECK] {line}", flush=True)
                return
            if key in (ord("q"), ord("Q"), 27):
                print(f"[EXIT] 기록 없이 종료합니다. 마지막 상태: {line}", flush=True)
                return
            if key in (ord("t"), ord("T")):
                mode = (mode + 1) % len(MODES)
            if key in (ord("s"), ord("S")):
                CONFIG.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(CONFIG / f"{args.camera}_camera_reference.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (CONFIG / f"{args.camera}_camera_reference.json").write_text(json.dumps({
                    "saved_at": datetime.now().isoformat(timespec="seconds"), "reference_episode": args.episode,
                    "target_basket_box": [round(v, 1) for v in target],
                    "basket_box": [round(v, 1) for v in box] if box else None,
                    "difference": [round(d, 1) for d in diffs] if diffs else None}, indent=2), encoding="utf-8")
                print(f"[SAVED] {CONFIG / f'{args.camera}_camera_reference.json'} 차이: {line}", flush=True)
                return
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
