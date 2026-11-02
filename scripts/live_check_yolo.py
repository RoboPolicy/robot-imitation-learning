"""SO-ARM101 YOLO 실시간 확인: 3개 카메라 화면에 검출 결과를 표시하고 기록합니다. 모터는 사용하지 않습니다.
Q/Esc 또는 시간이 끝나면 종료합니다. 결과는 모델 폴더의 live_check에 저장됩니다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from capture_label_yolo import CAMERAS, CLASSES, Camera

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "yolo"
COLORS = {0: (40, 40, 245), 1: (30, 230, 90)}
SAMPLE_SECONDS = 2.0
WINDOW = "YOLO live check"


def draw(frame, boxes, name, stats):
    image = frame.copy()
    for cls, conf, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls], 2)
        cv2.putText(image, f"{CLASSES[cls]} {conf:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls], 2)
    frames = max(1, stats["frames"])
    cv2.putText(image, f"{name.upper()} red {100 * stats['red'] / frames:.0f}%  basket {100 * stats['basket'] / frames:.0f}%",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="graybasket_v2")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run = MODEL_DIR / args.name
    model = YOLO(str(run / "weights" / "best.pt"))
    out = args.out or run / "live_check" / f"{datetime.now():%Y%m%d_%H%M%S}"
    (out / "samples").mkdir(parents=True, exist_ok=False)
    stats = {n: {"frames": 0, "red": 0, "basket": 0, "many_red": 0, "many_basket": 0} for n in CAMERAS}
    cameras = []
    latencies = []
    try:
        for name, index in CAMERAS.items():
            camera = Camera(name, index)
            cameras.append(camera)
            camera.start()
        time.sleep(1.5)
        print(f"[READY] {args.seconds:.0f}초 동안 확인합니다. Q/Esc: 종료", flush=True)
        start = last_sample = time.monotonic()
        while time.monotonic() - start < args.seconds:
            frames = {c.name_: c.latest()[0] for c in cameras}
            if any(f is None for f in frames.values()):
                continue
            names = list(frames)
            t0 = time.perf_counter()
            results = model.predict([frames[n] for n in names], conf=args.conf, imgsz=640, device=0, verbose=False)
            latencies.append(time.perf_counter() - t0)
            tiles = {}
            for name, result in zip(names, results):
                boxes = [(int(b.cls), float(b.conf), *b.xyxy[0].tolist()) for b in result.boxes]
                reds = sum(b[0] == 0 for b in boxes)
                baskets = sum(b[0] == 1 for b in boxes)
                s = stats[name]
                s["frames"] += 1
                s["red"] += reds > 0
                s["basket"] += baskets > 0
                s["many_red"] += reds > 1
                s["many_basket"] += baskets > 1
                tiles[name] = draw(frames[name], boxes, name, s)
            canvas = cv2.hconcat([tiles[n] for n in names])
            left = args.seconds - (time.monotonic() - start)
            cv2.putText(canvas, f"{left:.0f}s left | Q quit", (8, canvas.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if canvas.shape[1] > 1900:
                canvas = cv2.resize(canvas, None, fx=1900 / canvas.shape[1], fy=1900 / canvas.shape[1])
            cv2.imshow(WINDOW, canvas)
            if time.monotonic() - last_sample >= SAMPLE_SECONDS:
                last_sample = time.monotonic()
                cv2.imwrite(str(out / "samples" / f"{last_sample - start:06.1f}s.jpg"), cv2.hconcat([tiles[n] for n in names]),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()
    summary = {"model": str(run / "weights" / "best.pt"), "conf": args.conf, "cameras": stats,
               "mean_inference_ms_for_3_images": round(1000 * sum(latencies) / max(1, len(latencies)), 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[SAVED] {out}", flush=True)


if __name__ == "__main__":
    main()
