"""SO-ARM101 YOLO 학습과 시험: red_box, gray_basket 2클래스.
build: shots.csv의 구분대로 학습/검증/시험 폴더를 만듭니다.
train: yolo26n.pt에서 새로 학습합니다. 기존 결과는 덮어쓰지 않습니다.
test: 학습에 쓰지 않은 시험 사진으로 카메라별 검출 결과를 기록합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_DIR / "datasets" / "yolo"
DATASET_DIR = YOLO_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models" / "yolo"
BASE_WEIGHTS = Path(r"C:\Users\user\LeRobot\scripts\yolo_occlusion_3cam\weights\yolo26n.pt")
CLASSES = {0: "red_box", 1: "gray_basket"}
CAMERAS = ("top", "wrist", "belly")
SPLITS = ("train", "val", "test")
IOU_MATCH = 0.5
COLORS = {0: (40, 40, 245), 1: (30, 230, 90)}


def build():
    with (YOLO_DIR / "shots.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    counts = {s: 0 for s in SPLITS}
    for row in rows:
        split = row["split"]
        for cam in CAMERAS:
            stem = f"set{int(row['set']):02d}_{cam}"
            for sub, name, source in (("images", f"{stem}.jpg", YOLO_DIR / "images" / "raw"),
                                      ("labels", f"{stem}.txt", YOLO_DIR / "labels")):
                target = DATASET_DIR / sub / split
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target / name)
            counts[split] += 1
    config = {"path": DATASET_DIR.as_posix(), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": CLASSES}
    (DATASET_DIR / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"[BUILD] {counts}", flush=True)
    return counts


def train(name, epochs):
    from ultralytics import YOLO

    if (MODEL_DIR / name).exists():
        raise FileExistsError(f"이미 있는 모델입니다: {MODEL_DIR / name}")
    YOLO(str(BASE_WEIGHTS)).train(
        data=str(DATASET_DIR / "data.yaml"), epochs=epochs, patience=30, imgsz=640, batch=8,
        device=0, workers=0, seed=42, deterministic=True, hsv_h=0.015,
        project=str(MODEL_DIR), name=name, exist_ok=False, plots=True, verbose=False)
    print(f"[TRAINED] {MODEL_DIR / name / 'weights' / 'best.pt'}", flush=True)


def read_labels(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cls, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * width, float(cy) * height, float(bw) * width, float(bh) * height
            boxes.append((int(cls), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_image(truth, predicted):
    """같은 클래스끼리 IoU가 높은 순서로 짝지어 놓침과 오검출을 돌려줍니다."""
    pairs = sorted(((iou(t[1:], p[1:5]), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(predicted)
                    if t[0] == p[0]), reverse=True)
    used_t, used_p, matched = set(), set(), []
    for value, ti, pi in pairs:
        if value >= IOU_MATCH and ti not in used_t and pi not in used_p:
            used_t.add(ti)
            used_p.add(pi)
            matched.append((ti, pi, value))
    missed = [truth[i] for i in range(len(truth)) if i not in used_t]
    extra = [predicted[i] for i in range(len(predicted)) if i not in used_p]
    return matched, missed, extra


def test(name, conf):
    from ultralytics import YOLO

    run = MODEL_DIR / name
    model = YOLO(str(run / "weights" / "best.pt"))
    images = sorted((DATASET_DIR / "images" / "test").glob("*.jpg"))
    out = run / "test_predictions"
    out.mkdir(exist_ok=True)
    report = {"model": str(run / "weights" / "best.pt"), "conf": conf, "iou_match": IOU_MATCH, "images": []}
    totals = {"truth": 0, "matched": 0, "missed_red": 0, "missed_basket": 0, "false_red": 0, "false_basket": 0}
    for path in images:
        frame = cv2.imread(str(path))
        height, width = frame.shape[:2]
        truth = read_labels(DATASET_DIR / "labels" / "test" / f"{path.stem}.txt", width, height)
        result = model.predict(frame, conf=conf, imgsz=640, device=0, verbose=False)[0]
        predicted = [(int(b.cls), *b.xyxy[0].tolist(), float(b.conf)) for b in result.boxes]
        matched, missed, extra = match_image(truth, predicted)
        totals["truth"] += len(truth)
        totals["matched"] += len(matched)
        totals["missed_red"] += sum(m[0] == 0 for m in missed)
        totals["missed_basket"] += sum(m[0] == 1 for m in missed)
        totals["false_red"] += sum(e[0] == 0 for e in extra)
        totals["false_basket"] += sum(e[0] == 1 for e in extra)
        report["images"].append({"image": path.name, "camera": path.stem.split("_")[1],
                                 "truth": len(truth), "matched_iou": [round(m[2], 3) for m in matched],
                                 "missed": [CLASSES[m[0]] for m in missed],
                                 "extra": [[CLASSES[e[0]], round(e[5], 3)] for e in extra]})
        for t in truth:
            cv2.rectangle(frame, (int(t[1]), int(t[2])), (int(t[3]), int(t[4])), (255, 255, 255), 1)
        for p in predicted:
            cv2.rectangle(frame, (int(p[1]), int(p[2])), (int(p[3]), int(p[4])), COLORS[p[0]], 2)
            cv2.putText(frame, f"{CLASSES[p[0]]} {p[5]:.2f}", (int(p[1]), max(16, int(p[2]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[p[0]], 2)
        cv2.imwrite(str(out / path.name), frame)
        print(f"[TEST] {path.stem}: truth={len(truth)} matched={len(matched)} "
              f"missed={[CLASSES[m[0]] for m in missed]} extra={[(CLASSES[e[0]], round(e[5], 2)) for e in extra]}", flush=True)
    metrics = model.val(data=str(DATASET_DIR / "data.yaml"), split="test", imgsz=640, batch=8, device=0, workers=0,
                        project=str(run), name="test_metrics", exist_ok=True, verbose=False, plots=False)
    report["totals"] = totals
    report["map50"], report["map50_95"] = float(metrics.box.map50), float(metrics.box.map)
    (run / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TOTALS] {totals}", flush=True)
    print(f"[mAP] mAP50={report['map50']:.3f} mAP50-95={report['map50_95']:.3f} (사진이 적어 참고용)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "train", "test"), required=True)
    parser.add_argument("--name", default="graybasket_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if args.stage == "build":
        build()
    elif args.stage == "train":
        train(args.name, args.epochs)
    else:
        test(args.name, args.conf)


if __name__ == "__main__":
    main()
