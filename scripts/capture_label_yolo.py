"""SO-ARM101 YOLO 촬영 + 라벨링: 3개 카메라 x 20세트 = 총 60장.
Windows / Python + opencv-python 기준. 모터는 사용하지 않습니다.
실시간: Space 3대 화면 고정, Q/Esc 종료.
고정 후: 1/2 클래스, 드래그 상자, Tab 카메라, U 취소, X 선택 클래스 삭제, C 전체 삭제, E 대상 없음, R 다시 찍기, S 저장.
--relabel: 저장된 세트의 라벨만 수정합니다. R 다시 불러오기, N 건너뛰기, Q 종료.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import cv2
import numpy as np

CAMERAS ={"top": 0, "wrist": 1, "belly": 2}
CLASSES = {0: "red_box", 1: "gray_basket"}
CLASS_COLORS = {0: (40, 40, 245), 1: (30, 230, 90)}
PROJECT_DIR = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_DIR / "datasets" / "yolo"
WIDTH, HEIGHT = 640, 480
MIN_BOX = 4
STALE_SECONDS = 0.5
FIELDS = ["set", "split", "scene", "captured_at", "top_boxes", "wrist_boxes", "belly_boxes"]
# 화면 글자는 OpenCV 제약으로 영어만 표시합니다.
SETS = [
    (1, "train", "Shuffle the 3 boxes: new positions and order"),
    (2, "train", "Shuffle the 3 boxes: new positions and order"),
    (3, "train", "Shuffle the 3 boxes: new positions and order"),
    (4, "val", "New layout not used in sets 1-3"),
    (5, "train", "Arm approaching the red box"),
    (6, "val", "Gripper close to the red box"),
    (7, "train", "Gripper holding the red box, moving"),
    (8, "train", "Red box inside the basket"),
    (9, "train", "Move and rotate the basket"),
    (10, "train", "NO red box: black and blue only"),
    (11, "test", "New layout, arm away"),
    (12, "test", "New layout, gripper close to the red box"),
    (13, "test", "New layout, red box inside the basket"),
    # 2회차: graybasket_v1이 놓친 "바구니 안의 빨간 상자" 장면을 보강합니다.
    (14, "train", "Red box in LEFT side of basket, arm right above basket"),
    (15, "train", "Red box in RIGHT side of basket, wrist camera close"),
    (16, "train", "Red box in FRONT side of basket, visible through mesh in belly"),
    (17, "train", "Releasing red box: arm and hand over the basket"),
    (18, "train", "Basket moved a little, red box inside, wrist camera close"),
    (19, "test", "NEW TEST: red box inside basket, new spot and arm pose"),
    (20, "test", "NEW TEST: red box inside basket, another spot, wrist close"),
]


class Camera(threading.Thread):
    """프레임을 계속 받아 Space 시점의 최신 화면을 바로 쓸 수 있게 합니다."""

    def __init__(self, name, index):
        super().__init__(daemon=True)
        self.name_ = name
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"{name} 카메라(index={index}) 열기 실패. 다른 프로그램이 쓰고 있는지 확인하세요.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame, self.stamp = frame, time.monotonic()

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cap.release()


class Shot:
    """고정한 한 세트의 사진과 카메라별 라벨."""

    def __init__(self, frames):
        self.frames = frames
        self.boxes = {name: [] for name in frames}
        self.empty = set()

    def add(self, cam, cls, x1, y1, x2, y2):
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
            return False
        self.boxes[cam].append((cls, x1, y1, x2, y2))
        self.empty.discard(cam)
        return True

    def mark_empty(self, cam):
        if self.boxes[cam]:
            return False
        self.empty.add(cam)
        return True

    def unlabeled(self):
        return [c for c in self.frames if not self.boxes[c] and c not in self.empty]


def to_yolo_lines(boxes, width=WIDTH, height=HEIGHT):
    return [f"{cls} {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            for cls, x1, y1, x2, y2 in boxes]


def read_rows(root):
    path = root / "shots.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def next_set(root):
    done = {int(row["set"]) for row in read_rows(root)}
    return next((s for s in SETS if s[0] not in done), None)


def replace_bytes(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_shot(root, entry, shot):
    """사진과 라벨을 모두 쓴 뒤 마지막에 shots.csv에 기록합니다. 기록 전 중단되면 그 세트는 다시 찍습니다."""
    number, split, scene = entry
    missing = shot.unlabeled()
    if missing:
        raise ValueError(f"라벨이 없는 카메라: {missing}")
    images, labels = root / "images" / "raw", root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for cam, frame in shot.frames.items():
        stem = f"set{number:02d}_{cam}"
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"JPEG 변환 실패: {stem}")
        replace_bytes(images / f"{stem}.jpg", jpg.tobytes())
        lines = to_yolo_lines(shot.boxes[cam], frame.shape[1], frame.shape[0])
        replace_bytes(labels / f"{stem}.txt", ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    path = root / "shots.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"set": number, "split": split, "scene": scene,
                         "captured_at": datetime.now().isoformat(timespec="seconds"),
                         **{f"{c}_boxes": len(shot.boxes[c]) for c in CAMERAS if c in shot.boxes}})


def load_shot(root, number):
    """저장된 세트의 사진과 라벨을 다시 읽습니다."""
    frames, shot_boxes = {}, {}
    for cam in CAMERAS:
        stem = f"set{number:02d}_{cam}"
        frame = cv2.imread(str(root / "images" / "raw" / f"{stem}.jpg"))
        if frame is None:
            raise FileNotFoundError(f"사진을 읽을 수 없습니다: {stem}")
        height, width = frame.shape[:2]
        boxes = []
        for line in (root / "labels" / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                boxes.append((int(cls), round((cx - bw / 2) * width), round((cy - bh / 2) * height),
                              round((cx + bw / 2) * width), round((cy + bh / 2) * height)))
        frames[cam], shot_boxes[cam] = frame, boxes
    shot = Shot(frames)
    shot.boxes = shot_boxes
    shot.empty = {cam for cam, boxes in shot_boxes.items() if not boxes}
    return shot


def update_labels(root, entry, shot):
    """사진은 그대로 두고 라벨 파일과 shots.csv의 상자 수만 바꿉니다."""
    number = entry[0]
    missing = shot.unlabeled()
    if missing:
        raise ValueError(f"라벨이 없는 카메라: {missing}")
    for cam, frame in shot.frames.items():
        lines = to_yolo_lines(shot.boxes[cam], frame.shape[1], frame.shape[0])
        replace_bytes(root / "labels" / f"set{number:02d}_{cam}.txt",
                      ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    rows = read_rows(root)
    for row in rows:
        if int(row["set"]) == number:
            row.update({f"{c}_boxes": len(shot.boxes[c]) for c in shot.frames})
    tmp = root / "shots.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, root / "shots.csv")


def backup_labels(root):
    target = root / f"labels_backup_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(root / "labels", target)
    shutil.copy2(root / "shots.csv", target / "shots.csv")
    return target


class App:
    def __init__(self, root, save=save_shot):
        self.root = root
        self.save = save
        self.cams = list(CAMERAS)
        self.shot = None
        self.active = 0
        self.selected = self.cams[0]
        self.start = None
        self.drag = None
        self.message = ""
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            screen = ctypes.windll.user32.GetSystemMetrics(0)
        except (AttributeError, OSError):
            screen = 1920
        self.tile_w = min(WIDTH, (screen - 30) // len(self.cams))
        self.scale = self.tile_w / WIDTH
        self.tile_h = int(HEIGHT * self.scale)
        self.title = "YOLO capture + label"
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self.mouse)

    def locate(self, x, y):
        index = min(max(x, 0) // self.tile_w, len(self.cams) - 1)
        px = int((x - index * self.tile_w) / self.scale)
        py = int(y / self.scale)
        return self.cams[index], max(0, min(WIDTH - 1, px)), max(0, min(HEIGHT - 1, py))

    def mouse(self, event, x, y, flags, userdata):
        if self.shot is None:
            return
        cam, px, py = self.locate(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = cam
            self.start, self.drag = (cam, px, py), (px, py)
        elif self.start is None:
            return
        elif event == cv2.EVENT_MOUSEMOVE:
            if cam == self.start[0]:
                self.drag = (px, py)
        elif event == cv2.EVENT_LBUTTONUP:
            if cam == self.start[0]:
                self.drag = (px, py)
            self.shot.add(self.start[0], self.active, self.start[1], self.start[2], *self.drag)
            self.start = self.drag = None

    def tile(self, name, frame):
        image = frame.copy() if frame is not None else None
        if image is None:
            image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
        if self.shot is not None:
            boxes = list(self.shot.boxes[name])
            if self.start is not None and self.start[0] == name:
                boxes.append((self.active, self.start[1], self.start[2], *self.drag))
            for cls, x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), CLASS_COLORS[cls], 2)
                cv2.putText(image, CLASSES[cls], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, CLASS_COLORS[cls], 2)
        image = cv2.resize(image, (self.tile_w, self.tile_h))
        status = name.upper()
        color = (0, 255, 255)
        if self.shot is not None:
            done = self.shot.boxes[name] or name in self.shot.empty
            status += " EMPTY" if name in self.shot.empty else f" boxes={len(self.shot.boxes[name])}"
            color = (80, 230, 80) if done else (60, 60, 255)
            if name == self.selected:
                cv2.rectangle(image, (1, 1), (self.tile_w - 2, self.tile_h - 2), (255, 255, 255), 2)
        cv2.putText(image, status, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image

    def render(self, frames, entry, saved):
        canvas = cv2.hconcat([self.tile(n, frames.get(n)) for n in self.cams])
        canvas = cv2.copyMakeBorder(canvas, 0, 96, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        y = self.tile_h
        head = f"SET {entry[0]}/{len(SETS)} [{entry[1]}]  saved={saved}  {entry[2]}"
        cv2.putText(canvas, head, (8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        if self.shot is None:
            keys = "LIVE: SPACE freeze 3 views | Q quit"
        else:
            keys = (f"CLASS={CLASSES[self.active]} | 1 red 2 basket | DRAG box | TAB camera | U undo | "
                    "X delete class | C clear | E no target | R " + ("reload | N skip | Q quit" if self.save is update_labels
                                                                      else "retake") + " | S save")
        cv2.putText(canvas, keys, (8, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(canvas, self.message, (8, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
        return canvas

    def handle_label_key(self, key, entry):
        """라벨 입력 키를 처리하고, 세트가 끝났으면 True를 돌려줍니다."""
        shot = self.shot
        if key in (ord("1"), ord("2")):
            self.active = key - ord("1")
        elif key == 9:
            self.selected = self.cams[(self.cams.index(self.selected) + 1) % len(self.cams)]
        elif key in (ord("u"), ord("U")):
            if shot.boxes[self.selected]:
                shot.boxes[self.selected].pop()
        elif key in (ord("c"), ord("C")):
            shot.boxes[self.selected].clear()
            shot.empty.discard(self.selected)
        elif key in (ord("x"), ord("X")):
            shot.boxes[self.selected] = [b for b in shot.boxes[self.selected] if b[0] != self.active]
        elif key in (ord("e"), ord("E")):
            if not shot.mark_empty(self.selected):
                self.message = f"{self.selected}: remove boxes first (C), then E"
        elif key in (ord("r"), ord("R")):
            self.shot = None
            self.message = "Retake: nothing saved"
        elif key in (ord("s"), ord("S")):
            missing = shot.unlabeled()
            if missing:
                self.message = "Not saved. Label or press E for: " + ", ".join(missing)
            else:
                self.save(self.root, entry, shot)
                print(f"[SAVED] 세트 {entry[0]}/{len(SETS)} ({entry[1]})", flush=True)
                self.message = f"Saved set {entry[0]}"
                self.shot = None
                return True
        return False

    def run(self, cameras):
        entry = next_set(self.root)
        while entry is not None:
            saved = len(read_rows(self.root))
            frames = self.shot.frames if self.shot else {c.name_: c.latest()[0] for c in cameras}
            cv2.imshow(self.title, self.render(frames, entry, saved))
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue
            if self.shot is None:
                if key in (ord("q"), ord("Q"), 27):
                    print(f"[EXIT] 저장 세트 {saved}/{len(SETS)}. 다시 실행하면 이어서 진행합니다.", flush=True)
                    return False
                if key == 32:
                    grabbed = {c.name_: c.latest() for c in cameras}
                    now = time.monotonic()
                    stale = [n for n, (f, t) in grabbed.items() if f is None or now - t > STALE_SECONDS]
                    if stale:
                        self.message = "Cannot freeze, no fresh image: " + ", ".join(stale)
                    else:
                        self.shot = Shot({n: f.copy() for n, (f, _) in grabbed.items()})
                        self.selected, self.active, self.message = self.cams[0], 0, ""
            elif self.handle_label_key(key, entry):
                entry = next_set(self.root)
        print(f"[DONE] 전체 {len(SETS)}세트 완료: {self.root}", flush=True)
        return True

    def run_relabel(self, numbers):
        """저장된 세트를 차례로 열어 라벨만 고칩니다. R은 저장된 라벨을 다시 불러옵니다."""
        by_number = {s[0]: s for s in SETS}
        index = 0
        while index < len(numbers):
            entry = by_number[numbers[index]]
            if self.shot is None:
                self.shot = load_shot(self.root, entry[0])
                self.selected, self.active = self.cams[0], 1
            cv2.imshow(self.title, self.render(self.shot.frames, entry, index))
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), ord("Q"), 27):
                print("[EXIT] 저장한 세트의 수정은 유지됩니다.", flush=True)
                return
            if key in (ord("n"), ord("N")):
                self.shot = None
                self.message = f"Skipped set {entry[0]}: labels unchanged"
                index += 1
            elif self.handle_label_key(key, entry):
                index += 1
        print("[DONE] 라벨 수정 완료", flush=True)


def self_test():
    """카메라와 창 없이 저장 규칙만 검사합니다."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert next_set(root)[0] == 1
        shot = Shot({c: frame.copy() for c in CAMERAS})
        assert not shot.add("top", 0, 10, 10, 12, 12), "tiny box must be rejected"
        assert shot.add("top", 0, 320, 240, 160, 120), "reversed drag must be normalized"
        assert shot.boxes["top"] == [(0, 160, 120, 320, 240)]
        assert to_yolo_lines(shot.boxes["top"]) == ["0 0.375000 0.375000 0.250000 0.250000"]
        assert not shot.mark_empty("top"), "E must be refused while boxes exist"
        try:
            save_shot(root, SETS[0], shot)
        except ValueError:
            pass
        else:
            raise AssertionError("save must be refused with unlabeled cameras")
        assert next_set(root)[0] == 1 and not (root / "shots.csv").exists()
        assert shot.mark_empty("wrist") and shot.mark_empty("belly")
        save_shot(root, SETS[0], shot)
        assert (root / "labels" / "set01_top.txt").read_text() == "0 0.375000 0.375000 0.250000 0.250000\n"
        assert (root / "labels" / "set01_wrist.txt").read_bytes() == b""
        assert cv2.imread(str(root / "images" / "raw" / "set01_belly.jpg")).shape == (HEIGHT, WIDTH, 3)
        assert next_set(root)[0] == 2, "must resume at the next set"
        row = read_rows(root)[0]
        assert (row["split"], row["top_boxes"], row["wrist_boxes"]) == ("train", "1", "0")
        before_image = (root / "images" / "raw" / "set01_top.jpg").read_bytes()
        loaded = load_shot(root, 1)
        assert loaded.boxes["top"] == [(0, 160, 120, 320, 240)], "load must restore pixel boxes"
        assert loaded.empty == {"wrist", "belly"} and not loaded.unlabeled()
        target = backup_labels(root)
        assert (target / "set01_top.txt").exists() and (target / "shots.csv").exists()
        assert loaded.add("top", 1, 0, 0, 100, 100) and loaded.add("belly", 1, 0, 0, 64, 48)
        update_labels(root, SETS[0], loaded)
        assert (root / "images" / "raw" / "set01_top.jpg").read_bytes() == before_image, "relabel must not touch images"
        row = read_rows(root)[0]
        assert (row["top_boxes"], row["wrist_boxes"], row["belly_boxes"]) == ("2", "0", "1")
        assert (row["split"], row["scene"]) == ("train", SETS[0][2]), "relabel must keep split and scene"
        assert load_shot(root, 1).boxes["belly"] == [(1, 0, 0, 64, 48)]
        assert (target / "set01_belly.txt").read_bytes() == b"", "backup must keep the old labels"
    splits =[s[1] for s in SETS]
    assert (splits.count("train"), splits.count("val"), splits.count("test")) == (13, 2, 5)
    print("[SELF-TEST] 저장 규칙 검사 통과. 카메라와 모터는 사용하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--relabel", nargs="*", type=int, metavar="SET",
                        help="저장된 세트의 라벨만 수정합니다. 번호를 생략하면 전체 세트를 엽니다.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.relabel is not None:
        saved = sorted(int(row["set"]) for row in read_rows(YOLO_DIR))
        numbers = [n for n in saved if not args.relabel or n in args.relabel]
        if not numbers:
            print("[ERROR] 수정할 세트가 없습니다.")
            return
        print(f"[BACKUP] {backup_labels(YOLO_DIR)}", flush=True)
        print(f"[RELABEL] 세트 {numbers}. 키 입력은 영문 상태에서 하세요.", flush=True)
        try:
            App(YOLO_DIR, save=update_labels).run_relabel(numbers)
        finally:
            cv2.destroyAllWindows()
        return
    entry = next_set(YOLO_DIR)
    if entry is None:
        print(f"[DONE] 이미 {len(SETS)}세트를 모두 저장했습니다: {YOLO_DIR}")
        return
    print(f"[SAVE] {YOLO_DIR}", flush=True)
    print(f"[NEXT] 세트 {entry[0]}부터 시작합니다. 키 입력은 영문 상태에서 하세요.", flush=True)
    cameras = []
    try:
        for name, index in CAMERAS.items():
            camera = Camera(name, index)
            cameras.append(camera)
            camera.start()
        print("[READY] 카메라 3대 연결됨", flush=True)
        App(YOLO_DIR).run(cameras)
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
