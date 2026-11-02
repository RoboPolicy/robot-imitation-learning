"""SO-ARM101 YOLO 사진 수집: 3개 카메라 x 10개 배치 = 총 30장.
Windows / Python + opencv-python 기준.
Space: 3대 카메라 사진 동시 저장, Q/Esc: 종료.
"""

from datetime import datetime
from pathlib import Path
import cv2

CAMERAS = {"top": 1, "wrist": 0, "belly": 2}
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "datasets" / "yolo" / "images" / "raw"
TARGET_SETS = 10  # 1세트당 3장 -> 총 30장
WIDTH, HEIGHT = 640, 480


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    captures = {}
    try:
        for name, index in CAMERAS.items():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"{name} 카메라(index={index}) 열기 실패")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            captures[name] = cap

        print("[READY] Space: 현재 장면 3장 저장 / Q: 종료")
        print(f"[SAVE] {OUTPUT_DIR}")
        print("[TIP] Space를 누르기 전 상자/바구니 배치를 매번 바꾸세요.")
        saved_sets = 0
        while saved_sets < TARGET_SETS:
            frames = {}
            previews = []
            ok_all = True
            for name, cap in captures.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[WARN] {name} 프레임 수신 실패: 이번 저장 불가")
                    ok_all = False
                    break
                frames[name] = frame
                tile = cv2.resize(frame, (400, 300))
                cv2.putText(tile, f"{name.upper()}  {saved_sets}/{TARGET_SETS}",
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                previews.append(tile)

            if ok_all:
                cv2.imshow("YOLO capture | SPACE=save  Q=quit", cv2.hconcat(previews))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and ok_all:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                success = True
                for name, frame in frames.items():
                    path = OUTPUT_DIR / f"{stamp}_{name}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        print(f"[ERROR] 저장 실패: {path}")
                        success = False
                if success:
                    saved_sets += 1
                    print(f"[SAVED] {saved_sets}/{TARGET_SETS}세트, 이번 실행 {saved_sets*3}장")
        print(f"[DONE] 이번 실행 저장 세트: {saved_sets}, 이미지: {saved_sets*3}장")
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
