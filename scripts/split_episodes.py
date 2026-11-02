"""30 FPS 시연을 5 FPS 에피소드 여러 개로 나눕니다. 원본은 바꾸지 않습니다.
6분할이면 원본 프레임 0,6,12.. / 1,7,13.. 처럼 나뉩니다. 분할본은 서로 거의 같은 시연이므로 독립 시연 수는 원본 그대로입니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR / "datasets" / "lerobot"
TASK = "Pick up only the red box and place it inside the gray basket"
SOURCE_FPS = 30
CAMERAS = ("top_cam", "wrist_cam", "belly_cam")
KEEP_FEATURE_FIELDS = ("dtype", "shape", "names")


def decode_rgb(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def read_source(folder):
    table = pq.read_table(next((folder / "data").rglob("*.parquet")), columns=["action", "observation.state", "frame_index"]).to_pydict()
    action = np.array(table["action"], dtype=np.float32)
    state = np.array(table["observation.state"], dtype=np.float32)
    if table["frame_index"] != list(range(len(action))):
        raise RuntimeError(f"프레임 번호가 이어지지 않습니다: {folder}")
    frames = {}
    for camera in CAMERAS:
        frames[camera] = decode_rgb(next((folder / "videos" / f"observation.images.{camera}").rglob("*.mp4")))
        if len(frames[camera]) != len(action):
            raise RuntimeError(f"영상과 데이터 길이가 다릅니다: {folder} {camera}")
    return action, state, frames


def split(source_root, output_root, parts, limit=None):
    folders = sorted(source_root.glob("episode_*"))[:limit]
    if not folders:
        raise RuntimeError(f"원본 에피소드가 없습니다: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"이미 있습니다. 확인 후 옮겨 주세요: {output_root}")
    info = json.loads((folders[0] / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["fps"] != SOURCE_FPS or SOURCE_FPS % parts:
        raise RuntimeError(f"{info['fps']} FPS는 {parts}개로 나눌 수 없습니다.")
    fps = SOURCE_FPS // parts
    keys = ["action", "observation.state", *[f"observation.images.{c}" for c in CAMERAS]]
    features = {k: {f: info["features"][k][f] for f in KEEP_FEATURE_FIELDS} for k in keys}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=f"local/{output_root.name}", fps=fps, features=features, root=output_root,
                                    robot_type=info["robot_type"], use_videos=True,
                                    image_writer_processes=0, image_writer_threads=4)
    manifest = []
    started = time.time()
    try:
        for number, folder in enumerate(folders):
            action, state, frames = read_source(folder)
            for phase in range(parts):
                indices = list(range(phase, len(action), parts))
                for i in indices:
                    frame = {"action": action[i], "observation.state": state[i], "task": TASK}
                    for camera in CAMERAS:
                        frame[f"observation.images.{camera}"] = frames[camera][i]
                    dataset.add_frame(frame)
                dataset.save_episode()
                manifest.append({"episode": len(manifest), "source": folder.name, "phase": phase,
                                 "first_source_frame": indices[0], "frames": len(indices)})
            print(f"[SPLIT] {number + 1}/{len(folders)} {folder.name}: {len(action)}프레임 -> {parts}개, "
                  f"경과 {(time.time() - started) / 60:.1f}분", flush=True)
    finally:
        dataset.finalize()
    (output_root / "meta" / "split_manifest.json").write_text(json.dumps(
        {"source": str(source_root), "parts": parts, "source_fps": SOURCE_FPS, "fps": fps,
         "independent_demonstrations": len(folders), "episodes": manifest}, indent=2), encoding="utf-8")
    return manifest


def verify(source_root, output_root):
    """분할본의 관절값과 영상이 원본과 대응하는지, 빠진 프레임이 없는지 확인합니다."""
    manifest = json.loads((output_root / "meta" / "split_manifest.json").read_text(encoding="utf-8"))
    parts, fps = manifest["parts"], manifest["fps"]
    dataset = LeRobotDataset(f"local/{output_root.name}", root=output_root, video_backend="pyav")
    if dataset.num_episodes != len(manifest["episodes"]) or dataset.fps != fps:
        raise RuntimeError(f"에피소드 수 또는 FPS가 다릅니다: {dataset.num_episodes}, {dataset.fps}")
    tables = [pq.read_table(p, columns=["action", "observation.state", "episode_index", "frame_index", "timestamp"]).to_pydict()
              for p in sorted((output_root / "data").rglob("*.parquet"))]
    action = np.concatenate([np.array(t["action"]) for t in tables])
    state = np.concatenate([np.array(t["observation.state"]) for t in tables])
    episode = np.concatenate([np.array(t["episode_index"]) for t in tables])
    timestamp = np.concatenate([np.array(t["timestamp"]) for t in tables])
    by_source, total_source = {}, 0
    for item in manifest["episodes"]:
        by_source.setdefault(item["source"], []).append(item)
    for source, items in by_source.items():
        table = pq.read_table(next((source_root / source / "data").rglob("*.parquet")), columns=["action", "observation.state"]).to_pydict()
        source_action, source_state = np.array(table["action"]), np.array(table["observation.state"])
        total_source += len(source_action)
        used = []
        for item in items:
            mask = episode == item["episode"]
            indices = np.arange(item["phase"], len(source_action), parts)
            used.extend(indices.tolist())
            if mask.sum() != len(indices) or not np.array_equal(action[mask], source_action[indices]) \
                    or not np.array_equal(state[mask], source_state[indices]):
                raise RuntimeError(f"관절값이 원본과 다릅니다: {source} phase {item['phase']}")
            if not np.allclose(np.diff(timestamp[mask]), 1 / fps, atol=1e-3):
                raise RuntimeError(f"시간 간격이 다릅니다: {source} phase {item['phase']}")
        if sorted(used) != list(range(len(source_action))):
            raise RuntimeError(f"원본 프레임이 빠졌거나 중복됐습니다: {source}")
    if len(action) != total_source:
        raise RuntimeError(f"총 프레임이 다릅니다: {len(action)} != {total_source}")
    # 영상은 앞, 가운데, 끝 에피소드의 프레임을 원본과 비교합니다.
    errors = []
    episodes = manifest["episodes"]
    for item in (episodes[0], episodes[len(episodes) // 2 + 1], episodes[-1]):
        source_frames = decode_rgb(next((source_root / item["source"] / "videos" / "observation.images.top_cam").rglob("*.mp4")))
        start = int(dataset.meta.episodes[item["episode"]]["dataset_from_index"])
        for k in (0, item["frames"] // 2, item["frames"] - 1):
            image = (dataset[start + k]["observation.images.top_cam"].permute(1, 2, 0).numpy() * 255).astype(np.float32)
            original = source_frames[item["phase"] + k * parts].astype(np.float32)
            errors.append(float(np.abs(image - original).mean()))
            other = source_frames[(item["phase"] + k * parts + parts * 3) % len(source_frames)].astype(np.float32)
            errors.append(-float(np.abs(image - other).mean()))
    matched = [e for e in errors if e >= 0]
    print(f"[VERIFY] 에피소드 {dataset.num_episodes}개, 프레임 {len(action)}개, {fps} FPS, 원본 {len(by_source)}개와 일치", flush=True)
    print(f"[VERIFY] 영상 표본의 원본 대비 평균 픽셀 차이: 최대 {max(matched):.2f} (0~255)", flush=True)
    return dataset.num_episodes, len(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="redbox_30fps")
    parser.add_argument("--output", default="redbox_5fps_split6")
    parser.add_argument("--parts", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="시험용: 원본 에피소드 수를 제한합니다.")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT, help="시험용: 출력 위치")
    args = parser.parse_args()
    source_root, output_root = DATASET_ROOT / args.source, args.root / args.output
    started = time.time()
    split(source_root, output_root, args.parts, args.limit)
    print(f"[SPLIT] 완료, {(time.time() - started) / 60:.1f}분", flush=True)
    verify(source_root, output_root)
    if args.root != DATASET_ROOT:
        shutil.rmtree(output_root)
        print("[TEST] 시험 출력을 지웠습니다.", flush=True)


if __name__ == "__main__":
    main()
