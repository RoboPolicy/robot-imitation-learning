"""분할한 데이터셋 여러 개를 학습용 하나로 합칩니다. 원본은 바꾸지 않습니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR / "datasets" / "lerobot"


def merge(roots, output):
    if output.exists():
        raise FileExistsError(f"이미 있습니다. 확인 후 옮겨 주세요: {output}")
    aggregate_datasets([f"local/{r.name}" for r in roots], f"local/{output.name}", roots=roots, aggr_root=output)


def verify(roots, output):
    """에피소드와 프레임 수, 합쳐진 뒤의 관절값과 영상이 원본과 같은지 확인합니다."""
    sources = [LeRobotDataset(f"local/{r.name}", root=r, video_backend="pyav") for r in roots]
    merged = LeRobotDataset(f"local/{output.name}", root=output, video_backend="pyav")
    if merged.num_episodes != sum(s.num_episodes for s in sources) or merged.num_frames != sum(s.num_frames for s in sources):
        raise RuntimeError(f"수가 다릅니다: {merged.num_episodes}개, {merged.num_frames}프레임")
    if any(s.fps != merged.fps for s in sources):
        raise RuntimeError("FPS가 다릅니다.")
    offset, worst = 0, 0.0
    for source in sources:
        for k in (0, source.num_frames // 2, source.num_frames - 1):
            a, b = source[k], merged[offset + k]
            if not np.array_equal(a["action"].numpy(), b["action"].numpy()) or not np.array_equal(a["observation.state"].numpy(), b["observation.state"].numpy()):
                raise RuntimeError(f"관절값이 원본과 다릅니다: {source.root} 프레임 {k}")
            for key in merged.meta.camera_keys:
                worst = max(worst, float((a[key] - b[key]).abs().mean()) * 255)
        offset += source.num_frames
    if worst > 3.0:
        raise RuntimeError(f"영상이 원본과 다릅니다: 평균 픽셀 차이 {worst:.2f}")
    print(f"[VERIFY] 에피소드 {merged.num_episodes}개, 프레임 {merged.num_frames}개, {merged.fps} FPS, 영상 표본 차이 최대 {worst:.2f} (0~255)", flush=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["redbox_5fps_split6", "redbox_redonly_5fps_split6"])
    parser.add_argument("--output", default="redbox_pool_v2")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT, help="시험용: 출력 위치")
    args = parser.parse_args()
    roots = [DATASET_ROOT / s if not Path(s).is_absolute() else Path(s) for s in args.sources]
    output = args.root / args.output
    merge(roots, output)
    merged = verify(roots, output)
    (output / "meta" / "pool_manifest.json").write_text(json.dumps(
        {"sources": [r.name for r in roots], "episodes": merged.num_episodes, "frames": merged.num_frames}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
