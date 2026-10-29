"""Download lerobot/pusht and convert it to the Zarr schema in docs/03 section 1.

Source: https://huggingface.co/datasets/lerobot/pusht (v3.0, 206 episodes,
25 650 frames, 96x96 images, 10 fps). The dataset card/license must be read
before publishing; see docs/02 section 10.

Usage: uv run python scripts/data/prepare_pusht.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import zarr

from policybench.data.normalizer import Stats
from policybench.data.window import load_zarr, split_episodes

REPO = "lerobot/pusht"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
FILES = {
    "meta/info.json": "meta/info.json",
    "meta/tasks.parquet": "meta/tasks.parquet",
    "data/chunk-000/file-000.parquet": "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4": "videos/observation.image/chunk-000/file-000.mp4",
}


def download(raw_dir: Path, force: bool = False) -> None:
    # huggingface_hub respects the proxy but its xet path failed on this host;
    # the plain HTTP resolver works (docs/CHANGELOG 2026-10-19).
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for remote in FILES:
        target = raw_dir / remote
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(REPO, remote, repo_type="dataset", local_dir=str(raw_dir))


def decode_frames(path: Path) -> np.ndarray:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8)


def convert(raw_dir: Path, zarr_path: Path, stats_path: Path) -> None:
    table = pq.read_table(raw_dir / "data/chunk-000/file-000.parquet")
    df = table.to_pandas().sort_values("index").reset_index(drop=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    episodes = df["episode_index"].to_numpy(dtype=np.int64)
    frames = df["frame_index"].to_numpy(dtype=np.int64)

    episode_ends: list[int] = []
    last = -1
    for pos, ep in enumerate(episodes.tolist()):
        if ep != last:
            if last >= 0:
                episode_ends.append(pos)
            last = ep
    episode_ends.append(len(df))
    if episodes.min() != 0 or episodes.max() != len(episode_ends) - 1:
        raise RuntimeError("episode indices are not contiguous from 0")
    for ep, end in enumerate(episode_ends):
        start = 0 if ep == 0 else episode_ends[ep - 1]
        if not np.array_equal(frames[start:end], np.arange(end - start)):
            raise RuntimeError(f"frame_index is not contiguous in episode {ep}")

    image = decode_frames(raw_dir / "videos/observation.image/chunk-000/file-000.mp4")
    if image.shape[0] != len(df):
        raise RuntimeError(f"video has {image.shape[0]} frames, parquet has {len(df)} rows")

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_array("image", data=image, chunks=(64, 96, 96, 3))
    root.create_array("state", data=state, chunks=(4096, 2))
    root.create_array("action", data=action, chunks=(4096, 2))
    root.create_array("episode_ends", data=np.array(episode_ends, dtype=np.int64))
    root.attrs.update(
        {
            "source": f"https://huggingface.co/datasets/{REPO}",
            "license": "mit (dataset card, accessed 2026-10-19)",
            "fps": 10,
            "image_size": 96,
            "success_threshold": 0.95,
            "converted_at": "2026-10-19",
        }
    )

    train_eps, val_eps = split_episodes(len(episode_ends), 0.1, seed=0)
    arrays = load_zarr(zarr_path)
    train_frames = np.concatenate(
        [np.arange(0 if ep == 0 else episode_ends[ep - 1], episode_ends[ep]) for ep in train_eps]
    )
    stats = Stats.compute(arrays.state[train_frames], arrays.action[train_frames])
    stats.to_json(stats_path)
    summary = {
        "episodes": len(episode_ends),
        "frames": int(len(df)),
        "train_episodes": int(len(train_eps)),
        "val_episodes": int(len(val_eps)),
        "state_min": stats.state_min,
        "state_max": stats.state_max,
        "action_min": stats.action_min,
        "action_max": stats.action_max,
        "image_bytes": int(image.nbytes),
    }
    (zarr_path.parent / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/pusht"))
    parser.add_argument("--zarr", type=Path, default=Path("data/pusht/pusht.zarr"))
    parser.add_argument("--stats", type=Path, default=Path("data/pusht/stats.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.raw_dir, force=args.force)
    if args.zarr.exists() and not args.force:
        print(f"{args.zarr} exists; pass --force to rebuild")
        return
    convert(args.raw_dir, args.zarr, args.stats)


if __name__ == "__main__":
    main()
