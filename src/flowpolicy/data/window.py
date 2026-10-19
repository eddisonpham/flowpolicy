"""PushT Zarr access, episode-level splits and the window sampler.

Schema (docs/03 section 1): `episode_ends[E]`, `image[N,H,W,3] uint8`,
`state[N,Ds] f32`, `action[N,Da] f32`.

Splits are by episode, never by window. Windows pad at episode edges by
repeating the first observation and the last action (same convention at train
and eval, so behaviour matches).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zarr
from numpy.typing import NDArray
from torch.utils.data import Dataset


def split_episodes(n_episodes: int, val_fraction: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    n_val = int(round(n_episodes * val_fraction))
    if val_fraction > 0 and n_val == 0:
        n_val = 1
    val = np.sort(order[:n_val]).astype(np.int64)
    train = np.sort(order[n_val:]).astype(np.int64)
    if set(train.tolist()) & set(val.tolist()):
        raise AssertionError("episode split overlapped; this is a bug")
    return train, val


@dataclass
class ZarrArrays:
    image: NDArray[np.uint8]  # [N, H, W, 3]
    state: NDArray[np.float32]  # [N, Ds]
    action: NDArray[np.float32]  # [N, Da]
    episode_ends: NDArray[np.int64]  # [E]


def load_zarr(path: str | Path, load_images: bool = True, image_limit: int | None = None) -> ZarrArrays:
    root: Any = zarr.open_group(str(path), mode="r")
    episode_ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
    state = np.asarray(root["state"][:], dtype=np.float32)
    action = np.asarray(root["action"][:], dtype=np.float32)
    if load_images:
        n = image_limit if image_limit is not None else state.shape[0]
        image = np.asarray(root["image"][:n], dtype=np.uint8)
    else:
        image = np.empty((0, 0, 0, 0), dtype=np.uint8)
    return ZarrArrays(image=image, state=state, action=action, episode_ends=episode_ends)


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample windows of `obs_horizon` frames and `pred_horizon` actions."""

    def __init__(
        self,
        arrays: ZarrArrays,
        episodes: NDArray[np.int64],
        obs_horizon: int = 2,
        pred_horizon: int = 16,
    ) -> None:
        if obs_horizon < 1 or pred_horizon < 1:
            raise ValueError("horizons must be positive")
        self.arrays = arrays
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.episode_ends = arrays.episode_ends

        starts = np.zeros_like(self.episode_ends)
        starts[1:] = self.episode_ends[:-1]
        self._starts = starts
        self._ends = self.episode_ends

        frames: list[NDArray[np.int64]] = []
        frame_episode: list[NDArray[np.int64]] = []
        for ep in episodes.tolist():
            s = int(starts[ep])
            e = int(self.episode_ends[ep])
            if e <= s:
                raise ValueError(f"episode {ep} is empty")
            frames.append(np.arange(s, e, dtype=np.int64))
            frame_episode.append(np.full(e - s, ep, dtype=np.int64))
        self.frames = np.concatenate(frames) if frames else np.empty(0, dtype=np.int64)
        self.frame_episode = np.concatenate(frame_episode) if frame_episode else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def window_indices(self, sample: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return (obs_idx [To], action_idx [Tp]) for one sample center."""
        t = int(self.frames[sample])
        ep = int(self.frame_episode[sample])
        start, end = int(self._starts[ep]), int(self._ends[ep])
        obs = np.clip(np.arange(t - self.obs_horizon + 1, t + 1), start, end - 1)
        act = np.clip(np.arange(t, t + self.pred_horizon), start, end - 1)
        return obs.astype(np.int64), act.astype(np.int64)

    def __getitem__(self, sample: int) -> dict[str, torch.Tensor]:
        obs_idx, act_idx = self.window_indices(sample)
        image = self.arrays.image[obs_idx]  # [To,H,W,3] uint8
        state = self.arrays.state[obs_idx]
        action = self.arrays.action[act_idx]
        return {
            "image": torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0,  # [To,3,H,W]
            "state": torch.from_numpy(state.copy()),
            "action": torch.from_numpy(action.copy()),
        }

    def get_batch(self, samples: NDArray[np.int64]) -> dict[str, torch.Tensor]:
        """Vectorized batch build; named separately so tests can check indices."""
        obs_idx = np.empty((samples.shape[0], self.obs_horizon), dtype=np.int64)
        act_idx = np.empty((samples.shape[0], self.pred_horizon), dtype=np.int64)
        for row, sample in enumerate(samples.tolist()):
            o, a = self.window_indices(sample)
            obs_idx[row] = o
            act_idx[row] = a
        image = self.arrays.image[obs_idx]  # [B,To,H,W,3]
        batch = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).permute(0, 1, 4, 2, 3).float() / 255.0,
            "state": torch.from_numpy(self.arrays.state[obs_idx]),
            "action": torch.from_numpy(self.arrays.action[act_idx]),
        }
        return batch
