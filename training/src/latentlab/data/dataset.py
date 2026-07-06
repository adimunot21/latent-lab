"""Torch Dataset/DataLoader for the sharded Two Rooms trajectory dataset.

Loads all shards into RAM (60k transitions of 64x64 uint8 ~= 250 MB — fine on
32 GB) and serves transitions:

    {
      "frame":      float32 (1, H, W)  normalized frame_t
      "action":     float32 (2,)       (dx, dy) as applied
      "next_frame": float32 (1, H, W)  normalized frame_{t+1}
      "state":      float32 (2,)       ground-truth (x, y) at t   (for probes)
      "next_state": float32 (2,)       ground-truth (x, y) at t+1
    }

Normalization: frames are scaled to [0, 1] then standardized with dataset-wide
mean/std, which are computed once and saved to ``norm_stats.json`` next to the
shards. The same stats ship in the browser manifest (Phase 4), so KEEP THE
FORMULA IN SYNC: normalized = (raw / 255 - mean) / std.

Standalone check:
    uv run python -m latentlab.data.dataset --data data/two_rooms_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

NORM_STATS_FILENAME = "norm_stats.json"


def load_norm_stats(data_dir: Path) -> dict[str, float]:
    """Load the persisted normalization stats for a dataset dir."""
    stats_path = Path(data_dir) / NORM_STATS_FILENAME
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found — instantiate TwoRoomsTransitions once to compute it"
        )
    stats: dict[str, float] = json.loads(stats_path.read_text())
    return stats


def normalize_frames(frames: np.ndarray, stats: dict[str, float]) -> torch.Tensor:
    """uint8 frames (..., H, W) -> normalized float32 (..., 1, H, W).

    THE normalization used everywhere (training, planning eval, browser
    manifest): (raw / 255 - frame_mean) / frame_std.
    """
    x = torch.from_numpy(frames.astype(np.float32)) / 255.0
    x = (x - stats["frame_mean"]) / stats["frame_std"]
    return x.unsqueeze(-3)


def compute_norm_stats(frames: np.ndarray) -> dict[str, float]:
    """Dataset-wide scalar mean/std of frames scaled to [0, 1]."""
    scaled_mean = float(frames.mean()) / 255.0
    # E[x^2] - E[x]^2 on the scaled values; avoids materializing frames/255.
    scaled_sq_mean = float((frames.astype(np.float64) ** 2).mean()) / 255.0**2
    std = float(np.sqrt(max(scaled_sq_mean - scaled_mean**2, 1e-12)))
    return {"frame_mean": scaled_mean, "frame_std": std}


class TwoRoomsTransitions(Dataset[dict[str, torch.Tensor]]):
    """All (frame_t, action_t, frame_{t+1}) transitions from a dataset dir."""

    def __init__(
        self,
        data_dir: Path,
        normalize: bool = True,
        episode_range: tuple[int, int] | None = None,
    ) -> None:
        """episode_range: half-open [start, end) episode slice (train/eval split)."""
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — generate the dataset first "
                "(uv run python -m latentlab.data.generate)"
            )
        self.meta = json.loads(meta_path.read_text())

        shard_paths = sorted(self.data_dir.glob("shard_*.npz"))
        if len(shard_paths) != self.meta["n_shards"]:
            raise ValueError(f"expected {self.meta['n_shards']} shards, found {len(shard_paths)}")

        frames_parts, states_parts, actions_parts = [], [], []
        for path in shard_paths:
            with np.load(path) as shard:
                frames_parts.append(shard["frames"])
                states_parts.append(shard["states"])
                actions_parts.append(shard["actions"])
        self.frames: npt.NDArray[np.uint8] = np.concatenate(frames_parts)  # (E, T+1, H, W)
        self.states: npt.NDArray[np.float32] = np.concatenate(states_parts)  # (E, T+1, 2)
        self.actions: npt.NDArray[np.float32] = np.concatenate(actions_parts)  # (E, T, 2)
        if episode_range is not None:
            start, end = episode_range
            if not 0 <= start < end <= self.frames.shape[0]:
                raise ValueError(
                    f"episode_range {episode_range} out of bounds for {self.frames.shape[0]} episodes"
                )
            self.frames = self.frames[start:end]
            self.states = self.states[start:end]
            self.actions = self.actions[start:end]
        self.episode_len = int(self.actions.shape[1])

        # Load or compute+persist normalization stats (train-time convenience;
        # the authoritative copy for the browser goes into the Phase 4 manifest).
        self.normalize = normalize
        stats_path = self.data_dir / NORM_STATS_FILENAME
        if stats_path.exists():
            self.norm_stats = json.loads(stats_path.read_text())
        else:
            self.norm_stats = compute_norm_stats(self.frames)
            stats_path.write_text(json.dumps(self.norm_stats, indent=2))
            print(f"computed and saved {stats_path}: {self.norm_stats}")

    def __len__(self) -> int:
        return int(self.frames.shape[0]) * self.episode_len

    def _normalize_frame(self, frame: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frame.astype(np.float32)) / 255.0
        if self.normalize:
            x = (x - self.norm_stats["frame_mean"]) / self.norm_stats["frame_std"]
        return x.unsqueeze(0)  # (1, H, W)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, t = divmod(index, self.episode_len)
        return {
            "frame": self._normalize_frame(self.frames[episode, t]),
            "action": torch.from_numpy(self.actions[episode, t]),
            "next_frame": self._normalize_frame(self.frames[episode, t + 1]),
            "state": torch.from_numpy(self.states[episode, t]),
            "next_state": torch.from_numpy(self.states[episode, t + 1]),
        }


def make_dataloader(
    data_dir: Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    episode_range: tuple[int, int] | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """DataLoader over all transitions. pin_memory speeds host->GPU copies."""
    return DataLoader(
        TwoRoomsTransitions(data_dir, episode_range=episode_range),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    dataset = TwoRoomsTransitions(args.data)
    print(f"dataset: {len(dataset)} transitions from {dataset.frames.shape[0]} episodes")
    print(f"norm stats: {dataset.norm_stats}")

    loader = make_dataloader(args.data, batch_size=args.batch_size, num_workers=0)
    batch = next(iter(loader))
    print("\nfirst batch:")
    for key, value in batch.items():
        print(
            f"  {key:12s} shape={tuple(value.shape)!s:20s} dtype={value.dtype} "
            f"min={value.min().item():+.3f} max={value.max().item():+.3f}"
        )
