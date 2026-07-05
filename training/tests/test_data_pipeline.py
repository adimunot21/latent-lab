"""End-to-end data pipeline test: generate a tiny dataset -> Dataset -> batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from latentlab.data.dataset import TwoRoomsTransitions, make_dataloader
from latentlab.data.generate import generate

EPISODES = 4
EPISODE_LEN = 10


def make_tiny_dataset(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    generate(
        out_dir=out,
        n_episodes=EPISODES,
        episode_len=EPISODE_LEN,
        episodes_per_shard=2,
        seed=0,
        goal_fraction=0.5,
    )
    return out


def test_generate_writes_shards_and_meta(tmp_path: Path) -> None:
    out = make_tiny_dataset(tmp_path)
    assert (out / "meta.json").exists()
    assert len(list(out.glob("shard_*.npz"))) == 2
    with np.load(out / "shard_0000.npz") as shard:
        assert shard["frames"].shape == (2, EPISODE_LEN + 1, 64, 64)
        assert shard["actions"].shape == (2, EPISODE_LEN, 2)
        assert shard["states"].shape == (2, EPISODE_LEN + 1, 2)


def test_generation_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        generate(out, EPISODES, EPISODE_LEN, 2, seed=123, goal_fraction=0.5)
    with np.load(out_a / "shard_0000.npz") as a, np.load(out_b / "shard_0000.npz") as b:
        assert np.array_equal(a["frames"], b["frames"])
        assert np.array_equal(a["actions"], b["actions"])
        assert np.array_equal(a["states"], b["states"])


def test_dataset_and_dataloader(tmp_path: Path) -> None:
    out = make_tiny_dataset(tmp_path)
    dataset = TwoRoomsTransitions(out)
    assert len(dataset) == EPISODES * EPISODE_LEN

    sample = dataset[0]
    assert sample["frame"].shape == (1, 64, 64)
    assert sample["frame"].dtype == torch.float32
    assert sample["action"].shape == (2,)

    # Norm stats persisted alongside the shards.
    assert (out / "norm_stats.json").exists()

    loader = make_dataloader(out, batch_size=8, num_workers=0)
    batch = next(iter(loader))
    assert batch["frame"].shape == (8, 1, 64, 64)
    assert batch["next_frame"].shape == (8, 1, 64, 64)
    assert batch["action"].shape == (8, 2)
    assert batch["state"].shape == (8, 2)
    # Transition consistency: next_state differs from state for moving agents.
    assert not torch.equal(batch["state"], batch["next_state"])


def test_last_transition_of_episode_uses_final_frame(tmp_path: Path) -> None:
    out = make_tiny_dataset(tmp_path)
    dataset = TwoRoomsTransitions(out, normalize=False)
    # Index EPISODE_LEN - 1 is the last transition of episode 0: its
    # next_frame must be frame T of episode 0, not frame 0 of episode 1.
    sample = dataset[EPISODE_LEN - 1]
    expected = (
        torch.from_numpy(dataset.frames[0, EPISODE_LEN].astype("float32")).unsqueeze(0) / 255.0
    )
    assert torch.equal(sample["next_frame"], expected)
