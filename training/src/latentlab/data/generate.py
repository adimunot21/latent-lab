"""Generate an offline trajectory dataset from the Two Rooms env.

Rollouts use a mix of two behavior policies:
- ``random``: uniform actions in [-max_step, max_step]^2 (local coverage).
- ``goal``: greedy move toward a sampled goal, routed through the doorway
  when the goal is in the other room, plus Gaussian noise. This guarantees
  the dataset contains cross-room transitions through the door, which the
  world model must learn.

Episodes have fixed length T, so each shard is a set of dense arrays:
    frames  uint8   (episodes, T+1, frame, frame)   rendered observations
    states  float32 (episodes, T+1, 2)              ground-truth agent (x, y)
    actions float32 (episodes, T, 2)                actions as applied (clamped)

Transition t is (frames[e, t], actions[e, t], frames[e, t+1]).

Shards are written as compressed .npz plus a ``meta.json`` describing the
config, counts, and seed (dataset is NOT git-tracked; regenerate with this
script).

Usage:
    uv run python -m latentlab.data.generate --out data/two_rooms_v1
    uv run python -m latentlab.data.generate --out /tmp/tiny --episodes 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from latentlab.envs.two_rooms import TwoRoomsConfig, TwoRoomsEnv

DATASET_VERSION = 1


def goal_policy_action(
    env: TwoRoomsEnv,
    goal: tuple[float, float],
    rng: np.random.Generator,
    noise_scale: float,
) -> tuple[float, float]:
    """Greedy step toward the goal, via the door if it's in the other room."""
    cfg = env.config
    x, y = env.state
    gx, gy = goal
    same_room = (x < cfg.wall_x) == (gx < cfg.wall_x)
    in_door_band = abs(y - cfg.door_center_y) < cfg.door_half_height - cfg.agent_radius
    if same_room or in_door_band:
        target_x, target_y = gx, gy
    else:
        # Head for the door center first; once in the band, the branch above
        # will steer toward the true goal.
        target_x, target_y = cfg.wall_x, cfg.door_center_y
    dx = target_x - x + float(rng.normal(0.0, noise_scale))
    dy = target_y - y + float(rng.normal(0.0, noise_scale))
    return env.clamp_action((dx, dy))


def sample_free_position(env: TwoRoomsEnv, rng: np.random.Generator) -> tuple[float, float]:
    """Uniform collision-free position (same rejection scheme as env.reset)."""
    r = env.config.agent_radius
    for _ in range(1000):
        x = float(rng.uniform(r, 1.0 - r))
        y = float(rng.uniform(r, 1.0 - r))
        if env.is_free(x, y):
            return (x, y)
    raise RuntimeError("could not sample a free position")


def rollout_episode(
    env: TwoRoomsEnv,
    episode_len: int,
    rng: np.random.Generator,
    goal_directed: bool,
    noise_scale: float = 0.02,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """One fixed-length episode. Returns (frames, states, actions)."""
    cfg = env.config
    frames = np.zeros((episode_len + 1, cfg.frame_size, cfg.frame_size), dtype=np.uint8)
    states = np.zeros((episode_len + 1, 2), dtype=np.float32)
    actions = np.zeros((episode_len, 2), dtype=np.float32)

    env.reset(seed=int(rng.integers(0, 2**31)))
    goal = sample_free_position(env, rng)
    frames[0] = env.render()
    states[0] = env.state

    for t in range(episode_len):
        if goal_directed:
            gx, gy = goal
            x, y = env.state
            # Resample the goal once reached so the episode keeps moving.
            if (x - gx) ** 2 + (y - gy) ** 2 < (2 * cfg.agent_radius) ** 2:
                goal = sample_free_position(env, rng)
            action = goal_policy_action(env, goal, rng, noise_scale)
        else:
            action = env.clamp_action(
                (
                    float(rng.uniform(-cfg.max_step, cfg.max_step)),
                    float(rng.uniform(-cfg.max_step, cfg.max_step)),
                )
            )
        env.step(action)
        actions[t] = action
        frames[t + 1] = env.render()
        states[t + 1] = env.state

    return frames, states, actions


def generate(
    out_dir: Path,
    n_episodes: int,
    episode_len: int,
    episodes_per_shard: int,
    seed: int,
    goal_fraction: float,
) -> None:
    """Generate the full dataset: sharded .npz files + meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env_config = TwoRoomsConfig()
    env = TwoRoomsEnv(config=env_config, seed=seed)
    rng = np.random.default_rng(seed)

    n_shards = (n_episodes + episodes_per_shard - 1) // episodes_per_shard
    start_time = time.time()
    total_transitions = 0

    for shard_idx in range(n_shards):
        shard_episodes = min(episodes_per_shard, n_episodes - shard_idx * episodes_per_shard)
        frames_list, states_list, actions_list = [], [], []
        for _ in range(shard_episodes):
            goal_directed = bool(rng.uniform() < goal_fraction)
            frames, states, actions = rollout_episode(env, episode_len, rng, goal_directed)
            frames_list.append(frames)
            states_list.append(states)
            actions_list.append(actions)

        shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            frames=np.stack(frames_list),
            states=np.stack(states_list),
            actions=np.stack(actions_list),
        )
        total_transitions += shard_episodes * episode_len
        elapsed = time.time() - start_time
        print(
            f"shard {shard_idx + 1}/{n_shards}: {shard_episodes} episodes, "
            f"{total_transitions} transitions total, "
            f"{shard_path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s elapsed"
        )

    meta = {
        "dataset_version": DATASET_VERSION,
        "env_config": dataclasses.asdict(env_config),
        "n_episodes": n_episodes,
        "episode_len": episode_len,
        "episodes_per_shard": episodes_per_shard,
        "n_shards": n_shards,
        "total_transitions": total_transitions,
        "seed": seed,
        "goal_fraction": goal_fraction,
        "layout": {
            "frames": "uint8 (episodes, T+1, frame, frame)",
            "states": "float32 (episodes, T+1, 2) agent (x, y) in [0,1]",
            "actions": "float32 (episodes, T, 2) clamped (dx, dy)",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'meta.json'}")
    print(f"done: {n_episodes} episodes x {episode_len} steps = {total_transitions} transitions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-len", type=int, default=60)
    parser.add_argument("--episodes-per-shard", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--goal-fraction",
        type=float,
        default=0.5,
        help="fraction of episodes driven by the goal-directed policy",
    )
    args = parser.parse_args()
    generate(
        out_dir=args.out,
        n_episodes=args.episodes,
        episode_len=args.episode_len,
        episodes_per_shard=args.episodes_per_shard,
        seed=args.seed,
        goal_fraction=args.goal_fraction,
    )


if __name__ == "__main__":
    main()
