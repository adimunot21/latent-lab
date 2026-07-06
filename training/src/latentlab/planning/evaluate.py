"""Planning evaluation: MPC with the CEM planner on real Two Rooms episodes.

For each episode: sample random free start and goal (half the episodes force
start/goal in different rooms — the hard case), render+encode the goal frame
once, then run the MPC loop: encode current frame -> CEM plan -> execute first
action -> repeat. Success = true agent position within ``success_radius`` of
the goal within ``max_steps``.

Writes results.json and a few animated GIFs (agent + goal ring) to --out.

Usage:
    uv run python -m latentlab.planning.evaluate \
        --checkpoint checkpoints/two_rooms_v1/healthy_v1/final.pt \
        --episodes 100 --out eval_out/healthy_v1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from latentlab.data.dataset import load_norm_stats, normalize_frames
from latentlab.envs.two_rooms import TwoRoomsConfig, TwoRoomsEnv
from latentlab.models.encoder import Encoder
from latentlab.planning.cem import CEMConfig, CEMPlanner
from latentlab.train import load_checkpoint_models


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 100
    max_steps: int = 60
    success_radius: float = 0.08  # one max_step of slack around the goal
    cross_room_fraction: float = 0.5  # fraction of episodes forcing a door crossing
    gif_count: int = 3  # animated GIFs saved for the first N episodes
    seed: int = 0


def sample_position(
    env: TwoRoomsEnv, rng: np.random.Generator, room: str | None = None
) -> tuple[float, float]:
    """Random free position, optionally constrained to 'left' or 'right' room."""
    r = env.config.agent_radius
    for _ in range(1000):
        x = float(rng.uniform(r, 1.0 - r))
        y = float(rng.uniform(r, 1.0 - r))
        if not env.is_free(x, y):
            continue
        if room == "left" and x >= env.config.wall_x:
            continue
        if room == "right" and x <= env.config.wall_x:
            continue
        return (x, y)
    raise RuntimeError("could not sample a position")


def frame_with_goal(env: TwoRoomsEnv, goal: tuple[float, float]) -> npt.NDArray[np.uint8]:
    """Current frame plus a ring marking the goal (for GIFs only)."""
    frame = env.render().copy()
    n = env.config.frame_size
    centers = (np.arange(n, dtype=np.float64) + 0.5) / n
    xs, ys = centers[np.newaxis, :], centers[:, np.newaxis]
    dist_sq = (xs - goal[0]) ** 2 + (ys - goal[1]) ** 2
    r = env.config.agent_radius
    ring = (dist_sq <= (r * 1.4) ** 2) & (dist_sq >= r**2)
    frame[ring] = 200
    return frame


def save_gif(frames: list[npt.NDArray[np.uint8]], path: Path, scale: int = 4) -> None:
    images = [
        Image.fromarray(f).resize(
            (f.shape[1] * scale, f.shape[0] * scale), Image.Resampling.NEAREST
        )
        for f in frames
    ]
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)


@torch.no_grad()
def run_episode(
    env: TwoRoomsEnv,
    encoder: Encoder,
    planner: CEMPlanner,
    start: tuple[float, float],
    goal: tuple[float, float],
    stats: dict[str, float],
    eval_config: EvalConfig,
    record_frames: bool = False,
) -> dict[str, object]:
    device = next(encoder.parameters()).device
    env.set_state(*start)
    planner.reset()

    # Encode the goal frame once (teleport, render, restore).
    env.set_state(*goal)
    goal_frame = env.render()
    env.set_state(*start)
    z_goal = encoder(normalize_frames(goal_frame[np.newaxis], stats).to(device))[0]

    frames: list[npt.NDArray[np.uint8]] = []
    for step in range(eval_config.max_steps):
        if record_frames:
            frames.append(frame_with_goal(env, goal))
        x, y = env.state
        if (x - goal[0]) ** 2 + (y - goal[1]) ** 2 <= eval_config.success_radius**2:
            return {"success": True, "steps": step, "frames": frames}
        z = encoder(normalize_frames(env.render()[np.newaxis], stats).to(device))[0]
        actions, _ = planner.plan(z, z_goal)
        env.step((float(actions[0, 0].item()), float(actions[0, 1].item())))

    if record_frames:
        frames.append(frame_with_goal(env, goal))
    x, y = env.state
    success = (x - goal[0]) ** 2 + (y - goal[1]) ** 2 <= eval_config.success_radius**2
    return {"success": success, "steps": eval_config.max_steps, "frames": frames}


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    out_dir: Path,
    eval_config: EvalConfig,
    cem_config: CEMConfig,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, _ = load_checkpoint_models(checkpoint)
    encoder.to(device)
    predictor.to(device)
    stats = load_norm_stats(data_dir)
    env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=eval_config.seed)
    rng = np.random.default_rng(eval_config.seed)
    planner = CEMPlanner(predictor, cem_config)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    start_time = time.time()
    for episode in range(eval_config.episodes):
        cross_room = episode < int(eval_config.episodes * eval_config.cross_room_fraction)
        if cross_room:
            rooms = ["left", "right"] if rng.uniform() < 0.5 else ["right", "left"]
            start = sample_position(env, rng, rooms[0])
            goal = sample_position(env, rng, rooms[1])
        else:
            start = sample_position(env, rng)
            goal = sample_position(env, rng)

        record = episode < eval_config.gif_count
        outcome = run_episode(env, encoder, planner, start, goal, stats, eval_config, record)
        results.append(
            {
                "episode": episode,
                "cross_room": cross_room,
                "success": outcome["success"],
                "steps": outcome["steps"],
            }
        )
        if record:
            frames = outcome["frames"]
            assert isinstance(frames, list)
            tag = "cross" if cross_room else "same"
            save_gif(frames, out_dir / f"episode_{episode:03d}_{tag}.gif")
        if (episode + 1) % 10 == 0:
            rate = sum(1 for r in results if r["success"]) / len(results)
            print(
                f"episode {episode + 1}/{eval_config.episodes}: "
                f"running success {rate:.1%}, {time.time() - start_time:.0f}s"
            )

    successes = [r for r in results if r["success"]]
    cross = [r for r in results if r["cross_room"]]
    cross_success = [r for r in cross if r["success"]]
    same = [r for r in results if not r["cross_room"]]
    same_success = [r for r in same if r["success"]]
    summary: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "episodes": eval_config.episodes,
        "success_rate": len(successes) / len(results),
        "cross_room_success_rate": len(cross_success) / len(cross) if cross else None,
        "same_room_success_rate": len(same_success) / len(same) if same else None,
        "mean_steps_to_goal": (
            float(np.mean([r["steps"] for r in successes])) if successes else None
        ),
        "eval_config": asdict(eval_config),
        "cem_config": asdict(cem_config),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "episodes": results}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data,
        out_dir=args.out,
        eval_config=EvalConfig(episodes=args.episodes, seed=args.seed),
        cem_config=CEMConfig(),
    )


if __name__ == "__main__":
    main()
