"""DATA-VALIDATION GATE: inspect a generated Two Rooms dataset.

Prints field names, shapes, dtypes, value ranges, and per-field stats for
every shard field, checks basic invariants, and writes sample transition PNGs
(frame_t | frame_t+1 side by side) for visual review.

Run this and LOOK at the output before writing any training code
(CLAUDE.md golden rule #1).

Usage:
    uv run python -m latentlab.data.inspect --data data/two_rooms_v1 --png-out /tmp/inspect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def inspect_dataset(data_dir: Path, png_out: Path, n_samples: int = 3) -> None:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json in {data_dir} — is this a dataset dir?")
    meta = json.loads(meta_path.read_text())
    print("=== meta.json ===")
    print(json.dumps(meta, indent=2))

    shards = sorted(data_dir.glob("shard_*.npz"))
    print(f"\n=== shards: {len(shards)} found ===")
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz files in {data_dir}")

    total_episodes = 0
    for i, shard_path in enumerate(shards):
        with np.load(shard_path) as shard:
            episodes = shard["frames"].shape[0]
            total_episodes += episodes
            if i == 0:  # detailed stats for the first shard only
                print(f"\n--- {shard_path.name} (detailed) ---")
                for name in shard.files:
                    arr = shard[name]
                    print(
                        f"{name:8s} shape={arr.shape!s:24s} dtype={arr.dtype!s:8s} "
                        f"min={arr.min():+.4f} max={arr.max():+.4f} mean={arr.mean():+.4f}"
                    )
            else:
                print(f"{shard_path.name}: {episodes} episodes")

    print(f"\ntotal episodes across shards: {total_episodes} (meta says {meta['n_episodes']})")

    # ---- invariant checks on the first shard --------------------------------
    print("\n=== invariant checks (first shard) ===")
    with np.load(shards[0]) as shard:
        frames, states, actions = shard["frames"], shard["states"], shard["actions"]

    cfg = meta["env_config"]
    checks: list[tuple[str, bool]] = [
        ("frames dtype uint8", frames.dtype == np.uint8),
        ("states dtype float32", states.dtype == np.float32),
        ("actions dtype float32", actions.dtype == np.float32),
        ("frames T+1 vs actions T", frames.shape[1] == actions.shape[1] + 1),
        ("states T+1 matches frames", states.shape[1] == frames.shape[1]),
        (
            "frame values in {0, wall, agent}",
            set(np.unique(frames).tolist()) <= {0, cfg["wall_intensity"], cfg["agent_intensity"]},
        ),
        (
            "every frame shows the agent",
            bool((frames == cfg["agent_intensity"]).any(axis=(2, 3)).all()),
        ),
        ("|actions| <= max_step", bool(np.abs(actions).max() <= cfg["max_step"] + 1e-7)),
        (
            "states within [r, 1-r]",
            bool(
                (states >= cfg["agent_radius"] - 1e-6).all()
                and (states <= 1 - cfg["agent_radius"] + 1e-6).all()
            ),
        ),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        all_ok &= ok

    # Cross-room coverage: the model can't learn door transitions it never sees.
    left = states[:, :, 0] < cfg["wall_x"]
    crossings = int(np.sum(left[:, :-1] != left[:, 1:]))
    episodes_with_crossing = int(np.sum((left[:, :-1] != left[:, 1:]).any(axis=1)))
    print(
        f"  [info] room crossings in shard 0: {crossings} "
        f"({episodes_with_crossing}/{frames.shape[0]} episodes cross at least once)"
    )
    if episodes_with_crossing == 0:
        print("  [FAIL] no cross-room transitions — dataset can't teach the doorway")
        all_ok = False

    # Action distribution detail (never assume semantics — look at values).
    print("\n=== action stats (shard 0) ===")
    for axis, name in enumerate(["dx", "dy"]):
        a = actions[:, :, axis]
        print(
            f"  {name}: min={a.min():+.4f} max={a.max():+.4f} "
            f"mean={a.mean():+.4f} std={a.std():.4f}"
        )

    # ---- sample transition PNGs ---------------------------------------------
    png_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"\n=== writing {n_samples} sample transitions to {png_out} ===")
    for _ in range(n_samples):
        e = int(rng.integers(0, frames.shape[0]))
        t = int(rng.integers(0, actions.shape[1]))
        # frame_t and frame_{t+1} side by side with a 2px divider.
        divider = np.full((frames.shape[2], 2), 64, dtype=np.uint8)
        pair = np.concatenate([frames[e, t], divider, frames[e, t + 1]], axis=1)
        path = png_out / f"transition_e{e}_t{t}.png"
        Image.fromarray(pair).save(path)
        print(
            f"  {path.name}: state {states[e, t].round(3).tolist()} "
            f"--action {actions[e, t].round(3).tolist()}--> "
            f"{states[e, t + 1].round(3).tolist()}"
        )

    print(f"\n{'GATE PASSED' if all_ok else 'GATE FAILED'}: review the PNGs before training.")
    if not all_ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset directory")
    parser.add_argument("--png-out", type=Path, required=True, help="where to write sample PNGs")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    inspect_dataset(args.data, args.png_out, args.samples)


if __name__ == "__main__":
    main()
