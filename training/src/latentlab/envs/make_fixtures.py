"""Generate cross-language env parity fixtures into shared/fixtures/.

The TypeScript port of Two Rooms must reproduce these EXACTLY (float64
bit-for-bit for states; byte-for-byte for frames). Fixtures deliberately
contain no RNG dependency: starts and actions are stored explicitly, so both
languages just replay dynamics.

Cases cover: free-space motion, wall blocking, wall sliding, doorway
crossings both ways, boundary clamping, oversized-action clamping, and a
long seeded random walk (actions materialized into the file).

Also embeds rendered frames for a few states: the browser encodes its OWN
canvas-rendered frames, so rasterization must match Python byte-for-byte or
encoder latents drift off the training distribution.

Usage (from training/):
    uv run python -m latentlab.envs.make_fixtures
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from latentlab.envs.two_rooms import TwoRoomsConfig, TwoRoomsEnv

FIXTURE_VERSION = 1


def rollout_case(
    name: str, start: tuple[float, float], actions: list[tuple[float, float]]
) -> dict[str, Any]:
    env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=0)
    env.set_state(*start)
    states = [list(env.step(a)) for a in actions]
    return {
        "name": name,
        "start": list(start),
        "actions": [list(a) for a in actions],
        "states": states,  # state AFTER each action; exact float64 replay
    }


def build_cases() -> list[dict[str, Any]]:
    cfg = TwoRoomsConfig()
    m = cfg.max_step
    cases = [
        rollout_case("free_space_diagonal", (0.2, 0.2), [(0.05, 0.03)] * 10),
        rollout_case("wall_block_from_left", (0.35, 0.15), [(m, 0.0)] * 15),
        rollout_case("wall_slide_diagonal", (0.40, 0.20), [(m, m)] * 12),
        rollout_case("door_crossing_left_to_right", (0.30, cfg.door_center_y), [(m, 0.0)] * 10),
        rollout_case("door_crossing_right_to_left", (0.70, cfg.door_center_y), [(-m, 0.0)] * 10),
        rollout_case("boundary_clamp_corner", (0.10, 0.10), [(-m, -m)] * 8),
        rollout_case("oversized_action_clamp", (0.20, 0.80), [(10.0, -10.0), (-5.0, 5.0)]),
        # Doorway approach at an angle (exercises corner collision + slide).
        rollout_case("door_corner_graze", (0.40, 0.30), [(m, 0.04)] * 14),
    ]
    # Long random walk: actions sampled once here and stored explicitly.
    rng = np.random.default_rng(1234)
    walk = [(float(a[0]), float(a[1])) for a in rng.uniform(-m, m, size=(100, 2))]
    cases.append(rollout_case("random_walk_100", (0.25, 0.65), walk))
    return cases


def build_frames() -> list[dict[str, Any]]:
    """Byte-exact rendered frames at representative states."""
    env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=0)
    frames = []
    for name, state in [
        ("left_room", (0.25, 0.30)),
        ("in_doorway", (0.5, 0.5)),
        ("near_boundary", (0.05, 0.95)),
    ]:
        env.set_state(*state)
        frames.append(
            {
                "name": name,
                "state": list(state),
                "frame": env.render().flatten().tolist(),  # row-major uint8
            }
        )
    return frames


def main() -> None:
    fixtures_dir = Path(__file__).resolve().parents[4] / "shared" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    frames = build_frames()
    payload: dict[str, Any] = {
        "fixture_version": FIXTURE_VERSION,
        "env_config": dataclasses.asdict(TwoRoomsConfig()),
        "cases": cases,
        "frames": frames,
    }
    out = fixtures_dir / "two_rooms_parity.json"
    # Default json float repr is shortest-roundtrip: parses back bit-exact in
    # both Python and JS (both IEEE float64).
    out.write_text(json.dumps(payload, indent=1))
    n_states = sum(len(c["states"]) for c in cases)
    print(
        f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, "
        f"{len(cases)} cases, {n_states} states, {len(frames)} frames)"
    )


if __name__ == "__main__":
    main()
