"""Unit tests for the Two Rooms env: collision, doorway, determinism, render."""

from __future__ import annotations

import numpy as np
import pytest

from latentlab.envs.two_rooms import (
    TwoRoomsConfig,
    TwoRoomsEnv,
    circle_hits_rect,
    wall_rects,
)

CFG = TwoRoomsConfig()


def make_env_at(x: float, y: float) -> TwoRoomsEnv:
    env = TwoRoomsEnv(seed=0)
    env.set_state(x, y)
    return env


# ---- collision geometry ----------------------------------------------------


def test_wall_rects_leave_door_gap() -> None:
    lower, upper = wall_rects(CFG)
    assert lower[3] == pytest.approx(CFG.door_center_y - CFG.door_half_height)
    assert upper[1] == pytest.approx(CFG.door_center_y + CFG.door_half_height)
    assert lower[3] < upper[1]  # actual opening


def test_circle_hits_rect_basic() -> None:
    rect = (0.4, 0.0, 0.6, 1.0)
    assert circle_hits_rect(0.5, 0.5, 0.05, rect)  # inside
    assert circle_hits_rect(0.38, 0.5, 0.05, rect)  # overlapping edge
    assert not circle_hits_rect(0.3, 0.5, 0.05, rect)  # clear of it
    # Exactly touching (distance == radius) is NOT a hit (strict <).
    assert not circle_hits_rect(0.35, 0.5, 0.05, rect)


# ---- wall blocks movement ----------------------------------------------------


def test_wall_blocks_horizontal_movement() -> None:
    # Start left of the wall, well away from the door, and push right hard.
    env = make_env_at(0.35, 0.15)
    for _ in range(20):
        env.step((CFG.max_step, 0.0))
    x, y = env.state
    # Agent must stop before the wall face at wall_x - half_thickness.
    assert x < CFG.wall_x - CFG.wall_half_thickness
    assert x > 0.3  # but it did move toward the wall
    assert y == pytest.approx(0.15)


def test_sliding_along_wall() -> None:
    # Diagonal push into the wall: x blocked, y still moves (per-axis resolution).
    env = make_env_at(0.40, 0.2)
    initial_y = env.state[1]
    env.step((CFG.max_step, CFG.max_step))
    x, y = env.state
    assert x < CFG.wall_x - CFG.wall_half_thickness
    assert y > initial_y  # slid along the wall


def test_boundary_clamp() -> None:
    env = make_env_at(0.1, 0.1)
    for _ in range(10):
        env.step((-CFG.max_step, -CFG.max_step))
    x, y = env.state
    assert x == pytest.approx(CFG.agent_radius)
    assert y == pytest.approx(CFG.agent_radius)


# ---- doorway is passable -----------------------------------------------------


def test_doorway_passable() -> None:
    # Aligned with the door center: repeated right steps must cross rooms.
    env = make_env_at(0.3, CFG.door_center_y)
    for _ in range(10):
        env.step((CFG.max_step, 0.0))
    x, _ = env.state
    assert x > CFG.wall_x + CFG.wall_half_thickness, "agent failed to pass the doorway"


def test_doorway_passable_right_to_left() -> None:
    env = make_env_at(0.7, CFG.door_center_y)
    for _ in range(10):
        env.step((-CFG.max_step, 0.0))
    x, _ = env.state
    assert x < CFG.wall_x - CFG.wall_half_thickness


# ---- action clamping ---------------------------------------------------------


def test_action_clamped() -> None:
    env = make_env_at(0.2, 0.8)
    env.step((10.0, -10.0))  # absurd action
    x, y = env.state
    assert x == pytest.approx(0.2 + CFG.max_step)
    assert y == pytest.approx(0.8 - CFG.max_step)


# ---- determinism ---------------------------------------------------------------


def test_reset_deterministic_given_seed() -> None:
    a = TwoRoomsEnv(seed=123).state
    b = TwoRoomsEnv(seed=123).state
    assert a == b


def test_trajectory_deterministic() -> None:
    rng = np.random.default_rng(7)
    actions = rng.uniform(-CFG.max_step, CFG.max_step, size=(50, 2))

    def rollout() -> list[tuple[float, float]]:
        env = TwoRoomsEnv(seed=42)
        return [env.step((float(a[0]), float(a[1]))) for a in actions]

    traj_a, traj_b = rollout(), rollout()
    assert traj_a == traj_b  # exact float equality, not approx


def test_reset_positions_are_free_and_varied() -> None:
    env = TwoRoomsEnv(seed=0)
    states = {env.reset() for _ in range(50)}
    assert len(states) == 50  # all distinct
    for x, y in states:
        assert CFG.agent_radius <= x <= 1 - CFG.agent_radius
        assert CFG.agent_radius <= y <= 1 - CFG.agent_radius


# ---- rendering -----------------------------------------------------------------


def test_render_shape_dtype_range() -> None:
    env = TwoRoomsEnv(seed=0)
    frame = env.render()
    assert frame.shape == (CFG.frame_size, CFG.frame_size)
    assert frame.dtype == np.uint8
    values = set(np.unique(frame).tolist())
    assert values <= {0, CFG.wall_intensity, CFG.agent_intensity}
    assert CFG.agent_intensity in values, "agent not visible"
    assert CFG.wall_intensity in values, "wall not visible"


def test_render_tracks_agent() -> None:
    env = make_env_at(0.2, 0.2)
    frame_a = env.render()
    env.set_state(0.8, 0.8)
    frame_b = env.render()
    assert not np.array_equal(frame_a, frame_b)
    # Agent blob centroid should be in the correct quadrant.
    ys, xs = np.nonzero(frame_b == CFG.agent_intensity)
    n = CFG.frame_size
    assert xs.mean() / n > 0.6
    assert ys.mean() / n > 0.6


def test_set_state_rejects_collisions() -> None:
    env = TwoRoomsEnv(seed=0)
    with pytest.raises(ValueError):
        env.set_state(CFG.wall_x, 0.1)  # inside the lower wall segment
    with pytest.raises(ValueError):
        env.set_state(0.0, 0.5)  # agent circle would poke out of bounds
