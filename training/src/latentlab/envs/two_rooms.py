"""Two Rooms navigation environment.

A point agent (drawn as a circle) moves inside the unit square [0, 1]^2, which
is split into two rooms by a vertical wall at ``wall_x`` with a doorway gap in
the middle. Actions are (dx, dy) displacements, clamped per-axis.

PORTABILITY CONTRACT (this env is re-implemented in TypeScript for the browser
demo; cross-language parity is gated on shared fixtures in ``shared/fixtures/``):

- All state math uses Python floats (float64), which match JavaScript numbers
  bit-for-bit for these operations. No float32 anywhere in the dynamics.
- Coordinate system: x grows rightward, y grows DOWNWARD (image convention).
  Frame row i, column j covers world point ((j + 0.5)/N, (i + 0.5)/N).
- ``step`` resolves movement per-axis, x FIRST then y, rejecting an axis move
  that would collide (this yields wall-sliding). The x-then-y order is part of
  the contract.
- Collision test is circle-vs-AABB via closest-point clamping, with a STRICT
  ``<`` comparison against radius^2.
- Randomness exists only in ``reset`` (rejection-sampled start position); the
  dynamics in ``step`` are pure. Parity fixtures store explicit states, so RNG
  never needs to match across languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TwoRoomsConfig:
    """Environment geometry and rendering parameters (world units in [0, 1])."""

    wall_x: float = 0.5  # wall centerline; splits the square into two rooms
    wall_half_thickness: float = 0.03  # wall extends wall_x +/- this
    door_center_y: float = 0.5  # doorway centered vertically
    door_half_height: float = 0.12  # opening height 0.24 vs agent diameter 0.10
    agent_radius: float = 0.05  # agent circle radius
    max_step: float = 0.08  # per-axis action clamp; ~13 steps to cross a room
    frame_size: int = 64  # rendered frame is frame_size x frame_size grayscale
    wall_intensity: int = 128  # wall pixel value in the uint8 frame
    agent_intensity: int = 255  # agent pixel value (drawn over wall)


# Axis-aligned rectangle as (x_min, y_min, x_max, y_max).
Rect = tuple[float, float, float, float]


def wall_rects(config: TwoRoomsConfig) -> list[Rect]:
    """The two wall segments (below and above the doorway)."""
    x0 = config.wall_x - config.wall_half_thickness
    x1 = config.wall_x + config.wall_half_thickness
    return [
        (x0, 0.0, x1, config.door_center_y - config.door_half_height),
        (x0, config.door_center_y + config.door_half_height, x1, 1.0),
    ]


def circle_hits_rect(cx: float, cy: float, radius: float, rect: Rect) -> bool:
    """Circle-vs-AABB collision via closest-point clamping. Strict inequality."""
    x0, y0, x1, y1 = rect
    nearest_x = min(max(cx, x0), x1)
    nearest_y = min(max(cy, y0), y1)
    dx = cx - nearest_x
    dy = cy - nearest_y
    return dx * dx + dy * dy < radius * radius


@dataclass
class TwoRoomsEnv:
    """Deterministic Two Rooms env. See module docstring for the parity contract."""

    config: TwoRoomsConfig = field(default_factory=TwoRoomsConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._walls = wall_rects(self.config)
        self._x: float = 0.0
        self._y: float = 0.0
        self.reset()

    # ---- state access -----------------------------------------------------

    @property
    def state(self) -> tuple[float, float]:
        """Agent position (x, y) in world units."""
        return (self._x, self._y)

    def set_state(self, x: float, y: float) -> None:
        """Teleport the agent (used by planners/evals). Validates free space."""
        if not self._is_free(x, y):
            raise ValueError(f"state ({x}, {y}) collides with a wall or boundary")
        self._x = x
        self._y = y

    def _is_free(self, x: float, y: float) -> bool:
        r = self.config.agent_radius
        if not (r <= x <= 1.0 - r and r <= y <= 1.0 - r):
            return False
        return not any(circle_hits_rect(x, y, r, rect) for rect in self._walls)

    # ---- dynamics ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[float, float]:
        """Sample a uniform collision-free start position. Returns the state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self.config.agent_radius
        # Rejection sampling; free space is ~90% of the square so this is fast.
        for _ in range(1000):
            x = float(self._rng.uniform(r, 1.0 - r))
            y = float(self._rng.uniform(r, 1.0 - r))
            if self._is_free(x, y):
                self._x = x
                self._y = y
                return self.state
        raise RuntimeError("could not sample a free start position (bad geometry?)")

    def step(self, action: tuple[float, float]) -> tuple[float, float]:
        """Apply a clamped (dx, dy) displacement; returns the new state.

        Movement is resolved per-axis (x first, then y); an axis move that
        would collide with a wall is rejected, which produces sliding along
        walls. Boundary clamping keeps the full agent circle inside the square.
        """
        m = self.config.max_step
        r = self.config.agent_radius
        dx = min(max(float(action[0]), -m), m)
        dy = min(max(float(action[1]), -m), m)

        new_x = min(max(self._x + dx, r), 1.0 - r)
        if not any(circle_hits_rect(new_x, self._y, r, rect) for rect in self._walls):
            self._x = new_x

        new_y = min(max(self._y + dy, r), 1.0 - r)
        if not any(circle_hits_rect(self._x, new_y, r, rect) for rect in self._walls):
            self._y = new_y

        return self.state

    def clamp_action(self, action: tuple[float, float]) -> tuple[float, float]:
        """The action as ``step`` will actually apply it (per-axis clamp)."""
        m = self.config.max_step
        return (
            min(max(float(action[0]), -m), m),
            min(max(float(action[1]), -m), m),
        )

    # ---- rendering ---------------------------------------------------------

    def render(self) -> npt.NDArray[np.uint8]:
        """Rasterize to a (frame_size, frame_size) grayscale uint8 frame.

        Background 0, walls ``wall_intensity``, agent ``agent_intensity``
        (agent drawn over wall). A pixel belongs to a shape if its CENTER is
        inside it.
        """
        cfg = self.config
        n = cfg.frame_size
        centers = (np.arange(n, dtype=np.float64) + 0.5) / n
        xs = centers[np.newaxis, :]  # columns -> world x
        ys = centers[:, np.newaxis]  # rows -> world y (downward)

        frame = np.zeros((n, n), dtype=np.uint8)
        for x0, y0, x1, y1 in self._walls:
            mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            frame[mask] = cfg.wall_intensity

        agent = (xs - self._x) ** 2 + (ys - self._y) ** 2 <= cfg.agent_radius**2
        frame[agent] = cfg.agent_intensity
        return frame


if __name__ == "__main__":
    # Standalone sanity check: reset, take a few steps, print states, render.
    env = TwoRoomsEnv(seed=0)
    print(f"config: {env.config}")
    print(f"reset state: {env.state}")
    for i in range(5):
        s = env.step((0.05, -0.02))
        print(f"step {i}: state={s}")
    f = env.render()
    print(f"frame: shape={f.shape} dtype={f.dtype} min={f.min()} max={f.max()}")
    print(f"unique intensities: {np.unique(f).tolist()}")
