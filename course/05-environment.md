# 05 — The environment: dynamics designed to be ported

`training/src/latentlab/envs/two_rooms.py` is ~180 lines and looks almost too
simple to deserve a lesson. The lesson is *why* it looks that way: every
simplification is load-bearing, because this exact logic must later be
re-implemented in TypeScript (lesson 12) and produce **bit-identical**
results — a requirement most simulators can't meet and ours meets by
construction.

## The world

Unit square [0,1]². A vertical wall at x = 0.5 (half-thickness 0.03) with a
doorway gap: opening height 0.24 vs agent diameter 0.10 — passable with
margin, but a genuine bottleneck. The wall is stored as two axis-aligned
rectangles:

```python
def wall_rects(config: TwoRoomsConfig) -> list[Rect]:
    x0 = config.wall_x - config.wall_half_thickness
    x1 = config.wall_x + config.wall_half_thickness
    return [
        (x0, 0.0, x1, config.door_center_y - config.door_half_height),
        (x0, config.door_center_y + config.door_half_height, x1, 1.0),
    ]
```

Collision is circle-vs-AABB via closest-point clamping:

```python
def circle_hits_rect(cx, cy, radius, rect):
    x0, y0, x1, y1 = rect
    nearest_x = min(max(cx, x0), x1)
    nearest_y = min(max(cy, y0), y1)
    dx = cx - nearest_x
    dy = cy - nearest_y
    return dx * dx + dy * dy < radius * radius
```

Clamp the circle center into the rectangle to find the nearest rectangle
point; collide iff that point is strictly inside the circle. Four min/max,
two multiplies, one compare — nothing a floating-point unit can disagree with
itself about.

## The step function — read the order

```python
def step(self, action):
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
```

Movement is resolved **per axis, x first then y**, and a colliding axis-move
is simply *rejected* (not resolved to contact). Consequences:

- **Wall sliding for free.** Push diagonally into the wall: the x-component
  is rejected, the y-component still applies — the agent slides along the
  wall. A contact solver would need to compute penetration depths and
  projection; this needs an `if`.
- **Total determinism, trivially.** No iteration, no epsilon tolerances, no
  solver state. Same inputs → same floats, always.
- **The x-then-y order is observable** (there exist corner states where
  y-then-x lands elsewhere) and is therefore *part of the contract* — the
  docstring says so explicitly, and the TS port must match it.

## The portability contract

The module docstring is the most important documentation in the repo. Its
promises, and why each exists:

| promise | reason |
|---|---|
| all state math in Python floats (float64) | JS numbers *are* IEEE float64 — bit-identical arithmetic is possible; one numpy float32 anywhere would break it |
| strict `<` in collision | `<` vs `<=` differs on exact-touch states; pick one, write it down |
| x-then-y resolution order | observable, so contractual |
| pixel-center rasterization, exact intensity values | the browser encodes its own renders (lesson 12) — frames must match byte-for-byte |
| RNG only in `reset()` | Python and JS RNGs will never match; keeping dynamics RNG-free means parity fixtures can just store starts and actions |

Lesson 13 shows the enforcement: fixtures generated from Python that both
languages must replay with `===`-level equality, gated in both CIs.

## Rendering

```python
centers = (np.arange(n, dtype=np.float64) + 0.5) / n
xs = centers[np.newaxis, :]   # columns -> world x
ys = centers[:, np.newaxis]   # rows -> world y (downward)

frame = np.zeros((n, n), dtype=np.uint8)
for x0, y0, x1, y1 in self._walls:
    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    frame[mask] = cfg.wall_intensity           # 128

agent = (xs - self._x) ** 2 + (ys - self._y) ** 2 <= cfg.agent_radius**2
frame[agent] = cfg.agent_intensity             # 255, drawn over wall
```

A pixel belongs to a shape iff its **center** is inside it. Three intensities
only: 0 / 128 / 255. Deliberately no anti-aliasing — soft edges would make
byte-exact cross-language rendering a nightmare for zero modeling benefit.
Note the y-axis points **down** (image convention), matching both numpy row
order and the browser canvas; the docstring pins this too.

Also worth noticing: `set_state` *validates* (raises on wall/boundary
overlap) — planners and evals teleport the agent constantly, and a silent
invalid teleport would corrupt an experiment far from the actual bug.

## Try it

1. Run the standalone check (`uv run python -m latentlab.envs.two_rooms`) and
   the env tests (`uv run pytest tests/test_two_rooms.py -v`). Match each
   test name to the behavior it pins.
2. Find a corner state where x-then-y and y-then-x orders genuinely diverge:
   place the agent just below-left of the lower wall segment's bottom corner
   and step (+m, +m) under both orders (edit a scratch copy). This is why
   order is contractual.
3. Predict, then verify: what does `step((0.2, 0.0))` do from (0.5, 0.5) —
   the doorway center? (Remember the per-axis clamp *and* which rectangles
   exist there.)
