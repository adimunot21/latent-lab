# Two Rooms & the dataset

## The environment

`training/src/latentlab/envs/two_rooms.py`. A point agent (drawn as a radius-
0.05 circle) in the unit square, split by a vertical wall at x=0.5 with a
doorway (opening 0.24 vs agent diameter 0.10). Actions are (dx, dy) clamped
per-axis to ±0.08. Frames are 64×64 grayscale uint8 with exactly three
intensities: 0 (background), 128 (wall), 255 (agent).

Non-obvious decisions:

- **Per-axis movement resolution (x first, then y)** instead of a contact
  solver: an axis move that would collide is simply rejected. This gives
  wall-sliding "for free", is fully deterministic, and ports to TypeScript
  with zero numerical ambiguity. The order is part of the parity contract —
  do not "clean it up" into a vector resolution.
- **Everything in Python floats, never numpy scalars, in the dynamics.**
  float64 matches JS numbers bit-for-bit; float32 anywhere in `step()` would
  make bit-exact parity impossible.
- **Why the env is this simple**: the project's demo value is in the *latent*
  machinery. A 2-DoF env keeps training minutes-long on a 4 GB GPU, makes the
  lookup-table trick viable, and still exposes nontrivial structure (the
  doorway bottleneck shows up as a neck in latent space).

## Dataset

`data/generate.py`: 1000 fixed-length episodes × 60 steps = 60k transitions,
sharded `.npz` (compressed; the whole thing is ~2.3 MB because frames are
mostly zeros), plus `meta.json` describing layout and seed. Fixed-length
episodes were chosen so shards are dense arrays `(E, T+1, ...)` — no episode
boundary bookkeeping anywhere downstream.

**Policy mix matters**: 50% random walk, 50% goal-directed with a
door-waypoint (head for the door when the goal is in the other room). A pure
random walk rarely crosses rooms, and the model cannot learn doorway dynamics
it never sees. Shard 0 of the real dataset has 302 crossings; the inspection
gate fails if a shard has none.

## The validation gate

`data/inspect.py` is not optional tooling — it's the checklist that runs
before any training code touches the data: dtypes/shapes/ranges per field,
invariant checks (actions within bounds, states within walls, agent visible
in every frame, cross-room coverage), and sample transition PNGs for eyeballs.
It exits nonzero on failure. If you change the env or generation, run it and
*look at the pictures* before training.

`data/dataset.py` holds everything in RAM (~250 MB) and serves transition
dicts. Normalization stats (`frame_mean`, `frame_std` over frames/255) are
computed once, persisted next to the shards, and shipped in the browser
manifest — the formula `(raw/255 - mean)/std` must stay identical in
`dataset.py`, `train.py`'s eval tensors, and `web/src/inference/manifest.ts`.
