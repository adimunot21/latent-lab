# 06 — Data: generation, validation, loading

A world model is a compression of its dataset. Every property you want the
model to have must first exist as *evidence in the data*. This lesson covers
the three modules in `training/src/latentlab/data/` and the discipline that
connects them.

## Generation (`generate.py`): coverage beats size

The dataset is 1000 episodes × 60 steps = **60k transitions** — small on
purpose. What's engineered is not volume but *coverage*, via a 50/50 policy
mix:

```python
if goal_directed:
    ...
    action = goal_policy_action(env, goal, rng, noise_scale)
else:
    action = env.clamp_action((uniform, uniform))
```

The random-walk half provides broad local coverage of (state, action, next)
everywhere reachable. But a random walk in a two-room world almost never
crosses rooms — the doorway is a needle's eye for diffusion. If the model
never sees crossings, the predictor cannot learn doorway dynamics, and no
planner can ever route through it. Hence the goal-directed half, with a
**waypoint policy**:

```python
same_room = (x < cfg.wall_x) == (gx < cfg.wall_x)
in_door_band = abs(y - cfg.door_center_y) < cfg.door_half_height - cfg.agent_radius
if same_room or in_door_band:
    target_x, target_y = gx, gy
else:
    target_x, target_y = cfg.wall_x, cfg.door_center_y   # head for the door first
```

Greedy toward the goal, unless the goal is in the other room and we're not
aligned with the doorway — then head for the door center first. Plus Gaussian
noise so trajectories aren't ruler-straight. Result (measured): 302 crossings
in the first 100 episodes alone.

Storage design: **fixed-length episodes**, so a shard is three dense arrays —

```
frames  uint8   (episodes, T+1, 64, 64)
states  float32 (episodes, T+1, 2)      ← ground truth, for probes only
actions float32 (episodes, T, 2)
```

— and transition t is simply (frames[e,t], actions[e,t], frames[e,t+1]). No
episode-boundary bookkeeping anywhere downstream; `__getitem__` is a
`divmod`. Compressed npz makes the whole dataset 2.3 MB (frames are mostly
zeros). The `states` array is the evaluation backdoor: the model never trains
on it, but lesson 08's probes need it to ask what the encoder learned.

## The validation gate (`inspect.py`): look before you train

House rule, and the single most transferable habit in this course: **no
training code runs until an inspection script has printed the data and a
human has looked at it.** The gate checks, mechanically:

- dtypes/shapes/ranges for every field
- invariants: |actions| ≤ max_step, states within [r, 1−r], frame values
  ∈ {0, 128, 255}, agent visible in *every* frame, T+1/T alignment
- **cross-room coverage** — the gate *fails* if no episode crosses rooms
  (see above: a model can't learn what data doesn't contain)
- writes side-by-side transition PNGs for eyeball review

It exits nonzero on failure, so it can sit in a pipeline. When it ran on the
real dataset, two of the three random sample transitions happened to catch
edge cases — a boundary clamp and a wall-corner rejection — both correct.
That's the gate doing its job: forcing you to *see* behavior, not assume it.

Why so paranoid about such a simple dataset? Because data bugs are silent.
A swapped axis, an off-by-one in the next-frame index, actions stored
pre-clamp — none of these crash; they just make the model quietly worse, and
you'll spend days blaming the architecture. Minutes of inspection buy days.

## Loading (`dataset.py`): one normalization formula, everywhere

Everything fits in RAM (~250 MB), so the Dataset just concatenates shards and
serves dict batches. The part that deserves attention is normalization:

```python
def compute_norm_stats(frames):
    scaled_mean = float(frames.mean()) / 255.0
    scaled_sq_mean = float((frames.astype(np.float64) ** 2).mean()) / 255.0**2
    std = float(np.sqrt(max(scaled_sq_mean - scaled_mean**2, 1e-12)))
    return {"frame_mean": scaled_mean, "frame_std": std}
```

Dataset-wide scalar mean/std over frames/255 (computed via E[x²]−E[x]² to
avoid materializing a float copy of every frame), persisted to
`norm_stats.json`. Measured values: mean 0.0314, std 0.1366 — the mean is
tiny because frames are ~97% black, and consequently the agent's 255-pixels
normalize to ≈ +7.1. A z-scored sparse image having a large max is normal;
the encoder is fine with it.

The critical discipline: the formula `(raw/255 − mean)/std` is applied in
**four places** — training batches, eval/probe tensors, the planning
evaluator, and the browser (`web/src/inference/manifest.ts`). They must never
drift, so the stats ship inside the browser manifest and the repo exposes one
shared helper (`normalize_frames`) on the Python side. An encoder fed frames
normalized with slightly different stats isn't "slightly off" — it's being
shown inputs from outside its training distribution, and nothing will warn
you.

## Try it

1. Generate a tiny dataset and run the gate on it:
   `... generate --out /tmp/tiny --episodes 20` then
   `... inspect --data /tmp/tiny --png-out /tmp/tiny_png`. Open the PNGs.
2. Generate with `--goal-fraction 0.0` and run the gate. Watch which check
   responds (episodes-with-crossing count at 20 episodes of pure random
   walk). Now you've seen the gate catch a real coverage failure.
3. Break the pipeline on purpose: in a scratch copy of `dataset.py`, change
   `frames[episode, t + 1]` to `frames[episode, t]` in `__getitem__`
   (a classic off-by-one). Which existing test catches it? (Run
   `pytest tests/test_data_pipeline.py -v`.) Note *how far* from the bug the
   failure would have surfaced without that test.
