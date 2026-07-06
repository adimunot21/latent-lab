# Latent planning

## CEM in latent space

`planning/cem.py` (reference) / `web/src/planner/cem.ts` (port). Plan = sample
a population of action sequences from a per-step Gaussian, roll each through
the **predictor only** (no env), score against the encoded goal, refit the
Gaussian on the elite set, repeat 4×. MPC: execute the first action, re-plan,
warm-starting from the previous mean shifted one step.

Wall avoidance is emergent: the predictor learned from data that pushing into
a wall doesn't move the embedding, so tunneling candidates score badly and CEM
discovers the doorway route. The planner contains no environment knowledge.

## The dense-cost lesson (the phase's big find)

The first implementation scored the **endpoint only** — ‖z_H − z_goal‖² — and
planning succeeded 20% of the time. Instrumentation showed CEM reporting
near-perfect imagined costs while the agent random-walked. Root cause: when
the goal is reachable in fewer than `horizon` steps, *every* sequence that
arrives by step H ties — including ones that wander first — so the first
action (the only one executed) is underdetermined, and each re-plan picks a
different arbitrary one.

Fix: **dense cost** — sum the latent distance over all imagined steps, with
4× weight on the terminal step ("arrive AND stay"). Immediate progress becomes
optimal, the first action is pinned down, and success jumped to 97%. If you
ever swap the cost function, keep this property or expect the random walk
back. Secondary benefit: dense cost gives model-exploiting trajectories fewer
places to hide, since every intermediate state is charged.

Verified before touching the cost: latent distance is cleanly monotone in
position distance (36 → 1 as the gap closes), so the representation was never
the problem. Diagnose geometry before algorithms.

## Evaluation

`planning/evaluate.py`: 100 episodes, half forced cross-room, success = within
0.08 of the goal within 60 steps. Healthy final: **97%** (94% cross-room,
100% same-room, mean 6.7 steps). Collapsed final: **44%** — and that number
flatters it; it's mostly 60-step wanders that end near the goal by luck (mean
steps double). Planning GIFs with a goal ring are written per run.

## Lookup table & PCA (`export/latent_maps.py`)

Grid the free space at 64×64 (→ 2880 free cells), render + encode each, save
(states, latents) plus a top-2 PCA (71.5% explained variance). Sanity checks
are part of the build: NN-decoding a real trajectory's latents lands within
0.0063 world units of the truth, and the PCA scatter is visually inspected —
it shows two sheets (one per room) joined at a narrow neck, which is the
doorway appearing in latent geometry.
