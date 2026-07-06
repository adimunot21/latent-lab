# Architecture

## What this is

latent-lab is two codebases sharing one contract: a **Python training
pipeline** (`training/`) that produces a JEPA world model, and a **TypeScript
browser app** (`web/`) that runs it. They meet at three artifacts: the ONNX
models, `manifest.json`, and the env-parity fixtures in `shared/fixtures/`.

## The dual-env design

The Two Rooms environment exists twice — Python for data generation and
evaluation, TypeScript for the live demo. This is the project's biggest
structural risk: if the two drift, the browser encodes frames the model never
saw and everything degrades *silently*.

The mitigation is a **portability contract** (documented at the top of
`training/src/latentlab/envs/two_rooms.py` and mirrored in
`web/src/env/twoRooms.ts`): all state math in float64 (JS numbers *are*
float64, so bit-identical results are achievable, not aspirational), a fixed
x-then-y collision resolution order, strict-`<` collision comparisons,
pixel-center rasterization, and RNG confined to reset. Fixtures generated
from Python replay in TS with `===` equality on states and byte equality on
frames, in both CI pipelines. Frame parity matters as much as state parity
because the browser **encodes its own canvas renders** — one differing pixel
shifts latents off the training distribution.

## Data flow

```
env → dataset (npz shards) → encoder/predictor training → probes/planning eval
                                        │
                              ONNX export + manifest + lookup/PCA
                                        │
                              HF Hub (pinned revision)
                                        │
        browser: fetch+verify → encoder session (main thread)
                              → predictor session (worker) → CEM → MPC loop
```

## Why decoder-free

JEPA deliberately has no pixel decoder — prediction happens in embedding
space. That leaves visualization of *imagined* states unsolved: you can't
render a latent. Two devices fill the gap, both computed at export time:

1. **Latent↔state lookup table**: encode a dense grid of positions; at
   runtime, nearest-neighbor an imagined latent → its (x, y). Works because
   Two Rooms' state space is 2-D and compact (2880 grid points suffice;
   NN-decode error measured at 0.0063 world units).
2. **Fixed PCA projection**: 2-D projection of the healthy encoder's latents
   for the cloud panel. It is deliberately **never refit per checkpoint** —
   a fixed basis and fixed view box are what make collapse *visible* as a
   shrinking cloud rather than being normalized away.

Gotcha: both devices are built from the **healthy** encoder. Latents from
other checkpoints are projected through the same basis (that's the point),
but lookup-decoding a *collapsed* model's latents is meaningless — everything
NN-matches to noise.

## Failure modes considered

- **Env drift** → parity fixtures gate both CIs (build failure, not runtime bug).
- **Artifact tampering / CDN corruption** → sha256 of every file in
  `manifest.json`, verified before use.
- **Moving upstream weights** → the browser pins an HF revision (commit hash)
  in `web/src/config.ts`; `main` on the Hub can move freely.
- **No WebGPU** → WASM is a first-class path, gated in CI; WebGPU is a
  progressive enhancement.
