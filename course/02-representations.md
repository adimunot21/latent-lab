# 02 — Representations: the currency of intelligence

## Three candidate currencies

For Two Rooms, the agent's situation could be described three ways:

1. **Ground-truth state** — the pair (x, y). Two numbers. Perfect, minimal…
   and unavailable in any interesting problem. A robot camera does not emit
   (x, y); it emits pixels. We keep states in our *dataset* purely for
   evaluation (they let us ask "did the encoder discover position?" —
   lesson 08) but the model never trains on them.
2. **Raw observation** — the 64×64 frame, a 4096-dimensional vector. Complete
   but terrible as a working currency: distances in pixel space don't respect
   task structure (move the agent one pixel and thousands of values change;
   two far-apart positions can differ in exactly as many pixels as two nearby
   ones), and any model operating on raw frames pays the full 4096-D cost at
   every step.
3. **Learned representation** — z ∈ R¹²⁸, produced by an encoder trained so
   that z supports a task. This is the currency the whole project runs on.

The interesting question is: *trained so that z supports what task, exactly?*
That choice defines entire subfields.

## The self-supervised landscape (fast tour)

No labels exist here — nobody annotates "this frame is at (0.31, 0.7)". The
supervision must come from structure in the data itself:

- **Reconstruction (autoencoders, VAEs)**: force z to contain enough to
  redraw the input. Works, but the objective rewards storing *everything*,
  including irrelevant detail — the decoder budget goes where the pixels are,
  not where the task is. And you pay for building a decoder you may never
  need.
- **Contrastive (SimCLR, InfoNCE families)**: pull embeddings of "same thing"
  views together, push different things apart. Effective, but needs careful
  negative sampling and augmentation design.
- **Predictive / joint-embedding (JEPA, BYOL, SimSiam, VICReg lineage)**:
  make embeddings of related inputs predictable from one another, *in
  embedding space*. No decoder, no negatives. The danger — collapse — moves
  from "engineering nuisance" to "the central design problem" (lesson 03).

This project sits squarely in the third family, with a twist that makes it a
**world model**: the two related inputs are *consecutive frames*, and the
predictor is **conditioned on the action** taken between them. The
representation is therefore forced to contain whatever is needed to know how
actions change the world — which for Two Rooms is precisely position and its
interaction with walls.

## What "good representation" means here, concretely

The project's working definition is operational, and each criterion has a
measurement in the repo:

| criterion | measured by | healthy value |
|---|---|---|
| linearly exposes the true state | ridge probe z → (x, y), lesson 08 | R² = 0.9997 |
| respects task geometry | latent distance vs position distance (lesson 09 diagnostics) | monotone, near-linear |
| supports imagination | k-step open-loop rollout error | 0.098 world units @ 8 steps |
| supports control | CEM planning success | 97% |
| uses its capacity | latent std / effective rank | std ≈ 1, rank ≈ 6–14 |

Note what's *not* on the list: reconstruction quality (there is no decoder)
and loss value (lesson 08 shows why loss is actively misleading).

A subtlety worth internalizing early: the dataset's frames live on a
**2-dimensional manifold** inside R⁴⁰⁹⁶ — every frame is fully determined by
(x, y). The encoder's job is to *unfold* that manifold into R¹²⁸ nicely. It
cannot (and should not) produce 128 independent dimensions of information;
there are only 2 to be had. When lesson 04's regularizer asks latents to look
"isotropic Gaussian", the model can only comply approximately — the measured
effective rank of ~6 rather than 128 is the geometry's honest answer, not a
bug.

## Where this lives in the code

The encoder is deliberately boring — a small strided CNN
(`training/src/latentlab/models/encoder.py`):

```python
channels = [base_channels * m for m in (1, 2, 4, 8)]  # 32, 64, 128, 256
blocks: list[nn.Module] = []
prev = in_channels
for ch in channels:
    blocks += [
        nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=ch),
        nn.SiLU(),
    ]
    prev = ch
```

Four stride-2 convs take 64×64 → 4×4 spatially while widening channels, then
a single linear head flattens to z ∈ R¹²⁸. Two decisions worth noticing:

- **GroupNorm, not BatchNorm.** BatchNorm couples every sample's output to
  the batch statistics — different behavior in train vs eval, batch-size
  sensitivity, and (later) a messier ONNX export. GroupNorm normalizes within
  each sample. In a project where the same network must run identically in
  PyTorch, onnxruntime CPU, and a browser, "no hidden state" is worth a lot.
- **0.91M parameters.** Small enough to train in six minutes on a 4 GB GPU
  and to ship to a browser tab. The course's phenomena don't need scale.

## Try it

1. Run the encoder standalone: `uv run python -m latentlab.models.encoder`.
   Note the parameter count and the output std at initialization.
2. Compute a pixel-space sanity check yourself: render frames at
   (0.2, 0.5), (0.28, 0.5), and (0.8, 0.5) using `TwoRoomsEnv.set_state` +
   `render()`, and compare `np.abs(a - b).sum()` between the near pair and
   the far pair. Observe how badly pixel distance tracks position distance —
   then keep that number in mind for lesson 09, where *latent* distance
   tracks position almost linearly.
3. Change `base_channels` to 8 in a scratch script and re-instantiate the
   encoder. How many parameters now? (You'll train tiny variants in
   lesson 07's exercises.)
