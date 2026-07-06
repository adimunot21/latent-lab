# 03 — JEPA and the collapse problem

## The architecture, precisely

JEPA = **Joint-Embedding Predictive Architecture**. "Joint-embedding" means
both inputs (here: frame_t and frame_{t+1}) are mapped into the *same*
embedding space by an encoder; "predictive" means a separate network predicts
one embedding from the other. Ours is action-conditioned, making it a world
model:

- Encoder `f_θ`: frame → z ∈ R¹²⁸ (lesson 02's CNN)
- Predictor `g_φ`: (z_t, a_t) → ẑ_{t+1}

The predictor (`training/src/latentlab/models/predictor.py`) is a small MLP
with one structural idea worth reading:

```python
def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    delta: torch.Tensor = self.mlp(torch.cat([z, action], dim=-1))
    return z + delta
```

**Residual prediction**: output = z + f([z; a]). One env step moves the agent
at most 0.08 world units, so the next embedding is *close* to the current
one; making the network predict the small delta rather than the full vector
is the right inductive bias. It also matters downstream: planning (lesson 09)
chains the predictor 12 steps deep, and the imagination panel chains it 8 —
residual structure keeps those chains from drifting wildly at initialization
and stabilizes them after training.

The prediction loss is plain MSE in embedding space
(`models/losses.py`):

```python
pred_loss = torch.nn.functional.mse_loss(z_pred, z_next)
```

## Why no decoder is a feature, not a shortcut

A pixel-predicting world model must spend capacity on everything that makes
pixels: exact rasterization of the wall, the circle's anti-aliasing, all of
it, at every imagined step. A JEPA only has to get the *embedding* right —
and the embedding only contains what the training pressure put there. For a
planner that just needs "will I be near the goal?", pixels are pure overhead.

The price: you lose the free visualization a decoder gives you. Lesson 10
shows the two devices this project uses instead (a latent→state lookup table
and a fixed PCA projection) — decoder-free visualization is a solvable
problem, not a reason to add a decoder.

## Now the villain

Look at the training setup with adversarial eyes. Both `z_pred` and `z_next`
are functions of the *same trainable encoder* — and in this project,
deliberately, **gradients flow through both** (no stop-gradient, no separate
target network). The optimizer is asked to minimize

    E‖ g(f(x_t), a_t) − f(x_{t+1}) ‖²

over θ *and* φ jointly. There is an embarrassing global optimum:

    f(anything) = c            (a constant vector)
    g(c, any action) = c

Loss: exactly zero. Gradient descent does not care that this solution is
useless; it cares that it's easy — and it is *very* easy to approach, because
shrinking the overall scale of f's output shrinks the loss quadratically
while nothing pushes back.

This isn't hypothetical. The repo ships the experiment
(`configs/collapse.yaml`, `lambda_reg: 0`). From the real training log:

```
epoch 1 step  50 | pred 0.00023 | z_std 0.012
epoch 1 step 200 | pred 0.00007 | z_std 0.006
epoch 14         | pred 0.00000 | z_std 0.001
```

Within **50 gradient steps** the latent std had fallen 20×; the "world model"
was near-perfect at predicting nothing. Compare the healthy run's final
prediction loss: 0.008 — *eighty times worse*, and infinitely more useful.
Burn this into memory: **in joint-embedding training, a beautiful loss curve
is evidence of nothing.**

## The defense landscape

Every working JEPA-family system counters collapse somehow. Three families:

1. **Architectural asymmetry** — BYOL/I-JEPA-style: the target branch is a
   slowly-updated EMA copy of the encoder, and/or gradients are stopped on
   the target side. The moving target denies the optimizer the coordinated
   two-sided shrink. Effective, widely used — and famously mysterious about
   *why* it suffices.
2. **Statistical regularizers** — VICReg-style: add penalty terms that
   directly demand per-dimension variance and decorrelation in the batch of
   embeddings. Collapse becomes expensive by construction.
3. **Distribution matching** — the approach here (SIGReg, from the LeJEPA
   line of work): require the batch of embeddings to look like a fixed
   target distribution, an isotropic Gaussian. A constant (or any degenerate)
   embedding distribution is maximally far from N(0, I), so collapse is
   penalized — but so are subtler pathologies like dimensional collapse,
   because the *whole shape* of the distribution is constrained, not just
   its first two moments.

This project chooses (3) and *removes* everything from families (1) and (2) —
no EMA, no stop-grad, no variance/covariance terms. Not because those are
bad, but for pedagogy: with exactly one anti-collapse mechanism in the
system, `lambda_reg` becomes a clean causal switch, and the demo can show you
both sides of it. The entire loss is
(`models/losses.py:jepa_loss`):

```python
pred_loss = torch.nn.functional.mse_loss(z_pred, z_next)
if lambda_reg > 0:
    latents = torch.cat([z_current, z_next], dim=0)
    reg_loss = sigreg_loss(latents, ...)
total = pred_loss + lambda_reg * reg_loss
```

Note the regularizer sees `z_current` *and* `z_next` — the constraint acts on
the encoder's output distribution generally, not just on one side of the
prediction.

How SIGReg actually measures "looks like N(0, I)" — with characteristic
functions and random projections, in ~30 lines — is the whole next lesson.

## Try it

1. Prove the degenerate optimum to yourself in ten lines: take the untrained
   encoder, multiply its final linear layer's weights and bias by 0.01, and
   compute `jepa_loss` with `lambda_reg=0` on a real batch. Watch pred MSE
   crater while the representation dies (z_std ≈ 0).
2. Read the collapse run's full log (`/tmp/train_collapse.log` if you still
   have it, or retrain: `uv run python -m latentlab.train --config
   configs/collapse.yaml`, ~6 min). Find the step where z_std crosses 0.01.
3. In `losses.py`, the healthy loss also backpropagates through `z_next`
   (no `.detach()`). Add `.detach()` on `z_next` in a scratch copy and think
   through: is a stop-gradient *alone* (with `lambda_reg=0`) enough to
   prevent collapse here? Form a hypothesis; lesson 07's exercise 3 lets you
   test it.
