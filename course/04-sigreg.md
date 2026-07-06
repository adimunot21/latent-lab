# 04 — SIGReg: testing Gaussianity with characteristic functions

Goal of this lesson: by the end, `sigreg_loss` in
`training/src/latentlab/models/losses.py` should read like prose. It's ~30
lines implementing a genuinely elegant idea: turn a *statistical hypothesis
test* into a *differentiable loss*.

## The target: why an isotropic Gaussian?

We want the batch of latents {z_i} ⊂ R¹²⁸ to be distributed like N(0, I):
zero mean, unit variance in every direction, no privileged axes. Reasons:

- It's the maximally "spread out" (max-entropy) distribution for a given
  variance budget — the exact opposite of collapse.
- It's *isotropic*: no dimension is special, so information can't hide by
  concentrating in a few coordinates (dimensional collapse) without being
  penalized.
- It's a fixed, simple target — no moving parts, no learned statistics.

## Step 1: reduce 128 dimensions to 1, honestly

Testing "is this 128-D sample Gaussian?" directly is hard. But there's a
classical fact (Cramér–Wold): a distribution on R^D is determined by the set
of its **1-D projections**. And for our specific target: z ~ N(0, I) *if and
only if* ⟨z, u⟩ ~ N(0, 1) for every unit vector u.

So: sample a few random unit directions, project the batch onto each, and
test each 1-D sample against N(0, 1). That's the "sketched" in "sketched
isotropy test" — we don't test every direction, we spot-check K=64 random
ones, **resampled fresh every training step**. Over thousands of steps the
directions cover the sphere; a pathology in any direction keeps getting
caught eventually, so nothing can hide.

```python
directions = torch.randn(latent_dim, num_projections, device=z.device, generator=generator)
directions = directions / directions.norm(dim=0, keepdim=True)
projections = z @ directions  # (B, K) — K 1-D samples, B points each
```

## Step 2: the 1-D test — characteristic functions

Now: given a 1-D sample {p_1..p_B}, produce a differentiable number that is
small iff the sample looks like N(0, 1). Options like comparing histograms
(non-differentiable) or just matching mean/variance (blind to shape) are out.
Enter the **characteristic function** (CF):

    φ(t) = E[e^{itX}] = E[cos(tX)] + i·E[sin(tX)]

The CF is the Fourier transform of the distribution — it encodes *all* of it,
every moment. Two distributions are equal iff their CFs are equal. Crucially,
the **empirical** CF is a smooth, differentiable function of the sample:
means of cosines and sines. And N(0, 1) has a famous, real-valued CF:

    φ_gauss(t) = e^{−t²/2}

The **Epps–Pulley statistic** measures the weighted L² gap between the
empirical CF and the Gaussian CF:

    T = ∫ |φ̂(t) − e^{−t²/2}|² · w(t) dt,     w(t) = standard normal density

The weight w(t) kills the integrand beyond |t| ≈ 4 (where CF estimates get
noisy anyway), so a small fixed integration grid suffices. In code:

```python
t = torch.linspace(0.0, t_max, t_points, device=z.device, dtype=z.dtype)
tp = projections.unsqueeze(-1) * t          # (B, K, T)
ecf_real = torch.cos(tp).mean(dim=0)        # (K, T)  E[cos(tX)]
ecf_imag = torch.sin(tp).mean(dim=0)        # (K, T)  E[sin(tX)]
target = torch.exp(-0.5 * t * t)            # e^{-t²/2}, imaginary part 0

sq_diff = (ecf_real - target) ** 2 + ecf_imag**2
weight = torch.exp(-0.5 * t * t)
return torch.trapezoid(sq_diff * weight, t, dim=-1).mean()
```

Three details that reward a careful read:

- **Why integrate only t ∈ [0, t_max]?** CFs are Hermitian
  (φ(−t) = conj φ(t)) and both the target and weight are even functions, so
  the integrand is even — the negative half-axis contributes an identical
  copy. Half the grid, same information.
- **The imaginary term `ecf_imag**2`.** The Gaussian CF is purely real, so
  *any* nonzero imaginary part in the empirical CF is pure penalty. The
  imaginary part of a CF encodes asymmetry — this term is what pushes the
  mean to zero and the shape to symmetric.
- **All moments at once.** Expand e^{itX} as a power series and you'll see
  matching CFs means matching E[X], E[X²], E[X³], … simultaneously. Contrast
  VICReg-style penalties, which pin down exactly two moments.

## Why collapse specifically gets hammered

If every z_i equals a constant c, each projection is a constant p = ⟨c, u⟩,
and the empirical CF is e^{itp} — a pure phase spiral with |φ̂(t)| = 1
everywhere. The Gaussian target *decays*. The gap |e^{itp} − e^{−t²/2}|² is
large across the whole grid. Run the module's built-in check:

```
$ uv run python -m latentlab.models.losses
sigreg(gaussian ) = 0.00086
sigreg(collapsed) = 0.39395     ← 460× the Gaussian score
sigreg(low-rank ) = 0.12351     ← variance hiding in 2 dims: also caught
```

The low-rank case is the subtle one: its per-dimension variance could be
tuned to fool a naive variance penalty, but most random projections of a
2-D-supported distribution in 128-D space look nothing like N(0, 1), and the
CF test sees it.

## What it costs and how it's balanced

Per step: a (B×K) matmul plus cos/sin on a (B, K, T) = (256, 64, 17) tensor —
microseconds on GPU, invisible next to the encoder's convolutions.

Balance: `lambda_reg = 1.0` in `configs/healthy.yaml`. From the real logs,
healthy training settles around pred ≈ 0.008, sigreg ≈ 0.045 — the two terms
live within an order of magnitude, neither dominating. And remember from
lesson 02: the data manifold is 2-D, so the encoder *cannot* reach
sigreg ≈ 0; the residual ~0.04 is the geometry's tax, permanently disputed
territory between the two loss terms. That tension is healthy — literally.

## Try it

1. Reproduce the three-case table above, then add a fourth case:
   `z = torch.randn(512, 128) * 3` (right shape, wrong scale). Predict its
   score relative to the others before running.
2. Set `num_projections=4` and score the same low-rank batch 20 times with
   different seeds. Watch the variance of the estimate — this is why K=64
   and fresh directions each step, rather than a few fixed ones.
3. Break it on purpose: change the weight to `torch.ones_like(t)` (uniform)
   and t_max to 40. What happens to the gaussian-vs-collapsed contrast, and
   why? (Hint: where do empirical CFs of *any* finite sample go as t grows?)
