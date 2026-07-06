# 08 — Measuring representations (and the metrics that lie)

You've trained two models. One is excellent, one is dead. This lesson is
about *proving* which is which — and it contains the project's most
instructive empirical surprise: several respectable metrics **cannot tell
them apart**, and one actively prefers the dead one.

## Tool 1: the linear probe (`probes/linear_probe.py`)

Question: did the encoder discover position? Test: freeze the encoder, fit a
*linear* map z → (x, y) on training latents, measure R² on held-out episodes.
Linear is the point — if a linear readout recovers the state, the encoder has
done all the nonlinear work of organizing pixels by position, which is
exactly what latent planning needs.

The fit is closed-form ridge regression — no SGD, no tuning:

```python
def fit_linear_probe(latents, targets, alpha=1e-4):
    z = torch.cat([latents, torch.ones(latents.shape[0], 1, ...)], dim=1)  # bias column
    gram = z.T @ z + alpha * torch.eye(d, ...)
    solution = torch.linalg.solve(gram, z.T @ targets)
    return solution[:-1], solution[-1]        # weight (D,2), bias (2,)
```

(The bias-augmentation trick: append a 1s column so the intercept is just
another weight; `alpha` keeps the Gram matrix invertible.) R² is
1 − SS_res/SS_tot per output dimension: 1.0 = perfect, 0.0 = no better than
predicting the mean.

## Tool 2: collapse metrics (`probes/collapse_metrics.py`)

- `mean_latent_std(z)` — amplitude: is there any spread at all?
- `effective_rank(z)` — shape: over how many directions is the spread
  distributed? Definition: take the singular values of the centered batch,
  normalize them into a probability vector p, and compute exp(−Σ pᵢ log pᵢ).
  All variance in one direction → 1; spread evenly over k directions → k.
  It's "counting dimensions" made continuous.

## Tool 3: open-loop rollout (`probes/rollout.py`)

Encode frame 0, then apply the predictor k times using the *recorded*
actions — never re-encoding — and compare against the encoder's embeddings of
the real frames. Two error views, and the difference between them is the
heart of this lesson:

- **latent MSE**: ‖ẑ_k − z_k‖². Cheap, logged during training.
- **probe-space position error**: decode both imagined and true latents
  through the fitted linear probe and measure the *position* discrepancy in
  world units.

The module docstring carries a warning written after seeing real numbers:
a fully collapsed model scores **~0** on latent MSE — everything maps to the
same point, so imagined and "true" latents agree perfectly. The metric
rewards the failure.

## The real numbers, and how to read them

From `probes/report.py` run against the actual checkpoints:

| checkpoint | probe R² | z_std | eff_rank | rollout latent MSE | rollout pos err |
|---|---|---|---|---|---|
| healthy epoch 1 | 0.9991 | 0.688 | 13.9 | 0.567 | 0.268 |
| healthy final | 0.9997 | 1.163 | 6.4 | 0.154 | **0.098** |
| collapsed epoch 1 | 0.687 | 0.006 | 123.7 | 0.0001 | 0.283 |
| collapsed final | **0.858** | **0.001** | 123.2 | **0.00001** | **0.290** |

Work through the surprises; each teaches a general principle.

**Surprise 1: collapsed probe R² is 0.86, not ~0.** R² is scale-invariant —
it measures correlation structure, not amplitude. The CNN never becomes
*perfectly* constant; microscopic residual variation (std 0.001) still
correlates with position, and ridge regression happily amplifies it with
huge weights. Information wasn't annihilated; its amplitude was. *Principle:
scale-invariant metrics are blind to amplitude collapse.*

**Surprise 2: collapsed eff_rank is ~123 — higher than healthy's 6.4!** The
residual noise around the collapsed point is nearly isotropic, and effective
rank (also scale-invariant) measures the shape of whatever variance exists,
however microscopic. Meanwhile the *healthy* model's rank honestly reports
the 2-D data manifold. Read naively, this metric ranks the corpse above the
athlete. *Principle: know what your metric is invariant to; that's where
failures hide.*

**Surprise 3: rollout latent MSE prefers the collapsed model by 10,000×.**
Already explained — and it's the same pathology as the training loss itself.

**What actually separates them: z_std (1000×) and downstream utility** —
probed rollout position error (0.098 vs 0.290 — the collapsed rollout is
uninformative noise) and, most brutally, planning success (97% vs 44%,
lesson 09). The 44% itself needs honest reading: with 60 steps and a
success radius, a drunkard's walk sometimes stumbles onto the goal.

## The meta-lesson

Evaluate representations the way the system will *use* them. Every
scale-invariant or reconstruction-flavored metric here had a blind spot that
collapse walked straight through; the metrics tied to downstream function
(can a linear probe *and its amplitude* survive a rollout? can a planner
navigate?) were undeceived. `probes/report.py` prints exactly this caveat
under its table — documentation written by getting burned.

## Try it

1. Run the report yourself across all six checkpoints (command in the module
   docstring). Reproduce the table.
2. Verify Surprise 1's mechanism: fit the probe on collapsed-final latents
   and print `weight.norm()`. Compare against the healthy probe's weight
   norm. Ridge amplifying microscopic signal should be visible as a weight
   norm orders of magnitude larger.
3. Design-your-own-metric exercise: propose a *single* scalar that would have
   correctly ranked all six checkpoints. (One candidate: probe R² computed
   after adding N(0, 0.01) noise to the latents — why does that fix
   Surprise 1? What's its failure mode?)
