# Model, losses, training

## Architecture

- **Encoder** (`models/encoder.py`, 0.91M params): four stride-2 conv blocks
  (64→4 spatial), GroupNorm + SiLU, linear head → z ∈ R¹²⁸. GroupNorm instead
  of BatchNorm on purpose: no train/eval statistics gap, no batch-size
  coupling, and a simpler ONNX graph.
- **Predictor** (`models/predictor.py`, 0.13M params): MLP on [z; a] with a
  **residual** output, ẑ′ = z + f([z; a]). One env step barely moves the
  embedding, so predicting the delta is the right inductive bias and keeps
  k-step open-loop rollouts stable.

128 latent dims for a 2-DoF env is deliberately generous: the interesting
question is what SIGReg does with the excess capacity (spread it), and what
its absence does (collapse it).

## The losses — and why there's no EMA or stop-gradient

`models/losses.py`. Both frame_t and frame_{t+1} go through the **same live
encoder** and the prediction MSE backpropagates into both paths. That setup
has a trivial global minimizer: encode everything to a constant. Most JEPA
implementations dodge this with an EMA target encoder or stop-gradients; this
project deliberately omits both so that the *only* thing standing between the
model and collapse is the regularizer — which makes collapse a controllable,
demonstrable phenomenon rather than a mystery.

**SIGReg** (sketched form): sample K=64 random unit directions per step,
project the latent batch onto each, and score each 1-D sample against N(0,1)
with the Epps–Pulley statistic — the ∫|ECF − e^{−t²/2}|² weighted by a normal
density, where ECF is the empirical characteristic function. CF-matching
penalizes *all* moments (not just mean/var), is smooth, and costs almost
nothing at this scale. Sanity anchors from the module's `__main__`:
gaussian ≈ 0.0009, collapsed ≈ 0.39, low-rank in between.

One knob: `lambda_reg`. 1.0 for the healthy run; 0.0 *is the product feature*
(the collapse checkpoints the demo hot-swaps).

## Training loop

`train.py`: pydantic config from yaml, AMP on CUDA, TensorBoard, and a
**built-in end-to-end sanity stage** — the first batch always prints
shape/dtype/range at every stage before the run proceeds (`--sanity-only` to
stop there). Checkpoints carry model + optimizer + config + step.

Live diagnostics every 50 steps: latent std, effective rank, and 8-step
open-loop rollout MSE on held-out episodes. Both runs take ~6 min on a
GTX 1650 at 0.41 GB peak VRAM.

## The collapse nuance (read before trusting metrics)

Recorded observation: the collapsed model's z_std is 0.001 (vs 1.163), but
its **probe R² is still 0.86** and its **effective rank ~123**. Both metrics
are scale-invariant, and a CNN never becomes *perfectly* constant — microscopic
residual structure survives and remains linearly decodable after rescaling.
Meanwhile the collapsed model's *training loss* looks 10,000× better than the
healthy one's (prediction is trivial when everything is one point).

Consequences, wired into `probes/report.py`'s output: **z_std and downstream
utility (rollout position error, planning success) are the honest collapse
detectors.** Loss, R², effective rank, and latent-space rollout MSE can all
look fine or even flattering on a dead representation.
