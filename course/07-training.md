# 07 — Training: the loop, the guardrails, the diagnostics

`training/src/latentlab/train.py` is a deliberately ordinary training script
with three unusual habits worth stealing: a mandatory end-to-end sanity
stage, *live* collapse diagnostics, and checkpoints that carry everything
needed to resurrect a run.

## Configuration as a contract

Hyperparameters live in a pydantic model with inline justifications:

```python
class TrainConfig(BaseModel):
    # Loss. lambda_reg=1.0 balances pred MSE (~1e-2 scale) against SIGReg
    # (~1e-3 scale when near-Gaussian); 0 => deliberate collapse run.
    lambda_reg: float = 1.0
    sigreg_projections: int = 64  # CF sketch width; more = lower-variance test

    # Optimization. Small model + easy task: AdamW 3e-4 is the safe default;
    # batch 256 gives SIGReg a decent sample for its distribution test.
    epochs: int = 15
    batch_size: int = 256
```

Two of these interact in a way that's easy to miss: **batch size 256 is a
SIGReg parameter as much as an optimization one.** The regularizer is a
statistical test on the *batch* — its empirical CFs are estimated from B
points. At B=32 the test is noisy; at B=256 (and the loss concatenates z_t
and z_{t+1}, so 512 points) it's stable. Shrink the batch and you weaken the
anti-collapse pressure in a way the loss value won't obviously reveal.

Yaml files override defaults; `healthy.yaml` and `collapse.yaml` differ in
one meaningful line (`lambda_reg: 1.0` vs `0.0`) — a proper controlled
experiment.

## The sanity stage: pay 5 seconds, save 6 minutes

Every run starts by pushing one real batch through every stage and printing
shapes, dtypes, and value ranges (`--sanity-only` stops there):

```
frame        shape=(256, 1, 64, 64)  min=-0.230 max=+7.088 mean=-0.000
action       shape=(256, 2)          min=-0.080 max=+0.080
z            shape=(256, 128)        min=-1.399 max=+1.290
losses: total=0.27265 pred_mse=0.09591 sigreg=0.17675
latent std=0.2606 eff_rank=105.4 (untrained baseline)
```

Every number is checkable against expectations you now hold: actions clamp
at ±0.08 (lesson 05), the frame max ≈ +7.1 is the normalized agent pixel
(lesson 06), and both loss terms are nonzero at init — there is signal to
descend. The untrained baseline row matters too: eff_rank ≈ 105 *before*
training is your reference point for reading the diagnostics below.

## The loop itself

Standard AMP-on-CUDA training — autocast forward, scaled backward — with one
structural point: a **single optimizer over both networks**,

```python
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(predictor.parameters()), ...)
```

encoder and predictor co-adapt through a shared loss; there's no target
network to maintain because SIGReg is doing the anti-collapse work (lesson
03). AMP here is a VRAM courtesy more than a speed win (measured peak:
0.41 GB of the 4 GB budget — the config could grow 5× before trouble).

## Live diagnostics: watching for collapse in real time

Every 50 steps, three numbers beyond the loss:

```python
latent_std = mean_latent_std(z.float())
eff_rank = effective_rank(z.float())
rollout = open_loop_rollout(encoder, predictor, rollout_frames,
                            rollout_actions, horizon=config.rollout_horizon)
```

- **z_std** — mean per-dimension std of the batch latents. The collapse
  siren: healthy training climbs it toward 1.0 (SIGReg's unit-variance
  target); the collapse run cratered it to 0.006 within 200 steps.
- **effective rank** — exp(entropy of the normalized singular-value
  spectrum): "how many dimensions is the batch really using?" Healthy
  training shows a story: ~105 at init (random noise is high-rank) falling
  to ~6.4 as the encoder discovers the data manifold is intrinsically 2-D
  (lesson 02). Low rank here is *learning*, not dying — which is exactly why
  no single metric is trusted alone (lesson 08 formalizes this).
- **k-step rollout MSE** on held-out episodes — chain the predictor 8 steps
  open-loop from a real start, compare against encodings of the real frames.
  This is the world model's actual job description, checked live.

The full split discipline: episodes 0–899 train, 900–999 eval; the rollout
diagnostic and all of lesson 08's probes use only the held-out slice.

## Checkpoints that can be resurrected

```python
torch.save({
    "encoder": ..., "predictor": ..., "optimizer": ...,
    "config": json.loads(config.model_dump_json()),
    "step": step, "epoch": epoch,
}, path)
```

Config *inside* the checkpoint means `load_checkpoint_models(path)` can
rebuild the exact architecture with no external context — which is what makes
the probe report (lesson 08), the ONNX exporter (lesson 11), and the demo's
checkpoint switcher all one-liners. `checkpoint_epochs: [1, 5]` also saves
early snapshots: "healthy epoch 1" isn't debris, it's a demo asset (the
switcher's "representations forming" entry).

Both real runs: ~6 minutes each on a GTX 1650. The collapse run's log is
worth reading end-to-end once — pred loss marching to 0.00000 while z_std
dies is lesson 03 rendered in ASCII.

## Try it

1. `uv run python -m latentlab.train --config configs/healthy.yaml
   --sanity-only` — verify every printed number against your expectations
   from lessons 05–06.
2. Retrain healthy with `batch_size: 64` (copy the yaml). Watch z_std and
   sigreg variance across steps versus the batch-256 log. This is exercise
   evidence for the "batch size is a SIGReg parameter" claim.
3. Settle lesson 03's exercise 3 empirically: add `z_next = z_next.detach()`
   in `jepa_loss` (scratch copy), set `lambda_reg: 0`, train 3 epochs. Does
   stop-gradient alone hold z_std up — fully, partially, not at all?
4. Open TensorBoard (`uv run tensorboard --logdir runs`) and read the healthy
   run's `collapse/effective_rank` curve: init ~105 → ~6. Explain both ends
   in one sentence each.
