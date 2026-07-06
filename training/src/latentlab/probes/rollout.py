"""K-step open-loop rollout evaluation.

Encode frame_0, then apply the predictor k times with the recorded actions
(never re-encoding), and compare against the encoder's embedding of the true
frame at each step.

Two error views:
- ``latent``: MSE(z-hat_k, z_k). NOTE: a fully collapsed model scores ~0 here
  (everything maps to the same point), so latent MSE alone CANNOT certify a
  good world model. It is still useful during training to watch drift.
- ``probe``: decode both z-hat_k and the true state through a linear probe
  (latent -> (x, y)) and report position error. This is the honest metric for
  the healthy-vs-collapsed contrast and drives the Phase 6 imagination panel.
"""

from __future__ import annotations

import torch

from latentlab.models.encoder import Encoder
from latentlab.models.predictor import Predictor


@torch.no_grad()
def open_loop_rollout(
    encoder: Encoder,
    predictor: Predictor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Roll the predictor forward ``horizon`` steps from frame 0.

    Args:
        frames: (B, T+1, 1, H, W) normalized episode frames, T >= horizon.
        actions: (B, T, 2) recorded actions.
        horizon: number of open-loop steps k.

    Returns:
        z_pred: (B, horizon, D) imagined latents (steps 1..k).
        z_true: (B, horizon, D) encoder latents of the real frames 1..k.
        latent_mse_per_step: (horizon,) mean latent MSE at each step.
    """
    if frames.shape[1] < horizon + 1:
        raise ValueError(f"need at least {horizon + 1} frames, got {frames.shape[1]}")
    z = encoder(frames[:, 0])
    preds, trues = [], []
    for k in range(horizon):
        z = predictor(z, actions[:, k])
        preds.append(z)
        trues.append(encoder(frames[:, k + 1]))
    z_pred = torch.stack(preds, dim=1)
    z_true = torch.stack(trues, dim=1)
    latent_mse_per_step = ((z_pred - z_true) ** 2).mean(dim=(0, 2))
    return {"z_pred": z_pred, "z_true": z_true, "latent_mse_per_step": latent_mse_per_step}


@torch.no_grad()
def probe_position_error(
    z_pred: torch.Tensor,
    true_states: torch.Tensor,
    probe_weight: torch.Tensor,
    probe_bias: torch.Tensor,
) -> torch.Tensor:
    """Mean Euclidean position error (world units) of probed rollout latents.

    Args:
        z_pred: (B, K, D) imagined latents.
        true_states: (B, K, 2) ground-truth agent positions at those steps.
        probe_weight: (D, 2) linear probe weights; probe_bias: (2,).

    Returns (K,) mean position error per rollout step.
    """
    positions = z_pred @ probe_weight + probe_bias  # (B, K, 2)
    error: torch.Tensor = (positions - true_states).norm(dim=-1).mean(dim=0)
    return error


if __name__ == "__main__":
    # Shape sanity with untrained models and random data.
    torch.manual_seed(0)
    encoder, predictor = Encoder(), Predictor()
    frames = torch.randn(3, 6, 1, 64, 64)
    actions = torch.randn(3, 5, 2) * 0.08
    out = open_loop_rollout(encoder, predictor, frames, actions, horizon=5)
    print(f"z_pred {tuple(out['z_pred'].shape)}, z_true {tuple(out['z_true'].shape)}")
    print(f"latent mse per step: {[round(v, 4) for v in out['latent_mse_per_step'].tolist()]}")
    err = probe_position_error(
        out["z_pred"], torch.rand(3, 5, 2), torch.randn(128, 2) * 0.01, torch.zeros(2)
    )
    print(f"probe position error per step: {[round(v, 4) for v in err.tolist()]}")
