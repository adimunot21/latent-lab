"""JEPA losses: next-embedding prediction MSE + SIGReg anti-collapse regularizer.

This JEPA uses NO EMA target network and NO stop-gradient: both frame_t and
frame_{t+1} go through the same live encoder. Prediction MSE alone therefore
has a trivial minimizer — encode everything to a constant — and *will*
collapse. SIGReg (LeJEPA-style) prevents this by pushing the batch latent
distribution toward an isotropic standard Gaussian N(0, I).

SIGReg, sketched/random-projection form:
  1. Sample K random unit directions in latent space (fresh each call).
  2. Project the batch latents onto each direction -> K one-dimensional samples.
  3. Score each 1-D sample against N(0, 1) with the Epps-Pulley statistic:
     the weighted integral of |empirical CF - Gaussian CF|^2, with a standard
     normal weight. CF = characteristic function E[exp(itX)].
  4. Average over directions.

If every 1-D projection of z is N(0, 1), z is isotropic Gaussian — matching
in CF-space penalizes ALL moments, not just mean/variance, and the statistic
is smooth and differentiable. Collapse (z near-constant) gives a CF of
exp(it*c), which is maximally far from exp(-t^2/2) under this metric.

The single knob is ``lambda_reg`` in ``jepa_loss``: lambda_reg = 0 reproduces
representation collapse on purpose (a checkpoint we ship as a demo feature).
"""

from __future__ import annotations

import torch


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 64,
    t_max: float = 4.0,
    t_points: int = 17,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Epps-Pulley isotropy score of a latent batch against N(0, I).

    Args:
        z: (B, D) latent batch.
        num_projections: K random directions per call (resampled each call).
        t_max: CF evaluated on t in [0, t_max]; the N(0,1) weight makes
            contributions beyond ~4 negligible.
        t_points: integration grid size.
        generator: optional torch.Generator for reproducible directions.

    Returns scalar loss (mean over projections).
    """
    batch_size, latent_dim = z.shape
    directions = torch.randn(latent_dim, num_projections, device=z.device, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    projections = z @ directions  # (B, K)

    # CF is Hermitian (phi(-t) = conj(phi(t))) and the target/weight are even,
    # so integrating t >= 0 captures everything up to a constant factor.
    t = torch.linspace(0.0, t_max, t_points, device=z.device, dtype=z.dtype)
    tp = projections.unsqueeze(-1) * t  # (B, K, T)
    ecf_real = torch.cos(tp).mean(dim=0)  # (K, T)
    ecf_imag = torch.sin(tp).mean(dim=0)  # (K, T)
    target = torch.exp(-0.5 * t * t)  # CF of N(0, 1); imaginary part is 0

    sq_diff = (ecf_real - target) ** 2 + ecf_imag**2  # (K, T)
    weight = torch.exp(-0.5 * t * t)  # standard normal weight (unnormalized)
    return torch.trapezoid(sq_diff * weight, t, dim=-1).mean()


def jepa_loss(
    z_pred: torch.Tensor,
    z_next: torch.Tensor,
    z_current: torch.Tensor,
    lambda_reg: float,
    num_projections: int = 64,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Total JEPA loss and its components.

    Args:
        z_pred: predictor output for t+1, (B, D).
        z_next: encoder output for frame_{t+1}, (B, D). NOT detached — no
            stop-gradient is deliberate (SIGReg is what prevents collapse).
        z_current: encoder output for frame_t, (B, D); regularized too so the
            constraint acts on the full encoder distribution.
        lambda_reg: SIGReg weight. 0 => collapse run.
    """
    pred_loss = torch.nn.functional.mse_loss(z_pred, z_next)
    if lambda_reg > 0:
        latents = torch.cat([z_current, z_next], dim=0)
        reg_loss = sigreg_loss(latents, num_projections=num_projections, generator=generator)
    else:
        reg_loss = torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)
    total = pred_loss + lambda_reg * reg_loss
    return {"total": total, "pred_mse": pred_loss, "sigreg": reg_loss}


if __name__ == "__main__":
    torch.manual_seed(0)
    # A standard normal batch should score near zero...
    z_good = torch.randn(512, 128)
    # ...a collapsed batch (all points identical) should score high...
    z_collapsed = torch.zeros(512, 128) + 0.7
    # ...and a low-rank batch (all variance in 2 dims) lands in between.
    z_lowrank = torch.zeros(512, 128)
    z_lowrank[:, :2] = torch.randn(512, 2) * 3

    for name, z in [("gaussian", z_good), ("collapsed", z_collapsed), ("low-rank", z_lowrank)]:
        print(f"sigreg({name:9s}) = {sigreg_loss(z).item():.5f}")

    losses = jepa_loss(z_good, z_good + 0.1, z_good, lambda_reg=1.0)
    print("jepa_loss components:", {k: round(v.item(), 5) for k, v in losses.items()})
