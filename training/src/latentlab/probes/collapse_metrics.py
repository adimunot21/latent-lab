"""Collapse detectors for latent batches: per-dim std and effective rank.

Both are logged during training (cheap, every N steps) and used post-hoc to
contrast healthy vs collapsed checkpoints.

- ``mean_latent_std``: mean over dimensions of the per-dim std. Collapse to a
  constant drives this to ~0. SIGReg-healthy training keeps it near 1.
- ``effective_rank``: exp(entropy of the normalized singular-value spectrum)
  of the centered batch. A healthy isotropic latent uses ~D dimensions;
  dimensional collapse concentrates variance in a few directions and drives
  this toward 1 even when std stays nonzero — which is why we track both.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mean_latent_std(z: torch.Tensor) -> float:
    """Mean over latent dims of the per-dimension standard deviation."""
    return float(z.std(dim=0).mean().item())


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """exp(Shannon entropy) of the normalized singular value spectrum.

    Ranges from 1 (all variance in one direction) to D (isotropic).
    """
    centered = z - z.mean(dim=0, keepdim=True)
    # float32: fp16 inputs (AMP) make svdvals unstable.
    singular_values = torch.linalg.svdvals(centered.float())
    # Normalize to a probability distribution over directions.
    sv_sum = singular_values.sum()
    if sv_sum <= 0:
        return 1.0  # exactly-constant batch
    p = singular_values / sv_sum
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = {
        "isotropic N(0,I)": torch.randn(512, 128),
        "collapsed (const)": torch.full((512, 128), 0.7),
        "low-rank (2 dims)": torch.cat([torch.randn(512, 2) * 3, torch.zeros(512, 126)], dim=1),
    }
    for name, z in cases.items():
        print(f"{name:20s} std={mean_latent_std(z):7.4f} eff_rank={effective_rank(z):8.2f}")
