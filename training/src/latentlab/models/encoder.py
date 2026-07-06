"""CNN encoder: 64x64 grayscale frame -> latent vector z.

Small strided conv stack (64 -> 32 -> 16 -> 8 -> 4 spatial), GroupNorm + SiLU,
then a linear head to ``latent_dim``. GroupNorm instead of BatchNorm on
purpose: no train/eval statistics mismatch, no batch-size coupling, and a
cleaner ONNX export (Phase 4).
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Pixels (B, 1, 64, 64) -> latents (B, latent_dim)."""

    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 1,
        base_channels: int = 32,
        frame_size: int = 64,
    ) -> None:
        super().__init__()
        if frame_size % 16 != 0:
            raise ValueError(f"frame_size must be divisible by 16, got {frame_size}")
        self.latent_dim = latent_dim

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
        self.backbone = nn.Sequential(*blocks)

        final_spatial = frame_size // 16
        self.head = nn.Linear(channels[-1] * final_spatial * final_spatial, latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.backbone(frames)
        latent: torch.Tensor = self.head(features.flatten(start_dim=1))
        return latent


if __name__ == "__main__":
    encoder = Encoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder params: {n_params / 1e6:.2f}M")
    z = encoder(torch.randn(4, 1, 64, 64))
    print(f"input (4, 1, 64, 64) -> z {tuple(z.shape)}, std {z.std().item():.3f}")
    assert z.shape == (4, 128)
