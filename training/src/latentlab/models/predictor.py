"""Action-conditioned latent predictor: (z_t, a_t) -> z-hat_{t+1}.

Residual MLP: the output is z_t + f([z_t; a_t]). One env step moves the agent
at most ``max_step`` in world units, so the next embedding is close to the
current one — predicting the *delta* is the right inductive bias and makes
the k-step open-loop rollouts (planning, imagination panel) more stable.
"""

from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    """(B, latent_dim) x (B, action_dim) -> (B, latent_dim)."""

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = [nn.Linear(latent_dim + action_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta: torch.Tensor = self.mlp(torch.cat([z, action], dim=-1))
        return z + delta


if __name__ == "__main__":
    predictor = Predictor()
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"predictor params: {n_params / 1e6:.2f}M")
    z = torch.randn(4, 128)
    a = torch.randn(4, 2)
    z_next = predictor(z, a)
    print(f"z {tuple(z.shape)} + a {tuple(a.shape)} -> {tuple(z_next.shape)}")
    # Residual: zero-ish delta at init means z_next stays close to z.
    print(f"mean |delta| at init: {(z_next - z).abs().mean().item():.4f}")
    assert z_next.shape == z.shape
