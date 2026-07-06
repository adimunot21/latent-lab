"""Linear probe: frozen latents -> agent (x, y) via closed-form ridge regression.

The probe is the primary representation-quality metric: if a *linear* readout
recovers the agent position from z, the encoder has organized the latent space
by position (which is what latent planning needs). Collapsed encoders probe at
R^2 ~ 0.

Closed form (ridge with bias): W = (Z^T Z + alpha I)^-1 Z^T Y on bias-augmented
Z. No SGD, no hyperparameters to tune besides a tiny alpha.

Standalone:
    uv run python -m latentlab.probes.linear_probe --checkpoint checkpoints/<run>/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentlab.models.encoder import Encoder


def fit_linear_probe(
    latents: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge fit z -> y. Returns (weight (D, out), bias (out,))."""
    z = torch.cat([latents, torch.ones(latents.shape[0], 1, device=latents.device)], dim=1)
    d = z.shape[1]
    gram = z.T @ z + alpha * torch.eye(d, device=z.device, dtype=z.dtype)
    solution = torch.linalg.solve(gram, z.T @ targets)
    return solution[:-1], solution[-1]


def r_squared(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-output-dim R^2 = 1 - SS_res / SS_tot."""
    ss_res = ((targets - predictions) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2: torch.Tensor = 1.0 - ss_res / ss_tot
    return r2


@torch.no_grad()
def encode_in_batches(
    encoder: Encoder, frames: torch.Tensor, batch_size: int = 512
) -> torch.Tensor:
    """Encode frames through the (possibly GPU) encoder, results on CPU."""
    device = next(encoder.parameters()).device
    chunks = [
        encoder(frames[i : i + batch_size].to(device)).cpu()
        for i in range(0, frames.shape[0], batch_size)
    ]
    return torch.cat(chunks)


@torch.no_grad()
def evaluate_probe(
    encoder: Encoder,
    frames_train: torch.Tensor,
    states_train: torch.Tensor,
    frames_eval: torch.Tensor,
    states_eval: torch.Tensor,
    batch_size: int = 512,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Fit on train frames/states, report R^2/MSE on held-out eval.

    Returns (metrics, probe_weight (D, 2), probe_bias (2,)) — the fitted probe
    is reused for rollout position error and the Phase 3 latent->state viz.
    """
    z_train = encode_in_batches(encoder, frames_train, batch_size)
    z_eval = encode_in_batches(encoder, frames_eval, batch_size)
    weight, bias = fit_linear_probe(z_train, states_train)
    predictions = z_eval @ weight + bias
    r2 = r_squared(predictions, states_eval)
    metrics = {
        "r2_x": float(r2[0].item()),
        "r2_y": float(r2[1].item()),
        "r2_mean": float(r2.mean().item()),
        "mse": float(((predictions - states_eval) ** 2).mean().item()),
    }
    return metrics, weight, bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    args = parser.parse_args()

    from latentlab.train import load_checkpoint_models, load_probe_tensors

    encoder, _, _config = load_checkpoint_models(args.checkpoint)
    probe_data = load_probe_tensors(args.data)
    metrics, _w, _b = evaluate_probe(
        encoder,
        probe_data.frames_train,
        probe_data.states_train,
        probe_data.frames_eval,
        probe_data.states_eval,
    )
    print(
        f"probe R^2: x={metrics['r2_x']:.4f} y={metrics['r2_y']:.4f} "
        f"mean={metrics['r2_mean']:.4f} mse={metrics['mse']:.6f}"
    )
