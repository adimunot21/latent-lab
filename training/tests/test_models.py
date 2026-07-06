"""Unit tests for encoder, predictor, JEPA losses, and collapse metrics."""

from __future__ import annotations

import torch

from latentlab.models.encoder import Encoder
from latentlab.models.losses import jepa_loss, sigreg_loss
from latentlab.models.predictor import Predictor
from latentlab.probes.collapse_metrics import effective_rank, mean_latent_std
from latentlab.probes.linear_probe import fit_linear_probe, r_squared


def test_encoder_output_shape() -> None:
    encoder = Encoder(latent_dim=64)
    z = encoder(torch.randn(3, 1, 64, 64))
    assert z.shape == (3, 64)
    assert z.dtype == torch.float32


def test_predictor_residual_structure() -> None:
    predictor = Predictor(latent_dim=64)
    z = torch.randn(5, 64)
    action = torch.zeros(5, 2)
    z_next = predictor(z, action)
    assert z_next.shape == z.shape
    # Residual form: output stays near input at init (small random delta).
    assert (z_next - z).abs().mean() < 1.0


def test_sigreg_discriminates_collapse() -> None:
    torch.manual_seed(0)
    gaussian = torch.randn(512, 64)
    collapsed = torch.full((512, 64), 0.5)
    generator = torch.Generator().manual_seed(1)
    score_gaussian = sigreg_loss(gaussian, generator=generator).item()
    score_collapsed = sigreg_loss(collapsed, generator=generator).item()
    assert score_collapsed > 50 * score_gaussian


def test_sigreg_differentiable() -> None:
    z = torch.randn(64, 32, requires_grad=True)
    sigreg_loss(z).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_jepa_loss_lambda_zero_skips_reg() -> None:
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    losses = jepa_loss(z + 0.1, z, z, lambda_reg=0.0)
    assert losses["sigreg"].item() == 0.0
    assert losses["total"].item() == losses["pred_mse"].item()


def test_collapse_metrics_extremes() -> None:
    torch.manual_seed(0)
    healthy = torch.randn(256, 32)
    collapsed = torch.full((256, 32), 0.7)
    assert mean_latent_std(healthy) > 0.9
    assert mean_latent_std(collapsed) < 1e-6
    assert effective_rank(healthy) > 25
    # Constant batch: exactly 1 up to fp noise in the SVD of a ~zero matrix.
    assert effective_rank(collapsed) < 1.01


def test_linear_probe_recovers_linear_map() -> None:
    torch.manual_seed(0)
    true_weight = torch.randn(16, 2)
    z = torch.randn(500, 16)
    y = z @ true_weight + 0.5
    weight, bias = fit_linear_probe(z, y)
    r2 = r_squared(z @ weight + bias, y)
    assert (r2 > 0.999).all()
