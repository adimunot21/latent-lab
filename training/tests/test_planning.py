"""Unit tests for the CEM planner (toy world models, no checkpoints needed)."""

from __future__ import annotations

import torch

from latentlab.models.predictor import Predictor
from latentlab.planning.cem import CEMConfig, CEMPlanner


class IdentityWorld(Predictor):
    """Toy 2-D world model: latent IS the position, z' = z + a."""

    def __init__(self) -> None:
        super().__init__(latent_dim=2)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + action


def make_planner(horizon: int = 8, seed: int = 0) -> CEMPlanner:
    generator = torch.Generator().manual_seed(seed)
    return CEMPlanner(IdentityWorld(), CEMConfig(horizon=horizon), generator=generator)


def test_cem_reaches_reachable_goal() -> None:
    planner = make_planner()
    z0 = torch.zeros(2)
    z_goal = torch.tensor([0.3, -0.25])
    actions, _ = planner.plan(z0, z_goal)
    endpoint = z0 + actions.sum(dim=0)
    # Dense cost slightly biases the endpoint (it also charges the approach
    # path), so the tolerance is looser than pure endpoint optimization.
    assert (endpoint - z_goal).norm() < 0.1


def test_cem_respects_action_bounds() -> None:
    planner = make_planner()
    actions, _ = planner.plan(torch.zeros(2), torch.tensor([5.0, 5.0]))
    bound = planner.config.action_bound
    assert actions.abs().max() <= bound + 1e-6


def test_cem_first_action_points_toward_goal() -> None:
    # Dense cost property: with a far goal, the first action should make
    # immediate progress, not wander (the endpoint-only failure mode).
    planner = make_planner()
    z_goal = torch.tensor([1.0, 0.0])
    actions, _ = planner.plan(torch.zeros(2), z_goal)
    assert actions[0, 0] > 0.5 * planner.config.action_bound


def test_cem_warm_start_persists_and_resets() -> None:
    planner = make_planner()
    planner.plan(torch.zeros(2), torch.ones(2))
    assert planner._mean is not None
    planner.reset()
    assert planner._mean is None


def test_cem_deterministic_with_generator() -> None:
    a1, _ = make_planner(seed=7).plan(torch.zeros(2), torch.tensor([0.2, 0.1]))
    a2, _ = make_planner(seed=7).plan(torch.zeros(2), torch.tensor([0.2, 0.1]))
    assert torch.equal(a1, a2)
