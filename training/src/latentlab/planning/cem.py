"""Cross-Entropy Method planner over action sequences in latent space.

Plan: sample a population of action sequences from a per-step Gaussian,
roll each through the PREDICTOR ONLY (no env, no decoder — pure latent
imagination), score by distance between the imagined final latent and the
encoded goal latent, refit the Gaussian on the elite set, repeat.

Wall avoidance falls out of the world model: the predictor learned that
actions pushing into the wall don't move the embedding, so sequences that
try to tunnel score badly and CEM discovers the doorway route — nobody
hard-coded the wall into the planner.

MPC usage: call ``plan`` each env step, execute the first action, and the
planner warm-starts the next call by shifting its mean one step forward.
This class is the reference implementation for the TypeScript port
(Phase 6); keep the algorithm boring and the batching predictor-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from latentlab.models.predictor import Predictor


@dataclass(frozen=True)
class CEMConfig:
    """Planner hyperparameters (defaults tuned for Two Rooms)."""

    horizon: int = 12  # imagined steps; ~1 room-width at max_step
    population: int = 256  # candidates per iteration
    iterations: int = 4  # CEM refinement rounds
    elite_frac: float = 0.125  # top 32 of 256
    init_sigma: float = 0.05  # exploration noise ~ 0.6 * max_step
    min_sigma: float = 0.01  # sigma floor: never stop exploring entirely
    action_bound: float = 0.08  # env max_step; actions clamped to +/- this
    # Cost = sum of latent distance over ALL imagined steps, not just the
    # endpoint. Endpoint-only cost makes the first action underdetermined
    # whenever the goal is reachable in < horizon steps (any wander-then-
    # arrive sequence ties), which turns MPC into a random walk. Dense cost
    # rewards immediate progress and pins the first action down.
    terminal_weight: float = 4.0  # extra weight on the final step: arrive AND stay


class CEMPlanner:
    """Stateful (warm-started) CEM planner in latent space."""

    def __init__(
        self,
        predictor: Predictor,
        config: CEMConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config if config is not None else CEMConfig()
        self.device = next(predictor.parameters()).device
        self.generator = generator
        self._mean: torch.Tensor | None = None  # (H, 2) warm-start between MPC steps

    def reset(self) -> None:
        """Clear the warm-start (call at the start of each episode)."""
        self._mean = None

    @torch.no_grad()
    def rollout_costs(
        self, z0: torch.Tensor, actions: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        """Dense trajectory cost of (P, H, 2) action sequences from latent z0 (D,).

        cost_p = sum_t ||z_t - z_goal||^2 + (terminal_weight - 1) * ||z_H - z_goal||^2
        """
        z = z0.unsqueeze(0).expand(actions.shape[0], -1).contiguous()
        horizon = actions.shape[1]
        costs = torch.zeros(actions.shape[0], device=z0.device)
        for t in range(horizon):
            z = self.predictor(z, actions[:, t])
            step_cost = ((z - z_goal) ** 2).sum(dim=1)
            weight = self.config.terminal_weight if t == horizon - 1 else 1.0
            costs += weight * step_cost
        return costs

    @torch.no_grad()
    def plan(self, z0: torch.Tensor, z_goal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """One MPC planning call.

        Args:
            z0: (D,) current latent. z_goal: (D,) goal latent.

        Returns:
            actions: (H, 2) best action sequence found (execute actions[0]).
            info: best cost and final sigma (for latency/telemetry panels).
        """
        cfg = self.config
        n_elite = max(1, int(cfg.population * cfg.elite_frac))

        if self._mean is None:
            mean = torch.zeros(cfg.horizon, 2, device=self.device)
        else:
            # Warm start: shift last plan one step forward, repeat final row.
            mean = torch.cat([self._mean[1:], self._mean[-1:]], dim=0)
        sigma = torch.full((cfg.horizon, 2), cfg.init_sigma, device=self.device)

        best_actions = mean.clone()
        best_cost = float("inf")
        for _ in range(cfg.iterations):
            noise = torch.randn(
                cfg.population, cfg.horizon, 2, device=self.device, generator=self.generator
            )
            candidates = (mean + noise * sigma).clamp(-cfg.action_bound, cfg.action_bound)
            # Keep the current mean in the population so refinement is monotone.
            candidates[0] = mean.clamp(-cfg.action_bound, cfg.action_bound)

            costs = self.rollout_costs(z0, candidates, z_goal)
            elite_idx = costs.topk(n_elite, largest=False).indices
            elites = candidates[elite_idx]

            mean = elites.mean(dim=0)
            sigma = elites.std(dim=0).clamp_min(cfg.min_sigma)

            iter_best = int(elite_idx[0].item())
            iter_best_cost = float(costs[iter_best].item())
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_actions = candidates[iter_best].clone()

        self._mean = best_actions
        return best_actions, {"best_cost": best_cost, "mean_sigma": float(sigma.mean().item())}


if __name__ == "__main__":
    # Standalone sanity: toy 2-D "latent" where z' = z + a (identity world
    # model). CEM must steer toward the goal.
    torch.manual_seed(0)

    class IdentityPredictor(Predictor):
        def __init__(self) -> None:
            super().__init__(latent_dim=2)

        def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return z + action

    planner = CEMPlanner(IdentityPredictor(), CEMConfig(horizon=8))
    z0 = torch.tensor([0.0, 0.0])
    z_goal = torch.tensor([0.4, -0.2])
    actions, info = planner.plan(z0, z_goal)
    endpoint = z0 + actions.sum(dim=0)
    print(f"goal {z_goal.tolist()} -> planned endpoint {[round(v, 3) for v in endpoint.tolist()]}")
    error = (endpoint - z_goal).norm().item()
    print(f"endpoint error {error:.4f} (dense cost also charges the approach path,")
    print("so best_cost is not ~0 even for a perfect plan; endpoint error should be small)")
    assert error < 0.05, "CEM failed to steer the identity world model to the goal"
