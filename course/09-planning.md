# 09 — Planning: CEM, MPC, and the dense-cost lesson

This is the payoff lesson: the world model stops being a curiosity and
starts *doing work*. It also contains the project's best debugging story —
a planner that went from 20% to 97% success by changing one design decision,
diagnosed from instrumentation, not guesswork.

## The task, stated in latent space

Given: current frame, goal frame. Encode both: z₀, z_goal. Find an action
sequence a₀..a_{H−1} such that rolling the *predictor* forward from z₀
lands near z_goal. Note everything happens inside the model's imagination —
the environment is only touched to execute the chosen first action.

Two prerequisites make this sane, both established earlier:
- latent distance tracks position distance (verified before any planner
  debugging — see below),
- the predictor knows about walls, because the dataset contained collisions
  (lesson 06). Nobody tells the planner walls exist; candidate sequences that
  try to tunnel simply *don't make progress in imagination* and score badly.
  The doorway route is discovered, not programmed.

## CEM: optimization by repeated elitism

Cross-Entropy Method, `planning/cem.py`. Maintain a Gaussian over action
sequences (mean μ and per-step σ, each (H, 2)); iterate: sample a population,
score it, keep the elite fraction, refit μ/σ to the elites, repeat. Four
iterations of population 256 suffice here.

```python
noise = torch.randn(cfg.population, cfg.horizon, 2, ...)
candidates = (mean + noise * sigma).clamp(-cfg.action_bound, cfg.action_bound)
candidates[0] = mean.clamp(...)          # keep the mean in the population

costs = self.rollout_costs(z0, candidates, z_goal)
elite_idx = costs.topk(n_elite, largest=False).indices
mean = elites.mean(dim=0)
sigma = elites.std(dim=0).clamp_min(cfg.min_sigma)
```

Details that earn their lines: candidate 0 is always the current mean
(refinement can never lose ground); σ has a floor (`min_sigma=0.01`) so the
search never fully stops exploring; rollouts are **batched** — all 256
candidates advance through the predictor together, one (256, D) tensor per
horizon step, which is also exactly the shape the ONNX dynamic-batch axis
exists for (lesson 11).

**MPC** (model-predictive control) wraps it: plan, execute *only the first
action*, observe, re-plan. Re-planning every step forgives imagination drift.
The warm start makes it cheap: next call's initial mean is the previous best
plan shifted one step (`torch.cat([self._mean[1:], self._mean[-1:]])`).

## The failure: endpoint-only cost

First implementation scored candidates by final state only:
cost = ‖z_H − z_goal‖². Principled-looking. Result on the eval harness:
**20% success** — worse inside rooms than crossing them, agents jittering in
place. The instrumented MPC trace (preserved in the lesson because the
*method* is the lesson) showed, per step:

```
step 0: pos (0.604,0.410) true_d 0.531  best_cost 4.44  action (+0.004,+0.060)
step 2: pos (0.607,0.421) true_d 0.516  best_cost 3.99  action (-0.029,+0.057)
step 8: pos (0.558,0.425) true_d 0.456  best_cost 3.56  action (+0.029,-0.044)
```

Read it like the debugger did: CEM reports best_cost ≈ 4 — in latent units
that's *practically at the goal* (true distance 0.53 ≈ latent 26) — so the
optimizer believes its plans totally succeed. Yet executed first actions
point every which way, and the agent random-walks. Optimizer confident,
behavior incoherent: the objective itself must be wrong.

It was. **With endpoint-only cost, every sequence that reaches the goal by
step H ties at cost ≈ 0** — including sequences that wander for five steps
and then sprint. The goal was reachable in ~7 of the 12 horizon steps, so an
enormous tie-set existed, and the *first action* — the only one that ever
gets executed — was undetermined within it. Each MPC step re-broke the tie
randomly. The planner wasn't failing to optimize; it was optimizing a
function that didn't constrain the one decision that mattered.

One more diagnostic deserves note: *before* touching the cost, latent
geometry was checked directly (encode a line of positions, print latent
distance vs position distance: 36.4 → 1.2, cleanly monotone). That ruled out
the representation and pinned the search/objective. Diagnose geometry before
algorithms.

## The fix: dense cost

```python
def rollout_costs(self, z0, actions, z_goal):
    z = z0.expand(...)
    costs = torch.zeros(...)
    for t in range(horizon):
        z = self.predictor(z, actions[:, t])
        step_cost = ((z - z_goal) ** 2).sum(dim=1)
        weight = self.config.terminal_weight if t == horizon - 1 else 1.0
        costs += weight * step_cost
    return costs
```

Charge the distance to goal at **every** imagined step (terminal step ×4:
"arrive *and stay*"). Wander-then-arrive now costs strictly more than
straight-there — immediate progress is optimal, the first action is pinned,
ties dissolve. A secondary benefit: trajectories that exploit predictor error
("teleport at step 11") get charged for all the steps they *weren't* at the
goal, shrinking the model-exploitation surface.

Same scenario after the fix: goal reached in 7 steps, near-optimal. Full
eval (`planning/evaluate.py`: 100 episodes, half forced cross-room, success
= within 0.08 in ≤60 steps): **97%** — 94% cross-room, 100% same-room, mean
6.7 steps. Collapsed checkpoint, same harness: 44%, mean steps doubled —
and lesson 08 told you why 44% ≠ 0% (lucky wandering under a generous
horizon).

## Try it

1. Reproduce the failure: set `terminal_weight` very high (say 10⁶ — making
   intermediate cost negligible ≈ endpoint-only) and run a 10-episode eval.
   Then restore and re-run. You should see roughly the 20%↔97% swing.
2. Horizon sensitivity: run evals at horizon 6 and 16 (defaults elsewhere).
   Explain both degradation directions — what does a too-short horizon do to
   *cross-room* episodes specifically?
3. Run `uv run python -m latentlab.planning.cem` (toy identity world) and
   read its printed caveat about why best_cost isn't ~0 under dense cost even
   for a perfect plan. Make sure you can derive the same fact from the
   `rollout_costs` code.
