/**
 * CEM planner in latent space — TypeScript port of
 * training/src/latentlab/planning/cem.py (the reference implementation).
 *
 * Key property carried over: the cost is DENSE — the sum of latent distance
 * to goal over ALL imagined steps plus an extra-weighted terminal step.
 * Endpoint-only cost leaves the first action underdetermined whenever the
 * goal is reachable in fewer than `horizon` steps, which turns MPC into a
 * random walk (this failed at 20% success in Python before the fix; see the
 * Python module docstring).
 *
 * The predictor is abstracted as an async batched function so the planner is
 * unit-testable without ONNX and runs identically in the worker.
 */

export interface CEMConfig {
  horizon: number
  population: number
  iterations: number
  eliteFrac: number
  initSigma: number
  minSigma: number
  actionBound: number
  terminalWeight: number
}

export const DEFAULT_CEM_CONFIG: CEMConfig = {
  horizon: 12,
  population: 256,
  iterations: 4,
  eliteFrac: 0.125,
  initSigma: 0.05,
  minSigma: 0.01,
  actionBound: 0.08,
  terminalWeight: 4.0,
}

/** Batched one-step world model: (latents B*D, actions B*2, B) -> latents B*D. */
export type PredictFn = (
  latents: Float32Array,
  actions: Float32Array,
  batch: number,
) => Promise<Float32Array>

export interface PlanResult {
  /** Best action sequence (horizon * 2), execute [0], [1] first. */
  actions: Float32Array
  bestCost: number
  /** Imagined latent trajectory of the best plan (horizon * latentDim). */
  bestTrajectory: Float32Array
  /** Elite imagined trajectories for viz (nElites * horizon * latentDim). */
  eliteTrajectories: Float32Array
  nElitesReturned: number
}

/** mulberry32 PRNG — deterministic across platforms, good enough for CEM. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Box-Muller standard normal from a uniform source. */
function gaussianPair(rand: () => number): [number, number] {
  let u = 0
  while (u === 0) u = rand()
  const v = rand()
  const r = Math.sqrt(-2 * Math.log(u))
  const theta = 2 * Math.PI * v
  return [r * Math.cos(theta), r * Math.sin(theta)]
}

export class CEMPlanner {
  private mean: Float32Array | null = null // (H*2) warm start
  private rand: () => number

  constructor(
    private predict: PredictFn,
    private latentDim: number,
    public config: CEMConfig = { ...DEFAULT_CEM_CONFIG },
    seed = 0,
  ) {
    this.rand = mulberry32(seed)
  }

  reset(): void {
    this.mean = null
  }

  /** One MPC planning call. z0/zGoal are (latentDim,) latents. */
  async plan(z0: Float32Array, zGoal: Float32Array, vizElites = 8): Promise<PlanResult> {
    const { horizon: H, population: P, iterations, eliteFrac } = this.config
    const { initSigma, minSigma, actionBound, terminalWeight } = this.config
    const D = this.latentDim
    const nElite = Math.max(1, Math.floor(P * eliteFrac))

    // Warm start: shift the previous plan one step, repeat the last row.
    let mean = new Float32Array(H * 2)
    if (this.mean) {
      mean.set(this.mean.subarray(2), 0)
      mean[H * 2 - 2] = this.mean[H * 2 - 2]
      mean[H * 2 - 1] = this.mean[H * 2 - 1]
    }
    const sigma = new Float32Array(H * 2).fill(initSigma)

    const candidates = new Float32Array(P * H * 2)
    const costs = new Float64Array(P)
    // Trajectory latents for the current iteration: (P, H, D) — reused.
    const traj = new Float32Array(P * H * D)
    const stepLatents = new Float32Array(P * D)
    const stepActions = new Float32Array(P * 2)

    let bestCost = Infinity
    const bestActions = new Float32Array(H * 2)
    const bestTrajectory = new Float32Array(H * D)
    let eliteIdx: number[] = []

    for (let iter = 0; iter < iterations; iter++) {
      // Sample candidates ~ N(mean, sigma), clamped. Candidate 0 = mean.
      for (let p = 0; p < P; p++) {
        for (let k = 0; k < H * 2; k += 2) {
          if (p === 0) {
            candidates[p * H * 2 + k] = clamp(mean[k], actionBound)
            candidates[p * H * 2 + k + 1] = clamp(mean[k + 1], actionBound)
          } else {
            const [g0, g1] = gaussianPair(this.rand)
            candidates[p * H * 2 + k] = clamp(mean[k] + g0 * sigma[k], actionBound)
            candidates[p * H * 2 + k + 1] = clamp(mean[k + 1] + g1 * sigma[k + 1], actionBound)
          }
        }
      }

      // Batched rollout with dense cost accumulation.
      costs.fill(0)
      for (let p = 0; p < P; p++) stepLatents.set(z0, p * D)
      for (let t = 0; t < H; t++) {
        for (let p = 0; p < P; p++) {
          stepActions[p * 2] = candidates[p * H * 2 + t * 2]
          stepActions[p * 2 + 1] = candidates[p * H * 2 + t * 2 + 1]
        }
        const next = await this.predict(stepLatents, stepActions, P)
        stepLatents.set(next)
        const weight = t === H - 1 ? terminalWeight : 1.0
        for (let p = 0; p < P; p++) {
          let dist = 0
          for (let d = 0; d < D; d++) {
            const diff = next[p * D + d] - zGoal[d]
            dist += diff * diff
          }
          costs[p] += weight * dist
          traj.set(next.subarray(p * D, (p + 1) * D), p * H * D + t * D)
        }
      }

      // Elites: indices of the nElite lowest costs.
      eliteIdx = argsortAscending(costs).slice(0, nElite)

      // Refit mean/sigma from elites.
      mean = new Float32Array(H * 2)
      for (const p of eliteIdx) {
        for (let k = 0; k < H * 2; k++) mean[k] += candidates[p * H * 2 + k]
      }
      for (let k = 0; k < H * 2; k++) mean[k] /= nElite
      for (let k = 0; k < H * 2; k++) {
        let variance = 0
        for (const p of eliteIdx) {
          const diff = candidates[p * H * 2 + k] - mean[k]
          variance += diff * diff
        }
        // Match torch.std: unbiased (n-1) sample std.
        sigma[k] = Math.max(Math.sqrt(variance / Math.max(nElite - 1, 1)), minSigma)
      }

      const iterBest = eliteIdx[0]
      if (costs[iterBest] < bestCost) {
        bestCost = costs[iterBest]
        bestActions.set(candidates.subarray(iterBest * H * 2, (iterBest + 1) * H * 2))
        bestTrajectory.set(traj.subarray(iterBest * H * D, (iterBest + 1) * H * D))
      }
    }

    this.mean = bestActions.slice()

    // Elite trajectories from the FINAL iteration for candidate viz.
    const nReturn = Math.min(vizElites, eliteIdx.length)
    const eliteTrajectories = new Float32Array(nReturn * H * D)
    for (let i = 0; i < nReturn; i++) {
      const p = eliteIdx[i]
      eliteTrajectories.set(traj.subarray(p * H * D, (p + 1) * H * D), i * H * D)
    }

    return {
      actions: bestActions,
      bestCost,
      bestTrajectory,
      eliteTrajectories,
      nElitesReturned: nReturn,
    }
  }
}

function clamp(v: number, bound: number): number {
  return Math.min(Math.max(v, -bound), bound)
}

function argsortAscending(values: Float64Array): number[] {
  return [...values.keys()].sort((a, b) => values[a] - values[b])
}
