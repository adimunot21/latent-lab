/**
 * CEM planner unit tests with a stub world model (latent = 2-D position,
 * z' = z + a) — mirrors training/tests/test_planning.py.
 */
import { describe, expect, it } from 'vitest'
import { CEMPlanner, DEFAULT_CEM_CONFIG, type PredictFn } from '../src/planner/cem'

const identityWorld: PredictFn = (latents, actions, batch) => {
  const next = new Float32Array(batch * 2)
  for (let p = 0; p < batch; p++) {
    next[p * 2] = latents[p * 2] + actions[p * 2]
    next[p * 2 + 1] = latents[p * 2 + 1] + actions[p * 2 + 1]
  }
  return Promise.resolve(next)
}

function makePlanner(seed = 0): CEMPlanner {
  return new CEMPlanner(identityWorld, 2, { ...DEFAULT_CEM_CONFIG, horizon: 8 }, seed)
}

describe('CEM planner (identity world model)', () => {
  it('reaches a reachable goal', async () => {
    const planner = makePlanner()
    const result = await planner.plan(new Float32Array([0, 0]), new Float32Array([0.3, -0.25]))
    let x = 0
    let y = 0
    for (let t = 0; t < 8; t++) {
      x += result.actions[t * 2]
      y += result.actions[t * 2 + 1]
    }
    // Dense cost also charges the approach path — loose-ish tolerance.
    expect(Math.hypot(x - 0.3, y + 0.25)).toBeLessThan(0.1)
  })

  it('respects action bounds', async () => {
    const planner = makePlanner()
    const result = await planner.plan(new Float32Array([0, 0]), new Float32Array([5, 5]))
    for (const a of result.actions) {
      expect(Math.abs(a)).toBeLessThanOrEqual(DEFAULT_CEM_CONFIG.actionBound + 1e-6)
    }
  })

  it('first action points toward a far goal (dense-cost property)', async () => {
    const planner = makePlanner()
    const result = await planner.plan(new Float32Array([0, 0]), new Float32Array([1, 0]))
    expect(result.actions[0]).toBeGreaterThan(0.5 * DEFAULT_CEM_CONFIG.actionBound)
  })

  it('is deterministic given a seed', async () => {
    const a = await makePlanner(7).plan(new Float32Array([0, 0]), new Float32Array([0.2, 0.1]))
    const b = await makePlanner(7).plan(new Float32Array([0, 0]), new Float32Array([0.2, 0.1]))
    expect([...a.actions]).toEqual([...b.actions])
  })

  it('returns viz trajectories with the right shapes', async () => {
    const planner = makePlanner()
    const result = await planner.plan(new Float32Array([0, 0]), new Float32Array([0.2, 0]), 4)
    expect(result.bestTrajectory.length).toBe(8 * 2)
    expect(result.nElitesReturned).toBe(4)
    expect(result.eliteTrajectories.length).toBe(4 * 8 * 2)
  })
})
