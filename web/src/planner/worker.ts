/**
 * Planner Web Worker: owns its own onnxruntime predictor session and the
 * latent<->state lookup table, runs CEM off the main thread, and decodes
 * imagined latent trajectories to (x, y) paths HERE so only small state
 * arrays cross the thread boundary (not megabytes of latents).
 */

import * as ort from 'onnxruntime-web'
import { CEMPlanner, type PredictFn } from './cem'
import type { WorkerRequest, WorkerResponse } from './messages'

let session: ort.InferenceSession | null = null
let planner: CEMPlanner | null = null
let latentDim = 0
let lookupStates: Float32Array | null = null
let lookupLatents: Float32Array | null = null

const predict: PredictFn = async (latents, actions, batch) => {
  if (!session) throw new Error('worker not initialized')
  const output = await session.run({
    latent: new ort.Tensor('float32', latents, [batch, latentDim]),
    action: new ort.Tensor('float32', actions, [batch, 2]),
  })
  return output.next_latent.data as Float32Array
}

/** Nearest-neighbor decode: latent (D,) -> lookup state (x, y). */
function decodeLatent(latent: Float32Array, offset: number): [number, number] {
  const states = lookupStates!
  const table = lookupLatents!
  const n = states.length / 2
  let bestDist = Infinity
  let bestIdx = 0
  for (let i = 0; i < n; i++) {
    let dist = 0
    const base = i * latentDim
    for (let d = 0; d < latentDim; d++) {
      const diff = table[base + d] - latent[offset + d]
      dist += diff * diff
      if (dist >= bestDist) break // early exit: already worse than best
    }
    if (dist < bestDist) {
      bestDist = dist
      bestIdx = i
    }
  }
  return [states[bestIdx * 2], states[bestIdx * 2 + 1]]
}

function decodeTrajectories(latents: Float32Array, count: number, horizon: number): Float32Array {
  const paths = new Float32Array(count * horizon * 2)
  for (let i = 0; i < count * horizon; i++) {
    const [x, y] = decodeLatent(latents, i * latentDim)
    paths[i * 2] = x
    paths[i * 2 + 1] = y
  }
  return paths
}

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  try {
    if (msg.type === 'init') {
      latentDim = msg.latentDim
      lookupStates = new Float32Array(msg.lookupStates)
      lookupLatents = new Float32Array(msg.lookupLatents)
      const providers = msg.backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
      session = await ort.InferenceSession.create(new Uint8Array(msg.predictorModel), {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      })
      // Warmup so the first real plan isn't paying kernel compilation.
      await predict(new Float32Array(latentDim), new Float32Array(2), 1)
      planner = new CEMPlanner(predict, latentDim)
      post({ type: 'init-done' })
    } else if (msg.type === 'reset') {
      planner?.reset()
    } else if (msg.type === 'rollout') {
      if (!session) throw new Error('rollout before init')
      const steps = msg.actions.length / 2
      const latents = new Float32Array(steps * latentDim)
      let z = msg.z0
      for (let t = 0; t < steps; t++) {
        z = await predict(z, msg.actions.subarray(t * 2, t * 2 + 2), 1)
        latents.set(z, t * latentDim)
      }
      const path = decodeTrajectories(latents, 1, steps)
      post({ type: 'rollout-done', id: msg.id, latents, path }, [latents.buffer, path.buffer])
    } else if (msg.type === 'plan') {
      if (!planner) throw new Error('plan before init')
      planner.config = msg.config
      const start = performance.now()
      const result = await planner.plan(msg.z0, msg.zGoal)
      const bestPath = decodeTrajectories(result.bestTrajectory, 1, msg.config.horizon)
      const elitePaths = decodeTrajectories(
        result.eliteTrajectories,
        result.nElitesReturned,
        msg.config.horizon,
      )
      const latencyMs = performance.now() - start
      post(
        {
          type: 'plan-done',
          id: msg.id,
          actions: result.actions,
          bestCost: result.bestCost,
          latencyMs,
          bestPath,
          elitePaths,
          nElites: result.nElitesReturned,
        },
        [result.actions.buffer, bestPath.buffer, elitePaths.buffer],
      )
    }
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
