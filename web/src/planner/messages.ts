/** Typed messages between the main thread and the planner worker. */

import type { CEMConfig } from './cem'

export interface InitRequest {
  type: 'init'
  predictorModel: ArrayBuffer
  latentDim: number
  actionDim: number
  backend: 'webgpu' | 'wasm'
  /** Lookup table for decoder-free path decoding (worker-side NN). */
  lookupStates: ArrayBuffer // float32 (n, 2)
  lookupLatents: ArrayBuffer // float32 (n, latentDim)
}

export interface PlanRequest {
  type: 'plan'
  id: number
  z0: Float32Array
  zGoal: Float32Array
  config: CEMConfig
}

export interface ResetRequest {
  type: 'reset'
}

/** Open-loop rollout for the imagination panel: replay actions from z0. */
export interface RolloutRequest {
  type: 'rollout'
  id: number
  z0: Float32Array
  /** (steps * 2) recorded actions. */
  actions: Float32Array
}

export type WorkerRequest = InitRequest | PlanRequest | ResetRequest | RolloutRequest

export interface InitDone {
  type: 'init-done'
}

export interface PlanDone {
  type: 'plan-done'
  id: number
  /** Best action sequence (horizon * 2). */
  actions: Float32Array
  bestCost: number
  latencyMs: number
  /** Best plan decoded to states via lookup: (horizon * 2) of (x, y). */
  bestPath: Float32Array
  /** Elite plans decoded to states: (nElites * horizon * 2). */
  elitePaths: Float32Array
  nElites: number
}

export interface RolloutDone {
  type: 'rollout-done'
  id: number
  /** Imagined latents after each action: (steps * latentDim). */
  latents: Float32Array
  /** Imagined path decoded via lookup: (steps * 2). */
  path: Float32Array
}

export interface WorkerError {
  type: 'error'
  message: string
}

export type WorkerResponse = InitDone | PlanDone | RolloutDone | WorkerError
