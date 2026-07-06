/**
 * Main-thread client for the planner worker: promise-based plan() calls,
 * one in flight at a time (MPC is sequential by nature).
 */

import type { CEMConfig } from './cem'
import type { PlanDone, RolloutDone, WorkerRequest, WorkerResponse } from './messages'

export class PlannerClient {
  private worker: Worker
  private nextId = 1
  private pending = new Map<number | 'init', (value: unknown) => void>()
  private rejectAll: ((error: Error) => void) | null = null

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      if (msg.type === 'init-done') {
        this.pending.get('init')?.(undefined)
        this.pending.delete('init')
      } else if (msg.type === 'plan-done' || msg.type === 'rollout-done') {
        this.pending.get(msg.id)?.(msg)
        this.pending.delete(msg.id)
      } else if (msg.type === 'error') {
        const error = new Error(`planner worker: ${msg.message}`)
        this.pending.clear()
        this.rejectAll?.(error)
      }
    }
  }

  init(
    predictorModel: ArrayBuffer,
    latentDim: number,
    backend: 'webgpu' | 'wasm',
    lookupStates: ArrayBuffer,
    lookupLatents: ArrayBuffer,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      this.rejectAll = reject
      this.pending.set('init', resolve as (value: unknown) => void)
      const message: WorkerRequest = {
        type: 'init',
        predictorModel,
        latentDim,
        actionDim: 2,
        backend,
        lookupStates,
        lookupLatents,
      }
      this.worker.postMessage(message, [predictorModel, lookupStates, lookupLatents])
    })
  }

  plan(z0: Float32Array, zGoal: Float32Array, config: CEMConfig): Promise<PlanDone> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.rejectAll = reject
      this.pending.set(id, resolve as (value: unknown) => void)
      const message: WorkerRequest = { type: 'plan', id, z0, zGoal, config }
      this.worker.postMessage(message)
    })
  }

  rollout(z0: Float32Array, actions: Float32Array): Promise<RolloutDone> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.rejectAll = reject
      this.pending.set(id, resolve as (value: unknown) => void)
      const message: WorkerRequest = { type: 'rollout', id, z0, actions }
      this.worker.postMessage(message)
    })
  }

  reset(): void {
    this.worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
  }

  terminate(): void {
    this.worker.terminate()
  }
}
