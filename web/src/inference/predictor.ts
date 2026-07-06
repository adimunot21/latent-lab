/**
 * Typed wrapper for predictor.onnx: (latents, actions) -> next latents.
 * Inputs "latent" (B, D), "action" (B, 2); output "next_latent" (B, D).
 * The CEM planner calls this with B in the hundreds — the wrapper reuses
 * caller-provided buffers and never copies latents.
 */

import type { Manifest } from './manifest'
import { createSession, warmup, ort, type Backend } from './session'

export class PredictorSession {
  private constructor(
    private session: ort.InferenceSession,
    readonly latentDim: number,
    readonly actionDim: number,
  ) {}

  static async create(
    model: ArrayBuffer,
    manifest: Manifest,
    backend: Backend,
  ): Promise<PredictorSession> {
    const session = await createSession(model, backend)
    const predictor = new PredictorSession(session, manifest.latent_dim, manifest.action_dim)
    await warmup(session, {
      latent: new ort.Tensor('float32', new Float32Array(manifest.latent_dim), [
        1,
        manifest.latent_dim,
      ]),
      action: new ort.Tensor('float32', new Float32Array(manifest.action_dim), [
        1,
        manifest.action_dim,
      ]),
    })
    return predictor
  }

  /** One latent step for a batch: latents (B*D), actions (B*2). */
  async predict(
    latents: Float32Array,
    actions: Float32Array,
    batch: number,
  ): Promise<Float32Array> {
    const d = this.latentDim
    if (latents.length !== batch * d || actions.length !== batch * this.actionDim) {
      throw new Error(
        `bad buffer sizes: latents ${latents.length} (want ${batch * d}), ` +
          `actions ${actions.length} (want ${batch * this.actionDim})`,
      )
    }
    const output = await this.session.run({
      latent: new ort.Tensor('float32', latents, [batch, d]),
      action: new ort.Tensor('float32', actions, [batch, this.actionDim]),
    })
    return output.next_latent.data as Float32Array
  }
}
