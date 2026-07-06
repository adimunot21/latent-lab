/**
 * Typed wrapper for encoder.onnx: normalized frames -> latents.
 * Input "frame" (B, 1, N, N) float32; output "latent" (B, D) float32.
 */

import type { Manifest } from './manifest'
import { createSession, warmup, ort, type Backend } from './session'

export class EncoderSession {
  private constructor(
    private session: ort.InferenceSession,
    readonly frameSize: number,
    readonly latentDim: number,
  ) {}

  static async create(
    model: ArrayBuffer,
    manifest: Manifest,
    backend: Backend,
  ): Promise<EncoderSession> {
    const session = await createSession(model, backend)
    const encoder = new EncoderSession(session, manifest.frame_size, manifest.latent_dim)
    await warmup(session, {
      frame: new ort.Tensor(
        'float32',
        new Float32Array(manifest.frame_size * manifest.frame_size),
        [1, 1, manifest.frame_size, manifest.frame_size],
      ),
    })
    return encoder
  }

  /** Encode a batch of normalized frames (batch * N * N floats). */
  async encode(frames: Float32Array, batch: number): Promise<Float32Array> {
    const n = this.frameSize
    if (frames.length !== batch * n * n) {
      throw new Error(`expected ${batch * n * n} floats, got ${frames.length}`)
    }
    const feeds = { frame: new ort.Tensor('float32', frames, [batch, 1, n, n]) }
    const output = await this.session.run(feeds)
    return output.latent.data as Float32Array
  }
}
