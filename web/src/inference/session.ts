/**
 * onnxruntime-web session management: WebGPU when available, WASM as the
 * first-class fallback. Every session gets a warmup run (first inference
 * compiles kernels/uploads weights; keeping it out of interactive paths).
 */

import * as ort from 'onnxruntime-web'

export type Backend = 'webgpu' | 'wasm'

/** Detect the backend once; both code paths are supported and tested. */
export async function detectBackend(): Promise<Backend> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter()
      if (adapter) return 'webgpu'
    } catch {
      // fall through to wasm
    }
  }
  return 'wasm'
}

export async function createSession(
  model: ArrayBuffer,
  backend: Backend,
): Promise<ort.InferenceSession> {
  const providers: string[] = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
  return ort.InferenceSession.create(new Uint8Array(model), {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  })
}

/** Run a throwaway inference so kernel compilation doesn't hit the UI path. */
export async function warmup(
  session: ort.InferenceSession,
  feeds: Record<string, ort.Tensor>,
): Promise<void> {
  await session.run(feeds)
}

export { ort }
