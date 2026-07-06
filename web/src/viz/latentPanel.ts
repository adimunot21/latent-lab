/**
 * Latent-space panel: projects latents through the manifest's FIXED PCA basis
 * (fit on the healthy encoder — fixed so collapse is visible when checkpoints
 * swap) and draws a static background cloud (the lookup-table latents) plus a
 * live dot with a fading trail.
 */

import type { Manifest } from '../inference/manifest'

export class LatentPanel {
  private components: Float32Array // (D * 2), column-major pairs [c0_d, c1_d]
  private mean: Float32Array
  private latentDim: number
  private bounds: { lo: [number, number]; hi: [number, number] }
  private cloud: Array<[number, number]> = []
  private trail: Array<[number, number]> = []
  private maxTrail = 40

  constructor(manifest: Manifest, lookupLatents: Float32Array) {
    this.latentDim = manifest.latent_dim
    this.mean = Float32Array.from(manifest.pca.mean)
    this.components = new Float32Array(this.latentDim * 2)
    for (let d = 0; d < this.latentDim; d++) {
      this.components[d * 2] = manifest.pca.components[d][0]
      this.components[d * 2 + 1] = manifest.pca.components[d][1]
    }

    // Project the lookup cloud once; its extent defines the fixed view box
    // (again: fixed, so a collapsed cloud visibly shrinks instead of zooming).
    const n = lookupLatents.length / this.latentDim
    const lo: [number, number] = [Infinity, Infinity]
    const hi: [number, number] = [-Infinity, -Infinity]
    for (let i = 0; i < n; i++) {
      const p = this.project(lookupLatents.subarray(i * this.latentDim, (i + 1) * this.latentDim))
      this.cloud.push(p)
      lo[0] = Math.min(lo[0], p[0])
      lo[1] = Math.min(lo[1], p[1])
      hi[0] = Math.max(hi[0], p[0])
      hi[1] = Math.max(hi[1], p[1])
    }
    // 10% margin so the live dot isn't clipped at the cloud edge.
    const mx = 0.1 * (hi[0] - lo[0])
    const my = 0.1 * (hi[1] - lo[1])
    this.bounds = { lo: [lo[0] - mx, lo[1] - my], hi: [hi[0] + mx, hi[1] + my] }
  }

  project(latent: Float32Array): [number, number] {
    let px = 0
    let py = 0
    for (let d = 0; d < this.latentDim; d++) {
      const centered = latent[d] - this.mean[d]
      px += centered * this.components[d * 2]
      py += centered * this.components[d * 2 + 1]
    }
    return [px, py]
  }

  push(latent: Float32Array): void {
    this.trail.push(this.project(latent))
    if (this.trail.length > this.maxTrail) this.trail.shift()
  }

  clearTrail(): void {
    this.trail = []
  }

  /**
   * Replace the background cloud with latents from the CURRENT encoder
   * (checkpoint switcher). The view box stays fixed to the healthy cloud's
   * extent — that's what makes a collapsed cloud visibly shrink to a point
   * instead of the view auto-zooming into its noise.
   */
  setCloud(latents: Float32Array): void {
    const n = latents.length / this.latentDim
    this.cloud = []
    for (let i = 0; i < n; i++) {
      this.cloud.push(this.project(latents.subarray(i * this.latentDim, (i + 1) * this.latentDim)))
    }
  }

  /** Mean distance of cloud points from their centroid (collapse metric). */
  cloudSpread(): number {
    if (this.cloud.length === 0) return 0
    let cx = 0
    let cy = 0
    for (const [x, y] of this.cloud) {
      cx += x
      cy += y
    }
    cx /= this.cloud.length
    cy /= this.cloud.length
    let total = 0
    for (const [x, y] of this.cloud) total += Math.hypot(x - cx, y - cy)
    return total / this.cloud.length
  }

  private toCanvas(p: [number, number], w: number, h: number): [number, number] {
    const { lo, hi } = this.bounds
    return [((p[0] - lo[0]) / (hi[0] - lo[0])) * w, (1 - (p[1] - lo[1]) / (hi[1] - lo[1])) * h]
  }

  draw(ctx: CanvasRenderingContext2D): void {
    const { width: w, height: h } = ctx.canvas
    ctx.fillStyle = '#0b0f17'
    ctx.fillRect(0, 0, w, h)

    ctx.fillStyle = 'rgba(120, 140, 170, 0.35)'
    for (const p of this.cloud) {
      const [cx, cy] = this.toCanvas(p, w, h)
      ctx.fillRect(cx - 1, cy - 1, 2, 2)
    }

    this.trail.forEach((p, i) => {
      const alpha = (i + 1) / this.trail.length
      const [cx, cy] = this.toCanvas(p, w, h)
      ctx.fillStyle = `rgba(96, 200, 255, ${0.15 + 0.5 * alpha})`
      ctx.beginPath()
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI)
      ctx.fill()
    })

    const last = this.trail[this.trail.length - 1]
    if (last) {
      const [cx, cy] = this.toCanvas(last, w, h)
      ctx.fillStyle = '#ffd166'
      ctx.beginPath()
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI)
      ctx.fill()
    }
  }
}
