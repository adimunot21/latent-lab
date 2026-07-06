/**
 * Two Rooms environment — TypeScript port of training/src/latentlab/envs/two_rooms.py.
 *
 * PORTABILITY CONTRACT (mirrors the Python module docstring; parity is gated
 * on shared/fixtures/two_rooms_parity.json in BOTH CI pipelines):
 * - All state math in float64 (JS numbers) — matches Python floats bit-for-bit.
 * - x grows rightward, y grows DOWNWARD. Frame row i / col j covers world
 *   point ((j + 0.5)/N, (i + 0.5)/N).
 * - step() resolves per-axis, x FIRST then y, rejecting colliding axis moves
 *   (wall sliding). Order is part of the contract.
 * - Collision: circle-vs-AABB closest-point with STRICT `<` against radius^2.
 * - No RNG in dynamics. reset() here just takes an explicit state (the
 *   browser picks starts; parity fixtures store starts explicitly).
 */

export interface TwoRoomsConfig {
  wallX: number
  wallHalfThickness: number
  doorCenterY: number
  doorHalfHeight: number
  agentRadius: number
  maxStep: number
  frameSize: number
  wallIntensity: number
  agentIntensity: number
}

export const DEFAULT_CONFIG: TwoRoomsConfig = {
  wallX: 0.5,
  wallHalfThickness: 0.03,
  doorCenterY: 0.5,
  doorHalfHeight: 0.12,
  agentRadius: 0.05,
  maxStep: 0.08,
  frameSize: 64,
  wallIntensity: 128,
  agentIntensity: 255,
}

/** Axis-aligned rectangle: [xMin, yMin, xMax, yMax]. */
export type Rect = [number, number, number, number]

export function wallRects(config: TwoRoomsConfig): Rect[] {
  const x0 = config.wallX - config.wallHalfThickness
  const x1 = config.wallX + config.wallHalfThickness
  return [
    [x0, 0.0, x1, config.doorCenterY - config.doorHalfHeight],
    [x0, config.doorCenterY + config.doorHalfHeight, x1, 1.0],
  ]
}

export function circleHitsRect(cx: number, cy: number, radius: number, rect: Rect): boolean {
  const [x0, y0, x1, y1] = rect
  const nearestX = Math.min(Math.max(cx, x0), x1)
  const nearestY = Math.min(Math.max(cy, y0), y1)
  const dx = cx - nearestX
  const dy = cy - nearestY
  return dx * dx + dy * dy < radius * radius
}

export class TwoRoomsEnv {
  readonly config: TwoRoomsConfig
  private walls: Rect[]
  private x = 0
  private y = 0

  constructor(config: TwoRoomsConfig = DEFAULT_CONFIG) {
    this.config = config
    this.walls = wallRects(config)
  }

  get state(): [number, number] {
    return [this.x, this.y]
  }

  isFree(x: number, y: number): boolean {
    const r = this.config.agentRadius
    if (!(r <= x && x <= 1.0 - r && r <= y && y <= 1.0 - r)) return false
    return !this.walls.some((rect) => circleHitsRect(x, y, r, rect))
  }

  setState(x: number, y: number): void {
    if (!this.isFree(x, y)) {
      throw new Error(`state (${x}, ${y}) collides with a wall or boundary`)
    }
    this.x = x
    this.y = y
  }

  /** Apply a clamped (dx, dy) displacement. Mirrors Python step() exactly. */
  step(action: [number, number]): [number, number] {
    const m = this.config.maxStep
    const r = this.config.agentRadius
    const dx = Math.min(Math.max(action[0], -m), m)
    const dy = Math.min(Math.max(action[1], -m), m)

    const newX = Math.min(Math.max(this.x + dx, r), 1.0 - r)
    if (!this.walls.some((rect) => circleHitsRect(newX, this.y, r, rect))) {
      this.x = newX
    }

    const newY = Math.min(Math.max(this.y + dy, r), 1.0 - r)
    if (!this.walls.some((rect) => circleHitsRect(this.x, newY, r, rect))) {
      this.y = newY
    }

    return this.state
  }

  clampAction(action: [number, number]): [number, number] {
    const m = this.config.maxStep
    return [Math.min(Math.max(action[0], -m), m), Math.min(Math.max(action[1], -m), m)]
  }

  /**
   * Rasterize to a frameSize x frameSize grayscale byte array (row-major).
   * Must stay byte-exact with the Python render(): pixel centers, rect mask
   * [x0, x1) x [y0, y1), agent circle with <=, agent drawn over wall.
   */
  render(): Uint8Array {
    const cfg = this.config
    const n = cfg.frameSize
    const frame = new Uint8Array(n * n)
    for (let i = 0; i < n; i++) {
      const wy = (i + 0.5) / n
      for (let j = 0; j < n; j++) {
        const wx = (j + 0.5) / n
        let value = 0
        for (const [x0, y0, x1, y1] of this.walls) {
          if (wx >= x0 && wx < x1 && wy >= y0 && wy < y1) {
            value = cfg.wallIntensity
            break
          }
        }
        const ddx = wx - this.x
        const ddy = wy - this.y
        if (ddx * ddx + ddy * ddy <= cfg.agentRadius * cfg.agentRadius) {
          value = cfg.agentIntensity
        }
        frame[i * n + j] = value
      }
    }
    return frame
  }

  /** Draw the current frame (plus optional goal ring) onto a canvas, scaled. */
  draw(ctx: CanvasRenderingContext2D, goal?: [number, number]): void {
    const n = this.config.frameSize
    const frame = this.render()
    const scale = ctx.canvas.width / n
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = frame[i * n + j]
        ctx.fillStyle = `rgb(${v},${v},${v})`
        ctx.fillRect(j * scale, i * scale, scale, scale)
      }
    }
    if (goal) {
      ctx.strokeStyle = 'rgb(255,180,60)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(
        goal[0] * ctx.canvas.width,
        goal[1] * ctx.canvas.height,
        this.config.agentRadius * ctx.canvas.width,
        0,
        2 * Math.PI,
      )
      ctx.stroke()
    }
  }
}
