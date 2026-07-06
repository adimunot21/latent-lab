/**
 * ENV PARITY GATE: the TS env must reproduce shared/fixtures/ EXACTLY.
 *
 * States compare with === (bit-exact float64) and frames byte-for-byte.
 * If this fails, the TS port drifted from the Python reference — fix the
 * port (or, for an intentional env change, regenerate fixtures from Python
 * and expect BOTH CI pipelines to gate on the new file).
 */
import { describe, expect, it } from 'vitest'
import fixtures from '../../shared/fixtures/two_rooms_parity.json'
import { DEFAULT_CONFIG, TwoRoomsEnv } from '../src/env/twoRooms'

interface FixtureCase {
  name: string
  start: number[]
  actions: number[][]
  states: number[][]
}

interface FrameFixture {
  name: string
  state: number[]
  frame: number[]
}

describe('env config parity', () => {
  it('fixture config matches the TS defaults', () => {
    const c = fixtures.env_config
    expect(DEFAULT_CONFIG).toEqual({
      wallX: c.wall_x,
      wallHalfThickness: c.wall_half_thickness,
      doorCenterY: c.door_center_y,
      doorHalfHeight: c.door_half_height,
      agentRadius: c.agent_radius,
      maxStep: c.max_step,
      frameSize: c.frame_size,
      wallIntensity: c.wall_intensity,
      agentIntensity: c.agent_intensity,
    })
  })
})

describe('trajectory parity (bit-exact float64)', () => {
  for (const fixtureCase of fixtures.cases as FixtureCase[]) {
    it(fixtureCase.name, () => {
      const env = new TwoRoomsEnv()
      env.setState(fixtureCase.start[0], fixtureCase.start[1])
      fixtureCase.actions.forEach((action, i) => {
        const [x, y] = env.step([action[0], action[1]])
        const [ex, ey] = fixtureCase.states[i]
        // Exact equality, not toBeCloseTo: the portability contract promises
        // bit-identical float64 dynamics.
        expect(x, `${fixtureCase.name} step ${i} x`).toBe(ex)
        expect(y, `${fixtureCase.name} step ${i} y`).toBe(ey)
      })
    })
  }
})

describe('frame parity (byte-exact render)', () => {
  for (const frameFixture of fixtures.frames as FrameFixture[]) {
    it(frameFixture.name, () => {
      const env = new TwoRoomsEnv()
      env.setState(frameFixture.state[0], frameFixture.state[1])
      const frame = env.render()
      const expected = Uint8Array.from(frameFixture.frame)
      expect(frame.length).toBe(expected.length)
      // Find first mismatch for a useful failure message.
      for (let k = 0; k < frame.length; k++) {
        if (frame[k] !== expected[k]) {
          const n = DEFAULT_CONFIG.frameSize
          expect.fail(
            `frame '${frameFixture.name}' differs at row ${Math.floor(k / n)}, col ${k % n}: ` +
              `got ${frame[k]}, expected ${expected[k]}`,
          )
        }
      }
    })
  }
})
