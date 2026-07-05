import { describe, expect, it } from 'vitest'
import { APP_NAME, APP_TAGLINE } from '../src/meta'

// Trivial Phase 0 smoke test. Real env-parity and inference tests land in
// Phases 5–6 (see PROJECTPLAN.md).
describe('app metadata', () => {
  it('exposes the app name', () => {
    expect(APP_NAME).toBe('latent-lab')
  })

  it('has a non-empty tagline', () => {
    expect(APP_TAGLINE.length).toBeGreaterThan(0)
  })
})
