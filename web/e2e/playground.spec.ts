/**
 * Phase 6 e2e: in-browser planning success + collapse visibility, on both
 * execution providers (the chromium-webgpu project auto-skips where no GPU
 * adapter exists, e.g. CI runners; chromium-wasm must always pass).
 *
 * Uses the window.__latentlab test hooks exposed by App.svelte.
 */
import { expect, test, type Page } from '@playwright/test'

interface LatentLabHooks {
  getBackend(): string
  getState(): [number, number]
  setAgent(x: number, y: number): void
  planTo(x: number, y: number): Promise<{ success: boolean; steps: number }>
  switchCheckpoint(id: string): Promise<void>
  cloudSpread(): number
}

declare global {
  interface Window {
    __latentlab: LatentLabHooks
  }
}

// Model download + session init can take a while on cold CI caches.
const READY_TIMEOUT = 180_000

async function waitReady(page: Page): Promise<void> {
  await page.goto('/')
  await page
    .getByTestId('status-badge')
    .filter({ hasText: 'ready' })
    .waitFor({ timeout: READY_TIMEOUT })
}

test.describe('playground', () => {
  test.beforeEach(async ({ page }, testInfo) => {
    if (testInfo.project.name === 'chromium-webgpu') {
      // Auto-skip where no GPU adapter exists (e.g. CI runners): the WASM
      // project is the always-on gate; this project verifies WebGPU where
      // hardware allows.
      await page.goto('about:blank')
      const hasAdapter = await page.evaluate(async () => {
        const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
        if (!gpu) return false
        try {
          return (await gpu.requestAdapter()) !== null
        } catch {
          return false
        }
      })
      testInfo.skip(!hasAdapter, 'no WebGPU adapter available in this environment')
    }
  })

  test('loads models and shows a backend badge', async ({ page }, testInfo) => {
    await waitReady(page)
    const badge = await page.getByTestId('backend-badge').textContent()
    if (testInfo.project.name === 'chromium-webgpu') {
      expect(badge).toBe('WEBGPU')
    } else {
      expect(['WEBGPU', 'WASM']).toContain(badge)
    }
  })

  test('in-browser planning success (3 episodes incl. cross-room)', async ({ page }) => {
    test.setTimeout(600_000) // CEM in WASM on CI runners is slow but bounded
    await waitReady(page)

    const episodes: Array<{ start: [number, number]; goal: [number, number] }> = [
      { start: [0.25, 0.5], goal: [0.75, 0.5] }, // cross-room through the door
      { start: [0.8, 0.2], goal: [0.2, 0.8] }, // cross-room diagonal
      { start: [0.7, 0.7], goal: [0.9, 0.2] }, // same-room
    ]
    let successes = 0
    const steps: number[] = []
    for (const episode of episodes) {
      const result = await page.evaluate(async ({ start, goal }) => {
        window.__latentlab.setAgent(start[0], start[1])
        return window.__latentlab.planTo(goal[0], goal[1])
      }, episode)
      if (result.success) successes++
      steps.push(result.steps)
    }
    console.log(`planning: ${successes}/${episodes.length} succeeded, steps: ${steps.join(', ')}`)
    // Python eval is 97%; with 3 episodes require all 3 (each is >90% likely,
    // and the fixed scenarios are well within the model's competence).
    expect(successes).toBe(episodes.length)
  })

  test('collapse is visible when switching checkpoints', async ({ page }) => {
    test.setTimeout(300_000)
    await waitReady(page)

    const healthySpread = await page.evaluate(() => window.__latentlab.cloudSpread())
    await page.evaluate(() => window.__latentlab.switchCheckpoint('collapsed'))
    await page
      .getByTestId('status-badge')
      .filter({ hasText: 'ready' })
      .waitFor({ timeout: READY_TIMEOUT })
    const collapsedSpread = await page.evaluate(() => window.__latentlab.cloudSpread())

    console.log(
      `cloud spread: healthy ${healthySpread.toFixed(3)}, collapsed ${collapsedSpread.toFixed(5)}`,
    )
    // The collapsed cloud (projected through the FIXED healthy PCA) must
    // shrink dramatically — this is the product's headline demo.
    expect(collapsedSpread).toBeLessThan(healthySpread / 20)
  })
})
