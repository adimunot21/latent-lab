import { expect, test } from '@playwright/test'

// Phase 0 e2e placeholder: the page loads and renders its heading. Real
// in-browser planning-success evals arrive in Phase 6 (see PROJECTPLAN.md).
// Not run in CI yet; wired up when the app has models to load.
test('landing page renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'latent-lab' })).toBeVisible()
})
