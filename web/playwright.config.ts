import { defineConfig, devices } from '@playwright/test'

// E2E is exercised from Phase 6 onward. Config lives here now so the wiring is
// ready; both a WebGPU-capable and a WASM-only browser context are configured
// because the WASM fallback is a first-class, separately tested path.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
        },
      },
    },
    {
      name: 'chromium-wasm',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
