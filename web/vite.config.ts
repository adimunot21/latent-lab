/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  // Site is served from the repo subpath on GitHub Pages until the custom
  // deploy target is finalized (Phase 7). Relative base keeps asset URLs valid
  // regardless of where the static bundle is hosted.
  base: './',
  plugins: [svelte()],
  test: {
    // Vitest: jsdom so Svelte component tests can mount against a DOM.
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.{test,spec}.ts', 'src/**/*.{test,spec}.ts'],
  },
})
