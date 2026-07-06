/**
 * Service worker: cache-first for model artifacts from the pinned Hugging
 * Face revision. Those URLs contain a commit hash, so their content is
 * immutable — caching forever is safe, and revisits load models instantly
 * and offline. App assets are left to normal HTTP caching (Vite hashes them).
 */

const CACHE_NAME = 'latentlab-models-v1'
const HF_PREFIX = 'https://huggingface.co/adimunot/latent-lab/resolve/'

self.addEventListener('install', () => {
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim())
})

self.addEventListener('fetch', (event) => {
  const url = event.request.url
  // Only intercept pinned model downloads (incl. the CDN redirect target
  // stays cached under the original request URL).
  if (!url.startsWith(HF_PREFIX)) return

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(event.request)
      if (cached) return cached
      const response = await fetch(event.request)
      if (response.ok) {
        cache.put(event.request, response.clone())
      }
      return response
    }),
  )
})
