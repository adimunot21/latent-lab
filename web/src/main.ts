import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'

// Service worker: caches pinned-revision model downloads (see public/sw.js).
// Registration failure is non-fatal — the app just refetches on each visit.
if ('serviceWorker' in navigator && !import.meta.env.DEV) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch((error) => {
    console.warn('service worker registration failed:', error)
  })
}

const app = mount(App, {
  target: document.getElementById('app')!,
})

export default app
