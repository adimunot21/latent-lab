<script lang="ts">
  import { APP_NAME, APP_TAGLINE } from './meta'
</script>

<main>
  <h1>{APP_NAME}</h1>
  <p class="tagline">{APP_TAGLINE}</p>
  <p class="status" data-testid="phase-status">
    Phase 0 scaffold — model loading, the Two Rooms env, and the latent planner arrive next.
  </p>
</main>

<style>
  main {
    max-width: 42rem;
    margin: 4rem auto;
    padding: 0 1.5rem;
    font-family:
      system-ui,
      -apple-system,
      sans-serif;
    line-height: 1.5;
  }
  h1 {
    margin-bottom: 0.25rem;
    font-size: 2.5rem;
  }
  .tagline {
    color: #6b7280;
    margin-top: 0;
  }
  .status {
    margin-top: 2rem;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    background: #f9fafb;
  }
</style>
