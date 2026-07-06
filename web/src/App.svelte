<script lang="ts">
  import { onMount } from 'svelte'
  import { APP_NAME, APP_TAGLINE } from './meta'
  import { TwoRoomsEnv } from './env/twoRooms'
  import { loadManifest, fetchVerified, normalizeFrame, type Manifest } from './inference/manifest'
  import { detectBackend, type Backend } from './inference/session'
  import { EncoderSession } from './inference/encoder'
  import { LatentPanel } from './viz/latentPanel'

  let status = $state<'loading' | 'ready' | 'error'>('loading')
  let statusDetail = $state('fetching manifest…')
  let backend = $state<Backend | null>(null)
  let agentState = $state<[number, number]>([0.25, 0.5])

  let envCanvas: HTMLCanvasElement
  let latentCanvas: HTMLCanvasElement

  const env = new TwoRoomsEnv()
  let encoder: EncoderSession | null = null
  let manifest: Manifest | null = null
  let panel: LatentPanel | null = null
  let encoding = false
  let pendingEncode = false

  async function encodeCurrentFrame(): Promise<void> {
    if (!encoder || !manifest || !panel) return
    // Coalesce: if an encode is in flight, remember to run one more after.
    if (encoding) {
      pendingEncode = true
      return
    }
    encoding = true
    try {
      const frame = normalizeFrame(env.render(), manifest)
      const latent = await encoder.encode(frame, 1)
      panel.push(latent)
      panel.draw(latentCanvas.getContext('2d')!)
    } finally {
      encoding = false
      if (pendingEncode) {
        pendingEncode = false
        void encodeCurrentFrame()
      }
    }
  }

  function redraw(): void {
    env.draw(envCanvas.getContext('2d')!)
    agentState = env.state
    void encodeCurrentFrame()
  }

  function onKeydown(event: KeyboardEvent): void {
    if (status !== 'ready') return
    const m = env.config.maxStep
    const moves: Record<string, [number, number]> = {
      ArrowUp: [0, -m],
      ArrowDown: [0, m],
      ArrowLeft: [-m, 0],
      ArrowRight: [m, 0],
    }
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    env.step(move)
    redraw()
  }

  onMount(() => {
    void (async () => {
      try {
        manifest = await loadManifest()
        statusDetail = 'detecting backend…'
        backend = await detectBackend()

        statusDetail = 'downloading healthy encoder…'
        const healthy = manifest.checkpoints.find((c) => c.id === 'healthy')!
        const encoderBuffer = await fetchVerified(healthy.files.encoder)

        statusDetail = 'downloading lookup table…'
        const latentsBuffer = await fetchVerified(manifest.lookup.latents)
        const lookupLatents = new Float32Array(latentsBuffer)

        statusDetail = `starting ${backend} session…`
        encoder = await EncoderSession.create(encoderBuffer, manifest, backend)
        panel = new LatentPanel(manifest, lookupLatents)

        env.setState(0.25, 0.5)
        status = 'ready'
        redraw()
      } catch (error) {
        status = 'error'
        statusDetail = error instanceof Error ? error.message : String(error)
      }
    })()
  })
</script>

<svelte:window onkeydown={onKeydown} />

<main>
  <header>
    <h1>{APP_NAME}</h1>
    <p class="tagline">{APP_TAGLINE}</p>
    <div class="badges">
      {#if backend}
        <span class="badge" data-testid="backend-badge">{backend.toUpperCase()}</span>
      {/if}
      <span class="badge status-{status}" data-testid="status-badge">
        {status === 'loading' ? statusDetail : status}
      </span>
    </div>
  </header>

  <section class="panels">
    <figure>
      <canvas bind:this={envCanvas} width="320" height="320" data-testid="env-canvas"></canvas>
      <figcaption>
        Two Rooms — drive with arrow keys.
        <span class="coords">({agentState[0].toFixed(2)}, {agentState[1].toFixed(2)})</span>
      </figcaption>
    </figure>
    <figure>
      <canvas bind:this={latentCanvas} width="320" height="320" data-testid="latent-canvas"
      ></canvas>
      <figcaption>Latent space (fixed PCA) — the yellow dot is you.</figcaption>
    </figure>
  </section>

  {#if status === 'error'}
    <p class="error" data-testid="error-message">Failed to load: {statusDetail}</p>
  {/if}
</main>

<style>
  main {
    max-width: 48rem;
    margin: 2rem auto;
    padding: 0 1.5rem;
    font-family:
      system-ui,
      -apple-system,
      sans-serif;
  }
  h1 {
    margin-bottom: 0.25rem;
    font-size: 2rem;
  }
  .tagline {
    color: #6b7280;
    margin-top: 0;
  }
  .badges {
    display: flex;
    gap: 0.5rem;
  }
  .badge {
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #1f2937;
    color: #e5e7eb;
  }
  .badge.status-ready {
    background: #14532d;
    color: #bbf7d0;
  }
  .badge.status-error {
    background: #7f1d1d;
    color: #fecaca;
  }
  .panels {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
  }
  figure {
    margin: 0;
  }
  canvas {
    border: 1px solid #374151;
    border-radius: 0.5rem;
    image-rendering: pixelated;
  }
  figcaption {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #6b7280;
  }
  .coords {
    font-variant-numeric: tabular-nums;
  }
  .error {
    color: #dc2626;
  }
</style>
