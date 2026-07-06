<script lang="ts">
  import { onMount } from 'svelte'
  import { APP_NAME, APP_TAGLINE } from './meta'
  import { TwoRoomsEnv } from './env/twoRooms'
  import {
    loadManifest,
    fetchVerified,
    normalizeFrame,
    type CheckpointEntry,
    type Manifest,
  } from './inference/manifest'
  import { detectBackend, type Backend } from './inference/session'
  import { EncoderSession } from './inference/encoder'
  import { LatentPanel } from './viz/latentPanel'
  import { PlannerClient } from './planner/client'
  import { DEFAULT_CEM_CONFIG } from './planner/cem'

  const MAX_MPC_STEPS = 60
  const SUCCESS_RADIUS = 0.08
  const IMAGINATION_STEPS = 8
  const CLOUD_SAMPLE_STRIDE = 5 // every 5th lookup state -> ~576 cloud points

  let status = $state<'loading' | 'ready' | 'error'>('loading')
  let statusDetail = $state('fetching manifest…')
  let loadProgress = $state<number | null>(null) // 0..1 while downloading models
  let backend = $state<Backend | null>(null)
  let agentState = $state<[number, number]>([0.25, 0.5])
  let checkpointId = $state('healthy')
  let planning = $state(false)
  let planLatencyMs = $state<number | null>(null)
  let lastPlanResult = $state<string | null>(null)
  let cemPopulation = $state(DEFAULT_CEM_CONFIG.population)
  let cemIterations = $state(DEFAULT_CEM_CONFIG.iterations)
  let cemHorizon = $state(DEFAULT_CEM_CONFIG.horizon)
  let divergence = $state<number[]>([])

  let envCanvas: HTMLCanvasElement
  let latentCanvas: HTMLCanvasElement

  const env = new TwoRoomsEnv()
  // $state: the checkpoint <select> renders from manifest after async load.
  let manifest = $state<Manifest | null>(null)
  let encoder: EncoderSession | null = null
  let planner: PlannerClient | null = null
  let panel: LatentPanel | null = null

  // Pristine lookup buffers (worker init transfers a copy away each time).
  let lookupStatesBuf: ArrayBuffer | null = null
  let lookupLatentsBuf: ArrayBuffer | null = null

  let goal: [number, number] | null = null
  let planEpoch = 0 // bumped to cancel a running MPC loop
  let overlayBestPath: Float32Array | null = null
  let overlayElitePaths: { paths: Float32Array; n: number } | null = null
  let imaginedPath: Float32Array | null = null
  // (latent, action-taken-after) history for the imagination panel.
  let history: Array<{ latent: Float32Array; action: [number, number] }> = []

  function cemConfig() {
    return {
      ...DEFAULT_CEM_CONFIG,
      population: cemPopulation,
      iterations: cemIterations,
      horizon: cemHorizon,
    }
  }

  async function encodeState(): Promise<Float32Array> {
    const frame = normalizeFrame(env.render(), manifest!)
    return encoder!.encode(frame, 1)
  }

  function drawEnv(): void {
    const ctx = envCanvas.getContext('2d')!
    env.draw(ctx, goal ?? undefined)
    const w = ctx.canvas.width
    // Elite candidate paths (dim), then the chosen plan (orange).
    if (overlayElitePaths) {
      ctx.strokeStyle = 'rgba(96, 140, 200, 0.35)'
      ctx.lineWidth = 1
      const { paths, n } = overlayElitePaths
      const horizon = paths.length / (n * 2)
      for (let i = 0; i < n; i++) {
        ctx.beginPath()
        for (let t = 0; t < horizon; t++) {
          const x = paths[(i * horizon + t) * 2] * w
          const y = paths[(i * horizon + t) * 2 + 1] * w
          if (t === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
    }
    if (overlayBestPath) {
      ctx.strokeStyle = 'rgb(255, 160, 40)'
      ctx.lineWidth = 2
      ctx.beginPath()
      const horizon = overlayBestPath.length / 2
      for (let t = 0; t < horizon; t++) {
        const x = overlayBestPath[t * 2] * w
        const y = overlayBestPath[t * 2 + 1] * w
        if (t === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }
    if (imaginedPath) {
      ctx.fillStyle = 'rgba(190, 120, 255, 0.8)'
      const steps = imaginedPath.length / 2
      for (let t = 0; t < steps; t++) {
        ctx.beginPath()
        ctx.arc(imaginedPath[t * 2] * w, imaginedPath[t * 2 + 1] * w, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    }
    agentState = env.state
  }

  async function afterStep(latent: Float32Array, action: [number, number]): Promise<void> {
    panel!.push(latent)
    panel!.draw(latentCanvas.getContext('2d')!)
    history.push({ latent, action })
    if (history.length > IMAGINATION_STEPS + 1) history.shift()
    if (history.length === IMAGINATION_STEPS + 1 && planner) {
      // Open-loop rollout from the oldest latent with the executed actions;
      // divergence = per-step latent distance to what the encoder actually saw.
      const actions = new Float32Array(IMAGINATION_STEPS * 2)
      history.slice(0, IMAGINATION_STEPS).forEach((h, i) => {
        actions[i * 2] = h.action[0]
        actions[i * 2 + 1] = h.action[1]
      })
      const rollout = await planner.rollout(history[0].latent, actions)
      const d = manifest!.latent_dim
      divergence = Array.from({ length: IMAGINATION_STEPS }, (_, t) => {
        let dist = 0
        const actual = history[t + 1].latent
        for (let k = 0; k < d; k++) {
          const diff = rollout.latents[t * d + k] - actual[k]
          dist += diff * diff
        }
        return Math.sqrt(dist)
      })
      imaginedPath = rollout.path
    }
  }

  /** MPC loop: plan -> execute first action -> repeat until goal or budget. */
  async function runPlanningLoop(): Promise<{ success: boolean; steps: number }> {
    const epoch = ++planEpoch
    planning = true
    lastPlanResult = null
    planner!.reset()
    try {
      // Encode the goal frame once (teleport pattern, like the Python eval).
      const [sx, sy] = env.state
      env.setState(goal![0], goal![1])
      const zGoal = (await encodeState()).slice()
      env.setState(sx, sy)

      for (let step = 0; step < MAX_MPC_STEPS; step++) {
        if (epoch !== planEpoch) return { success: false, steps: step }
        const [x, y] = env.state
        if (goal && Math.hypot(x - goal[0], y - goal[1]) <= SUCCESS_RADIUS) {
          lastPlanResult = `reached in ${step} steps`
          return { success: true, steps: step }
        }
        const z0 = await encodeState()
        const result = await planner!.plan(z0.slice(), zGoal, cemConfig())
        if (epoch !== planEpoch) return { success: false, steps: step }
        planLatencyMs = result.latencyMs
        overlayBestPath = result.bestPath
        overlayElitePaths = { paths: result.elitePaths, n: result.nElites }
        const action: [number, number] = [result.actions[0], result.actions[1]]
        env.step(action)
        drawEnv()
        await afterStep(z0, action)
      }
      lastPlanResult = `gave up after ${MAX_MPC_STEPS} steps`
      return { success: false, steps: MAX_MPC_STEPS }
    } finally {
      if (epoch === planEpoch) {
        planning = false
        overlayElitePaths = null
        drawEnv()
      }
    }
  }

  function onEnvClick(event: MouseEvent): void {
    if (status !== 'ready') return
    const rect = envCanvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    if (!env.isFree(x, y)) return
    goal = [x, y]
    drawEnv()
    void runPlanningLoop()
  }

  function onKeydown(event: KeyboardEvent): void {
    if (status !== 'ready' || planning) return
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
    drawEnv()
    void encodeState().then((z) => afterStep(z, move))
  }

  /** Re-encode a grid sample with the CURRENT encoder -> latent cloud swap. */
  async function refreshCloud(): Promise<void> {
    const states = new Float32Array(lookupStatesBuf!)
    const n = states.length / 2
    const frameSize = manifest!.frame_size
    const probe = new TwoRoomsEnv()
    const sample: number[] = []
    for (let i = 0; i < n; i += CLOUD_SAMPLE_STRIDE) sample.push(i)

    const batchSize = 64
    const latents = new Float32Array(sample.length * manifest!.latent_dim)
    for (let start = 0; start < sample.length; start += batchSize) {
      const ids = sample.slice(start, start + batchSize)
      const frames = new Float32Array(ids.length * frameSize * frameSize)
      ids.forEach((idx, j) => {
        probe.setState(states[idx * 2], states[idx * 2 + 1])
        frames.set(normalizeFrame(probe.render(), manifest!), j * frameSize * frameSize)
      })
      const z = await encoder!.encode(frames, ids.length)
      latents.set(z, start * manifest!.latent_dim)
    }
    panel!.setCloud(latents)
    panel!.draw(latentCanvas.getContext('2d')!)
  }

  async function loadCheckpoint(entry: CheckpointEntry): Promise<void> {
    statusDetail = `downloading ${entry.label}…`
    status = 'loading'
    planEpoch++ // cancel any MPC loop
    planning = false
    // Combined progress across both model files (manifest knows the sizes).
    const totals = [entry.files.encoder.bytes, entry.files.predictor.bytes]
    const loaded = [0, 0]
    const report = (slot: number) => (bytes: number) => {
      loaded[slot] = bytes
      loadProgress = (loaded[0] + loaded[1]) / (totals[0] + totals[1])
    }
    loadProgress = 0
    const [encoderBuffer, predictorBuffer] = await Promise.all([
      fetchVerified(entry.files.encoder, report(0)),
      fetchVerified(entry.files.predictor, report(1)),
    ])
    loadProgress = null
    statusDetail = `starting ${backend} sessions…`
    encoder = await EncoderSession.create(encoderBuffer, manifest!, backend!)
    planner?.terminate()
    planner = new PlannerClient()
    await planner.init(
      predictorBuffer,
      manifest!.latent_dim,
      backend!,
      lookupStatesBuf!.slice(0),
      lookupLatentsBuf!.slice(0),
    )
    statusDetail = 'encoding latent cloud…'
    panel!.clearTrail()
    history = []
    divergence = []
    imaginedPath = null
    await refreshCloud()
    status = 'ready'
    drawEnv()
  }

  async function onCheckpointChange(): Promise<void> {
    const entry = manifest!.checkpoints.find((c) => c.id === checkpointId)!
    try {
      await loadCheckpoint(entry)
    } catch (error) {
      status = 'error'
      statusDetail = error instanceof Error ? error.message : String(error)
    }
  }

  onMount(() => {
    void (async () => {
      try {
        manifest = await loadManifest()
        statusDetail = 'detecting backend…'
        backend = await detectBackend()
        statusDetail = 'downloading lookup table…'
        const [statesBuf, latentsBuf] = await Promise.all([
          fetchVerified(manifest.lookup.states),
          fetchVerified(manifest.lookup.latents),
        ])
        lookupStatesBuf = statesBuf
        lookupLatentsBuf = latentsBuf
        panel = new LatentPanel(manifest, new Float32Array(latentsBuf))
        env.setState(0.25, 0.5)
        await loadCheckpoint(manifest.checkpoints.find((c) => c.id === 'healthy')!)

        // E2E hooks (harmless in production; used by Playwright).
        ;(window as unknown as Record<string, unknown>).__latentlab = {
          getBackend: () => backend,
          getState: () => env.state,
          setAgent: (x: number, y: number) => {
            env.setState(x, y)
            drawEnv()
          },
          planTo: (x: number, y: number) => {
            goal = [x, y]
            drawEnv()
            return runPlanningLoop()
          },
          switchCheckpoint: async (id: string) => {
            checkpointId = id
            await onCheckpointChange()
          },
          cloudSpread: () => panel!.cloudSpread(),
        }
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
      {#if planLatencyMs !== null}
        <span class="badge" data-testid="latency-badge">plan {planLatencyMs.toFixed(0)} ms</span>
      {/if}
    </div>
    {#if loadProgress !== null}
      <div class="progress" data-testid="load-progress">
        <div class="progress-fill" style="width: {(loadProgress * 100).toFixed(1)}%"></div>
      </div>
    {/if}
  </header>

  <section class="controls">
    <label>
      model
      <select
        bind:value={checkpointId}
        onchange={onCheckpointChange}
        disabled={status === 'loading'}
        data-testid="checkpoint-select"
      >
        {#each manifest?.checkpoints ?? [] as entry (entry.id)}
          <option value={entry.id}>{entry.label}</option>
        {/each}
      </select>
    </label>
    <label>
      population {cemPopulation}
      <input type="range" min="64" max="512" step="64" bind:value={cemPopulation} />
    </label>
    <label>
      iterations {cemIterations}
      <input type="range" min="2" max="6" step="1" bind:value={cemIterations} />
    </label>
    <label>
      horizon {cemHorizon}
      <input type="range" min="6" max="16" step="1" bind:value={cemHorizon} />
    </label>
  </section>

  <section class="panels">
    <figure>
      <canvas
        bind:this={envCanvas}
        width="320"
        height="320"
        onclick={onEnvClick}
        data-testid="env-canvas"
      ></canvas>
      <figcaption>
        <strong>Two Rooms.</strong> Arrow keys drive; <em>click to set a goal</em> and the CEM
        planner takes over (orange = chosen plan, blue = elite candidates, purple = imagined path).
        <span class="coords">({agentState[0].toFixed(2)}, {agentState[1].toFixed(2)})</span>
        {#if lastPlanResult}<span class="result" data-testid="plan-result">{lastPlanResult}</span
          >{/if}
      </figcaption>
    </figure>
    <figure>
      <canvas bind:this={latentCanvas} width="320" height="320" data-testid="latent-canvas"
      ></canvas>
      <figcaption>
        <strong>Latent space</strong> (fixed healthy-model PCA). Swap models above — without SIGReg the
        cloud collapses to a point.
      </figcaption>
    </figure>
    <figure class="imagination">
      <div class="bars" data-testid="divergence-bars">
        {#each divergence as d, i (i)}
          <div
            class="bar"
            style="height: {Math.min(d * 8, 100)}%"
            title="step {i + 1}: {d.toFixed(2)}"
          ></div>
        {/each}
      </div>
      <figcaption>
        <strong>Imagination drift.</strong> Latent distance between an 8-step open-loop rollout and what
        the encoder actually saw. Healthy stays low; collapsed looks deceptively perfect.
      </figcaption>
    </figure>
  </section>

  {#if status === 'error'}
    <p class="error" data-testid="error-message">Failed to load: {statusDetail}</p>
  {/if}
</main>

<style>
  main {
    max-width: 72rem;
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
  .progress {
    margin-top: 0.5rem;
    height: 4px;
    max-width: 24rem;
    background: #1f2937;
    border-radius: 999px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: #60c8ff;
    transition: width 0.15s ease;
  }
  .controls {
    display: flex;
    gap: 1.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
    align-items: end;
  }
  .controls label {
    display: flex;
    flex-direction: column;
    font-size: 0.8rem;
    color: #6b7280;
    gap: 0.25rem;
  }
  .panels {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
  }
  figure {
    margin: 0;
    max-width: 20rem;
  }
  canvas {
    border: 1px solid #374151;
    border-radius: 0.5rem;
    image-rendering: pixelated;
    cursor: crosshair;
  }
  figcaption {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.4;
  }
  .coords {
    font-variant-numeric: tabular-nums;
  }
  .result {
    color: #059669;
    font-weight: 600;
  }
  .imagination .bars {
    width: 320px;
    height: 320px;
    border: 1px solid #374151;
    border-radius: 0.5rem;
    display: flex;
    align-items: flex-end;
    gap: 4px;
    padding: 8px;
    box-sizing: border-box;
    background: #0b0f17;
  }
  .imagination .bar {
    flex: 1;
    background: linear-gradient(to top, #60c8ff, #be78ff);
    border-radius: 2px 2px 0 0;
    min-height: 2px;
  }
  .error {
    color: #dc2626;
  }
</style>
