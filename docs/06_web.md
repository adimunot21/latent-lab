# The browser app

## Inference plumbing (`web/src/inference/`)

`session.ts` detects WebGPU (`navigator.gpu` + a real `requestAdapter()`
probe) and falls back to WASM; every session gets a warmup inference so
kernel compilation never lands on an interactive path. `manifest.ts` types
the manifest, streams downloads with progress (byte totals come from the
manifest, not Content-Length — the service worker cache doesn't always
surface headers), and sha256-verifies every artifact before use.
`encoder.ts`/`predictor.ts` are thin typed wrappers around named
inputs/outputs.

Chrome on Linux still gates WebGPU behind flags, so a Linux dev machine
showing the WASM badge is expected, not a bug. WASM is the CI-gated path;
WebGPU is an enhancement verified where an adapter exists.

## The planner worker (`web/src/planner/`)

`cem.ts` is a line-faithful port of the Python planner — including the dense
cost (see docs/04; endpoint-only cost demonstrably breaks MPC). The hot loop
reuses preallocated Float32Arrays; the RNG is seedable (mulberry32 +
Box-Muller) so unit tests are deterministic.

The worker (`worker.ts`) owns its **own** ONNX predictor session (sessions
can't cross threads) *and* the lookup table. That second part is the
non-obvious design choice: elite/best imagined trajectories are
nearest-neighbor-decoded to (x, y) paths **inside the worker**, so a plan
response carries a few hundred floats instead of megabytes of latents.
`messages.ts` is the typed protocol; `client.ts` wraps it in promises. The
worker also serves `rollout` requests for the imagination panel — the
predictor exists in exactly one place.

Measured: ~230 ms per plan on WASM at defaults (population 256 × 4 iterations
× horizon 12 = ~12k predictor calls, batched 256 at a time).

## App composition (`App.svelte`)

One component on purpose — three canvases and a control row don't warrant a
component tree yet. Key mechanics:

- **MPC loop**: click → encode goal frame once (teleport-render-restore, same
  pattern as the Python eval) → plan/execute/redraw until within 0.08 or 60
  steps. A `planEpoch` counter cancels stale loops when the user re-clicks or
  swaps checkpoints mid-run.
- **Checkpoint switcher**: downloads that checkpoint's encoder+predictor
  (verified), rebuilds sessions (new worker), then re-encodes a stride-5
  sample of the lookup grid with the *current* encoder and projects it
  through the **fixed healthy PCA**. Fixed basis + fixed view box = collapse
  renders as the cloud shrinking ~10,000× instead of being autoscaled away.
- **Imagination panel**: keeps the last 8 (latent, action) pairs; each step,
  the worker replays the actions open-loop from the oldest latent and the
  panel shows per-step divergence bars plus the imagined path as ghost dots.
- **Svelte 5 gotcha that bit once**: anything read by the template after an
  async load must be `$state` — `manifest` initially wasn't, and the
  checkpoint `<select>` rendered permanently empty while all tests passed
  (the e2e hooks bypassed the UI). The compiler warning
  `non_reactive_update` is worth treating as an error.
- `window.__latentlab` exposes test hooks (setAgent/planTo/switchCheckpoint/
  cloudSpread) — harmless in production, load-bearing for e2e.

## Loading UX

Progress bar driven by manifest byte counts; `public/sw.js` caches pinned-
revision HF downloads cache-first (immutable URLs → safe forever), so
revisits and checkpoint re-switches are instant and work offline.
