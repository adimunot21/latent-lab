# 12 — The browser: inference, the worker planner, the app

Everything so far ran where ML usually runs. This lesson is about making it
run where *users* are — a browser tab — without a server, and without the
TypeScript half quietly diverging from the Python truth.

## The env port and why it can be *proven* correct

`web/src/env/twoRooms.ts` mirrors `two_rooms.py` operation-for-operation:
same clamps, same x-then-y order, same strict `<`, same pixel-center
rasterization. Lesson 05's portability contract makes an unusual promise
achievable: JS numbers are IEEE float64, exactly like Python floats, so the
port is verified with **bit-exact equality** — fixtures generated from
Python (9 trajectories, 181 states, 3 rendered frames) replay in TS under
`expect(x).toBe(ex)` (not `toBeCloseTo`!) and byte-for-byte frame comparison.
All 13 parity tests passed on the port's first run — that's the contract
paying out, not luck.

Frame parity is as important as state parity for a subtle reason: the
browser **encodes its own canvas renders**. If TS rasterization differed by
one pixel, the encoder would see inputs from outside its training
distribution on every single frame — no error, just quiet degradation.

## Inference plumbing (`web/src/inference/`)

**Backend selection** (`session.ts`): probe `navigator.gpu` and actually
request an adapter; WebGPU if it answers, else WASM. Both are first-class —
WASM is the CI-gated path (headless browsers and CI runners expose no GPU),
WebGPU the enhancement. Every fresh session gets a **warmup inference** so
kernel compilation/weight upload never lands inside a user interaction.

**Integrity** (`manifest.ts`): every artifact download is verified —

```typescript
const digest = await crypto.subtle.digest('SHA-256', buffer)
const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
if (hex !== entry.sha256) throw new Error(`integrity check failed for ${entry.path}...`)
```

— against hashes from the manifest, which itself comes from a *pinned*
revision. Downloads stream with progress computed from the manifest's byte
counts rather than Content-Length (the service worker cache doesn't reliably
surface headers; the manifest always knows).

**Normalization**: `(raw/255 − mean)/std` with manifest stats — the fourth
and final appearance of lesson 06's One True Formula.

## The planner worker (`web/src/planner/`)

`cem.ts` ports lesson 09's planner faithfully — dense cost included (the
comment block retells the 20%→97% story so no future refactorer "simplifies"
it away) — with two browser-specific disciplines: a seedable RNG
(mulberry32 + Box–Muller) so tests are deterministic, and preallocated
`Float32Array`s reused across iterations, because the hot loop runs
~12k predictor calls per plan and GC pauses are user-visible.

The architectural decision worth studying is **what lives in the worker**:

- The predictor's ONNX session — sessions aren't transferable, so the worker
  builds its own from transferred model bytes. Planning (~230 ms of compute
  per MPC step on WASM) never touches the UI thread.
- **The lookup table.** Naively, the worker would return elite trajectories
  as latents for the main thread to visualize: 8 elites × 12 steps × 128
  floats per plan step. Instead the worker NN-decodes trajectories to (x, y)
  paths *itself* (with an early-exit distance loop) and posts back a few
  hundred floats. Move the computation to where the data is; ship only
  answers across the boundary.

`messages.ts` types the protocol; `client.ts` wraps it in promises. The
worker also answers `rollout` requests — lesson 08's open-loop imagination,
reused by the UI's divergence panel. The predictor exists in exactly one
place.

## The app (`App.svelte`)

One component, three canvases, and the MPC loop from lesson 09 driving the
real DOM: click → encode goal (teleport-render-restore, same pattern as the
Python evaluator) → plan → draw overlays (orange best path, blue elites,
purple imagined ghost — all lookup-decoded) → execute first action → repeat.
A `planEpoch` counter cancels stale loops when the user re-clicks or swaps
models mid-plan — the async-cancellation pattern in its simplest form.

**The checkpoint switcher** is lesson 10's fixed-PCA decision made
interactive: on switch, download+verify that checkpoint's models, rebuild
sessions, then re-encode a stride-5 sample of the lookup grid with the *new*
encoder and project through the *healthy* basis in the *fixed* viewport.
Healthy → the two-sheet cloud; collapsed → measured spread 11.53 → 0.0012, a
dot. The e2e suite asserts exactly that ratio.

**A bug worth remembering**: `manifest` was initially a plain variable, not
Svelte `$state`. Every test passed — unit, parity, e2e — because the e2e
drove the app through JS hooks. But the checkpoint `<select>`, whose options
render from `manifest` *after* async load, would have been permanently empty
for every real user. The build's `non_reactive_update` warning caught it.
Two morals: framework warnings about reactivity are load-bearing, and test
hooks that bypass the UI also bypass the UI's bugs.

## Loading UX

`public/sw.js`: a ~30-line service worker, cache-first for pinned HF URLs
only. Safe *because* of the pin — a URL containing a commit hash serves
immutable bytes forever, so there is no invalidation problem, the hardest
problem in caching having been deleted by upstream design. Revisits and
checkpoint re-switches hit cache instantly; the site works offline after
first load.

## Try it

1. Corrupt one hex digit of a sha256 in a local manifest copy (serve it via
   the dev server override of your choice, or temporarily edit
   `fetchVerified` to check against a wrong hash) and reload — watch the
   integrity path produce a *graceful* error state.
2. In DevTools → Application → Service Workers + Network: reload the live
   site twice and compare model fetch timings (network vs SW cache).
3. Read `worker.ts:decodeLatent` and explain the `if (dist >= bestDist) break`
   early exit: why is it correct (distances only grow within the loop), and
   roughly what speedup do you expect on 128-D vectors when most candidates
   are far?
4. Kill the reactivity fix in a scratch checkout (`let manifest: Manifest |
   null = null` without `$state`) and load the page: confirm the dropdown is
   empty while everything else "works". Feel the bug class viscerally.
