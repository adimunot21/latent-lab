# 13 — The engineering spine: gates, CI, deploy, trust

The previous twelve lessons each had a "and this is enforced by…" clause.
This closing lesson collects the enforcement layer itself — the part that
makes the project *stay* correct after everyone stops paying attention.

## The gate philosophy

Every risky boundary in the system has a mechanical gate, and the gates share
a design: **cheap to run, impossible to ignore, failing close to the cause.**

| boundary | gate | where it runs |
|---|---|---|
| data → training | `data/inspect.py` (exits nonzero) | manually, before any training (house rule) |
| Python env ↔ TS env | shared fixtures, bit/byte-exact replay | *both* CIs — `python.yml` and `web.yml` both watch `shared/**` |
| PyTorch ↔ ONNX | parity < 1e-4 (random weights) | python CI |
| exported bundle ↔ browser | sha256 per file at fetch time | every user's browser, every load |
| app behavior | Playwright e2e incl. planning success + collapse ratio | web CI, on real pinned models |

The double-registration of `shared/**` deserves a highlight: an env change
that regenerates fixtures breaks the *web* CI until the TS port is updated,
and a TS-only "fix" breaks against fixtures the *python* CI still validates.
Neither side can drift alone. That's what "treat parity drift as a build
failure" means operationally.

## CI design notes (`.github/workflows/`)

- **One lockfile, two torch flavors.** PyTorch ships as conflicting uv
  extras: `--extra gpu` resolves 2.6.0+cu124 locally, `--extra cpu` resolves
  2.6.0+cpu in CI — same `uv.lock`, same everything-else, so CI tests the
  dependency graph users actually get. (Pinned to 2.6.x after newer cu13x
  wheels shipped a broken cuDNN layout on the dev box — an afternoon of
  debugging preserved as a version constraint comment.)
- **Python pinned to 3.11** (`.python-version`) after CI resolved 3.12 and
  mypy choked on numpy's 3.12-flavored stubs. Interpreter drift between dev
  and CI is a whole bug class; one file deletes it.
- **e2e in CI fetches real models** from the pinned HF revision (~1 min
  total). Testing against the true artifact chain was judged worth the
  network dependency — the pin makes it reproducible, unlike testing against
  a moving `main`.
- The WebGPU Playwright project **self-skips** when no adapter exists
  (hosted runners, headless sandboxes) rather than failing or lying — WASM
  is the always-on gate, and the skip is visible in reports, not silent.

## The artifact trust chain

The deployed site is static files; the models arrive at runtime. The chain
that makes that safe:

1. `web/src/config.ts` pins a **revision** — a git commit hash on the HF
   repo. `/resolve/<hash>/...` URLs are immutable by construction.
2. That revision's `manifest.json` lists **sha256 + bytes** for every file.
3. `fetchVerified` refuses any byte-stream whose digest mismatches.
4. The service worker caches *only* those pinned URLs, cache-first — safe
   precisely because of (1); immutability deletes cache invalidation.

Rotating models is a procedure, not an adventure: new bundle → `push_to_hub`
prints the new hash → update `HF_REVISION` → deploy. Old deployments keep
working against old revisions forever. Secrets footprint: one HF write token
in a gitignored `.env`, used only at publish time; CI holds no secrets at
all.

## Deploy, and a war story

`deploy.yml` builds `web/` and publishes `dist/` via `actions/deploy-pages`.
The one non-textbook event: the **first** deployment failed with a generic
"Deployment failed, try again later" — the Pages site had been created via
API seconds earlier and was half-provisioned. A retry failed too; deleting
and recreating the Pages site fixed it permanently. Recorded in
PROJECTPLAN.md, because "generic platform error on first-ever deploy →
re-provision, don't debug your own config" is exactly the kind of knowledge
that evaporates.

Acceptance was verified the only way that counts: a **fresh browser context
against the public URL** — models fetched cold from HF, sha256-verified,
sessions warm, and a cross-room plan executed successfully in 7 steps, on
production.

## The reproducibility ledger

What it takes to rebuild this project from a bare clone, in order:
`uv sync` (lockfile) → `generate` (seeded: dataset is deterministic) →
`inspect` (gate) → `train` ×2 (seeded configs, checkpoints embed their
config) → `evaluate`/`report` (the README's numbers regenerate) →
`latent_maps` → `to_onnx` + parity → `manifest` → `push_to_hub` → update pin.
Every artifact that isn't in git is either derived by a seeded script or
hosted at a content-addressed pin. That property didn't happen by accident;
it was a golden rule from commit one.

## Course epilogue: the five lessons under the lessons

1. **Loss is not evidence.** The collapsed model had the best loss curve in
   the repo. Evaluate representations by what the system does with them.
2. **Know your metrics' invariances.** Scale-invariant metrics were blind to
   amplitude collapse — in numbers (R², rank) *and* in pictures
   (per-checkpoint PCA would have hidden the demo's own punchline).
3. **Optimizer confident + behavior incoherent ⇒ wrong objective.** The
   dense-cost fix came from instrumentation showing CEM succeeding at the
   wrong problem. Check the geometry before blaming the algorithm.
4. **Contracts before ports.** Bit-exact cross-language parity was cheap
   because the Python env was *written for it*; retrofitting it onto a
   numpy-float32, solver-based sim would have been miserable.
5. **Gate the boundaries.** Every silent-failure surface (data semantics,
   env drift, export fidelity, artifact integrity, UI reactivity) got a
   mechanical check that fails loudly and early. The ones that existed
   caught real bugs; the one that was missing (rendering the actual
   `<select>` from real state) let one through.

## Try it

1. Trip the double gate: change `door_half_height` to 0.13, regenerate
   fixtures, and run *both* test suites. Watch what fails where, then revert.
2. Audit the trust chain end-to-end in DevTools on the live site: find the
   pinned revision in the JS bundle, the manifest fetch, one model fetch,
   and confirm the SW cache entry. Write the chain down from memory.
3. Capstone: pick any constant in the system you now believe you understand
   (door width, latent dim, horizon, λ, batch size) and predict — in writing,
   with mechanisms — everything that changes if you double it. Then run the
   pipeline and grade yourself. That exercise *is* the course.
