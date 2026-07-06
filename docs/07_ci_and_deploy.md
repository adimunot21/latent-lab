# CI, deploy, security

## Pipelines (`.github/workflows/`)

- **python.yml** — uv sync (CPU torch via the `cpu` extra; same lockfile as
  the local GPU install), ruff lint + format check, mypy strict, pytest.
  Triggers on `training/**` *and* `shared/**` — an env change that
  invalidates parity fixtures must fail here too.
- **web.yml** — npm ci, eslint + prettier, svelte-check + vite build, vitest
  (includes the bit-exact env parity gate), then Playwright e2e on the
  chromium-wasm project (fetches real pinned models from the HF CDN;
  ~1 min total). The chromium-webgpu project exists but auto-skips where no
  GPU adapter is exposed — which is all hosted CI runners.
- **deploy.yml** — builds `web/` and publishes `web/dist` to GitHub Pages via
  actions/deploy-pages. Pages is configured with `build_type=workflow`. The
  deploy never touches model weights; those come from the Hub at runtime.

Non-obvious choices:

- **CPU/GPU torch as conflicting uv extras**: one `uv.lock` serves the
  GTX 1650 dev box (`--extra gpu` → 2.6.0+cu124) and CI (`--extra cpu`).
  Pinned to 2.6.x because newer cu13x wheels had a broken cuDNN layout on the
  dev machine — re-verify `torch.cuda.is_available()` before bumping.
- **ONNX parity in CI uses random weights**: checkpoints aren't in git, and
  the property under test is the export path, not the weights.
- **Python pinned to 3.11** (`training/.python-version`): CI once resolved
  3.12 and mypy choked on numpy's 3.12-flavored stubs under
  `python_version = 3.11`. One interpreter everywhere.

## Deploy & runtime security model

The site is static assets on GitHub Pages — no backend, no auth surface, no
secrets in the client. The trust chain for models:

1. `web/src/config.ts` pins an HF **revision** (commit hash). `/resolve/<hash>/`
   URLs are immutable.
2. `manifest.json` (fetched from that revision) lists sha256 + bytes for every
   artifact; `fetchVerified` checks the digest before a byte reaches
   onnxruntime.
3. `public/sw.js` caches only those pinned URLs, cache-first — safe *because*
   of immutability, and it makes revisits/offline work.

Rotating models = upload a new bundle (`export/push_to_hub.py`), take the
printed commit hash, update `HF_REVISION` in `web/src/config.ts`, ship. Old
deployments keep working against the old revision forever.

Secrets: only the HF write token, which lives in `training/.env` (gitignored)
and is needed exclusively for publishing bundles. CI needs no secrets at all.
