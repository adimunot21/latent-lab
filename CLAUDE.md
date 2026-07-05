# CLAUDE.md

Guidance for Claude Code working in this repo. Read this fully before acting. When in doubt, follow the rules here over general habits.

## What we're building

**latent-lab** — an interactive, browser-based playground for understanding JEPA (Joint-Embedding Predictive Architecture) world models. We train small action-conditioned JEPA world models in PyTorch on a "Two Rooms" navigation env, export to ONNX, and run the encoder + predictor + CEM planner + latent-space visualization live in the browser via WebGPU (WASM fallback). Think *TensorFlow Playground, but for JEPA world models and latent planning*.

Full roadmap and acceptance criteria live in `PROJECTPLAN.md`. That file is the source of truth for **what** to build and in **what order**. This file is the source of truth for **how** to work.

## Golden rules (non-negotiable)

1. **Data-validation gate before any training or processing code.** Before writing code that consumes a dataset, env output, or exported model, first write/run an inspection script that prints field names, shapes, dtypes, value ranges, and 2–3 raw examples (render frames to PNG where visual). Never assume data semantics — if a value is called `score`, verify its range. This rule exists because a past project wasted iterations on assumed formats. It is not optional.
2. **End-to-end sanity check before heavy runs.** Push one real sample through every pipeline stage and print shape/dtype/value at each stage before launching a long training run.
3. **The env exists twice and must stay in sync.** `two_rooms` is implemented in Python (training data) and TypeScript (live demo). A cross-language parity test against shared fixtures in `shared/fixtures/` must pass before the browser demo is trusted. Treat parity drift as a build failure.
4. **No browser-side training.** Collapse and training dynamics are shown by hot-swapping pre-trained checkpoints, never by training in-browser.
5. **Respect the 4GB VRAM ceiling.** Everything in Phases 0–7 must run on a GTX 1650 (4GB). If a config OOMs, shrink the model or use gradient accumulation — do not silently change the plan. RunPod is Phase 8 only.
6. **WASM fallback is first-class.** Every browser feature must work without WebGPU. Both paths are tested.
7. **Small, testable steps.** Never dump many files with no way to verify until all are in place. Get something visible working, then extend. Each step should be runnable and checkable immediately.
8. **Never tell the human to "download an artifact and move it into a folder."** Write files directly with the tools, or give exact `cat`/editor steps.
9. **GitHub repo is created on github.com FIRST**, then `git remote add origin` + push. Remind the human before the first push.

## Repository layout

```
latent-lab/
├── training/          # Python, uv-managed
│   ├── src/latentlab/{envs,data,models,probes,planning,export}/
│   ├── configs/       # yaml, including ablation/collapse configs
│   └── tests/
├── web/               # TypeScript, Vite + Svelte
│   ├── src/{env,inference,planner,viz,ui}/
│   ├── public/        # manifest.json; model weights fetched from HF CDN
│   └── tests/ + e2e/
├── shared/fixtures/   # cross-language env parity trajectories (JSON)
├── docs/              # codebase walkthrough (written at ship time)
└── .github/workflows/ # python.yml, web.yml, deploy.yml
```

## Commands

Python (run from `training/`):
- Install/sync: `uv sync`
- Run anything: `uv run <cmd>` (e.g. `uv run python -m latentlab.data.generate`)
- Test: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Types: `uv run mypy src`

Web (run from `web/`):
- Install: `npm ci`
- Dev server: `npm run dev`
- Build: `npm run build`
- Unit tests: `npm run test` (vitest)
- E2E: `npm run test:e2e` (playwright)
- Lint/format: `npm run lint && npm run format`

Keep these commands accurate — if you add a script, update this section in the same change.

## Code style

**General:** working code over clever code; use established libraries; config in one place (yaml/env/top-of-file), no magic numbers scattered around; handle errors and edge cases; brief comments only for non-obvious decisions.

**Python:** ruff + mypy clean, type hints on public functions, no bare `except`, dataclasses/pydantic for configs, pathlib over os.path. Keep model definitions, losses, and training loop in separate modules.

**TypeScript:** strict mode on, eslint + prettier clean, no `any` unless justified with a comment, keep the planner hot loop allocation-free where practical (it runs many predictor calls per plan), all ONNX inference behind a typed wrapper in `inference/`.

## Testing requirements

- Every env method (collision, determinism, render) has unit tests.
- ONNX export has a **parity test**: same inputs through PyTorch and onnxruntime agree within tolerance (fp32 < 1e-4). This runs in CI.
- Env parity (Python vs TS) tested from `shared/fixtures/` in both CI pipelines.
- Planning success rate has an automated eval (Python), and an in-browser Playwright eval in Phase 6.
- Don't mark a phase done until its acceptance criteria in `PROJECTPLAN.md` pass.

## Git & CI

- Conventional commits (`feat:`, `fix:`, `chore:`, `test:`, `docs:`).
- Commit after each working milestone with a descriptive message; suggest the commit to the human, don't auto-push without saying so.
- CI (GitHub Actions) must be green to merge: Python (pytest/ruff/mypy) + Web (vitest/eslint/build) + parity tests.
- `pip-audit` / `npm audit` and Dependabot enabled.
- Large files (datasets, weights) are **not** git-tracked. Datasets are regenerated via script; weights live on Hugging Face Hub. Document how to obtain both.

## Security

- No backend. Site is static assets → no auth surface.
- No secrets in the repo. HF push token only via local `.env` (gitignored) and GitHub Actions secrets.
- Browser fetches model weights from a **pinned HF revision (commit hash)**, never a moving `main`.
- `manifest.json` integrity-checked; service-worker caching.

## How to work with me (the human)

- Be direct and concise. Explain a decision in one sentence when it affects what I need to run; don't teach theory unless I ask.
- Give me exact, copy-paste terminal commands for install/run/build/deploy.
- Tell me what to expect before I run something (output, behavior, URL).
- When something breaks, diagnose from my actual error output — don't guess. Give the minimal fix, not a full-file rewrite for a one-line bug.
- If you told me to do 3 tasks and I got stuck on task 1, after we fix it, confirm 2 and 3 are done before moving to task 4. Don't skip them.
- If a feature is unreasonably complex for the scope, say so and propose a simpler alternative.
- Build momentum — after a working milestone, hand me the next step; don't ask permission at every turn.

## Current status

Track progress in `PROJECTPLAN.md` by checking off items. At the start of each session, read the plan, report which phase we're in and what's next, then proceed.
