# PROJECTPLAN.md — latent-lab

Executable roadmap. Work top to bottom. Do not start a phase until the previous phase's **Acceptance** boxes are all checked. Check boxes as you complete them and reference this file at the start of every session.

Read `CLAUDE.md` for how-to-work rules (data-validation gate, env parity, VRAM ceiling, commit discipline). This file is the what/when.

---

## Context recap (so this file stands alone)

We build a browser playground for JEPA world models. Pipeline: custom **Two Rooms** navigation env → offline trajectory dataset → **action-conditioned JEPA world model** (CNN encoder + predictor, next-embedding MSE + SIGReg anti-collapse loss, no EMA / no stop-gradient) → probes + **CEM latent planner** → **ONNX export** → **Hugging Face Hub** → **browser app** (onnxruntime-web, WebGPU + WASM fallback) that runs encoder/predictor/planner live and visualizes the latent space, planning, and representation collapse.

**Definition of done (whole project):**
- [ ] Two Rooms JEPA world model, ≥90% planning success in Python eval
- [ ] ONNX export passing parity tests, hosted on HF Hub with model cards
- [ ] Public static site: goal-drag planning + latent panel + collapse checkpoint switcher, working on WebGPU and WASM
- [ ] Green CI (Python + Web + parity), README + `docs/` walkthrough

---

## Phase 0 — Environment & Repo Setup
Goal: a clean scaffold with green CI and nothing else.

- [x] Confirm project name availability (`latent-lab`) on GitHub before first commit
- [x] Create monorepo structure (see `CLAUDE.md` layout) with `mkdir -p`; show final tree
- [x] `training/`: init `uv` project, add torch (CUDA build for local 1650), numpy, onnx, onnxruntime, huggingface_hub, pytest, ruff, mypy, tensorboard (or wandb). Pin versions.
- [x] `web/`: init Vite + Svelte + TypeScript (strict), add onnxruntime-web, vitest, playwright, eslint, prettier
- [x] `.gitignore` (Python + Node + data/weights/artifacts), `.env.example`, `LICENSE` (Apache-2.0), `README.md` stub
- [x] `.pre-commit-config.yaml`: ruff, ruff-format, mypy, eslint, prettier
- [x] `.github/workflows/`: `python.yml` (sync/lint/type/test), `web.yml` (ci/lint/build/test), `deploy.yml` (stub, disabled)
- [x] `git init`, initial commit. **Remind human to create GitHub repo on github.com first**, then `git remote add origin` + push.

**Acceptance:**
- [x] `uv run pytest` passes (one trivial test) and `uv run ruff check .` clean
- [x] `npm run build` and `npm run test` pass (one trivial test)
- [x] CI green on the initial commit

## Phase 1 — Two Rooms Env + Data Generation (Python)
Goal: a validated offline trajectory dataset.

- [x] Implement `envs/two_rooms.py`: state = agent (x,y); action = (dx,dy) clamped; two rooms + wall with a doorway; `reset()`, `step(action)`, `render()` → 64×64 (grayscale or RGB) frame. Deterministic given seed.
- [x] Unit tests: wall collision blocks movement, doorway passable, determinism (same seed+actions → same trajectory), render shape/dtype/range
- [x] `data/generate.py`: rollout policy (mix of random + goal-directed) → ~50–100k transitions `(frame_t, action_t, frame_{t+1})` saved as sharded `.npz` (not git-tracked). Include state (x,y) alongside frames for probing. *(1000 eps × 60 steps = 60k transitions, `data/two_rooms_v1`, seed 0)*
- [x] **DATA-VALIDATION GATE:** `data/inspect.py` prints field names, shapes, dtypes, action min/max/mean, frame value range, and writes 2–3 sample transition PNGs to a scratch dir. Run it and confirm sane before any training code. *(GATE PASSED: all invariants ok, 302 door crossings in shard 0, PNGs visually verified)*
- [x] `data/dataset.py`: torch Dataset/DataLoader reading shards, with normalization stats computed and saved *(frame_mean=0.0314, frame_std=0.1366 over frames/255)*

**Acceptance:**
- [x] Env unit tests pass *(14 env tests + 4 pipeline tests)*
- [x] `inspect.py` output reviewed; frames look like the env; actions in expected range *(±0.08, symmetric)*
- [x] DataLoader yields a batch with correct shapes/dtypes (printed) *(frame (64,1,64,64) f32, action (64,2) f32)*

## Phase 2 — Train the JEPA World Model (Python)
Goal: a model with useful representations + a deliberately collapsed counterpart.

- [x] `models/encoder.py`: small CNN (pixels → latent z, dim ~128–256). `models/predictor.py`: (z_t, a_t) → ẑ_{t+1} (MLP or small transformer). *(GroupNorm CNN 0.91M → z∈R¹²⁸; residual MLP predictor 0.13M)*
- [x] `models/losses.py`: next-embedding MSE + **SIGReg** (isotropic-Gaussian regularizer; sketched/random-projection form). Single `lambda_reg` knob. *(Epps–Pulley CF statistic on random 1-D projections)*
- [x] `train.py`: AMP, config-driven (yaml in `configs/`), TensorBoard/W&B logging. **End-to-end sanity check first** (one batch through every stage, print shapes/values) before the full run. *(sanity stage built into every run; `--sanity-only` flag)*
- [x] Built-in diagnostics logged every N steps: latent std + effective rank (collapse detector), k-step open-loop rollout MSE.
- [x] `probes/linear_probe.py`: frozen z → agent (x,y) linear regression; report R²/MSE. `probes/collapse_metrics.py`, `probes/rollout.py`. *(+ probes/report.py comparison table)*
- [x] Train the **healthy** model. Then train **ablation/collapse checkpoints**: `lambda_reg = 0` saved at a few epochs (early → collapsed), plus early/late healthy checkpoints. Save all under a versioned dir. *(checkpoints/two_rooms_v1/{healthy_v1,collapse_v1}/, ~6 min per run)*
- [x] Confirm training fits in 4GB (report peak VRAM). If not: shrink model / grad-accum, note it. *(peak VRAM 0.41 GB)*

**Acceptance:**
- [x] Healthy model: linear position probe R² above an agreed threshold (record it) *(R² = 0.9997 held-out; threshold 0.9)*
- [x] Collapse run: latent std/rank visibly collapse in logs (this is a *feature* — the checkpoint is a deliverable) *(z_std 1.163 → 0.001, a 1000× amplitude collapse. Nuance recorded: eff_rank and probe R² are scale-invariant and stay deceptively moderate — std + downstream utility are the honest detectors)*
- [x] Rollout error is low for healthy, high for collapsed *(8-step probed position error: 0.098 vs 0.290 world units; collapsed latent-MSE misleadingly ~0 — documented in probes/report.py)*
- [x] Checkpoints saved with configs *(model+optimizer+config+step in each .pt, config.yaml per run dir)*

## Phase 3 — Latent Planning in Python
Goal: solve navigation by planning in latent space.

- [x] `planning/cem.py`: CEM over action sequences; cost = latent distance between predicted rollout end-state and encoded goal frame; horizon/population/iterations configurable *(NOTE: endpoint-only cost made the first action underdetermined → MPC random-walked at 20% success. Switched to dense trajectory cost (sum over all imagined steps + 4× terminal weight) → 97%. Rationale documented in cem.py)*
- [x] `planning/evaluate.py`: N random (start, goal) episodes → success rate + steps-to-goal; save planning GIFs *(eval_out/<run>/: results.json + GIFs; cross-room GIF visually verified — agent routes through the door)*
- [x] Build + save the **latent↔state lookup table** and **PCA projection matrix** (from a latent sample) for the browser's decoder-free visualization *(export/latent_maps.py: 2880-state grid → latent_maps.npz; PCA top-2 explains 71.5%)*
- [x] Sanity-check the lookup/PCA visually (does a latent trajectory map to a sensible path?) *(NN-decode error 0.0063 world units — sub-pixel; PCA scatter shows two room-sheets joined at a doorway neck)*

**Acceptance:**
- [x] Planning success rate ≥ 90% on Two Rooms (record exact number) *(97% overall: 94% cross-room, 100% same-room, mean 6.7 steps, N=100, seed 0)*
- [x] Collapsed checkpoint plans poorly (contrast documented) *(44% at N=50 — mostly lucky wandering; mean steps double at 13.5)*
- [x] PCA matrix + lookup table exported *(export_out/healthy_v1/latent_maps.npz + meta.json)*

## Phase 4 — ONNX Export + Parity
Goal: portable, verified inference artifacts.

- [ ] `export/to_onnx.py`: export `encoder.onnx` and `predictor.onnx` with **dynamic batch dim** (planner batches candidates). Also emit int8-quantized variants.
- [ ] `export/manifest.py`: `manifest.json` = latent dim, input normalization stats, PCA matrix, env config, checkpoint list + labels, HF revision placeholder
- [ ] `export/parity_test.py`: identical inputs through PyTorch vs onnxruntime → max abs diff < 1e-4 (fp32); record quantized error too. Wire into CI.
- [ ] Push models + `manifest.json` to Hugging Face Hub with **model cards** (what/how trained/limitations); note the pinned commit hash

**Acceptance:**
- [ ] Parity test passes in CI
- [ ] Models load from a pinned HF revision via a throwaway script
- [ ] Quantized-vs-fp32 error recorded and acceptable

## Phase 5 — Browser Core (TypeScript)
Goal: first visible in-browser wow moment.

- [ ] `env/twoRooms.ts`: port of the Python env + canvas renderer
- [ ] **ENV PARITY GATE:** generate shared trajectory fixtures in `shared/fixtures/` (fixed seed + action sequence + resulting states) from Python; test that TS reproduces them exactly. Gate the phase on this.
- [ ] `inference/session.ts`: onnxruntime-web session with WebGPU detect → WASM fallback; warmup run; typed wrappers `encoder.ts`, `predictor.ts`
- [ ] Load models from pinned HF revision + `manifest.json`; apply normalization from manifest
- [ ] Wire arrow-key control: drive agent, encode current frame, show live latent PCA dot moving in a panel

**Acceptance:**
- [ ] Env parity test passes in web CI
- [ ] Page loads models, shows WebGPU/WASM badge
- [ ] Driving the agent moves the latent dot sensibly (screenshot/GIF)

## Phase 6 — Playground Features
Goal: the actual product.

- [ ] Goal-drag → CEM planning loop in a **Web Worker**; animate candidate action trajectories; render chosen path; agent executes and re-plans (MPC loop)
- [ ] Candidate/rollout visualization uses the latent↔state lookup (decoder-free)
- [ ] **Checkpoint switcher:** healthy / no-regularizer / early / late → latent-cloud panel updates so users *watch collapse* (same inputs, cloud shrinks to a point)
- [ ] **Imagination panel:** k-step open-loop latent rollout vs ground-truth divergence
- [ ] Controls: CEM population / iterations / horizon sliders; latency readout; explainer-mode annotations
- [ ] Playwright e2e: automated in-browser planning success measurement

**Acceptance:**
- [ ] In-browser planning success rate measured and close to Python eval
- [ ] Collapse is visually obvious when switching checkpoints
- [ ] Works on both WebGPU and WASM (both tested in Playwright)

## Phase 7 — Polish, Deploy, Ship
- [ ] Enable `deploy.yml` → Cloudflare Pages (or GH Pages); public URL
- [ ] Model-loading UX: progress bar, service-worker caching, graceful errors
- [ ] README: what it is, live link, GIFs, local dev, how to regenerate data / obtain weights
- [ ] `docs/` walkthrough (see below), README links to it
- [ ] Finalize HF model cards with the live-site link

**Acceptance:**
- [ ] Public URL works cold (fresh browser, no cache)
- [ ] README + docs complete; CI (incl. deploy) green

## Phase 8 — Stretch: PushT Track (RunPod) — only if 0–7 shipped
- [ ] PushT dataset gen (pymunk), larger model trained on RunPod
- [ ] Browser dynamics via planck.js or precomputed-dynamics approach
- [ ] Decide scope at this point; do not start before Phase 7 ships

---

## docs/ walkthrough (write in Phase 7)
One markdown file per major component — a maintainer's guide, not a tutorial. For each: what it does + why it exists, why this approach over alternatives (one paragraph), a walk through the non-obvious decisions, how it connects to the rest, and gotchas/tradeoffs. Suggested files:
- `01_architecture.md` — the dual-env design, data flow, why decoder-free
- `02_env_and_data.md` — Two Rooms, dataset format, the validation gate
- `03_model_and_training.md` — encoder/predictor, MSE + SIGReg, collapse
- `04_planning.md` — CEM, latent cost, the lookup/PCA trick
- `05_export.md` — ONNX, dynamic batch, parity, HF hosting
- `06_web.md` — onnxruntime-web, WebGPU/WASM, the worker planner, viz
- `07_ci_and_deploy.md` — pipelines, parity gates, static deploy, security notes

## Open questions to resolve early (don't let these block silently)
1. Env parity drift → fixtures gate (Phase 5). 2. CEM browser latency → batch candidates, quantize, worker; benchmark in Phase 5 before UI. 3. 4GB ceiling for SIGReg batch stats → grad-accum or RunPod; test Phase 2 week 1. 4. Decoder-free imagined-latent viz → lookup table; validate Phase 3. 5. WebGPU coverage → WASM first-class. 6. Name availability → Phase 0.
