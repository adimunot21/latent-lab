# latent-lab (web)

Browser playground for JEPA world models: runs the ONNX encoder + predictor and
a CEM latent planner live via [onnxruntime-web](https://onnxruntime.ai/) (WebGPU
with a first-class WASM fallback), and visualizes the latent space. Vite + Svelte
5 + TypeScript (strict).

## Quickstart

```bash
npm ci            # install (or `npm install` to (re)generate the lockfile)
npm run dev       # dev server
npm run build     # type-check + production build
npm run test      # unit tests (vitest)
npm run test:e2e  # end-to-end (playwright) — wired up from Phase 6
npm run lint      # eslint + prettier --check
npm run format    # prettier --write
```

> Node is provided via a dedicated conda env: `conda activate latent-lab-node`
> (Node 22). See the repo root `README.md`.

Model weights are fetched at runtime from a pinned Hugging Face revision (not
git-tracked). See `PROJECTPLAN.md` for the roadmap.
