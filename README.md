# latent-lab

> An interactive, browser-based playground for understanding **JEPA**
> (Joint-Embedding Predictive Architecture) world models — *TensorFlow
> Playground, but for JEPA world models and latent planning*.

We train small action-conditioned JEPA world models in PyTorch on a **Two Rooms**
navigation environment, export them to ONNX, and run the encoder + predictor +
CEM planner + latent-space visualization **live in the browser** via WebGPU
(with a first-class WASM fallback).

🚧 **Status:** Phase 0 (scaffold). See [`PROJECTPLAN.md`](PROJECTPLAN.md) for the
full roadmap and [`CLAUDE.md`](CLAUDE.md) for working conventions.

## Repository layout

```
training/   Python (uv) — env, dataset, JEPA model, planning, ONNX export
web/        TypeScript (Vite + Svelte) — in-browser inference, planner, viz
shared/     cross-language env-parity fixtures (Python ↔ TypeScript)
docs/       component walkthroughs (written at ship time)
```

## Prerequisites

- **Python** tooling via [uv](https://docs.astral.sh/uv/).
- **Node 22** via the project's conda env (Node is not installed globally):
  ```bash
  conda activate latent-lab-node    # env created in Phase 0
  ```
  Recreate it if needed: `mamba create -n latent-lab-node -c conda-forge nodejs=22`.

## Develop

Python (from `training/`):

```bash
uv sync --extra gpu --extra dev            # CUDA wheels locally (GTX 1650)
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Web (from `web/`, with the conda env active):

```bash
npm ci
npm run dev            # dev server
npm run build          # type-check + production build
npm run test           # unit tests (vitest)
npm run lint           # eslint + prettier
```

## Data & weights

Datasets are **regenerated via script** (Phase 1) and model weights live on the
**Hugging Face Hub** (Phase 4) — neither is git-tracked. Instructions land with
those phases.

## License

[Apache-2.0](LICENSE).
