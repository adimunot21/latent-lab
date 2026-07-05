# latentlab (training)

Python training pipeline for **latent-lab**: the Two Rooms env, offline dataset
generation, the action-conditioned JEPA world model, latent CEM planning, and
ONNX export. Managed with [uv](https://docs.astral.sh/uv/).

## Quickstart

```bash
uv sync --extra gpu --extra dev   # venv + deps with CUDA torch (GTX 1650)
uv run pytest                     # tests
uv run ruff check .               # lint
uv run ruff format .              # format
uv run mypy src                   # types
```

PyTorch is split into mutually-exclusive `cpu`/`gpu` extras so one lockfile
serves both machines: use `--extra gpu` locally (CUDA), and `--extra cpu` in CI
(no GPU). Both pin the same torch version (2.6.x).

See the repo root `PROJECTPLAN.md` for the phase-by-phase roadmap and
`CLAUDE.md` for working conventions (data-validation gate, VRAM ceiling, etc.).
