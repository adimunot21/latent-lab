# ONNX export, parity, and hosting

## Export (`export/to_onnx.py`)

Encoder and predictor export separately (opset 17), each with a **dynamic
batch dimension** — the planner batches hundreds of candidates through the
predictor, and lookup-table building batches the encoder. The export dummy
uses batch=2 so no dimension accidentally folds to a constant 1; a CI test
pushes batch 300 through a graph traced at batch 2 to keep that honest.

Weight-only dynamic-int8 variants are emitted alongside fp32. Sizes per
checkpoint: encoder 3.66 MB fp32 / 0.94 int8; predictor 0.53 / 0.14.

## Parity (`export/parity_test.py` + `tests/test_onnx_parity.py`)

Same inputs through PyTorch and onnxruntime, batch sizes 1/7/64, gate at
max-abs-diff < 1e-4 for fp32 (measured ~1e-6 on the real checkpoints). CI
runs the gate on **randomly-initialized** models — checkpoints aren't in git,
and parity is a property of the export path, not of particular weights. The
CLI variant checks real artifacts locally.

**Quantization verdict (recorded, and worth re-checking if models change):**
int8 encoder error 0.034 is negligible against latent scale ~1. int8
predictor error 0.24 is *not* — per-step latent deltas are ~4.6 and rollout
errors compound multiplicatively over horizon steps. `manifest.json` carries
a note: prefer the fp32 predictor (it's 0.53 MB; there is no size argument).
The int8 files exist for latency experiments, not as defaults.

## The bundle and manifest (`export/manifest.py`)

One folder, uploaded verbatim to the Hub: `manifest.json`, `models/<id>/*`
for four checkpoints (healthy, healthy_early, collapsed, collapsed_early),
and the lookup table as raw little-endian float32 `.bin` (a 1.5 MB array has
no business being JSON). The manifest is the Python↔browser contract:
normalization stats, env config, the fixed PCA basis, the checkpoint registry
with UI labels, and **sha256 + byte size for every file** — the browser
verifies each download and uses the byte counts for progress bars.

## Hugging Face hosting (`export/push_to_hub.py`)

Uploads the bundle with a model card and prints the resulting commit hash.
The browser pins that hash in `web/src/config.ts` and fetches via
`/resolve/<hash>/...` — immutable URLs, which is also why the service worker
can cache them forever. `hf.revision` inside the *uploaded* manifest is null
by necessity (a commit can't contain its own hash); the pin lives browser-side.

Gotcha for maintainers: the Hub namespace is **`adimunot`** (the HF account),
not `adimunot21` (the GitHub account). The token lives in `training/.env`
(gitignored).
