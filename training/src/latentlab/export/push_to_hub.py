"""Upload the browser bundle to the Hugging Face Hub and pin the revision.

Writes a model card (README.md) into the bundle, uploads the folder to the
model repo, then fetches the resulting commit hash and re-writes
``manifest.json``'s ``hf.revision`` locally so the browser config can pin it.
(The manifest inside THAT upload keeps revision=null — the hash of a commit
can't be known before the commit exists; the browser takes the pin from its
own config, not from the manifest.)

Requires HF_TOKEN in the environment or in training/.env (gitignored).

Usage:
    uv run python -m latentlab.export.push_to_hub --bundle export_out/bundle
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

MODEL_CARD = """\
---
license: apache-2.0
tags:
  - jepa
  - world-model
  - onnx
  - reinforcement-learning
library_name: onnx
---

# latent-lab — Two Rooms JEPA world models (ONNX)

Action-conditioned JEPA world models trained on a "Two Rooms" navigation
environment, exported to ONNX for in-browser inference (onnxruntime-web,
WebGPU/WASM). Part of [latent-lab](https://github.com/adimunot21/latent-lab),
an interactive playground for understanding JEPA world models and latent
planning.

## What's here

- `models/<id>/encoder.onnx` — CNN encoder: 64x64 grayscale frame -> 128-d latent (0.91M params)
- `models/<id>/predictor.onnx` — residual MLP: (latent, action) -> next latent (0.13M params)
- `models/<id>/*.int8.onnx` — weight-only dynamic-int8 variants
- `lookup/{states,latents}.bin` — latent<->state lookup table (float32 LE) for decoder-free visualization
- `manifest.json` — normalization stats, env config, PCA projection, per-file sha256

Checkpoints: `healthy` (MSE + SIGReg, 97% planning success), `healthy_early`
(epoch 1), `collapsed` (lambda_reg = 0 — deliberate representation collapse,
a demo feature), `collapsed_early`.

## How they were trained

Joint-embedding predictive architecture with NO EMA target and NO
stop-gradient; collapse is prevented solely by SIGReg (Epps-Pulley
characteristic-function statistic on random 1-D projections of the latent
batch, pushing toward an isotropic Gaussian). Next-embedding MSE + SIGReg,
AdamW, AMP, 15 epochs on 60k transitions from a scripted mixed
random/goal-directed policy. Trained on a single GTX 1650 (peak VRAM 0.41 GB).

Recorded metrics (held-out): healthy linear position probe R^2 = 0.9997;
CEM planning success 97% (N=100). Collapsed: latent std 0.001 (vs 1.16
healthy), planning 44%. fp32 ONNX parity vs PyTorch < 2e-6 max abs diff;
int8 errors recorded in `manifest.json`.

## Limitations

- Toy environment: a 2-DoF point agent; these weights model nothing else.
- The int8 predictor's quantization error (~0.24 max abs) compounds over
  multi-step rollouts; prefer the fp32 predictor.
- Deterministic env: the world model has never seen stochastic dynamics.

## Use

Fetch from a **pinned revision** (see the latent-lab site config for the
current pin), verify sha256 against `manifest.json`, run with onnxruntime.
Input normalization: `(uint8_frame / 255 - frame_mean) / frame_std` with the
stats in `manifest.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-id", default="adimunot/latent-lab")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                    token = line.split("=", 1)[1].strip()
    if not token:
        raise SystemExit("HF_TOKEN not set (env var or training/.env). Aborting.")

    (args.bundle / "README.md").write_text(MODEL_CARD)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload Two Rooms JEPA bundle (models + lookup + manifest)",
    )
    revision = commit.oid
    print(f"uploaded to https://huggingface.co/{args.repo_id}")
    print(f"pinned revision: {revision}")

    # Record the pin locally so the web config can pick it up.
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hf"]["revision"] = revision
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"local manifest updated with revision: {manifest_path}")


if __name__ == "__main__":
    main()
