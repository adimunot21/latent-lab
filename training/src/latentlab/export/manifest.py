"""Assemble the browser bundle: manifest.json + models + lookup binaries.

Bundle layout (uploaded as-is to the HF Hub; the browser fetches from a
pinned revision):

    bundle/
      manifest.json               <- everything the browser needs to know
      models/<id>/encoder.onnx , predictor.onnx , *.int8.onnx
      lookup/states.bin           <- float32 LE (n_states, 2)
      lookup/latents.bin          <- float32 LE (n_states, latent_dim)
      README.md                   <- HF model card

manifest.json carries: model/env geometry, THE normalization formula inputs,
the fixed PCA projection (from the healthy encoder — a fixed basis is what
makes collapse visible when checkpoints are swapped), the checkpoint
registry with labels for the UI switcher, per-file byte sizes + sha256 for
integrity checks, recorded parity/quantization errors, and an ``hf.revision``
placeholder that is pinned after the first upload.

Usage:
    uv run python -m latentlab.export.manifest \
        --onnx-root export_out/onnx --latent-maps export_out/healthy_v1/latent_maps.npz \
        --out export_out/bundle
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from latentlab.data.dataset import load_norm_stats
from latentlab.envs.two_rooms import TwoRoomsConfig

MANIFEST_VERSION = 1

# Registry of demo checkpoints: id -> UI label + story. Order = UI order.
CHECKPOINTS: list[dict[str, str]] = [
    {
        "id": "healthy",
        "label": "Healthy (final)",
        "description": "MSE + SIGReg, 15 epochs. Plans at 97% success.",
    },
    {
        "id": "healthy_early",
        "label": "Healthy (epoch 1)",
        "description": "Same recipe, barely trained. Representations forming.",
    },
    {
        "id": "collapsed",
        "label": "No regularizer (final)",
        "description": "lambda_reg = 0. Fully collapsed: latent cloud is a point.",
    },
    {
        "id": "collapsed_early",
        "label": "No regularizer (epoch 1)",
        "description": "lambda_reg = 0, epoch 1. Collapse already underway.",
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path, bundle_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(bundle_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def build_bundle(
    onnx_root: Path,
    latent_maps_path: Path,
    data_dir: Path,
    out_dir: Path,
    parity_report: dict[str, dict[str, float]] | None = None,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ---- models ----
    model_entries: dict[str, dict[str, Any]] = {}
    for ckpt in CHECKPOINTS:
        src = onnx_root / ckpt["id"]
        if not src.exists():
            raise FileNotFoundError(f"missing ONNX dir for checkpoint '{ckpt['id']}': {src}")
        dst = out_dir / "models" / ckpt["id"]
        dst.mkdir(parents=True)
        files = {}
        for name in ("encoder.onnx", "predictor.onnx", "encoder.int8.onnx", "predictor.int8.onnx"):
            shutil.copy2(src / name, dst / name)
            files[name.replace(".onnx", "").replace(".", "_")] = file_entry(dst / name, out_dir)
        model_entries[ckpt["id"]] = {**ckpt, "files": files}

    # ---- lookup table (from the healthy encoder's latent maps) ----
    maps = np.load(latent_maps_path)
    lookup_dir = out_dir / "lookup"
    lookup_dir.mkdir()
    states = np.ascontiguousarray(maps["states"], dtype="<f4")
    latents = np.ascontiguousarray(maps["latents"], dtype="<f4")
    (lookup_dir / "states.bin").write_bytes(states.tobytes())
    (lookup_dir / "latents.bin").write_bytes(latents.tobytes())

    # ---- manifest ----
    env_config = TwoRoomsConfig()
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "latent_dim": int(latents.shape[1]),
        "frame_size": env_config.frame_size,
        "action_dim": 2,
        # normalized = (raw_uint8 / 255 - frame_mean) / frame_std
        "norm_stats": load_norm_stats(data_dir),
        "env_config": dataclasses.asdict(env_config),
        # Fixed PCA basis from the HEALTHY encoder. Do not refit per
        # checkpoint: a fixed basis is what makes collapse visible.
        "pca": {
            "components": maps["pca_components"].astype(float).tolist(),  # (D, 2)
            "mean": maps["pca_mean"].astype(float).tolist(),  # (D,)
            "explained_variance_ratio": maps["pca_explained"].astype(float).tolist(),
        },
        "lookup": {
            "n_states": int(states.shape[0]),
            "states": file_entry(lookup_dir / "states.bin", out_dir),
            "latents": file_entry(lookup_dir / "latents.bin", out_dir),
            "dtype": "float32-le",
        },
        "checkpoints": [model_entries[c["id"]] for c in CHECKPOINTS],
        "quantization_note": (
            "int8 predictor max-abs error ~0.24 vs per-step latent deltas ~4.6; "
            "compounds over rollouts. Prefer fp32 predictor (0.53 MB); int8 encoder ok."
        ),
        "parity": parity_report or {},
        "hf": {"repo_id": "adimunot21/latent-lab", "revision": None},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"bundle: {out_dir} ({total / 1e6:.1f} MB total)")
    print(f"manifest: {out_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-root", type=Path, required=True)
    parser.add_argument("--latent-maps", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_bundle(args.onnx_root, args.latent_maps, args.data, args.out)


if __name__ == "__main__":
    main()
