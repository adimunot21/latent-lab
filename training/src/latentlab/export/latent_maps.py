"""Latent<->state lookup table + PCA projection for decoder-free visualization.

The browser never decodes latents to pixels (no decoder exists — that's the
JEPA point). Instead it visualizes imagined latents two ways:
1. Nearest-neighbor against a LOOKUP TABLE of (state, latent) pairs built from
   a dense grid of free positions -> draw the matched (x, y) on the map.
2. A fixed 2-D PCA PROJECTION of latent space -> draw the latent cloud panel
   (this is where collapse becomes visible when checkpoints are hot-swapped).

Both artifacts are computed here from a trained encoder and saved as .npz
(+ sanity PNGs). Phase 4 packs them into the browser manifest.

Usage:
    uv run python -m latentlab.export.latent_maps \
        --checkpoint checkpoints/two_rooms_v1/healthy_v1/final.pt --out export_out/healthy_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from latentlab.data.dataset import TwoRoomsTransitions, load_norm_stats, normalize_frames
from latentlab.envs.two_rooms import TwoRoomsConfig, TwoRoomsEnv
from latentlab.models.encoder import Encoder
from latentlab.train import load_checkpoint_models, normalized_frames_tensor


def grid_free_states(env: TwoRoomsEnv, resolution: int = 64) -> npt.NDArray[np.float32]:
    """All collision-free positions on a resolution x resolution grid."""
    coords = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    states = [(x, y) for y in coords for x in coords if env.is_free(float(x), float(y))]
    return np.asarray(states, dtype=np.float32)


@torch.no_grad()
def encode_states(
    encoder: Encoder,
    env: TwoRoomsEnv,
    states: npt.NDArray[np.float32],
    stats: dict[str, float],
    batch_size: int = 256,
) -> npt.NDArray[np.float32]:
    """Render + encode each state -> (N, D) latents."""
    device = next(encoder.parameters()).device
    latents = []
    for start in range(0, len(states), batch_size):
        chunk = states[start : start + batch_size]
        frames = np.stack([_render_at(env, float(x), float(y)) for x, y in chunk])
        z = encoder(normalize_frames(frames, stats).to(device))
        latents.append(z.cpu().numpy())
    return np.concatenate(latents).astype(np.float32)


def _render_at(env: TwoRoomsEnv, x: float, y: float) -> npt.NDArray[np.uint8]:
    env.set_state(x, y)
    return env.render()


def fit_pca(
    latents: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Top-2 PCA. Returns (components (D, 2), mean (D,), explained_var_ratio (2,))."""
    z = torch.from_numpy(latents)
    mean = z.mean(dim=0)
    centered = z - mean
    _, s, v = torch.linalg.svd(centered, full_matrices=False)
    components = v[:2].T.contiguous()  # (D, 2)
    explained = (s**2) / (s**2).sum()
    return (
        components.numpy().astype(np.float32),
        mean.numpy().astype(np.float32),
        explained[:2].numpy().astype(np.float32),
    )


def sanity_check_lookup(
    lookup_states: npt.NDArray[np.float32],
    lookup_latents: npt.NDArray[np.float32],
    encoder: Encoder,
    data_dir: Path,
    out_dir: Path,
) -> float:
    """Nearest-neighbor-decode a real eval trajectory; return mean position error.

    Also writes a PNG comparing the true path against the decoded path.
    """
    eval_ds = TwoRoomsTransitions(data_dir, episode_range=(900, 902))
    frames = normalized_frames_tensor(eval_ds)[0]  # (T+1, 1, H, W)
    true_states = eval_ds.states[0]  # (T+1, 2)
    device = next(encoder.parameters()).device
    with torch.no_grad():
        z_traj = encoder(frames.to(device)).cpu().numpy()

    table = torch.from_numpy(lookup_latents)
    decoded = []
    for z in z_traj:
        dist = ((table - torch.from_numpy(z)) ** 2).sum(dim=1)
        decoded.append(lookup_states[int(dist.argmin().item())])
    decoded_arr = np.asarray(decoded)
    error = float(np.linalg.norm(decoded_arr - true_states, axis=1).mean())

    # Visual: true path (white) vs decoded path (gray) on a 256px canvas.
    canvas = np.zeros((256, 256), dtype=np.uint8)
    for pts, intensity in [(true_states, 255), (decoded_arr, 140)]:
        for x, y in pts:
            px, py = int(x * 255), int(y * 255)
            canvas[max(py - 1, 0) : py + 2, max(px - 1, 0) : px + 2] = intensity
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_dir / "lookup_sanity_path.png")
    return error


def pca_scatter_png(
    pca_xy: npt.NDArray[np.float32],
    states: npt.NDArray[np.float32],
    path: Path,
) -> None:
    """Scatter of PCA-projected grid latents, colored by room side (x < 0.5)."""
    lo, hi = pca_xy.min(axis=0), pca_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (pca_xy - lo) / span
    canvas = np.zeros((256, 256), dtype=np.uint8)
    for (px, py), (sx, _sy) in zip(norm, states, strict=True):
        cx, cy = int(px * 255), int((1 - py) * 255)
        canvas[max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2] = 255 if sx < 0.5 else 128
    Image.fromarray(canvas).save(path)


def build_latent_maps(checkpoint: Path, data_dir: Path, out_dir: Path, resolution: int) -> None:
    encoder, _, _ = load_checkpoint_models(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    stats = load_norm_stats(data_dir)
    env = TwoRoomsEnv(config=TwoRoomsConfig(), seed=0)

    states = grid_free_states(env, resolution)
    print(f"grid: {len(states)} free states at resolution {resolution}")
    latents = encode_states(encoder, env, states, stats)
    print(f"latents: {latents.shape} {latents.dtype}")

    components, mean, explained = fit_pca(latents)
    pca_xy = ((latents - mean) @ components).astype(np.float32)
    print(f"PCA explained variance (top 2): {explained.tolist()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "latent_maps.npz",
        states=states,
        latents=latents,
        pca_components=components,
        pca_mean=mean,
        pca_explained=explained,
    )
    meta = {
        "checkpoint": str(checkpoint),
        "grid_resolution": resolution,
        "n_states": int(len(states)),
        "latent_dim": int(latents.shape[1]),
        "pca_explained": [float(v) for v in explained],
        "norm_stats": stats,
    }
    (out_dir / "latent_maps_meta.json").write_text(json.dumps(meta, indent=2))

    error = sanity_check_lookup(states, latents, encoder, data_dir, out_dir)
    print(f"lookup sanity: mean NN-decode position error {error:.4f} world units")
    pca_scatter_png(pca_xy, states, out_dir / "pca_scatter.png")
    print(f"wrote {out_dir}/latent_maps.npz, lookup_sanity_path.png, pca_scatter.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    build_latent_maps(args.checkpoint, args.data, args.out, args.resolution)


if __name__ == "__main__":
    main()
