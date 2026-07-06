"""ONNX parity check: identical inputs through PyTorch and onnxruntime.

fp32 exports must agree with PyTorch to < 1e-4 max abs diff (gate). int8
errors are recorded for the quantization-quality decision, not gated here.
Runs with multiple batch sizes to exercise the dynamic batch dim.

The reusable checks live here; CI runs them via tests/test_onnx_parity.py on
randomly-initialized models (checkpoints are not in git — parity is a
property of the export path, not of particular weights). This CLI checks
real exported artifacts:

    uv run python -m latentlab.export.parity_test \
        --checkpoint checkpoints/two_rooms_v1/healthy_v1/final.pt \
        --onnx-dir export_out/onnx/healthy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from latentlab.models.encoder import Encoder
from latentlab.models.predictor import Predictor
from latentlab.train import load_checkpoint_models

FP32_TOLERANCE = 1e-4
BATCH_SIZES = (1, 7, 64)  # exercise the dynamic batch dim


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@torch.no_grad()
def encoder_max_diff(encoder: Encoder, onnx_path: Path, frame_size: int = 64) -> float:
    session = _session(onnx_path)
    worst = 0.0
    rng = np.random.default_rng(0)
    for batch in BATCH_SIZES:
        frames = rng.standard_normal((batch, 1, frame_size, frame_size)).astype(np.float32)
        torch_out = encoder(torch.from_numpy(frames)).numpy()
        ort_out = session.run(None, {"frame": frames})[0]
        worst = max(worst, float(np.abs(torch_out - ort_out).max()))
    return worst


@torch.no_grad()
def predictor_max_diff(predictor: Predictor, onnx_path: Path) -> float:
    session = _session(onnx_path)
    worst = 0.0
    rng = np.random.default_rng(1)
    for batch in BATCH_SIZES:
        z = rng.standard_normal((batch, predictor.latent_dim)).astype(np.float32)
        action = (rng.standard_normal((batch, predictor.action_dim)) * 0.08).astype(np.float32)
        torch_out = predictor(torch.from_numpy(z), torch.from_numpy(action)).numpy()
        ort_out = session.run(None, {"latent": z, "action": action})[0]
        worst = max(worst, float(np.abs(torch_out - ort_out).max()))
    return worst


def check_parity(encoder: Encoder, predictor: Predictor, onnx_dir: Path) -> dict[str, float]:
    """All four artifacts vs PyTorch. Missing int8 files are skipped."""
    report = {
        "encoder_fp32": encoder_max_diff(encoder, onnx_dir / "encoder.onnx"),
        "predictor_fp32": predictor_max_diff(predictor, onnx_dir / "predictor.onnx"),
    }
    for name, fn, model in [
        ("encoder_int8", encoder_max_diff, encoder),
        ("predictor_int8", predictor_max_diff, predictor),
    ]:
        path = onnx_dir / f"{name.split('_')[0]}.int8.onnx"
        if path.exists():
            report[name] = fn(model, path)  # type: ignore[operator]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    args = parser.parse_args()

    encoder, predictor, _ = load_checkpoint_models(args.checkpoint)
    report = check_parity(encoder, predictor, args.onnx_dir)
    failed = False
    for name, diff in report.items():
        gated = name.endswith("fp32")
        ok = diff < FP32_TOLERANCE if gated else True
        status = "ok" if ok else "FAIL"
        note = f"(gate < {FP32_TOLERANCE})" if gated else "(recorded, not gated)"
        print(f"  [{status}] {name:16s} max abs diff {diff:.2e} {note}")
        failed |= not ok
    if failed:
        raise SystemExit(1)
    print("parity: PASS")


if __name__ == "__main__":
    main()
