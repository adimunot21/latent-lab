"""CI parity test: PyTorch vs onnxruntime on freshly-initialized models.

Checkpoints are not git-tracked, so CI exercises the export path itself:
random weights are as good as trained ones for verifying that the exported
graph computes the same function. fp32 is gated at 1e-4; int8 error is
printed for the record (its magnitude depends on weights, so no tight gate).
"""

from __future__ import annotations

from pathlib import Path

import torch

from latentlab.export.parity_test import FP32_TOLERANCE, check_parity
from latentlab.export.to_onnx import export_models
from latentlab.models.encoder import Encoder
from latentlab.models.predictor import Predictor


def test_onnx_parity_fp32_and_int8(tmp_path: Path) -> None:
    torch.manual_seed(0)
    encoder = Encoder(latent_dim=128)
    predictor = Predictor(latent_dim=128)
    export_models(encoder, predictor, tmp_path, quantize=True)

    report = check_parity(encoder, predictor, tmp_path)

    assert report["encoder_fp32"] < FP32_TOLERANCE, report
    assert report["predictor_fp32"] < FP32_TOLERANCE, report

    # int8: recorded, sanity-bounded only (dynamic quantization error scale
    # depends on weight distributions).
    print(
        f"int8 max abs diffs: encoder {report['encoder_int8']:.4f}, "
        f"predictor {report['predictor_int8']:.4f}"
    )
    assert report["encoder_int8"] < 1.0
    assert report["predictor_int8"] < 1.0


def test_onnx_dynamic_batch(tmp_path: Path) -> None:
    """Exported graphs must accept batch sizes other than the export dummy's."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(1)
    encoder = Encoder(latent_dim=32)
    predictor = Predictor(latent_dim=32)
    export_models(encoder, predictor, tmp_path, quantize=False)

    enc = ort.InferenceSession(str(tmp_path / "encoder.onnx"), providers=["CPUExecutionProvider"])
    pred = ort.InferenceSession(
        str(tmp_path / "predictor.onnx"), providers=["CPUExecutionProvider"]
    )
    for batch in (1, 3, 300):
        z = enc.run(None, {"frame": np.zeros((batch, 1, 64, 64), dtype=np.float32)})[0]
        assert z.shape == (batch, 32)
        z_next = pred.run(None, {"latent": z, "action": np.zeros((batch, 2), dtype=np.float32)})[0]
        assert z_next.shape == (batch, 32)
