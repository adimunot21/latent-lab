"""Export a trained checkpoint to ONNX: encoder.onnx + predictor.onnx (+ int8).

Both graphs get a DYNAMIC batch dimension — the CEM planner batches hundreds
of candidates through the predictor, and the encoder is also called with
batch > 1 when building lookup tables. Weight-only dynamic int8 variants are
emitted alongside fp32 (MatMul/Gemm weights quantize; conv stays fp32 under
dynamic quantization — the size win is mostly in the predictor and encoder
head, and the browser can pick per-network).

Usage:
    uv run python -m latentlab.export.to_onnx \
        --checkpoint checkpoints/two_rooms_v1/healthy_v1/final.pt \
        --out export_out/onnx/healthy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from latentlab.models.encoder import Encoder
from latentlab.models.predictor import Predictor
from latentlab.train import load_checkpoint_models

OPSET = 17  # broadly supported by onnxruntime-web (WebGPU + WASM)


def export_models(
    encoder: Encoder,
    predictor: Predictor,
    out_dir: Path,
    frame_size: int = 64,
    quantize: bool = True,
) -> dict[str, Path]:
    """Export an encoder/predictor pair to out_dir. Returns artifact paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder.eval()
    predictor.eval()
    latent_dim = encoder.latent_dim
    action_dim = predictor.action_dim
    paths: dict[str, Path] = {}

    encoder_path = out_dir / "encoder.onnx"
    torch.onnx.export(
        encoder,
        (torch.zeros(2, 1, frame_size, frame_size),),  # batch=2 so nothing folds to 1
        str(encoder_path),
        input_names=["frame"],
        output_names=["latent"],
        dynamic_axes={"frame": {0: "batch"}, "latent": {0: "batch"}},
        opset_version=OPSET,
    )
    paths["encoder"] = encoder_path

    predictor_path = out_dir / "predictor.onnx"
    torch.onnx.export(
        predictor,
        (torch.zeros(2, latent_dim), torch.zeros(2, action_dim)),
        str(predictor_path),
        input_names=["latent", "action"],
        output_names=["next_latent"],
        dynamic_axes={
            "latent": {0: "batch"},
            "action": {0: "batch"},
            "next_latent": {0: "batch"},
        },
        opset_version=OPSET,
    )
    paths["predictor"] = predictor_path

    for path in (encoder_path, predictor_path):
        onnx.checker.check_model(onnx.load(str(path)))

    if quantize:
        for name in ("encoder", "predictor"):
            int8_path = out_dir / f"{name}.int8.onnx"
            quantize_dynamic(
                model_input=str(paths[name]),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
            )
            paths[f"{name}_int8"] = int8_path

    return paths


def export_checkpoint(checkpoint: Path, out_dir: Path, quantize: bool = True) -> dict[str, Path]:
    encoder, predictor, config = load_checkpoint_models(checkpoint)
    paths = export_models(encoder, predictor, out_dir, quantize=quantize)
    for name, path in paths.items():
        print(f"  {name:16s} {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    print(f"exporting {args.checkpoint} -> {args.out}")
    export_checkpoint(args.checkpoint, args.out, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
