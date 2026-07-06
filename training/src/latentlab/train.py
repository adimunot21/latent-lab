"""Train the action-conditioned JEPA world model on Two Rooms.

Config-driven (yaml in configs/), AMP on CUDA, TensorBoard logging, live
collapse diagnostics (latent std, effective rank) and k-step open-loop rollout
MSE every ``log_every`` steps. Checkpoints carry model + optimizer + config +
step so any of them can be resumed or exported.

The first batch always runs through an END-TO-END SANITY CHECK that prints
shapes/dtypes/value ranges at every stage (CLAUDE.md golden rule #2);
``--sanity-only`` stops after it.

Usage:
    uv run python -m latentlab.train --config configs/healthy.yaml --sanity-only
    uv run python -m latentlab.train --config configs/healthy.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, Field
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from latentlab.data.dataset import TwoRoomsTransitions, make_dataloader
from latentlab.models.encoder import Encoder
from latentlab.models.losses import jepa_loss
from latentlab.models.predictor import Predictor
from latentlab.probes.collapse_metrics import effective_rank, mean_latent_std
from latentlab.probes.rollout import open_loop_rollout


class TrainConfig(BaseModel):
    """All training hyperparameters. Values justified inline."""

    run_name: str
    data_dir: Path = Path("data/two_rooms_v1")
    checkpoint_root: Path = Path("checkpoints/two_rooms_v1")

    # Split: last chunk of episodes held out for probes/rollout eval.
    train_episodes: tuple[int, int] = (0, 900)
    eval_episodes: tuple[int, int] = (900, 1000)

    # Model. 128 dims is plenty for a 2-DoF env while keeping ONNX small.
    latent_dim: int = 128
    encoder_base_channels: int = 32
    predictor_hidden_dim: int = 256
    predictor_layers: int = 2

    # Loss. lambda_reg=1.0 balances pred MSE (~1e-2 scale) against SIGReg
    # (~1e-3 scale when near-Gaussian); 0 => deliberate collapse run.
    lambda_reg: float = 1.0
    sigreg_projections: int = 64  # CF sketch width; more = lower-variance test

    # Optimization. Small model + easy task: AdamW 3e-4 is the safe default;
    # batch 256 gives SIGReg a decent sample for its distribution test.
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    amp: bool = True  # fp16 autocast on CUDA; saves VRAM on the 4GB card

    # Diagnostics / checkpoints.
    log_every: int = 50  # steps between scalar logs
    rollout_horizon: int = 8  # k for open-loop rollout diagnostics
    rollout_episodes: int = 32  # eval episodes used for the rollout metric
    checkpoint_epochs: list[int] = Field(default_factory=list)  # also always saves final
    seed: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(config: TrainConfig, device: torch.device) -> tuple[Encoder, Predictor]:
    encoder = Encoder(latent_dim=config.latent_dim, base_channels=config.encoder_base_channels).to(
        device
    )
    predictor = Predictor(
        latent_dim=config.latent_dim,
        hidden_dim=config.predictor_hidden_dim,
        n_hidden_layers=config.predictor_layers,
    ).to(device)
    return encoder, predictor


@dataclasses.dataclass
class ProbeTensors:
    """Flattened frame/state tensors for probe fitting and eval."""

    frames_train: torch.Tensor  # (N, 1, H, W) normalized
    states_train: torch.Tensor  # (N, 2)
    frames_eval: torch.Tensor
    states_eval: torch.Tensor


def normalized_frames_tensor(dataset: TwoRoomsTransitions) -> torch.Tensor:
    """All frames of a dataset as normalized float32 (E, T+1, 1, H, W)."""
    frames = torch.from_numpy(dataset.frames.astype("float32")) / 255.0
    if dataset.normalize:
        stats = dataset.norm_stats
        frames = (frames - stats["frame_mean"]) / stats["frame_std"]
    return frames.unsqueeze(2)


def load_probe_tensors(
    data_dir: Path,
    train_range: tuple[int, int] = (0, 900),
    eval_range: tuple[int, int] = (900, 1000),
    probe_train_episodes: int = 200,
) -> ProbeTensors:
    """Probe-fitting tensors. 200 episodes x 61 frames = 12.2k samples is
    plenty to fit a 129-parameter-per-output ridge probe."""
    train_ds = TwoRoomsTransitions(
        data_dir,
        episode_range=(train_range[0], min(train_range[0] + probe_train_episodes, train_range[1])),
    )
    eval_ds = TwoRoomsTransitions(data_dir, episode_range=eval_range)
    frames_train = normalized_frames_tensor(train_ds).flatten(0, 1)
    states_train = torch.from_numpy(train_ds.states).flatten(0, 1)
    frames_eval = normalized_frames_tensor(eval_ds).flatten(0, 1)
    states_eval = torch.from_numpy(eval_ds.states).flatten(0, 1)
    return ProbeTensors(frames_train, states_train, frames_eval, states_eval)


def sanity_check(
    batch: dict[str, torch.Tensor],
    encoder: Encoder,
    predictor: Predictor,
    config: TrainConfig,
    device: torch.device,
) -> None:
    """One batch through every stage with shapes/dtypes/ranges printed."""
    print("=== END-TO-END SANITY CHECK (one batch through every stage) ===")

    def describe(name: str, tensor: torch.Tensor) -> None:
        print(
            f"  {name:12s} shape={tuple(tensor.shape)!s:22s} dtype={tensor.dtype} "
            f"min={tensor.min().item():+.3f} max={tensor.max().item():+.3f} "
            f"mean={tensor.mean().item():+.3f}"
        )

    frame = batch["frame"].to(device)
    action = batch["action"].to(device)
    next_frame = batch["next_frame"].to(device)
    describe("frame", frame)
    describe("action", action)
    describe("next_frame", next_frame)

    with torch.no_grad():
        z = encoder(frame)
        z_next = encoder(next_frame)
        z_pred = predictor(z, action)
        describe("z", z)
        describe("z_next", z_next)
        describe("z_pred", z_pred)
        losses = jepa_loss(z_pred, z_next, z, lambda_reg=config.lambda_reg)
        print(
            f"  losses: total={losses['total'].item():.5f} "
            f"pred_mse={losses['pred_mse'].item():.5f} sigreg={losses['sigreg'].item():.5f}"
        )
        print(
            f"  latent std={mean_latent_std(z):.4f} eff_rank={effective_rank(z):.1f} "
            f"(untrained baseline)"
        )
    print("=== sanity check done ===\n")


def save_checkpoint(
    path: Path,
    encoder: Encoder,
    predictor: Predictor,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": json.loads(config.model_dump_json()),
            "step": step,
            "epoch": epoch,
        },
        path,
    )
    print(f"saved checkpoint: {path}")


def load_checkpoint_models(path: Path) -> tuple[Encoder, Predictor, TrainConfig]:
    """Rebuild encoder/predictor (eval mode, CPU) from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**payload["config"])
    encoder, predictor = build_models(config, torch.device("cpu"))
    encoder.load_state_dict(payload["encoder"])
    predictor.load_state_dict(payload["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, config


def train(config: TrainConfig, sanity_only: bool = False) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    print(f"device: {device}, run: {config.run_name}, lambda_reg: {config.lambda_reg}")

    loader = make_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        episode_range=config.train_episodes,
        num_workers=2,
    )
    encoder, predictor = build_models(config, device)
    n_params = sum(p.numel() for m in (encoder, predictor) for p in m.parameters())
    print(f"total params: {n_params / 1e6:.2f}M, steps/epoch: {len(loader)}")

    # Fixed eval episodes for the k-step rollout diagnostic.
    eval_ds = TwoRoomsTransitions(config.data_dir, episode_range=config.eval_episodes)
    rollout_frames = normalized_frames_tensor(eval_ds)[: config.rollout_episodes].to(device)
    rollout_actions = torch.from_numpy(eval_ds.actions[: config.rollout_episodes]).to(device)

    first_batch = next(iter(loader))
    sanity_check(first_batch, encoder, predictor, config, device)
    if sanity_only:
        return Path()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    run_dir = config.checkpoint_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(config.model_dump_json())))
    writer = SummaryWriter(log_dir=str(Path("runs") / config.run_name))

    step = 0
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        for batch in loader:
            frame = batch["frame"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            next_frame = batch["next_frame"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                z = encoder(frame)
                z_next = encoder(next_frame)
                z_pred = predictor(z, action)
                losses = jepa_loss(
                    z_pred,
                    z_next,
                    z,
                    lambda_reg=config.lambda_reg,
                    num_projections=config.sigreg_projections,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % config.log_every == 0:
                with torch.no_grad():
                    latent_std = mean_latent_std(z.float())
                    eff_rank = effective_rank(z.float())
                    rollout = open_loop_rollout(
                        encoder,
                        predictor,
                        rollout_frames,
                        rollout_actions,
                        horizon=config.rollout_horizon,
                    )
                    rollout_mse = float(rollout["latent_mse_per_step"][-1].item())
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch:3d} step {step:5d} | total {losses['total'].item():.5f} "
                    f"pred {losses['pred_mse'].item():.5f} sigreg {losses['sigreg'].item():.5f} | "
                    f"z_std {latent_std:.3f} eff_rank {eff_rank:6.1f} "
                    f"rollout@{config.rollout_horizon} {rollout_mse:.5f} | {elapsed:6.1f}s"
                )
                for key, value in [
                    ("loss/total", losses["total"].item()),
                    ("loss/pred_mse", losses["pred_mse"].item()),
                    ("loss/sigreg", losses["sigreg"].item()),
                    ("collapse/latent_std", latent_std),
                    ("collapse/effective_rank", eff_rank),
                    (f"rollout/latent_mse_at_{config.rollout_horizon}", rollout_mse),
                ]:
                    writer.add_scalar(key, value, step)

        if epoch in config.checkpoint_epochs:
            save_checkpoint(
                run_dir / f"epoch_{epoch:03d}.pt",
                encoder,
                predictor,
                optimizer,
                config,
                step,
                epoch,
            )

    final_path = run_dir / "final.pt"
    save_checkpoint(final_path, encoder, predictor, optimizer, config, step, config.epochs)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM: {peak_gb:.2f} GB (budget: 4 GB)")
    writer.close()
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="yaml config path")
    parser.add_argument("--sanity-only", action="store_true", help="stop after the sanity check")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    config = TrainConfig(**raw)
    train(config, sanity_only=args.sanity_only)


if __name__ == "__main__":
    main()
