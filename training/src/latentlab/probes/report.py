"""Checkpoint comparison report: probe R^2, collapse metrics, rollout errors.

Runs every probe against one or more checkpoints and prints a table. This is
the Phase 2 acceptance evidence: healthy checkpoints should show high probe
R^2, effective rank >> 1, and low probed rollout position error; collapse
checkpoints the opposite.

Usage:
    uv run python -m latentlab.probes.report --checkpoints checkpoints/two_rooms_v1/*/final.pt
    uv run python -m latentlab.probes.report --checkpoints checkpoints/two_rooms_v1/collapse_v1/*.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentlab.probes.collapse_metrics import effective_rank, mean_latent_std
from latentlab.probes.linear_probe import encode_in_batches, evaluate_probe
from latentlab.probes.rollout import open_loop_rollout, probe_position_error
from latentlab.train import load_checkpoint_models, load_probe_tensors, normalized_frames_tensor


@torch.no_grad()
def report_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    rollout_horizon: int = 8,
    rollout_episodes: int = 64,
) -> dict[str, float]:
    encoder, predictor, config = load_checkpoint_models(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    predictor.to(device)

    probe_data = load_probe_tensors(data_dir, config.train_episodes, config.eval_episodes)
    probe_metrics, weight, bias = evaluate_probe(
        encoder,
        probe_data.frames_train,
        probe_data.states_train,
        probe_data.frames_eval,
        probe_data.states_eval,
    )

    # Collapse metrics over the eval latents.
    z_eval = encode_in_batches(encoder, probe_data.frames_eval)
    latent_std = mean_latent_std(z_eval)
    eff_rank = effective_rank(z_eval)

    # Open-loop rollout on eval episodes; position error via the fitted probe.
    from latentlab.data.dataset import TwoRoomsTransitions

    eval_ds = TwoRoomsTransitions(data_dir, episode_range=config.eval_episodes)
    frames = normalized_frames_tensor(eval_ds)[:rollout_episodes].to(device)
    actions = torch.from_numpy(eval_ds.actions[:rollout_episodes]).to(device)
    states = torch.from_numpy(eval_ds.states[:rollout_episodes])
    rollout = open_loop_rollout(encoder, predictor, frames, actions, horizon=rollout_horizon)
    position_error = probe_position_error(
        rollout["z_pred"].cpu(),
        states[:, 1 : rollout_horizon + 1],
        weight,
        bias,
    )

    return {
        "probe_r2": probe_metrics["r2_mean"],
        "latent_std": latent_std,
        "eff_rank": eff_rank,
        "rollout_latent_mse": float(rollout["latent_mse_per_step"][-1].item()),
        "rollout_pos_err": float(position_error[-1].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/two_rooms_v1"))
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    header = (
        f"{'checkpoint':44s} {'probe_R2':>9s} {'z_std':>7s} {'eff_rank':>9s} "
        f"{'roll_mse':>9s} {'roll_pos_err':>13s}"
    )
    print(header)
    print("-" * len(header))
    for path in args.checkpoints:
        metrics = report_checkpoint(path, args.data, rollout_horizon=args.horizon)
        label = f"{path.parent.name}/{path.name}"
        print(
            f"{label:44s} {metrics['probe_r2']:9.4f} {metrics['latent_std']:7.3f} "
            f"{metrics['eff_rank']:9.1f} {metrics['rollout_latent_mse']:9.5f} "
            f"{metrics['rollout_pos_err']:13.4f}"
        )
    print(
        f"\n(roll_* at horizon {args.horizon}; pos_err in world units, 1.0 = room width)"
        "\nReading collapsed checkpoints: z_std ~0 is the collapse signature. Scale-"
        "\ninvariant metrics stay deceptively healthy — R2 can stay moderate (microscopic"
        "\nresiduals remain linearly decodable) and eff_rank high (residual noise is"
        "\nisotropic). roll_mse is misleadingly tiny (everything maps to ~one point);"
        "\nroll_pos_err and planning success are the honest utility metrics."
    )


if __name__ == "__main__":
    main()
