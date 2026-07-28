"""Quantify signal concentration and path SNR in the SPC guided subspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


from experiments.rae_latent_cache import CachedRAELatentDataset
from experiments.rae_layerwise_path import spatial_center
from experiments.run_rae_path_gradient_interference import _load_basis
from experiments.run_rae_spc_directional_sensitivity import orthogonal_control_basis


DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_layerwise_path_streams/seed3407_n160000_fp32"
)
DEFAULT_SUBSPACES = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path/"
    "gate1_imagenet_train1024_val256_mid9/subspaces.pt"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/subspace_signal"
)


def projected_variance_per_active_dimension(
    residual: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    rows = residual.permute(0, 2, 3, 1).reshape(-1, residual.shape[1]).double()
    coefficients = rows @ basis.double()
    return coefficients.square().mean()


def accumulate_subspace_signal(
    dataset: CachedRAELatentDataset,
    guided_basis: torch.Tensor,
    control_basis: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    channels, rank = guided_basis.shape
    totals = {
        "guided_coefficient_square": 0.0,
        "control_coefficient_square": 0.0,
        "residual_square": 0.0,
        "row_count": 0,
    }
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        latent = torch.stack([dataset[index][0] for index in range(start, end)])
        _, residual = spatial_center(latent)
        rows = residual.permute(0, 2, 3, 1).reshape(-1, channels).double()
        guided_coefficients = rows @ guided_basis.double()
        control_coefficients = rows @ control_basis.double()
        totals["guided_coefficient_square"] += float(
            guided_coefficients.square().sum()
        )
        totals["control_coefficient_square"] += float(
            control_coefficients.square().sum()
        )
        totals["residual_square"] += float(rows.square().sum())
        totals["row_count"] += int(len(rows))
    row_count = int(totals["row_count"])
    guided_sum = totals["guided_coefficient_square"]
    control_sum = totals["control_coefficient_square"]
    residual_sum = totals["residual_square"]
    complement_sum = residual_sum - guided_sum
    return {
        "sample_count": int(len(dataset)),
        "token_count": row_count,
        "channels": int(channels),
        "rank": int(rank),
        "rank_fraction": float(rank / channels),
        "guided_variance_per_active_dim": guided_sum / (row_count * rank),
        "control_variance_per_active_dim": control_sum / (row_count * rank),
        "complement_variance_per_active_dim": complement_sum
        / (row_count * (channels - rank)),
        "residual_variance_per_dim": residual_sum / (row_count * channels),
        "guided_residual_energy_fraction": guided_sum / residual_sum,
        "control_residual_energy_fraction": control_sum / residual_sum,
    }


def snr_rows(
    signal: dict[str, float],
    *,
    times: tuple[float, ...],
    floor: float,
    power: float,
) -> pd.DataFrame:
    variances = {
        "guided": signal["guided_variance_per_active_dim"],
        "control": signal["control_variance_per_active_dim"],
        "complement": signal["complement_variance_per_active_dim"],
    }
    rows = []
    for time in times:
        if not 0 < time < 1:
            raise ValueError("SNR times must lie strictly inside (0,1)")
        static_scale = ((1.0 - time) / time) ** 2
        fading = floor + (1.0 - floor) * (1.0 - time) ** power
        for name, variance in variances.items():
            rows.append(
                {
                    "time": float(time),
                    "subspace": name,
                    "variance_per_active_dim": float(variance),
                    "static_state_snr": float(static_scale * variance),
                    "spc_state_snr": float(
                        static_scale
                        * variance
                        * (fading**2 if name == "guided" else 1.0)
                    ),
                    "spc_over_static_snr": float(
                        fading**2 if name == "guided" else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--control-seed", type=int, default=20_260_731)
    parser.add_argument("--times", default="0.85,0.3,0.1")
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--power", type=float, default=2.0)
    args = parser.parse_args()

    payload = torch.load(
        args.subspaces.expanduser(), map_location="cpu", weights_only=False
    )
    entry = payload["subspaces"].get(
        args.rank, payload["subspaces"].get(str(args.rank))
    )
    if entry is None:
        raise KeyError(f"rank {args.rank} is absent from {args.subspaces}")
    guided = entry["basis"].float().contiguous()
    control = orthogonal_control_basis(guided, seed=args.control_seed).cpu()
    dataset = CachedRAELatentDataset(
        args.cache.expanduser(), start=args.start, stop=args.start + args.count
    )
    signal = accumulate_subspace_signal(
        dataset, guided, control, batch_size=args.batch_size
    )
    signal["fit_explained_final_fraction"] = float(
        entry["explained_final_fraction"]
    )
    signal["fit_explained_predictable_fraction"] = float(
        entry["explained_predictable_fraction"]
    )
    signal["guided_over_control_variance"] = (
        signal["guided_variance_per_active_dim"]
        / signal["control_variance_per_active_dim"]
    )
    signal["guided_over_complement_variance"] = (
        signal["guided_variance_per_active_dim"]
        / signal["complement_variance_per_active_dim"]
    )
    times = tuple(float(value) for value in args.times.split(",") if value.strip())
    table = snr_rows(
        signal, times=times, floor=args.floor, power=args.power
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "signal_concentration.json").write_text(
        json.dumps(signal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table.to_csv(output / "state_snr.csv", index=False)
    print(json.dumps(signal, indent=2, ensure_ascii=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
