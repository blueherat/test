"""Measure raw-latent and decoder-condition covariance spectra post hoc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_imagenette_latent_prior_tradeoff import (  # noqa: E402
    condition_embeddings,
    load_run_config,
)
from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    load_frozen_models,
)


CAPACITIES = (16, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"


def spectrum_metrics(values: torch.Tensor) -> dict[str, float | int]:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("spectrum input must be a rank-two sample matrix")
    centered = values.detach().double().cpu()
    centered = centered - centered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0).flip(0)
    total = eigenvalues.sum()
    if float(total) <= 0.0:
        raise ValueError("spectrum input has zero variance")
    cumulative = eigenvalues.cumsum(dim=0) / total
    result: dict[str, float | int] = {
        "effective_rank": float(
            total.square() / eigenvalues.square().sum().clamp_min(1e-18)
        )
    }
    for fraction in (0.50, 0.80, 0.90, 0.95, 0.99):
        result[f"k{int(fraction * 100)}"] = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(fraction, dtype=cumulative.dtype),
            ).item()
            + 1
        )
    for components in (16, 32, 64):
        result[f"top{components}_fraction"] = float(
            eigenvalues[: min(components, len(eigenvalues))].sum() / total
        )
    return result


@torch.no_grad()
def analyze(root: Path, device_name: str) -> pd.DataFrame:
    device = torch.device(device_name)
    records = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            run = root / f"d{capacity}_seed{seed}_p0"
            config = load_run_config(run, device_name)
            _encoder, decoder, _frozen = load_frozen_models(config, device)
            latent = torch.load(
                run / "latent_cache.pt", map_location="cpu", weights_only=True
            )["train_latent"]
            condition = condition_embeddings(
                decoder, latent, batch_size=max(256, int(config.eval_batch_size))
            )
            records.append(
                {
                    "latent_dim": capacity,
                    "frozen_seed": seed,
                    **{
                        f"raw_{key}": value
                        for key, value in spectrum_metrics(latent).items()
                    },
                    **{
                        f"condition_{key}": value
                        for key, value in spectrum_metrics(condition).items()
                    },
                }
            )
            del decoder, condition
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"complete d{capacity} seed{seed}", flush=True)
    table = pd.DataFrame(records).sort_values(["frozen_seed", "latent_dim"])
    output = root / "comparison_p0"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "latent_condition_spectrum_posthoc.csv", index=False)
    columns = [
        "raw_effective_rank",
        "raw_k90",
        "raw_k99",
        "condition_effective_rank",
        "condition_k90",
        "condition_k99",
    ]
    print(table.groupby("latent_dim")[columns].mean().round(3).to_string())
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    return analyze(args.root, args.device)


if __name__ == "__main__":
    main()
