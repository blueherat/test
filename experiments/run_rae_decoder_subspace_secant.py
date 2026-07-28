"""Probe frozen RAE decoder secant sensitivity along matched latent subspaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.analyze_rae_predictability_gain import MATCHED_PAIRS  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.run_rae_decoder_risk_phase0 import (  # noqa: E402
    _decode_image_and_hidden,
    _load_full_rae,
    decoder_embed_metric,
)
from experiments.run_rae_latent_trust_decoder_spotcheck import (  # noqa: E402
    DEFAULT_BASES,
    DEFAULT_RESULTS,
    per_sample_lpips,
)
from experiments.run_rae_latent_trust_rollout import selected_block_names  # noqa: E402
from experiments.run_rae_spc_directional_sensitivity import (  # noqa: E402
    match_per_sample_norm,
    projected_residual,
)
from experiments.train_rae_layerwise_path import configure_determinism  # noqa: E402


DEFAULT_BRANCH = DEFAULT_RESULTS / "seed1201_static_s0_to5000"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "evaluation/decoder_subspace_secant_v1"


@torch.no_grad()
def decode_with_hidden(
    rae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    images = []
    hidden_batches: list[list[torch.Tensor]] = []
    for start in range(0, len(latents), int(batch_size)):
        image, hidden = _decode_image_and_hidden(
            rae, latents[start : start + int(batch_size)].to(device)
        )
        images.append(image.float().clamp(0.0, 1.0).cpu())
        if not hidden_batches:
            hidden_batches = [[] for _ in hidden]
        for layer_batches, features in zip(hidden_batches, hidden):
            layer_batches.append(features.float().cpu())
    return torch.cat(images), tuple(torch.cat(values) for values in hidden_batches)


def decoder_secant_metrics(
    clean_latent: torch.Tensor,
    delta: torch.Tensor,
    clean_image: torch.Tensor,
    perturbed_image: torch.Tensor,
    clean_hidden: tuple[torch.Tensor, ...],
    perturbed_hidden: tuple[torch.Tensor, ...],
    lpips: LearnedPerceptualImagePatchSimilarity,
    *,
    lpips_device: torch.device,
) -> dict[str, torch.Tensor]:
    latent_mse = delta.square().flatten(1).mean(1).clamp_min(1e-20)
    image_difference = perturbed_image - clean_image
    image_mse = image_difference.square().flatten(1).mean(1)
    image_l1 = image_difference.abs().flatten(1).mean(1)
    image_lpips = per_sample_lpips(
        lpips,
        perturbed_image.to(lpips_device),
        clean_image.to(lpips_device),
    ).cpu()
    rows = {
        "clean_latent_rms": clean_latent.square().flatten(1).mean(1).sqrt(),
        "latent_delta_rms": latent_mse.sqrt(),
        "image_shift_mse": image_mse,
        "image_shift_l1": image_l1,
        "image_shift_lpips": image_lpips,
        "decoder_pixel_mse_gain": image_mse / latent_mse,
        "decoder_l1_secant": image_l1 / latent_mse.sqrt(),
        "decoder_lpips_secant": image_lpips / latent_mse,
    }
    for layer_index, (base, perturbed) in enumerate(
        zip(clean_hidden, perturbed_hidden), start=1
    ):
        hidden_mse = (perturbed - base).square().flatten(1).mean(1)
        rows[f"decoder_hidden_{layer_index}_mse_gain"] = hidden_mse / latent_mse
    return rows


def pair_ratios(table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in table
        if column.startswith("decoder_") and column != "decoder_layer"
    ]
    grouped = table.groupby(["basis", "fraction", "sign"], as_index=False)[
        metric_columns
    ].mean()
    rows = []
    for high, low in MATCHED_PAIRS:
        high_rows = grouped[grouped["basis"] == high].set_index(["fraction", "sign"])
        low_rows = grouped[grouped["basis"] == low].set_index(["fraction", "sign"])
        for fraction, sign in high_rows.index.intersection(low_rows.index):
            row: dict[str, float | str] = {
                "higher_predictability_basis": high,
                "lower_predictability_basis": low,
                "fraction": float(fraction),
                "sign": float(sign),
            }
            for metric in metric_columns:
                row[f"{metric}_ratio"] = float(high_rows.loc[(fraction, sign), metric]) / max(
                    float(low_rows.loc[(fraction, sign), metric]), 1e-20
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(ratios: pd.DataFrame, output: Path) -> None:
    metrics = (
        ("decoder_pixel_mse_gain_ratio", "pixel MSE gain ratio"),
        ("decoder_l1_secant_ratio", "pixel L1 secant ratio"),
        ("decoder_lpips_secant_ratio", "LPIPS secant ratio"),
        ("decoder_hidden_1_mse_gain_ratio", "decoder hidden 1 ratio"),
        ("decoder_hidden_4_mse_gain_ratio", "decoder hidden 4 ratio"),
    )
    averaged = ratios.groupby(
        ["higher_predictability_basis", "lower_predictability_basis", "fraction"],
        as_index=False,
    )[[field for field, _ in metrics]].mean()
    fig, axes_grid = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    axes = axes_grid.flatten()
    colors = ("#2678a8", "#2f855a", "#8b5a9f")
    for axis, (field, title) in zip(axes, metrics):
        for (high, low), color in zip(MATCHED_PAIRS, colors):
            frame = averaged[
                (averaged["higher_predictability_basis"] == high)
                & (averaged["lower_predictability_basis"] == low)
            ]
            axis.plot(
                frame["fraction"] * 100.0,
                frame[field],
                marker="o",
                color=color,
                label=f"{high} / {low}",
            )
        axis.axhline(1.0, color="#888888", linestyle="--", linewidth=1)
        axis.set_xlabel("latent perturbation RMS (%)")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[-2].legend(frameon=False, fontsize=8)
    fig.savefig(output / "decoder_subspace_secant.png", dpi=180)
    plt.close(fig)


def summarize(ratios: pd.DataFrame, count: int) -> dict[str, object]:
    largest = float(ratios["fraction"].max())
    selected = ratios[ratios["fraction"] == largest]
    rows = []
    for (high, low), frame in selected.groupby(
        ["higher_predictability_basis", "lower_predictability_basis"]
    ):
        rows.append(
            {
                "higher_predictability_basis": high,
                "lower_predictability_basis": low,
                "fraction": largest,
                "pixel_mse_gain_ratio": float(
                    frame["decoder_pixel_mse_gain_ratio"].mean()
                ),
                "l1_secant_ratio": float(
                    frame["decoder_l1_secant_ratio"].mean()
                ),
                "lpips_secant_ratio": float(
                    frame["decoder_lpips_secant_ratio"].mean()
                ),
                "hidden_1_ratio": float(
                    frame["decoder_hidden_1_mse_gain_ratio"].mean()
                ),
                "hidden_4_ratio": float(
                    frame["decoder_hidden_4_mse_gain_ratio"].mean()
                ),
            }
        )
    agreement = sum(
        row["l1_secant_ratio"] > 1.0 and row["lpips_secant_ratio"] > 1.0
        for row in rows
    )
    return {
        "exploratory": True,
        "sample_count": int(count),
        "largest_fraction": largest,
        "higher_predictability_more_sensitive_l1_and_lpips_pairs": agreement,
        "all_three_pairs": agreement == len(MATCHED_PAIRS),
        "pair_summary": rows,
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_determinism(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    branch = args.branch.expanduser().resolve()
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    dataset = CachedRAELatentDataset(
        Path(str(manifest["latent_cache"])),
        start=args.cache_start,
        stop=args.cache_start + args.count,
    )
    clean = torch.stack([dataset[index][0] for index in range(len(dataset))]).float()
    basis_payload = torch.load(
        args.bases.expanduser(), map_location="cpu", weights_only=False
    )
    selected = (
        {
            name
            for name in basis_payload["blocks"]
            if not name.startswith("random_")
        }
        if args.all_blocks
        else selected_block_names()
    )
    bases = {
        name: basis.float()
        for name, basis in basis_payload["blocks"].items()
        if name in selected
    }
    config = OmegaConf.load(branch / "config.yaml")
    rae = _load_full_rae(config, device)
    embed_metric = decoder_embed_metric(rae).cpu()
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device).requires_grad_(False).eval()
    clean_image, clean_hidden = decode_with_hidden(
        rae, clean, device=device, batch_size=args.decode_batch_size
    )
    fractions = tuple(float(value) for value in args.fractions.split(",") if value)
    signs = (-1.0, 1.0) if args.both_signs else (-1.0,)
    rows: list[dict[str, float | int | str]] = []
    for basis_name, basis in bases.items():
        embed_gain = torch.trace(basis.T @ embed_metric @ basis) / float(
            basis.shape[1]
        )
        component = projected_residual(clean, basis)
        matched = match_per_sample_norm(component, clean)
        for fraction in fractions:
            for sign in signs:
                delta = float(sign * fraction) * matched
                image, hidden = decode_with_hidden(
                    rae,
                    clean + delta,
                    device=device,
                    batch_size=args.decode_batch_size,
                )
                metrics = decoder_secant_metrics(
                    clean,
                    delta,
                    clean_image,
                    image,
                    clean_hidden,
                    hidden,
                    lpips,
                    lpips_device=device,
                )
                metrics["decoder_embed_gain"] = torch.full(
                    (len(clean),), float(embed_gain)
                )
                for sample_index in range(len(clean)):
                    row: dict[str, float | int | str] = {
                        "sample_index": sample_index,
                        "basis": basis_name,
                        "fraction": fraction,
                        "sign": sign,
                    }
                    row.update(
                        {
                            name: float(values[sample_index])
                            for name, values in metrics.items()
                        }
                    )
                    rows.append(row)
                print(
                    f"basis={basis_name} fraction={fraction:.4f} sign={sign:+.0f}",
                    flush=True,
                )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "decoder_subspace_secant_samples.csv", index=False)
    ratios = pair_ratios(table)
    ratios.to_csv(output / "decoder_subspace_pair_ratios.csv", index=False)
    plot_results(ratios, output)
    summary = summarize(ratios, len(clean))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, default=DEFAULT_BRANCH)
    parser.add_argument("--bases", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--fractions", default="0.005,0.01")
    parser.add_argument("--both-signs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--all-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=20_260_730)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
