"""Test whether predictability can cheaply approximate decoder perceptual gradients."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_decoder_risk_phase0 import (  # noqa: E402
    channel_metric_loss,
    clean_from_velocity,
    decoder_hidden_features,
    decoder_hidden_loss,
    gradient_cosine,
    static_linear_state,
    trace_normalize_channel_metric,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    load_models,
)
from experiments.run_rae_decoder_risk_phase0 import (  # noqa: E402
    DEFAULT_AUDIT_CACHE,
    DEFAULT_STATIC_BRANCH,
    deterministic_noise,
    split_dataset,
    _time_values,
)


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
DEFAULT_BASES = DEFAULT_RESULTS / "evaluation/predictability_basis_v1/bases.pt"
DEFAULT_BLOCK_METRICS = (
    DEFAULT_RESULTS / "evaluation/predictability_basis_v1/basis_block_metrics.csv"
)
DEFAULT_DECODER_ATLAS = (
    DEFAULT_RESULTS
    / "evaluation/decoder_predictability_alignment_v1/decoder_basis_alignment.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "evaluation/predictability_lpl_proxy_gate_v1"


def build_block_metric(
    bases: dict[str, torch.Tensor],
    weights: dict[str, float],
    *,
    family: str = "fractional",
    floor: float,
) -> torch.Tensor:
    selected = {
        name: basis.float()
        for name, basis in bases.items()
        if name.startswith(f"{family}_") and name in weights
    }
    if len(selected) != 8:
        raise ValueError(f"expected eight {family} blocks, got {sorted(selected)}")
    channels = next(iter(selected.values())).shape[0]
    metric = torch.eye(channels, dtype=torch.float32) * float(floor)
    for name, basis in selected.items():
        metric = metric + (float(weights[name]) - float(floor)) * (basis @ basis.T)
    return trace_normalize_channel_metric(metric)


def predictability_and_oracle_metrics(
    basis_path: Path,
    block_metrics_path: Path,
    decoder_atlas_path: Path,
) -> dict[str, torch.Tensor]:
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    blocks = payload["blocks"]
    block_metrics = pd.read_csv(block_metrics_path)
    predictability = block_metrics[block_metrics["basis_family"] == "fractional"]
    predictability_weights = dict(zip(predictability["basis"], predictability["val_r2"]))
    random_floor = float(
        block_metrics[block_metrics["basis_family"] == "random"]["val_r2"].median()
    )
    atlas = pd.read_csv(decoder_atlas_path)
    atlas = atlas[atlas["basis_family"] == "fractional"]
    oracle_weights = dict(zip(atlas["basis"], atlas["decoder_hidden_2_mse_gain"]))
    oracle_floor = min(oracle_weights.values())
    return {
        "latent_mse": torch.eye(768),
        "predictability": build_block_metric(
            blocks, predictability_weights, floor=random_floor
        ),
        "decoder_atlas_oracle": build_block_metric(
            blocks, oracle_weights, floor=oracle_floor
        ),
    }


def proxy_losses(error: torch.Tensor, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: channel_metric_loss(error, metric.to(error))
        for name, metric in metrics.items()
    }


def summarize(
    scores: pd.DataFrame, gradients: pd.DataFrame
) -> dict[str, object]:
    rows = []
    for (proxy, time), frame in scores.groupby(["proxy", "time"]):
        gradient_frame = gradients[
            (gradients["proxy"] == proxy) & (gradients["time"] == time)
        ]
        rows.append(
            {
                "proxy": proxy,
                "time": float(time),
                "score_spearman": float(
                    frame[["l_proxy", "l_dec"]].corr(method="spearman").iloc[0, 1]
                ),
                "gradient_cosine_median": float(
                    gradient_frame["gradient_cosine_proxy_dec"].median()
                ),
                "gradient_cosine_mean": float(
                    gradient_frame["gradient_cosine_proxy_dec"].mean()
                ),
                "score_count": int(len(frame)),
                "gradient_count": int(len(gradient_frame)),
            }
        )
    per_time = pd.DataFrame(rows)
    proxy_rows = []
    for proxy, frame in per_time.groupby("proxy"):
        proxy_rows.append(
            {
                "proxy": proxy,
                "median_time_spearman": float(frame["score_spearman"].median()),
                "median_time_gradient_cosine": float(
                    frame["gradient_cosine_median"].median()
                ),
            }
        )
    proxy_summary = pd.DataFrame(proxy_rows)
    predictability = proxy_summary.set_index("proxy").loc["predictability"]
    oracle = proxy_summary.set_index("proxy").loc["decoder_atlas_oracle"]
    dynamic_prefixes = proxy_summary[
        proxy_summary["proxy"].str.startswith("decoder_prefix_")
    ].sort_values("median_time_gradient_cosine", ascending=False)
    gate = {
        "pass": bool(
            predictability["median_time_spearman"] >= 0.80
            and predictability["median_time_gradient_cosine"] >= 0.70
        ),
        "thresholds": {"median_time_spearman": 0.80, "gradient_cosine": 0.70},
        "predictability": predictability.to_dict(),
        "decoder_atlas_oracle": oracle.to_dict(),
        "dynamic_decoder_prefixes": dynamic_prefixes.to_dict(orient="records"),
        "oracle_static_metric_also_fails": bool(
            oracle["median_time_spearman"] < 0.80
            or oracle["median_time_gradient_cosine"] < 0.70
        ),
        "per_time": per_time.to_dict(orient="records"),
    }
    return gate


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_fp32(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model, rae, config = load_models(args.static_branch.expanduser(), device)
    metrics = predictability_and_oracle_metrics(
        args.bases.expanduser(),
        args.block_metrics.expanduser(),
        args.decoder_atlas.expanduser(),
    )
    all_times = _time_values(config, 5)
    time_indices = tuple(
        int(value) for value in args.time_indices.split(",") if value
    )
    times = all_times[list(time_indices)].to(device)
    dataset, _, _ = split_dataset(args.audit_cache.expanduser(), "test", 0, 1)
    count = min(int(args.count), len(dataset))
    samples = [dataset[index] for index in range(count)]
    clean_all = torch.stack([sample[0] for sample in samples]).float()
    labels_all = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    noise_all = deterministic_noise(
        clean_all.shape, seed=args.noise_seed, split="test", first_index=0
    )
    score_rows: list[dict[str, float | int | str]] = []
    gradient_rows: list[dict[str, float | int | str]] = []
    for start in range(0, count, args.model_batch_size):
        stop = min(start + args.model_batch_size, count)
        clean = clean_all[start:stop].to(device)
        labels = labels_all[start:stop].to(device)
        noise = noise_all[start:stop].to(device)
        with torch.no_grad():
            reference = tuple(
                feature.detach() for feature in decoder_hidden_features(rae, clean)
            )
        for scalar_time in times:
            batch_time = torch.full((len(clean),), float(scalar_time), device=device)
            state, _ = static_linear_state(clean, noise, batch_time)
            with torch.no_grad():
                prediction = model(state, batch_time, y=labels)
                estimate = clean_from_velocity(state, prediction, batch_time)
                errors = estimate - clean
                estimate_features = decoder_hidden_features(rae, estimate)
                l_dec = decoder_hidden_loss(estimate_features, reference)
                losses = proxy_losses(errors, metrics)
                prefix_losses = {
                    f"decoder_prefix_{count}": decoder_hidden_loss(
                        estimate_features[:count], reference[:count]
                    )
                    for count in range(1, len(estimate_features))
                }
            for offset in range(len(clean)):
                sample_index = start + offset
                for proxy, values in losses.items():
                    score_rows.append(
                        {
                            "proxy": proxy,
                            "sample_index": sample_index,
                            "time": float(scalar_time),
                            "l_proxy": float(values[offset]),
                            "l_dec": float(l_dec[offset]),
                        }
                    )
                for proxy, values in prefix_losses.items():
                    score_rows.append(
                        {
                            "proxy": proxy,
                            "sample_index": sample_index,
                            "time": float(scalar_time),
                            "l_proxy": float(values[offset]),
                            "l_dec": float(l_dec[offset]),
                        }
                    )
                if sample_index >= args.exact_count:
                    continue
                candidate = estimate[offset : offset + 1].detach().clone().requires_grad_(True)
                candidate_reference = tuple(
                    feature[offset : offset + 1] for feature in reference
                )
                candidate_features = decoder_hidden_features(rae, candidate)
                exact_loss = decoder_hidden_loss(
                    candidate_features, candidate_reference
                ).sum()
                exact_gradient = torch.autograd.grad(
                    exact_loss, candidate, retain_graph=True
                )[0].detach()
                error = errors[offset : offset + 1].detach()
                for proxy, metric in metrics.items():
                    variable = error.clone().requires_grad_(True)
                    proxy_loss = channel_metric_loss(variable, metric.to(device)).sum()
                    proxy_gradient = torch.autograd.grad(proxy_loss, variable)[0]
                    gradient_rows.append(
                        {
                            "proxy": proxy,
                            "sample_index": sample_index,
                            "time": float(scalar_time),
                            "gradient_cosine_proxy_dec": float(
                                gradient_cosine(proxy_gradient, exact_gradient).item()
                            ),
                        }
                    )
                for prefix_count in range(1, len(candidate_features)):
                    prefix_loss = decoder_hidden_loss(
                        candidate_features[:prefix_count],
                        candidate_reference[:prefix_count],
                    ).sum()
                    prefix_gradient = torch.autograd.grad(
                        prefix_loss,
                        candidate,
                        retain_graph=prefix_count < len(candidate_features) - 1,
                    )[0]
                    gradient_rows.append(
                        {
                            "proxy": f"decoder_prefix_{prefix_count}",
                            "sample_index": sample_index,
                            "time": float(scalar_time),
                            "gradient_cosine_proxy_dec": float(
                                gradient_cosine(prefix_gradient, exact_gradient).item()
                            ),
                        }
                    )
        print(f"processed {stop}/{count}", flush=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(score_rows)
    gradients = pd.DataFrame(gradient_rows)
    scores.to_csv(output / "proxy_scores.csv", index=False)
    gradients.to_csv(output / "proxy_gradients.csv", index=False)
    summary = summarize(scores, gradients)
    summary["protocol"] = {
        "data": "ImageNet-1K validation latent cache",
        "score_count": count,
        "exact_gradient_count": int(args.exact_count),
        "times": [float(value) for value in times.cpu()],
        "predictability_basis_fit": "disjoint ImageNet train split",
        "decoder_oracle_fit": "disjoint held-out ImageNet train latents",
        "fp32": True,
        "tf32": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-branch", type=Path, default=DEFAULT_STATIC_BRANCH)
    parser.add_argument("--audit-cache", type=Path, default=DEFAULT_AUDIT_CACHE)
    parser.add_argument("--bases", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--block-metrics", type=Path, default=DEFAULT_BLOCK_METRICS)
    parser.add_argument("--decoder-atlas", type=Path, default=DEFAULT_DECODER_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--exact-count", type=int, default=8)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--time-indices", default="0,2,4")
    parser.add_argument("--noise-seed", type=int, default=20_260_718)
    parser.add_argument("--seed", type=int, default=20_260_720)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
