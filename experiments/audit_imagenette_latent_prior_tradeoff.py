"""Independent NFE and decoder-interface audit for the latent-prior trade-off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    INTERFACE_DIM,
    LatentPriorTradeoffConfig,
    OrthogonalLatentInterface,
    ResNet18Evaluator,
    build_prior,
    covariance_statistics,
    deterministic_datasets,
    evaluate_rollout,
    fixed_eval_subset,
    fixed_orthogonal_basis,
    frechet_distance,
    latent_distribution_metrics,
    load_frozen_models,
    sample_prior_coordinates,
    sliced_wasserstein_distance,
    state_dict_sha256,
)
from experiments.mnist_spectral_rollout_toy import configure_fp32  # noqa: E402


def load_run_config(run: Path, device: str) -> LatentPriorTradeoffConfig:
    values = json.loads((run / "config.json").read_text())
    for key in ("data_root", "checkpoint_root", "output_root"):
        values[key] = Path(values[key])
    values.update({"device": device, "overwrite": False, "resume": True, "save": True})
    return LatentPriorTradeoffConfig(**values)


@torch.no_grad()
def condition_embeddings(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    device = next(decoder.parameters()).device
    return torch.cat(
        [decoder.condition_embedding(batch.to(device)).cpu() for batch in latent.split(batch_size)]
    )


def distribution_comparison(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    seed: int,
    prefix: str,
) -> dict[str, float]:
    values = covariance_statistics(real, generated)
    values["latent_sliced_wasserstein"] = sliced_wasserstein_distance(
        real, generated, directions=256, seed=seed
    )
    return {f"{prefix}_{key}": value for key, value in values.items()}


def audit_run(
    run: Path,
    *,
    device_name: str,
    nfe: int = 200,
    recompute_100_fid: bool = False,
    overwrite: bool = False,
) -> Path:
    output = run / f"nfe{int(nfe)}_audit.json"
    if output.is_file() and not overwrite:
        print(f"audit already complete: {output}", flush=True)
        return output
    config = load_run_config(run, device_name)
    configure_fp32(config.prior_seed)
    device = torch.device(device_name)
    _, val_dataset = deterministic_datasets(config.data_root, config.image_size)
    _encoder, decoder, frozen = load_frozen_models(config, device)
    cache = torch.load(run / "latent_cache.pt", map_location="cpu", weights_only=True)
    prior_state = torch.load(run / "prior_state.pt", map_location="cpu", weights_only=True)
    prior = build_prior(config, device)
    prior.load_state_dict(prior_state["prior_ema"])
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    interface = OrthogonalLatentInterface(
        config.latent_dim,
        fixed_orthogonal_basis(INTERFACE_DIM, config.basis_seed),
    ).to(device)
    count = min(config.quality_count, len(val_dataset))
    eval_subset = fixed_eval_subset(val_dataset, count, seed=2_027)
    eval_indices = torch.as_tensor(eval_subset.indices, dtype=torch.long)
    val_latent = cache["val_latent"][eval_indices]
    prior100 = sample_prior_coordinates(
        prior,
        interface,
        count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )
    prior_audit = sample_prior_coordinates(
        prior,
        interface,
        count,
        int(nfe),
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )
    saved_summary = json.loads((run / "summary.json").read_text())
    regenerated_metrics = latent_distribution_metrics(val_latent, prior100, config)
    regenerated_differences = {
        f"regenerated100_{key}_abs_diff": abs(float(value) - float(saved_summary[key]))
        for key, value in regenerated_metrics.items()
    }
    feature_state = torch.load(
        run / "features_prior.pt", map_location="cpu", weights_only=True
    )
    formal_fid = frechet_distance(
        feature_state["real_features"], feature_state["generated_features"]
    )
    real_labels = torch.as_tensor(val_dataset.targets, dtype=torch.long)[eval_indices]
    evaluator = ResNet18Evaluator().to(device).eval()
    dummy_images = torch.empty((count, 0), dtype=torch.float32)
    metrics_nfe, features_nfe, _preview = evaluate_rollout(
        "prior",
        decoder,
        prior_audit,
        dummy_images,
        real_labels,
        feature_state["real_features"],
        evaluator,
        frozen["class_to_idx"],
        config,
    )
    metrics_100 = None
    if recompute_100_fid:
        metrics_100, _features100, _preview100 = evaluate_rollout(
            "prior",
            decoder,
            prior100,
            dummy_images,
            real_labels,
            feature_state["real_features"],
            evaluator,
            frozen["class_to_idx"],
            config,
        )

    empirical_generator = torch.Generator(device="cpu").manual_seed(config.prior_seed + 1_101)
    empirical_indices = torch.randint(
        len(cache["train_latent"]), (count,), generator=empirical_generator
    )
    empirical_latent = cache["train_latent"][empirical_indices]
    real_condition = condition_embeddings(
        decoder, val_latent, batch_size=config.eval_batch_size
    )
    empirical_condition = condition_embeddings(
        decoder, empirical_latent, batch_size=config.eval_batch_size
    )
    prior100_condition = condition_embeddings(
        decoder, prior100, batch_size=config.eval_batch_size
    )
    prior_nfe_condition = condition_embeddings(
        decoder, prior_audit, batch_size=config.eval_batch_size
    )
    payload = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "prior_replicate": int(config.prior_replicate),
        "audit_nfe": int(nfe),
        "formal_nfe100_fid_saved": float(saved_summary["end_to_end_feature_fid"]),
        "formal_nfe100_fid_from_saved_features": float(formal_fid),
        "formal_saved_feature_fid_abs_diff": abs(
            float(formal_fid) - float(saved_summary["end_to_end_feature_fid"])
        ),
        "audit_nfe_fid": float(metrics_nfe["feature_fid"]),
        "audit_nfe_minus_formal_fid": float(
            metrics_nfe["feature_fid"] - saved_summary["end_to_end_feature_fid"]
        ),
        "audit_nfe_predicted_class_entropy": float(
            metrics_nfe["predicted_class_entropy"]
        ),
        "frozen_decoder_sha256": state_dict_sha256(decoder),
        "frozen_decoder_matches_formal": bool(
            state_dict_sha256(decoder) == saved_summary["frozen_decoder_sha256"]
        ),
        **regenerated_differences,
        **distribution_comparison(
            val_latent,
            prior_audit,
            seed=config.prior_seed + 1_701,
            prefix=f"raw_nfe{int(nfe)}",
        ),
        **distribution_comparison(
            real_condition,
            empirical_condition,
            seed=config.prior_seed + 1_703,
            prefix="condition_empirical",
        ),
        **distribution_comparison(
            real_condition,
            prior100_condition,
            seed=config.prior_seed + 1_709,
            prefix="condition_prior100",
        ),
        **distribution_comparison(
            real_condition,
            prior_nfe_condition,
            seed=config.prior_seed + 1_711,
            prefix=f"condition_prior{int(nfe)}",
        ),
    }
    if metrics_100 is not None:
        payload["independent_nfe100_fid"] = float(metrics_100["feature_fid"])
        payload["independent_nfe100_fid_abs_diff"] = abs(
            float(metrics_100["feature_fid"])
            - float(saved_summary["end_to_end_feature_fid"])
        )
    if not all(
        torch.isfinite(torch.tensor(float(value)))
        for value in payload.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        raise FloatingPointError("non-finite audit metric")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    torch.save(
        {"real_features": feature_state["real_features"], "generated_features": features_nfe},
        run / f"features_prior_nfe{int(nfe)}.pt",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=200)
    parser.add_argument("--recompute-100-fid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return audit_run(
        args.run,
        device_name=args.device,
        nfe=args.nfe,
        recompute_100_fid=args.recompute_100_fid,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
