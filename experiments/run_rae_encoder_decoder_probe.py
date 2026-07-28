"""Fit clean reverse-hierarchy projectors and test them on RAE latent shifts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    configure_fp32,
    load_named_dataset,
    pick_dataset_images,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.rae_encoder_decoder_atlas import (  # noqa: E402
    RidgeMoments,
    fit_ridge_map,
    linear_probe_scores,
    spearman_correlation,
)
from experiments.run_rae_encoder_decoder_atlas import (  # noqa: E402
    _decode_image_and_states,
    _encoder_states,
    _setup_device,
    _stable_noise,
)


DEFAULT_ATLAS = (
    Path.home()
    / "data/eqvae/experiments/rae_encoder_decoder_atlas/"
    "dinov2_clean512_generated256_seed20260718"
)


def _reduce_fit_moments(moments: list[RidgeMoments]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for item in moments:
        for tensor in item.tensors():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _append_scores(
    rows: list[dict[str, object]],
    source: str,
    sample_ids: list[int],
    encoder_states: tuple[torch.Tensor, ...],
    decoder_states: tuple[torch.Tensor, ...],
    anchors: list[tuple[int, int]],
    mappings: list[torch.Tensor],
) -> None:
    for (decoder_layer, encoder_layer), mapping in zip(anchors, mappings):
        error, cosine = linear_probe_scores(
            decoder_states[decoder_layer],
            encoder_states[encoder_layer],
            mapping,
        )
        for offset, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "source": source,
                    "sample_id": int(sample_id),
                    "decoder_layer": int(decoder_layer),
                    "encoder_layer": int(encoder_layer),
                    "relative_error": float(error[offset].cpu()),
                    "cosine": float(cosine[offset].cpu()),
                }
            )


@torch.no_grad()
def run(
    atlas_dir: Path,
    *,
    batch_size: int,
    ridge: float,
    anchor_decoder_layers: tuple[int, ...],
    device_name: str,
    run_name: str,
) -> dict[str, object]:
    configure_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, world_size, device = _setup_device(device_name)
    atlas_dir = atlas_dir.expanduser()
    config = json.loads((atlas_dir / "config.json").read_text(encoding="utf-8"))
    matrices = torch.load(atlas_dir / "atlas_matrices.pt", map_location="cpu")
    calibration_atlas = matrices["clean_source_calibration"]
    anchors = [
        (decoder, int(calibration_atlas[:, decoder].argmax()))
        for decoder in anchor_decoder_layers
    ]
    output_dir = atlas_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_named_dataset(
        config["dataset_name"],
        config["data_root"],
        split=config["dataset_split"],
        dataset_path=config["dataset_path"],
    )
    indices = [int(index) for index in config["indices"]]
    calibration_count = int(config["calibration_count"])
    clean_count = int(config["clean_count"])
    generated_count = int(config["generated_count"])
    generated_root = Path(config["generated_root"]).expanduser()
    generated_paths = tuple(config["generated_paths"])
    sigma = max(float(value) for value in config["gaussian_sigmas"])
    shrink_scale = float(config["shrink_scale"])
    seed = int(config["seed"])

    adapter = load_rae_adapter(
        "rae_dinov2",
        repo_path=config["rae_repo_path"],
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    adapter.model.requires_grad_(False)
    fit_moments = [RidgeMoments.zeros(1152, 768, device=device) for _ in anchors]
    calibration_positions = list(range(calibration_count))[rank::world_size]
    calibration_indices = [indices[position] for position in calibration_positions]
    started = time.time()
    for start in range(0, len(calibration_indices), batch_size):
        batch_indices = calibration_indices[start : start + batch_size]
        images, _ = pick_dataset_images(
            dataset,
            indices=batch_indices,
            count=len(batch_indices),
            image_size=int(config["image_size"]),
        )
        images = images.to(device=device, dtype=torch.float32)
        encoder_states, latent = _encoder_states(adapter, images)
        _, decoder_states = _decode_image_and_states(adapter.model, latent)
        for moments, (decoder_layer, encoder_layer) in zip(fit_moments, anchors):
            moments.update(
                decoder_states[decoder_layer],
                encoder_states[encoder_layer],
            )
    _reduce_fit_moments(fit_moments)
    mappings = []
    ridge_scales = []
    for moments in fit_moments:
        mapping, scale = fit_ridge_map(moments, ridge=ridge)
        mappings.append(mapping)
        ridge_scales.append(scale)
    if rank == 0:
        print(
            f"[probe fit] anchors={anchors}, elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    score_rows: list[dict[str, object]] = []
    positions = list(range(clean_count))[rank::world_size]
    local_indices = [indices[position] for position in positions]
    for start in range(0, len(local_indices), batch_size):
        batch_indices = local_indices[start : start + batch_size]
        batch_positions = positions[start : start + batch_size]
        images, _ = pick_dataset_images(
            dataset,
            indices=batch_indices,
            count=len(batch_indices),
            image_size=int(config["image_size"]),
        )
        images = images.to(device=device, dtype=torch.float32)
        encoder_source, latent = _encoder_states(adapter, images)
        reconstructed, decoder_clean = _decode_image_and_states(adapter.model, latent)
        encoder_cycle, _ = _encoder_states(adapter, reconstructed)
        for split_name, use_calibration in (("calibration", True), ("test", False)):
            mask = torch.tensor(
                [
                    (position < calibration_count) == use_calibration
                    for position in batch_positions
                ],
                device=device,
            )
            if not bool(mask.any()):
                continue
            sample_ids = [
                position
                for position in batch_positions
                if (position < calibration_count) == use_calibration
            ]
            _append_scores(
                score_rows,
                f"clean_source_{split_name}",
                sample_ids,
                tuple(state[mask] for state in encoder_source),
                tuple(state[mask] for state in decoder_clean),
                anchors,
                mappings,
            )
            if not use_calibration:
                _append_scores(
                    score_rows,
                    "clean_cycle_test",
                    sample_ids,
                    tuple(state[mask] for state in encoder_cycle),
                    tuple(state[mask] for state in decoder_clean),
                    anchors,
                    mappings,
                )
                test_latent = latent[mask]
                noise = _stable_noise(test_latent, sample_ids, seed + 17)
                _, decoder_noisy = _decode_image_and_states(
                    adapter.model, test_latent + sigma * noise
                )
                _append_scores(
                    score_rows,
                    f"gaussian_{sigma:.2f}_source_test",
                    sample_ids,
                    tuple(state[mask] for state in encoder_source),
                    decoder_noisy,
                    anchors,
                    mappings,
                )
                _, decoder_shrunk = _decode_image_and_states(
                    adapter.model, shrink_scale * test_latent
                )
                _append_scores(
                    score_rows,
                    f"shrink_{shrink_scale:.2f}_source_test",
                    sample_ids,
                    tuple(state[mask] for state in encoder_source),
                    decoder_shrunk,
                    anchors,
                    mappings,
                )

    for path_name in generated_paths:
        payload_path = generated_root / f"0b_generated_latents_{path_name}_n256_s50.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        endpoints = payload["latents"][:generated_count]
        generated_indices = list(range(generated_count))[rank::world_size]
        local_endpoints = endpoints[rank::world_size]
        for start in range(0, len(local_endpoints), batch_size):
            latent = local_endpoints[start : start + batch_size].to(device)
            sample_ids = generated_indices[start : start + batch_size]
            decoded, decoder_states = _decode_image_and_states(adapter.model, latent)
            encoder_states, _ = _encoder_states(adapter, decoded)
            _append_scores(
                score_rows,
                f"generated_{path_name}_cycle",
                sample_ids,
                encoder_states,
                decoder_states,
                anchors,
                mappings,
            )

    rank_path = output_dir / f"probe_scores_rank{rank:02d}.csv"
    pd.DataFrame(score_rows).to_csv(rank_path, index=False)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    result: dict[str, object] = {"rank": rank, "world_size": world_size}
    if rank == 0:
        scores = pd.concat(
            [pd.read_csv(output_dir / f"probe_scores_rank{item:02d}.csv") for item in range(world_size)],
            ignore_index=True,
        )
        scores.to_csv(output_dir / "probe_scores.csv", index=False)
        summary = (
            scores.groupby(["source", "decoder_layer", "encoder_layer"], as_index=False)
            .agg(
                relative_error_mean=("relative_error", "mean"),
                relative_error_median=("relative_error", "median"),
                cosine_mean=("cosine", "mean"),
                cosine_median=("cosine", "median"),
                n=("relative_error", "size"),
            )
        )
        summary.to_csv(output_dir / "probe_anchor_summary.csv", index=False)
        source_summary = (
            scores.groupby("source", as_index=False)
            .agg(
                relative_error_mean=("relative_error", "mean"),
                relative_error_median=("relative_error", "median"),
                cosine_mean=("cosine", "mean"),
                cosine_median=("cosine", "median"),
                n=("relative_error", "size"),
            )
        )
        quality_path = Path(config["generation_metrics"]).expanduser()
        ordering: dict[str, object] = {"available": False}
        if quality_path.exists():
            quality = pd.read_csv(quality_path)
            quality["source"] = "generated_" + quality["source"].astype(str) + "_cycle"
            source_summary = source_summary.merge(
                quality[["source", "fid_5k", "kid_5k"]],
                on="source",
                how="left",
            )
            generated = source_summary.dropna(subset=["fid_5k"])
            if len(generated) >= 3:
                worst_fid_source = str(
                    generated.loc[generated["fid_5k"].idxmax(), "source"]
                )
                worst_probe_source = str(
                    generated.loc[
                        generated["relative_error_mean"].idxmax(), "source"
                    ]
                )
                ordering = {
                    "available": True,
                    "n": len(generated),
                    "probe_error_vs_fid_spearman": spearman_correlation(
                        torch.tensor(generated["relative_error_mean"].to_numpy()),
                        torch.tensor(generated["fid_5k"].to_numpy()),
                    ),
                    "probe_cosine_vs_fid_spearman": spearman_correlation(
                        torch.tensor(generated["cosine_mean"].to_numpy()),
                        torch.tensor(generated["fid_5k"].to_numpy()),
                    ),
                    "worst_fid_source": worst_fid_source,
                    "worst_probe_source": worst_probe_source,
                    "worst_path_identified": worst_fid_source == worst_probe_source,
                    "rows": generated.to_dict(orient="records"),
                }
        source_summary.to_csv(output_dir / "probe_source_summary.csv", index=False)
        values = source_summary.set_index("source")
        train_error = float(values.loc["clean_source_calibration", "relative_error_mean"])
        test_error = float(values.loc["clean_source_test", "relative_error_mean"])
        gate = {
            "clean_test_error": test_error,
            "train_test_ratio": test_error / max(train_error, 1e-12),
            "clean_generalization_passed": test_error <= 1.15 * train_error,
            "generation_quality_tracking_passed": bool(
                ordering.get("available")
                and float(ordering["probe_error_vs_fid_spearman"]) >= 0.80 - 1e-12
                and bool(ordering["worst_path_identified"])
            ),
        }
        gate["passed"] = bool(
            gate["clean_generalization_passed"]
            and gate["generation_quality_tracking_passed"]
        )
        torch.save(
            {
                "anchors": anchors,
                "mappings": [mapping.cpu() for mapping in mappings],
                "ridge": ridge,
                "ridge_scales": ridge_scales,
            },
            output_dir / "probe_maps.pt",
        )
        report = {
            "anchors": anchors,
            "ridge": ridge,
            "ridge_scales": ridge_scales,
            "gate": gate,
            "generation_ordering": ordering,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        (output_dir / "probe_decision.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"run_dir": str(output_dir), **report}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--anchor-decoder-layers", nargs="+", type=int, default=[0, 7, 14, 21, 28])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default="full_linear_probe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.atlas_dir,
        batch_size=args.batch_size,
        ridge=args.ridge,
        anchor_decoder_layers=tuple(args.anchor_decoder_layers),
        device_name=args.device,
        run_name=args.run_name,
    )
