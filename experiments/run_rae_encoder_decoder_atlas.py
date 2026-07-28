"""Run a no-training RAE encoder-decoder cross-layer alignment atlas."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    configure_fp32,
    extract_vit_stage_latents,
    load_named_dataset,
    pick_dataset_images,
    split_indices,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.rae_decoder_risk_phase0 import _latent_to_decoder_tokens  # noqa: E402
from experiments.rae_encoder_decoder_atlas import (  # noqa: E402
    CKAMoments,
    compare_atlases,
    paired_and_mismatched_cka,
    spearman_correlation,
    summarize_atlas,
)


DEFAULT_GENERATED_ROOT = (
    Path.home() / "data/eqvae/experiments/rae_decoder_risk_phase0"
)


@dataclass(frozen=True)
class AtlasConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "test"
    image_size: int = 256
    clean_count: int = 512
    calibration_count: int = 256
    batch_size: int = 4
    seed: int = 20260718
    gaussian_sigmas: tuple[float, ...] = (0.1, 0.3)
    shrink_scale: float = 0.75
    generated_count: int = 256
    generated_paths: tuple[str, ...] = ("static", "random", "annealed", "reverse")
    generated_root: Path = DEFAULT_GENERATED_ROOT
    generation_metrics: Path = DEFAULT_GENERATED_ROOT / "0b_closure_vs_generation.csv"
    device: str = "cuda:0"
    rae_repo_path: str = "external/RAE"
    output_root: Path = Path.home() / "data/eqvae/experiments/rae_encoder_decoder_atlas"
    run_name: str = "dinov2_clean512_generated256_seed20260718"


def _setup_device(requested: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        return dist.get_rank(), world_size, device
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def _encoder_states(
    adapter,
    images: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    stages = extract_vit_stage_latents(
        adapter,
        images,
        hidden_indices=tuple(range(13)),
        include_rae_normalized=False,
    )
    states = tuple(
        stages[f"hidden_{index}"].flatten(2).transpose(1, 2).contiguous()
        for index in range(13)
    )
    final_raw = stages["final_raw"]
    rae = adapter.model
    if bool(getattr(rae, "do_normalization", False)):
        latent_mean = getattr(rae, "latent_mean", None)
        latent_var = getattr(rae, "latent_var", None)
        mean = latent_mean.to(final_raw) if latent_mean is not None else 0.0
        var = latent_var.to(final_raw) if latent_var is not None else 1.0
        latent = (final_raw - mean) / torch.sqrt(
            var + float(getattr(rae, "eps", 1e-5))
        )
    else:
        latent = final_raw
    return states, latent.contiguous()


def _decode_image_and_states(
    rae: torch.nn.Module,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    output = rae.decoder(
        _latent_to_decoder_tokens(rae, latent),
        drop_cls_token=False,
        output_hidden_states=True,
    )
    if output.hidden_states is None:
        raise RuntimeError("RAE decoder did not expose hidden states")
    image = rae.decoder.unpatchify(output.logits)
    image = image * rae.encoder_std.to(image) + rae.encoder_mean.to(image)
    image_m11 = image.clamp(0.0, 1.0) * 2.0 - 1.0
    states = tuple(state[:, 1:].float() for state in output.hidden_states)
    return image_m11, states


def _select_states(
    states: Sequence[torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(state[mask] for state in states)


def _accumulate(
    moments: CKAMoments,
    encoder_states: Sequence[torch.Tensor],
    decoder_states: Sequence[torch.Tensor],
) -> None:
    if not encoder_states or len(encoder_states[0]) == 0:
        return
    paired, mismatched = paired_and_mismatched_cka(encoder_states, decoder_states)
    moments.update(paired, mismatched)


def _stable_noise(
    latent: torch.Tensor,
    positions: Sequence[int],
    seed: int,
) -> torch.Tensor:
    samples = []
    shape = tuple(latent.shape[1:])
    for position in positions:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * int(position)
        )
        samples.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(samples).to(device=latent.device)


def _source_names(config: AtlasConfig) -> list[str]:
    names = [
        "clean_source_calibration",
        "clean_source_test",
        "clean_cycle_calibration",
        "clean_cycle_test",
    ]
    names.extend(
        f"gaussian_{sigma:.2f}_source_test" for sigma in config.gaussian_sigmas
    )
    names.append(f"shrink_{config.shrink_scale:.2f}_source_test")
    names.extend(f"generated_{path}_cycle" for path in config.generated_paths)
    return names


def _reduce_moments(moments: dict[str, CKAMoments]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for source in sorted(moments):
        for tensor in moments[source].tensors():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _rows_and_matrices(
    moments: dict[str, CKAMoments],
) -> tuple[pd.DataFrame, dict[str, torch.Tensor], list[dict[str, object]]]:
    rows = []
    matrices = {}
    summaries = []
    for source, source_moments in moments.items():
        stats = source_moments.summary()
        excess = stats["excess_mean"].detach().cpu()
        matrices[source] = excess
        structural = summarize_atlas(excess)
        summaries.append(
            {
                "source": source,
                "paired_count": stats["paired_count"],
                "mismatched_count": stats["mismatched_count"],
                **structural,
            }
        )
        for encoder in range(excess.shape[0]):
            for decoder in range(excess.shape[1]):
                rows.append(
                    {
                        "source": source,
                        "encoder_layer": encoder,
                        "decoder_layer": decoder,
                        "paired_cka": float(stats["paired_mean"][encoder, decoder]),
                        "paired_std": float(stats["paired_std"][encoder, decoder]),
                        "mismatched_cka": float(
                            stats["mismatched_mean"][encoder, decoder]
                        ),
                        "mismatched_std": float(
                            stats["mismatched_std"][encoder, decoder]
                        ),
                        "excess_cka": float(excess[encoder, decoder]),
                    }
                )
    return pd.DataFrame(rows), matrices, summaries


def _comparison_rows(
    config: AtlasConfig,
    matrices: dict[str, torch.Tensor],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []

    def add(reference: str, candidate: str) -> None:
        rows.append(
            {
                "reference": reference,
                "candidate": candidate,
                **compare_atlases(matrices[reference], matrices[candidate]),
            }
        )

    add("clean_source_calibration", "clean_source_test")
    add("clean_cycle_calibration", "clean_cycle_test")
    for sigma in config.gaussian_sigmas:
        add("clean_source_test", f"gaussian_{sigma:.2f}_source_test")
    add("clean_source_test", f"shrink_{config.shrink_scale:.2f}_source_test")
    for path in config.generated_paths:
        add("clean_cycle_test", f"generated_{path}_cycle")

    frame = pd.DataFrame(rows)
    generation = frame[frame["candidate"].str.startswith("generated_")].copy()
    ordering: dict[str, object] = {"available": False}
    if config.generation_metrics.exists() and not generation.empty:
        quality = pd.read_csv(config.generation_metrics)
        quality = quality.copy()
        quality["candidate"] = "generated_" + quality["source"].astype(str) + "_cycle"
        generation = generation.merge(
            quality[["candidate", "fid_5k", "kid_5k"]],
            on="candidate",
            how="left",
        )
        frame = frame.merge(
            quality[["candidate", "fid_5k", "kid_5k"]],
            on="candidate",
            how="left",
        )
        valid = generation.dropna(subset=["fid_5k"])
        if len(valid) >= 3:
            distance = torch.tensor(valid["rms_distance"].to_numpy())
            fid = torch.tensor(valid["fid_5k"].to_numpy())
            ordering = {
                "available": True,
                "n": len(valid),
                "atlas_distance_vs_fid_spearman": spearman_correlation(distance, fid),
                "rows": valid.to_dict(orient="records"),
            }
    return frame, ordering


def _plot_atlases(
    matrices: dict[str, torch.Tensor],
    summaries: list[dict[str, object]],
    path: Path,
) -> None:
    names = list(matrices)
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(24, 6.4 * rows), squeeze=False)
    values = torch.cat([matrix.flatten() for matrix in matrices.values()])
    limit = max(float(torch.quantile(values.abs(), 0.99)), 1e-4)
    summary_by_name = {str(row["source"]): row for row in summaries}
    image = None
    for axis, name in zip(axes.flat, names):
        matrix = matrices[name]
        image = axis.imshow(
            matrix.numpy(),
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        summary = summary_by_name[name]
        axis.set_title(
            f"{name}\nreverse rho={summary['reverse_spearman_soft']:.3f}, "
            f"peak={summary['mean_peak_score']:.3f}"
        )
        axis.set_xlabel("Decoder hidden state")
        axis.set_ylabel("Encoder hidden state")
        axis.set_xticks(range(0, matrix.shape[1], 4))
        axis.set_yticks(range(matrix.shape[0]))
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.06, top=0.94, hspace=0.48, wspace=0.24)
    if image is not None:
        color_axis = figure.add_axes((0.91, 0.12, 0.018, 0.76))
        figure.colorbar(image, cax=color_axis, label="Paired CKA - mismatched CKA")
    figure.suptitle("RAE encoder-decoder cross-layer atlas", fontsize=18, y=0.985)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _premise_gate(
    matrices: dict[str, torch.Tensor],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    summary = {str(row["source"]): row for row in summaries}
    stability = compare_atlases(
        matrices["clean_source_calibration"], matrices["clean_source_test"]
    )
    test = summary["clean_source_test"]
    checks = {
        "sample_specific_peak": float(test["mean_peak_score"]) >= 0.05,
        "split_atlas_pearson": float(stability["pearson"]) >= 0.90,
        "split_mapping_within_one": float(stability["within_one_mapping_rate"]) >= 0.75,
        "reverse_soft_hierarchy": float(test["reverse_spearman_soft"]) >= 0.50,
        "reverse_argmax_hierarchy": float(test["reverse_spearman_argmax"]) >= 0.50,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "clean_split_stability": stability,
        "interpretation": (
            "stable reverse hierarchy exists; a small hierarchical-distillation pilot is justified"
            if all(checks.values())
            else "the clean reverse-hierarchy premise is not strong enough to justify decoder training"
        ),
    }


@torch.no_grad()
def run(config: AtlasConfig) -> dict[str, object]:
    if not 0 < config.calibration_count < config.clean_count:
        raise ValueError("calibration_count must be between zero and clean_count")
    configure_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, world_size, device = _setup_device(config.device)
    output_dir = config.output_root.expanduser() / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_named_dataset(
        config.dataset_name,
        config.data_root,
        split=config.dataset_split,
        dataset_path=config.dataset_path,
    )
    indices = split_indices(len(dataset), config.clean_count, config.seed)
    positions = list(range(config.clean_count))[rank::world_size]
    local_indices = [indices[position] for position in positions]
    adapter = load_rae_adapter(
        "rae_dinov2",
        repo_path=config.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    adapter.model.requires_grad_(False)

    source_names = _source_names(config)
    moments = {
        source: CKAMoments.zeros(13, 29, device=device) for source in source_names
    }
    started = time.time()
    processed = 0
    for start in range(0, len(local_indices), config.batch_size):
        batch_indices = local_indices[start : start + config.batch_size]
        batch_positions = positions[start : start + config.batch_size]
        images, _ = pick_dataset_images(
            dataset,
            indices=batch_indices,
            count=len(batch_indices),
            image_size=config.image_size,
        )
        images = images.to(device=device, dtype=torch.float32)
        encoder_source, clean_latent = _encoder_states(adapter, images)
        reconstructed, decoder_clean = _decode_image_and_states(
            adapter.model, clean_latent
        )
        encoder_cycle, _ = _encoder_states(adapter, reconstructed)

        position_tensor = torch.tensor(batch_positions, device=device)
        calibration = position_tensor < config.calibration_count
        test = ~calibration
        for split_name, mask in (("calibration", calibration), ("test", test)):
            _accumulate(
                moments[f"clean_source_{split_name}"],
                _select_states(encoder_source, mask),
                _select_states(decoder_clean, mask),
            )
            _accumulate(
                moments[f"clean_cycle_{split_name}"],
                _select_states(encoder_cycle, mask),
                _select_states(decoder_clean, mask),
            )

        if bool(test.any()):
            test_positions = [
                position for position in batch_positions if position >= config.calibration_count
            ]
            test_latent = clean_latent[test]
            source_test = _select_states(encoder_source, test)
            noise = _stable_noise(test_latent, test_positions, config.seed + 17)
            for sigma in config.gaussian_sigmas:
                _, decoder_noisy = _decode_image_and_states(
                    adapter.model, test_latent + float(sigma) * noise
                )
                _accumulate(
                    moments[f"gaussian_{sigma:.2f}_source_test"],
                    source_test,
                    decoder_noisy,
                )
            _, decoder_shrunk = _decode_image_and_states(
                adapter.model, float(config.shrink_scale) * test_latent
            )
            _accumulate(
                moments[f"shrink_{config.shrink_scale:.2f}_source_test"],
                source_test,
                decoder_shrunk,
            )

        processed += len(batch_indices)
        if rank == 0 and (
            processed % max(config.batch_size * 8, 1) == 0
            or processed == len(local_indices)
        ):
            print(
                f"[rank0 clean] {processed}/{len(local_indices)}, "
                f"elapsed={(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    for path_name in config.generated_paths:
        payload_path = (
            config.generated_root.expanduser()
            / f"0b_generated_latents_{path_name}_n256_s50.pt"
        )
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        endpoints = payload["latents"][: config.generated_count]
        local_endpoints = endpoints[rank::world_size]
        generated_processed = 0
        for start in range(0, len(local_endpoints), config.batch_size):
            latent = local_endpoints[start : start + config.batch_size].to(
                device=device,
                dtype=torch.float32,
            )
            decoded, decoder_states = _decode_image_and_states(adapter.model, latent)
            encoder_states, _ = _encoder_states(adapter, decoded)
            _accumulate(
                moments[f"generated_{path_name}_cycle"],
                encoder_states,
                decoder_states,
            )
            generated_processed += len(latent)
        if rank == 0:
            print(
                f"[rank0 generated:{path_name}] {generated_processed}/{len(local_endpoints)}, "
                f"elapsed={(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    _reduce_moments(moments)
    result: dict[str, object] = {"rank": rank, "world_size": world_size}
    if rank == 0:
        config_payload = asdict(config)
        for key in ("generated_root", "generation_metrics", "output_root"):
            config_payload[key] = str(config_payload[key])
        config_payload["indices"] = indices
        config_payload["world_size"] = world_size
        (output_dir / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metric_frame, matrices, summaries = _rows_and_matrices(moments)
        comparison_frame, ordering = _comparison_rows(config, matrices)
        gate = _premise_gate(matrices, summaries)
        metric_frame.to_csv(output_dir / "atlas_metrics.csv", index=False)
        comparison_frame.to_csv(output_dir / "atlas_comparisons.csv", index=False)
        pd.DataFrame(summaries).to_csv(output_dir / "atlas_summary.csv", index=False)
        torch.save(matrices, output_dir / "atlas_matrices.pt")
        _plot_atlases(matrices, summaries, output_dir / "atlas_heatmaps.png")
        report = {
            "premise_gate": gate,
            "generation_ordering": ordering,
            "summaries": summaries,
        }
        (output_dir / "decision.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "run_dir": str(output_dir),
            "world_size": world_size,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "premise_gate": gate,
            "generation_ordering": ordering,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args() -> AtlasConfig:
    defaults = AtlasConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=defaults.dataset_name)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--dataset-path", default=defaults.dataset_path)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--clean-count", type=int, default=defaults.clean_count)
    parser.add_argument("--calibration-count", type=int, default=defaults.calibration_count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--gaussian-sigmas", nargs="+", type=float, default=list(defaults.gaussian_sigmas))
    parser.add_argument("--shrink-scale", type=float, default=defaults.shrink_scale)
    parser.add_argument("--generated-count", type=int, default=defaults.generated_count)
    parser.add_argument("--generated-paths", nargs="+", default=list(defaults.generated_paths))
    parser.add_argument("--generated-root", type=Path, default=defaults.generated_root)
    parser.add_argument("--generation-metrics", type=Path, default=defaults.generation_metrics)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--rae-repo-path", default=defaults.rae_repo_path)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    args = parser.parse_args()
    return AtlasConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        clean_count=args.clean_count,
        calibration_count=args.calibration_count,
        batch_size=args.batch_size,
        seed=args.seed,
        gaussian_sigmas=tuple(args.gaussian_sigmas),
        shrink_scale=args.shrink_scale,
        generated_count=args.generated_count,
        generated_paths=tuple(args.generated_paths),
        generated_root=args.generated_root,
        generation_metrics=args.generation_metrics,
        device=args.device,
        rae_repo_path=args.rae_repo_path,
        output_root=args.output_root,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    run(parse_args())
