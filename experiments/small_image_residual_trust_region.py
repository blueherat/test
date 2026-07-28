"""Unseen validation/test trust-region selection for frozen residual adapters."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    evaluate_rollouts,
    train_feature_classifier,
)
from experiments.small_image_basis_mechanism import (  # noqa: E402
    _load_run,
    _load_study_config,
)
from experiments.small_image_basis_transport import load_small_image_tensors  # noqa: E402
from experiments.small_image_residual_adapter import ProjectedResidualField  # noqa: E402
from experiments.small_image_training_rescue import _toy_config  # noqa: E402


@dataclass(frozen=True)
class ResidualTrustRegionConfig:
    residual_dir: Path
    output_root: Path = (
        Path.home() / "data/eqvae/experiments/small_image_residual_trust_region"
    )
    bases: tuple[str, ...] = ("dct",)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    source_test_count: int = 1024
    validation_count: int = 1024
    test_count: int = 1024
    validation_seed_offset: int = 31_000
    test_seed_offset: int = 41_000
    adapter_variant: str = "teacher_residual"
    save: bool = True


def unseen_pool_slices(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    source_count: int,
    validation_count: int,
    test_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = int(source_count) + int(validation_count) + int(test_count)
    if len(values) < required or len(indices) < required:
        raise ValueError("test pool is too small for disjoint source/validation/test slices")
    validation_start = int(source_count)
    test_start = validation_start + int(validation_count)
    return (
        values[validation_start:test_start],
        values[test_start:required],
        indices[validation_start:test_start],
        indices[test_start:required],
    )


def select_scale(validation: pd.DataFrame, scales: Sequence[float]) -> float:
    expected = {f"scale_{float(scale):.2f}" for scale in scales}
    available = set(validation["variant"])
    if expected != available:
        raise ValueError(f"validation scales disagree: expected={expected}, got={available}")
    ranked = validation.assign(
        scale=validation["variant"].str.removeprefix("scale_").astype(float)
    ).sort_values(["feature_fid", "scale"], ascending=[True, True])
    return float(ranked.iloc[0]["scale"])


def _scaled_model(
    baseline,
    analyzer,
    adapter_state: dict[str, torch.Tensor],
    *,
    adapter_width: int,
    depth: int,
    scale: float,
    device: torch.device,
) -> ProjectedResidualField:
    model = ProjectedResidualField(
        copy.deepcopy(baseline),
        analyzer,
        adapter_width=int(adapter_width),
        depth=int(depth),
        residual_scale=float(scale),
    ).to(device)
    model.adapter.load_state_dict(adapter_state)
    return model.eval()


def run_trust_region_pair(
    config: ResidualTrustRegionConfig,
    *,
    adapter_run_dir: Path,
    device_name: str,
    result_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    adapter_metadata = json.loads(
        (adapter_run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    seed = int(adapter_metadata["seed"])
    basis = str(adapter_metadata["basis"])
    source_run_dir = Path(adapter_metadata["source_run_dir"])
    study_dir = source_run_dir.parent
    study_config = _load_study_config(study_dir)
    root_config = json.loads(
        (config.residual_dir / "config.json").read_text(encoding="utf-8")
    )
    configure_fp32(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu"
    )
    total_test_count = (
        int(config.source_test_count)
        + int(config.validation_count)
        + int(config.test_count)
    )
    loaded = load_small_image_tensors(
        study_config.dataset,
        study_config.data_root,
        study_config.train_size,
        total_test_count,
        seed,
        download=False,
    )
    train = loaded["train"].to(device)
    validation, test, validation_indices, test_indices = unseen_pool_slices(
        loaded["test"],
        loaded["test_indices"],
        source_count=config.source_test_count,
        validation_count=config.validation_count,
        test_count=config.test_count,
    )
    validation = validation.to(device)
    test = test.to(device)
    test_labels = loaded["test_labels"][
        int(config.source_test_count + config.validation_count) : total_test_count
    ].to(device)
    raw_models, analyzer, _ = _load_run(source_run_dir, study_config, device)
    state = torch.load(adapter_run_dir / "state.pt", map_location="cpu", weights_only=False)
    adapter_state = state["adapters"][config.adapter_variant]
    adapter_width = int(root_config["adapter_width"])

    classifier, classifier_accuracy = train_feature_classifier(
        train,
        loaded["train_labels"].to(device),
        test,
        test_labels,
        epochs=study_config.classifier_epochs,
        batch_size=study_config.classifier_batch_size,
        seed=seed,
    )
    validation_models = {
        f"scale_{scale:.2f}": _scaled_model(
            raw_models["baseline"],
            analyzer,
            adapter_state,
            adapter_width=adapter_width,
            depth=study_config.depth,
            scale=scale,
            device=device,
        )
        for scale in config.scales
    }
    validation_summary, validation_bands, _ = evaluate_rollouts(
        validation_models,
        validation,
        classifier,
        analyzer,
        _toy_config(
            study_config,
            seed + int(config.validation_seed_offset),
            device,
            config.validation_count,
        ),
        loaded["normalization"],
    )
    selected_scale = select_scale(validation_summary, config.scales)
    for model in validation_models.values():
        model.cpu()
    test_models = {
        "baseline": _scaled_model(
            raw_models["baseline"],
            analyzer,
            adapter_state,
            adapter_width=adapter_width,
            depth=study_config.depth,
            scale=0.0,
            device=device,
        ),
        "selected": _scaled_model(
            raw_models["baseline"],
            analyzer,
            adapter_state,
            adapter_width=adapter_width,
            depth=study_config.depth,
            scale=selected_scale,
            device=device,
        ),
        "full": _scaled_model(
            raw_models["baseline"],
            analyzer,
            adapter_state,
            adapter_width=adapter_width,
            depth=study_config.depth,
            scale=1.0,
            device=device,
        ),
    }
    test_summary, test_bands, _ = evaluate_rollouts(
        test_models,
        test,
        classifier,
        analyzer,
        _toy_config(
            study_config,
            seed + int(config.test_seed_offset),
            device,
            config.test_count,
        ),
        loaded["normalization"],
    )
    baseline_values = test_summary[test_summary["variant"].eq("baseline")].iloc[0]
    lower_better = (
        "latent_swd",
        "decoded_pixel_swd",
        "feature_swd",
        "feature_fid",
    )
    for metric in lower_better:
        test_summary[f"{metric}_over_baseline"] = (
            test_summary[metric] / max(float(baseline_values[metric]), 1e-12)
        )
    identity = {
        "dataset": study_config.dataset,
        "basis": basis,
        "seed": seed,
        "selected_scale": selected_scale,
    }
    for frame in (validation_summary, validation_bands, test_summary, test_bands):
        for key, value in reversed(identity.items()):
            frame.insert(0, key, value)
    source_indices = set(
        int(value) for value in loaded["test_indices"][: config.source_test_count]
    )
    validation_index_set = set(int(value) for value in validation_indices)
    test_index_set = set(int(value) for value in test_indices)
    if (
        source_indices & validation_index_set
        or source_indices & test_index_set
        or validation_index_set & test_index_set
    ):
        raise RuntimeError("source, validation, and final-test image pools overlap")
    metadata: dict[str, object] = {
        **identity,
        "adapter_run_dir": str(adapter_run_dir),
        "source_run_dir": str(source_run_dir),
        "adapter_variant": config.adapter_variant,
        "classifier_accuracy": float(classifier_accuracy),
        "source_test_indices": sorted(source_indices),
        "validation_indices": [int(value) for value in validation_indices],
        "test_indices": [int(value) for value in test_indices],
        "validation_evaluation_seed": seed + int(config.validation_seed_offset),
        "test_evaluation_seed": seed + int(config.test_seed_offset),
        "scope": "unseen validation-selected frozen residual scale; independent final test",
    }
    if config.save:
        if result_dir is None:
            raise ValueError("result_dir is required when saving")
        run_dir = result_dir / f"{basis}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        validation_summary.to_csv(run_dir / "validation_summary.csv", index=False)
        validation_bands.to_csv(run_dir / "validation_bands.csv", index=False)
        test_summary.to_csv(run_dir / "test_summary.csv", index=False)
        test_bands.to_csv(run_dir / "test_bands.csv", index=False)
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    for model in test_models.values():
        model.cpu()
    raw_models["baseline"].cpu()
    analyzer.cpu()
    del train, validation, test, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return validation_summary, test_summary, metadata


def _run_group(
    tasks: Sequence[tuple[ResidualTrustRegionConfig, Path, str, Path | None]],
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    results = []
    for config, adapter_run_dir, device, result_dir in tasks:
        result = run_trust_region_pair(
            config,
            adapter_run_dir=adapter_run_dir,
            device_name=device,
            result_dir=result_dir,
        )
        selected = result[1].set_index("variant")
        metadata = result[2]
        print(
            f"done {metadata['basis']} seed={metadata['seed']}: "
            f"scale={metadata['selected_scale']:.2f}, "
            f"fid={selected.loc['selected', 'feature_fid_over_baseline']:.3f}, "
            f"full={selected.loc['full', 'feature_fid_over_baseline']:.3f}",
            flush=True,
        )
        results.append(result)
    return results


def run_trust_region_study(
    config: ResidualTrustRegionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    config = ResidualTrustRegionConfig(
        **{**asdict(config), "residual_dir": config.residual_dir.expanduser().resolve()}
    )
    adapter_dirs = sorted(
        path
        for basis in config.bases
        for path in config.residual_dir.glob(f"{basis}_seed*")
        if (path / "metadata.json").is_file()
    )
    if not adapter_dirs:
        raise ValueError("no matching residual-adapter runs found")
    result_dir = None
    if config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = (
            config.output_root.expanduser()
            / f"residual_trust_region_preregistered_{timestamp}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        serialized = asdict(config)
        serialized["residual_dir"] = str(config.residual_dir)
        serialized["output_root"] = str(config.output_root.expanduser())
        (result_dir / "config.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
    devices = config.devices or ("cpu",)
    tasks = [
        (config, path, devices[index % len(devices)], result_dir)
        for index, path in enumerate(adapter_dirs)
    ]
    grouped = [[] for _ in devices]
    for index, task in enumerate(tasks):
        grouped[index % len(devices)].append(task)
    results = []
    if len(devices) == 1:
        results = _run_group(grouped[0])
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_run_group, group) for group in grouped if group]
            for future in as_completed(futures):
                results.extend(future.result())
    validation = pd.concat([item[0] for item in results], ignore_index=True)
    test = pd.concat([item[1] for item in results], ignore_index=True)
    metadata = pd.DataFrame([item[2] for item in results]).sort_values(["basis", "seed"])
    if result_dir is not None:
        validation.to_csv(result_dir / "validation_summary.csv", index=False)
        test.to_csv(result_dir / "test_summary.csv", index=False)
        metadata.to_csv(result_dir / "metadata.csv", index=False)
    return validation, test, metadata, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bases", default="dct")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = ResidualTrustRegionConfig(
        residual_dir=args.residual_dir,
        output_root=args.output_root or ResidualTrustRegionConfig.output_root,
        bases=tuple(value.strip() for value in args.bases.split(",") if value.strip()),
        devices=tuple(value.strip() for value in args.devices.split(",") if value.strip())
        or ("cpu",),
        save=not args.no_save,
    )
    _, test, metadata, result_dir = run_trust_region_study(config)
    print(f"result_dir={result_dir}")
    selected = test[test["variant"].eq("selected")]
    metrics = [
        "feature_fid_over_baseline",
        "feature_swd_over_baseline",
        "latent_swd_over_baseline",
        "decoded_pixel_swd_over_baseline",
    ]
    print(selected.groupby("basis")[metrics].agg(["mean", "std", "min", "max"]).round(4))
    print(metadata[["basis", "seed", "selected_scale"]].to_string(index=False))


if __name__ == "__main__":
    main()
