"""Phase-0 numerical audit for the latent transport research protocol."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (
    configure_fp32,
    load_named_dataset,
    pick_dataset_images,
)
from baselines.visual_adapters import load_rae_adapter
from experiments.decoder_adapted_rae import _load_adapter_from_checkpoint
from experiments.latent_transport_paths import (
    bridge_commutation_defect,
    jvp_relative_error,
    relative_l2_per_sample,
)


@dataclass(frozen=True)
class ProtocolAuditConfig:
    adapter_checkpoint: Path = (
        Path.home()
        / "data/eqvae/artifacts/latent_adapter/"
        "dinov2_adapter_imagenet_train32768_val2048_testval2048_e6_seq_noleak/adapter.pt"
    )
    data_root: Path = Path("/data/shared")
    dataset_path: Path = Path("/data/shared/imagenet-1k")
    dataset_name: str = "imagenet_parquet"
    dataset_split: str = "validation"
    rae_repo_path: Path = Path("external/RAE")
    device: str = "cuda:0"
    seed: int = 20260718
    count: int = 2
    image_size: int = 256
    finite_difference_step: float = 1e-2
    output_root: Path = Path.home() / "data/eqvae/audits/latent_transport_phase0"


def _mean_max(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    return {"mean": float(value.mean()), "max": float(value.max())}


def _orthogonal_channel_map(value: torch.Tensor) -> torch.Tensor:
    signs = torch.ones(value.shape[1], device=value.device, dtype=value.dtype)
    signs[1::2] = -1
    return value.flip(1) * signs.view(1, -1, 1, 1)


def _pixel_errors(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = (prediction - target).abs().detach().float().cpu()
    return {"mean": float(error.mean()), "max": float(error.max())}


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    return {
        "channels": int(checkpoint.get("channels", 0)),
        "height": int(checkpoint.get("height", 0)),
        "width": int(checkpoint.get("width", 0)),
        "training_config": config,
    }


def run_protocol_audit(config: ProtocolAuditConfig) -> dict[str, object]:
    configure_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not config.adapter_checkpoint.exists():
        raise FileNotFoundError(config.adapter_checkpoint)

    dataset = load_named_dataset(
        config.dataset_name,
        str(config.data_root),
        config.dataset_split,
        download=False,
        dataset_path=str(config.dataset_path),
    )
    images, indices = pick_dataset_images(
        dataset,
        count=config.count,
        seed=config.seed,
        image_size=config.image_size,
    )
    images = images.to(device=device, dtype=torch.float32)

    rae = load_rae_adapter(
        "rae_dinov2",
        repo_path=config.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    for parameter in rae.model.parameters():
        parameter.requires_grad_(False)
    adapter = _load_adapter_from_checkpoint(
        str(config.adapter_checkpoint),
        channels=768,
        hidden_channels=None,
        blocks=None,
    ).to(device=device, dtype=torch.float32).eval()
    adapter.requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(config.seed)
    with torch.no_grad():
        latent = rae.encode(images)
        adapted = adapter(latent)
        cycle = adapter.inverse(adapted)
        base_reconstruction = rae.decode(latent)
        repeated_base_reconstruction = rae.decode(latent)
        cycle_reconstruction = rae.decode(cycle)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)

    cycle_error = relative_l2_per_sample(cycle, latent)
    times = torch.tensor(
        [0.1 + 0.8 * index / max(1, len(latent) - 1) for index in range(len(latent))],
        device=device,
        dtype=latent.dtype,
    )
    identity_defect = bridge_commutation_defect(latent, noise, times, lambda value: value)
    orthogonal_defect = bridge_commutation_defect(
        latent,
        noise,
        times,
        _orthogonal_channel_map,
    )
    adapter_defect = bridge_commutation_defect(latent, noise, times, adapter)

    # A single sample is enough for the expensive real-adapter finite-difference audit.
    point = ((1.0 - times[:1, None, None, None]) * latent[:1] + times[:1, None, None, None] * noise[:1]).detach()
    direction = (noise[:1] - latent[:1]).detach()
    actual_jvp_error = jvp_relative_error(
        adapter,
        point,
        direction,
        step=config.finite_difference_step,
    )

    checkpoint_metadata = _checkpoint_metadata(config.adapter_checkpoint)
    results = {
        "protocol_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "dataset": {
            "name": config.dataset_name,
            "split": config.dataset_split,
            "indices": [int(index) for index in indices],
            "count": len(indices),
        },
        "checkpoint": checkpoint_metadata,
        "execution_graph": {
            "latent": "z = normalized RAE.encode(x)",
            "adapted_latent": "y = f(z)",
            "decode": "x_hat = RAE.decode(f^{-1}(y))",
            "legacy_training_source": "eta ~ N(0, I)",
            "legacy_training_path": "(1-t) * f(z) + t * eta",
            "legacy_sampling_source": "eta ~ N(0, I)",
            "legacy_branch": "gaussian_straight",
        },
        "frozen": {
            "rae_all_requires_grad_false": not any(
                parameter.requires_grad for parameter in rae.model.parameters()
            ),
            "adapter_all_requires_grad_false": not any(
                parameter.requires_grad for parameter in adapter.parameters()
            ),
        },
        "numerics": {
            "cycle_relative_l2": _mean_max(cycle_error),
            "decoder_identity_pixel_abs": _pixel_errors(
                cycle_reconstruction,
                base_reconstruction,
            ),
            "decoder_repeat_pixel_abs": _pixel_errors(
                repeated_base_reconstruction,
                base_reconstruction,
            ),
            "jvp_finite_difference_relative_l2": _mean_max(actual_jvp_error),
            "identity_bridge_defect": _mean_max(identity_defect),
            "orthogonal_bridge_defect": _mean_max(orthogonal_defect),
            "adapter_bridge_defect": _mean_max(adapter_defect),
        },
    }
    thresholds = {
        "cycle_relative_l2_max": 1e-6,
        "decoder_identity_pixel_abs_mean": 1e-6,
        "decoder_repeat_pixel_abs_max": 1e-7,
        "jvp_finite_difference_relative_l2_max": 1e-4,
        "identity_bridge_defect_max": 1e-6,
        "orthogonal_bridge_defect_max": 1e-6,
    }
    observed = {
        "cycle_relative_l2_max": results["numerics"]["cycle_relative_l2"]["max"],
        "decoder_identity_pixel_abs_mean": results["numerics"]["decoder_identity_pixel_abs"]["mean"],
        "decoder_repeat_pixel_abs_max": results["numerics"]["decoder_repeat_pixel_abs"]["max"],
        "jvp_finite_difference_relative_l2_max": results["numerics"]["jvp_finite_difference_relative_l2"]["max"],
        "identity_bridge_defect_max": results["numerics"]["identity_bridge_defect"]["max"],
        "orthogonal_bridge_defect_max": results["numerics"]["orthogonal_bridge_defect"]["max"],
    }
    checks = {
        name: {"value": float(observed[name]), "threshold": threshold, "passed": observed[name] <= threshold}
        for name, threshold in thresholds.items()
    }
    results["acceptance"] = {
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }

    del adapter, rae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def save_results(results: dict[str, object], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "protocol_audit.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    defaults = ProtocolAuditConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", type=Path, default=defaults.adapter_checkpoint)
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--dataset-path", type=Path, default=defaults.dataset_path)
    parser.add_argument("--dataset-name", default=defaults.dataset_name)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--rae-repo-path", type=Path, default=defaults.rae_repo_path)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--count", type=int, default=defaults.count)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--finite-difference-step", type=float, default=defaults.finite_difference_step)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProtocolAuditConfig(**vars(args))
    results = run_protocol_audit(config)
    path = save_results(results, config.output_root)
    print(json.dumps({"result": str(path), "acceptance": results["acceptance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
