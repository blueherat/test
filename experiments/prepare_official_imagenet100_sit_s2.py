#!/usr/bin/env python3
"""Verify and adapt the official ImageNet-1K SiT-S/2 to ImageNet-100 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.official_imagenet100_sit_s2 import (
        DEFAULT_CACHE_MANIFEST,
        DEFAULT_INDEX_MANIFEST,
        DEFAULT_RAW_CHECKPOINT,
        DEFAULT_SUBSET_CHECKPOINT,
        HF_FILENAME,
        HF_REPOSITORY,
        HF_REVISION,
        PRETRAINED_SOURCE_FORMAT,
        RAW_CHECKPOINT_BYTES,
        RAW_CHECKPOINT_SHA256,
        SUBSET_CHECKPOINT_FORMAT,
        load_imagenet100_class_mapping,
        subset_official_state_dict,
        subset_original_labels,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from official_imagenet100_sit_s2 import (
        DEFAULT_CACHE_MANIFEST,
        DEFAULT_INDEX_MANIFEST,
        DEFAULT_RAW_CHECKPOINT,
        DEFAULT_SUBSET_CHECKPOINT,
        HF_FILENAME,
        HF_REPOSITORY,
        HF_REVISION,
        PRETRAINED_SOURCE_FORMAT,
        RAW_CHECKPOINT_BYTES,
        RAW_CHECKPOINT_SHA256,
        SUBSET_CHECKPOINT_FORMAT,
        load_imagenet100_class_mapping,
        subset_official_state_dict,
        subset_original_labels,
    )


def verify_raw_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != RAW_CHECKPOINT_BYTES:
        raise ValueError(
            f"official checkpoint size mismatch: expected {RAW_CHECKPOINT_BYTES}, "
            f"found {size}"
        )
    digest = base.sha256_file(path)
    if digest != RAW_CHECKPOINT_SHA256:
        raise ValueError(
            "official checkpoint SHA256 mismatch: "
            f"expected {RAW_CHECKPOINT_SHA256}, found {digest}"
        )
    return digest


@torch.inference_mode()
def equivalence_audit(
    *,
    full_model: torch.nn.Module,
    subset_model: torch.nn.Module,
    original_labels: list[int],
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(int(seed))
    state = torch.randn((4, *base.LATENT_SHAPE), generator=generator).to(device)
    time_value = torch.tensor([0.05, 0.3, 0.7, 0.95], device=device)
    subset_labels = torch.tensor([0, 17, 53, 99], device=device)
    full_labels = torch.tensor(
        [original_labels[index] for index in subset_labels.tolist()],
        device=device,
    )
    full_output = full_model(state, time_value, full_labels)
    subset_output = subset_model(state, time_value, subset_labels)
    class_max_abs = float((full_output - subset_output).abs().max().item())

    full_unconditional = full_model(
        state[:1], time_value[:1], torch.tensor([1_000], device=device)
    )
    subset_unconditional = subset_model(
        state[:1], time_value[:1], torch.tensor([100], device=device)
    )
    unconditional_max_abs = float(
        (full_unconditional - subset_unconditional).abs().max().item()
    )
    tolerance = 1e-6
    if class_max_abs > tolerance or unconditional_max_abs > tolerance:
        raise AssertionError(
            "class-subset checkpoint changes official outputs: "
            f"class={class_max_abs}, unconditional={unconditional_max_abs}"
        )
    return {
        "seed": int(seed),
        "device": str(device),
        "class_labels": subset_labels.cpu().tolist(),
        "original_imagenet_labels": full_labels.cpu().tolist(),
        "class_output_max_abs": class_max_abs,
        "unconditional_output_max_abs": unconditional_max_abs,
        "tolerance": tolerance,
        "passed": True,
    }


def existing_output_is_valid(path: Path, raw_sha256: str) -> bool:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("format") == SUBSET_CHECKPOINT_FORMAT
        and metadata.get("raw_sha256") == raw_sha256
        and metadata.get("output_sha256") == base.sha256_file(path)
        and metadata.get("equivalence_audit", {}).get("passed") is True
    )


def prepare(args: argparse.Namespace) -> dict[str, object]:
    raw_path = args.raw_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    index_manifest_path = args.index_manifest.expanduser().resolve()
    cache_manifest_path = args.cache_manifest.expanduser().resolve()
    official_repo = args.official_sit_repo.expanduser().resolve()
    raw_sha256 = verify_raw_checkpoint(raw_path)
    if not args.force and existing_output_is_valid(output_path, raw_sha256):
        metadata = json.loads(
            output_path.with_suffix(output_path.suffix + ".json").read_text(
                encoding="utf-8"
            )
        )
        print(json.dumps({"event": "reuse", **metadata}, indent=2), flush=True)
        return metadata

    classes = load_imagenet100_class_mapping(index_manifest_path)
    original_labels = subset_original_labels(classes)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
        raise ValueError(f"unsupported latent cache manifest: {cache_manifest_path}")

    raw_state = torch.load(
        raw_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("official checkpoint is not a non-empty state dict")
    subset_state = subset_official_state_dict(raw_state, original_labels)

    sit_module, source_metadata = base.load_official_sit_module(
        official_repo,
        verify_source=args.verify_sit_source,
    )
    full_model = sit_module.SiT_models["SiT-S/2"](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=1_000,
        class_dropout_prob=0.1,
    )
    subset_model = sit_module.SiT_models["SiT-S/2"](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=100,
        class_dropout_prob=0.1,
    )
    full_model.load_state_dict(raw_state, strict=True)
    subset_model.load_state_dict(subset_state, strict=True)
    device = torch.device(args.device)
    full_model = full_model.to(device).eval()
    subset_model = subset_model.to(device).eval()
    audit = equivalence_audit(
        full_model=full_model,
        subset_model=subset_model,
        original_labels=original_labels,
        device=device,
        seed=args.audit_seed,
    )
    subset_state = {
        key: value.detach().cpu() for key, value in subset_model.state_dict().items()
    }
    del full_model, subset_model, raw_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pretrained_source = {
        "format": PRETRAINED_SOURCE_FORMAT,
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "raw_checkpoint": str(raw_path),
        "raw_bytes": RAW_CHECKPOINT_BYTES,
        "raw_sha256": raw_sha256,
        "training_step": None,
        "training_step_note": "not published with the final state dict",
    }
    payload = {
        "protocol": base.LEGACY_PROTOCOL,
        "step": 0,
        "model": subset_state,
        "ema": subset_state,
        "config": {
            "cache_dir": str(cache_manifest_path.parent),
            "output_dir": str(output_path.parent),
            "official_sit_repo": str(official_repo),
            "model_name": "SiT-S/2",
            "prediction_target": "velocity",
            "loss_space": "velocity",
            "denominator_floor": 0.001,
            "cfg_dropout": 0.1,
            "num_classes": 100,
            "source_num_classes": 1_000,
            "seed": None,
        },
        "official_sit": source_metadata,
        "data_manifest_sha256": base.sha256_file(cache_manifest_path),
        "pretrained_source": pretrained_source,
        "class_subset": {
            "format": SUBSET_CHECKPOINT_FORMAT,
            "index_manifest": str(index_manifest_path),
            "index_manifest_sha256": base.sha256_file(index_manifest_path),
            "subset_labels": list(range(100)),
            "original_imagenet_labels": original_labels,
            "unconditional_source_label": 1_000,
            "unconditional_subset_label": 100,
            "equivalence_audit": audit,
        },
    }
    base.atomic_torch_save(payload, output_path)
    metadata = {
        "format": SUBSET_CHECKPOINT_FORMAT,
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": base.sha256_file(output_path),
        "raw_sha256": raw_sha256,
        "official_sit": source_metadata,
        "pretrained_source": pretrained_source,
        "class_count": 100,
        "equivalence_audit": audit,
    }
    base.atomic_json_dump(
        metadata,
        output_path.with_suffix(output_path.suffix + ".json"),
    )
    print(json.dumps({"event": "prepared", **metadata}, indent=2), flush=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBSET_CHECKPOINT)
    parser.add_argument("--index-manifest", type=Path, default=DEFAULT_INDEX_MANIFEST)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument(
        "--official-sit-repo",
        type=Path,
        default=base.DEFAULT_OFFICIAL_SIT_REPO,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-seed", type=int, default=20260817)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
