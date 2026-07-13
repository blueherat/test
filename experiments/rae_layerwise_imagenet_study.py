from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    P,
    configure_fp32,
    extract_vit_stage_latents,
    load_named_dataset,
    pick_dataset_images,
    relative_token_error,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402


@dataclass
class LayerwiseStudyConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "test"
    image_size: int = 256
    model_keys: Tuple[str, ...] = ("rae_dinov2", "rae_mae", "rae_siglip2")
    transforms: Tuple[str, ...] = ("rot90", "rot180", "rot270", "flip_h", "flip_v")
    hidden_indices: Tuple[int, ...] = tuple(range(13))
    count: int = 512
    batch_size: int = 8
    seed: int = 0
    center: str = "sample"
    device: str = "cuda:0"
    rae_repo_path: str = "external/RAE"
    rae_auto_clone: bool = False
    rae_auto_download: bool = False
    output_dir: str = "artifacts/layerwise_imagenet"
    run_name: str = ""
    save_detail: bool = True


def layer_order(stages: Dict[str, torch.Tensor], hidden_indices: Sequence[int]) -> List[str]:
    order = ["patch_pre_pos", "post_pos"]
    order.extend(f"hidden_{i}" for i in hidden_indices)
    order.extend(["final_raw", "rae_normalized"])
    return [name for name in order if name in stages]


def split_indices(total: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if total < count:
        raise ValueError(f"dataset has {total} images, less than requested {count}")
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(total)[:count]]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["model", "transform", "layer"], sort=False)["direct_error"]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


def write_config(run_dir: Path, cfg: LayerwiseStudyConfig, indices: Sequence[int], dataset_len: int) -> None:
    payload = asdict(cfg)
    payload["indices"] = [int(i) for i in indices]
    payload["dataset_len"] = int(dataset_len)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_summary(summary: pd.DataFrame, run_dir: Path, cfg: LayerwiseStudyConfig) -> Path:
    short = {
        "patch_pre_pos": "pre-pos",
        "post_pos": "post-pos",
        "final_raw": "final raw",
        "rae_normalized": "RAE norm",
    }
    first = next(iter(summary.groupby(["model", "transform"], sort=False)))[1]
    order = list(first["layer"])
    transforms = [g for g in cfg.transforms if g in set(summary["transform"])]
    fig, axes = plt.subplots(
        len(transforms),
        1,
        figsize=(max(14, 0.8 * len(order) + 4), 3.4 * len(transforms)),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(order))
    for row, transform in enumerate(transforms):
        ax = axes[row, 0]
        sub_t = summary[summary["transform"] == transform]
        for model, sub_m in sub_t.groupby("model", sort=False):
            rows = sub_m.set_index("layer").reindex(order)
            ax.errorbar(
                x,
                rows["mean"].to_numpy(dtype=float),
                yerr=rows["std"].fillna(0.0).to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                capsize=2,
                label=model,
            )
        ax.set_title(transform)
        ax.set_ylabel(f"direct error\ncenter={cfg.center}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=min(3, len(cfg.model_keys)))
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([short.get(name, name.replace("hidden_", "h")) for name in order], rotation=35, ha="right")
    axes[-1, 0].set_xlabel("layer")
    fig.suptitle(
        f"RAE layerwise direct equivariance on {cfg.dataset_name}/{cfg.dataset_split}, n={cfg.count}",
        y=0.995,
    )
    fig.tight_layout()
    out = run_dir / "layerwise_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def run_study(cfg: LayerwiseStudyConfig) -> Dict[str, str]:
    configure_fp32()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    dataset = load_named_dataset(
        cfg.dataset_name,
        cfg.data_root,
        split=cfg.dataset_split,
        dataset_path=cfg.dataset_path,
    )
    indices = split_indices(len(dataset), cfg.count, cfg.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name.strip() or f"{cfg.dataset_split}_n{cfg.count}_{stamp}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, cfg, indices, len(dataset))

    rows: List[Dict[str, float | int | str]] = []
    adapters = {}
    for model_key in cfg.model_keys:
        print(f"[model] {model_key}", flush=True)
        adapters[model_key] = load_rae_adapter(
            model_key,
            repo_path=cfg.rae_repo_path,
            device=device,
            dtype=torch.float32,
            auto_clone=cfg.rae_auto_clone,
            auto_download=cfg.rae_auto_download,
        )
        adapter = adapters[model_key]
        for start in range(0, len(indices), cfg.batch_size):
            batch_indices = indices[start : start + cfg.batch_size]
            x_cpu, _ = pick_dataset_images(
                dataset,
                count=len(batch_indices),
                indices=batch_indices,
                image_size=cfg.image_size,
            )
            x = x_cpu.to(device=device, dtype=torch.float32)
            base = extract_vit_stage_latents(adapter, x, hidden_indices=cfg.hidden_indices)
            order = layer_order(base, cfg.hidden_indices)
            for transform in cfg.transforms:
                target = extract_vit_stage_latents(adapter, P(x, transform), hidden_indices=cfg.hidden_indices)
                for layer in order:
                    per_image = relative_token_error(
                        target[layer],
                        P(base[layer], transform),
                        center=cfg.center,
                    ).detach().cpu().numpy()
                    for image_index, value in zip(batch_indices, per_image):
                        rows.append(
                            {
                                "model": model_key,
                                "transform": transform,
                                "layer": layer,
                                "index": int(image_index),
                                "direct_error": float(value),
                            }
                        )
            done = start + len(batch_indices)
            if done % max(cfg.batch_size * 8, 1) == 0 or done == len(indices):
                print(f"  {model_key}: {done}/{len(indices)} images", flush=True)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary_path = run_dir / "summary.csv"
    detail_path = run_dir / "detail.csv"
    summary.to_csv(summary_path, index=False)
    if cfg.save_detail:
        detail.to_csv(detail_path, index=False)
    plot_path = plot_summary(summary, run_dir, cfg)
    metadata = {
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "detail": str(detail_path) if cfg.save_detail else "",
        "plot": str(plot_path),
        "rows": int(len(detail)),
        "summary_rows": int(len(summary)),
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def parse_args() -> LayerwiseStudyConfig:
    parser = argparse.ArgumentParser(description="Run RAE layerwise direct equivariance study on ImageNet parquet.")
    parser.add_argument("--dataset-name", default=LayerwiseStudyConfig.dataset_name)
    parser.add_argument("--data-root", default=LayerwiseStudyConfig.data_root)
    parser.add_argument("--dataset-path", default=LayerwiseStudyConfig.dataset_path)
    parser.add_argument("--dataset-split", default=LayerwiseStudyConfig.dataset_split)
    parser.add_argument("--image-size", type=int, default=LayerwiseStudyConfig.image_size)
    parser.add_argument("--model-keys", nargs="+", default=list(LayerwiseStudyConfig.model_keys))
    parser.add_argument("--transforms", nargs="+", default=list(LayerwiseStudyConfig.transforms))
    parser.add_argument("--hidden-indices", nargs="+", type=int, default=list(LayerwiseStudyConfig.hidden_indices))
    parser.add_argument("--count", type=int, default=LayerwiseStudyConfig.count)
    parser.add_argument("--batch-size", type=int, default=LayerwiseStudyConfig.batch_size)
    parser.add_argument("--seed", type=int, default=LayerwiseStudyConfig.seed)
    parser.add_argument("--center", default=LayerwiseStudyConfig.center)
    parser.add_argument("--device", default=LayerwiseStudyConfig.device)
    parser.add_argument("--rae-repo-path", default=LayerwiseStudyConfig.rae_repo_path)
    parser.add_argument("--rae-auto-clone", action="store_true")
    parser.add_argument("--rae-auto-download", action="store_true")
    parser.add_argument("--output-dir", default=LayerwiseStudyConfig.output_dir)
    parser.add_argument("--run-name", default=LayerwiseStudyConfig.run_name)
    parser.add_argument("--no-detail", action="store_true")
    args = parser.parse_args()
    return LayerwiseStudyConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        model_keys=tuple(args.model_keys),
        transforms=tuple(args.transforms),
        hidden_indices=tuple(args.hidden_indices),
        count=args.count,
        batch_size=args.batch_size,
        seed=args.seed,
        center=args.center,
        device=args.device,
        rae_repo_path=args.rae_repo_path,
        rae_auto_clone=args.rae_auto_clone,
        rae_auto_download=args.rae_auto_download,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_detail=not args.no_detail,
    )


if __name__ == "__main__":
    run_study(parse_args())
