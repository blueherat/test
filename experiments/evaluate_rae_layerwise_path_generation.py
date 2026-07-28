"""Fixed-seed 5k KID/FID evaluation for layerwise-path branches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external/RAE"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
SAMPLING_SEED = 20_260_718


class NumpyRGBDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = np.load(path, mmap_mode="r")
        self.images = payload["arr_0"]
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"expected NHWC RGB images in {path}, got {self.images.shape}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.images[int(index)]).copy()).permute(2, 0, 1)


def branches(results: Path, branch_name: str = "") -> list[Path]:
    if branch_name:
        candidate = results / branch_name
        return [candidate] if (candidate / "manifest.json").exists() else []
    return sorted(
        path
        for path in results.glob("seed*")
        if path.is_dir() and (path / "manifest.json").exists()
    )


def sample_folder_name(
    sample_count: int, endpoint: int, steps: int, weight_source: str = "ema"
) -> str:
    suffix = "" if weight_source == "ema" else f"_{weight_source}"
    return f"fixed_seed{SAMPLING_SEED}_n{sample_count}_step{endpoint}_{steps}steps{suffix}"


def prepare_sampling_config(
    branch: Path, checkpoint: Path, steps: int, weight_source: str = "ema"
) -> Path:
    if weight_source not in {"ema", "model"}:
        raise ValueError(f"unknown weight source: {weight_source}")
    generation = branch / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    materialized = generation / f"{weight_source}_{checkpoint.stem}.pt"
    if not materialized.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        torch.save(state[weight_source], materialized)
    config = OmegaConf.load(branch / "config.yaml")
    config.stage_2.ckpt = str(materialized)
    config.sampler.params.num_steps = int(steps)
    config.guidance.method = "cfg"
    config.guidance.scale = 1.0
    if "training" in config:
        del config["training"]
    if "eval" in config:
        del config["eval"]
    output = generation / f"sampling_{weight_source}_{checkpoint.stem}_{steps}steps.yaml"
    OmegaConf.save(config, output)
    return output


def sample_branch(
    branch: Path,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    devices: str,
    processes: int,
    per_process_batch: int,
    weight_source: str = "ema",
) -> Path:
    checkpoint = branch / "checkpoints" / f"step-{endpoint:07d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config = prepare_sampling_config(branch, checkpoint, steps, weight_source)
    sample_root = branch / "generation"
    folder_name = sample_folder_name(sample_count, endpoint, steps, weight_source)
    sample_folder = sample_root / folder_name
    sample_npz = sample_folder.with_suffix(".npz")
    if sample_npz.exists():
        return sample_folder
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={processes}",
        "src/sample_ddp.py",
        "--config",
        str(config),
        "--sample-dir",
        str(sample_root),
        "--per-proc-batch-size",
        str(per_process_batch),
        "--num-fid-samples",
        str(sample_count),
        "--global-seed",
        str(SAMPLING_SEED),
        "--precision",
        "fp32",
        "--no-tf32",
        "--label-sampling",
        "equal",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = devices
    environment["SAVE_FOLDER"] = folder_name
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=RAE_ROOT, env=environment, check=True)
    if not sample_npz.exists():
        raise RuntimeError(f"sampling did not create {sample_npz}")
    return sample_folder


def fidelity_metrics(
    samples: Dataset,
    reference: Dataset,
    *,
    batch_size: int,
) -> dict[str, float]:
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=samples,
        input2=reference,
        cuda=torch.cuda.is_available(),
        batch_size=batch_size,
        isc=True,
        fid=True,
        kid=True,
        kid_subsets=100,
        kid_subset_size=1000,
        rng_seed=SAMPLING_SEED,
        input2_cache_name="imagenet256_virtual_reference_layerwise_path",
        cache=True,
        verbose=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "metrics", "all"), default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--branch-name", default="")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--endpoint", type=int, default=10_000)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--per-process-batch", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--weight-source", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    if args.sample_count % 1000:
        raise ValueError("equal ImageNet labels require sample_count divisible by 1000")
    selected = branches(args.results, args.branch_name)
    if not selected:
        raise RuntimeError("no completed layerwise-path branches found")

    folders: dict[str, Path] = {}
    for branch in selected:
        if args.mode in {"sample", "all"}:
            folders[branch.name] = sample_branch(
                branch,
                endpoint=args.endpoint,
                sample_count=args.sample_count,
                steps=args.steps,
                devices=args.devices,
                processes=args.processes,
                per_process_batch=args.per_process_batch,
                weight_source=args.weight_source,
            )
        else:
            folders[branch.name] = branch / "generation" / sample_folder_name(
                args.sample_count, args.endpoint, args.steps, args.weight_source
            )
    if args.mode == "sample":
        return

    reference = NumpyRGBDataset(args.reference)
    rows = []
    for branch in selected:
        sample_npz = folders[branch.name].with_suffix(".npz")
        samples = NumpyRGBDataset(sample_npz)
        if len(samples) != args.sample_count:
            raise ValueError(f"expected {args.sample_count} images in {sample_npz}")
        metrics = fidelity_metrics(samples, reference, batch_size=args.metric_batch_size)
        manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        row = {
            "branch": branch.name,
            "seed": int(manifest["global_seed"]),
            "path_mode": manifest["path_mode"],
            "subspace_kind": manifest["subspace_kind"],
            "subspace_rank": int(manifest["subspace_rank"]),
            "endpoint": args.endpoint,
            "sample_count": args.sample_count,
            "sampling_seed": SAMPLING_SEED,
            "sampling_steps": args.steps,
            "precision": "fp32",
            "tf32": False,
            "weight_source": args.weight_source,
            **metrics,
        }
        output = branch / "generation" / f"generation_metrics_{args.weight_source}.json"
        output.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["seed", "path_mode", "subspace_kind"])
    output = args.results / f"layerwise_path_generation_metrics_{args.weight_source}.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
