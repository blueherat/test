"""Fixed-noise 5k generation and KID/FID proxy evaluation for tiny RAE branches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spectral_tiny"
DEFAULT_REFERENCE = Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz")
SAMPLING_SEED = 20260715


class NumpyRGBDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = np.load(path)
        self.images = payload["arr_0"]
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"expected NHWC RGB images in {path}, got {self.images.shape}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.images[int(index)])).permute(2, 0, 1)


def branches(results: Path, branch_name: str = "") -> list[Path]:
    selected = sorted(
        path
        for path in results.glob("seed*_*_from_s5000")
        if (path / "manifest.json").exists()
    )
    if branch_name:
        selected = [path for path in selected if path.name == branch_name]
    return selected


def sample_folder_name(sample_count: int, endpoint: int, steps: int) -> str:
    base = f"fixed_seed{SAMPLING_SEED}_{int(sample_count)}_step{int(endpoint)}"
    return base if int(steps) == 50 else f"{base}_{int(steps)}steps"


def endpoint_checkpoint(branch: Path, endpoint: int) -> Path:
    path = branch / "checkpoints" / f"step-{int(endpoint):07d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_sampling_config(branch: Path, checkpoint: Path, steps: int) -> tuple[Path, Path]:
    evaluation = branch / "generation"
    evaluation.mkdir(parents=True, exist_ok=True)
    materialized = evaluation / f"ema_{checkpoint.stem}.pt"
    if not materialized.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        torch.save(state["ema"], materialized)
    config = OmegaConf.load(branch / "config.yaml")
    config.stage_2.ckpt = str(materialized)
    config.sampler.params.num_steps = int(steps)
    config.guidance.method = "cfg"
    config.guidance.scale = 1.0
    if "training" in config:
        del config["training"]
    if "eval" in config:
        del config["eval"]
    output = evaluation / f"sampling_{checkpoint.stem}_{steps}steps.yaml"
    OmegaConf.save(config, output)
    return output, materialized


def sample_branch(
    branch: Path,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    devices: str,
    processes: int,
    per_process_batch: int,
) -> Path:
    checkpoint = endpoint_checkpoint(branch, endpoint)
    config, _ = prepare_sampling_config(branch, checkpoint, steps)
    sample_root = branch / "generation"
    folder_name = sample_folder_name(sample_count, endpoint, steps)
    sample_folder = sample_root / folder_name
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(processes)}",
        "src/sample_ddp.py",
        "--config",
        str(config),
        "--sample-dir",
        str(sample_root),
        "--per-proc-batch-size",
        str(int(per_process_batch)),
        "--num-fid-samples",
        str(int(sample_count)),
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
    print(f"sampling {branch.name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=RAE_ROOT, env=environment, check=True)
    if not sample_folder.exists() or not sample_folder.with_suffix(".npz").exists():
        raise RuntimeError(f"sampling did not create {sample_folder} and its npz")
    return sample_folder


def torch_fidelity_metrics(
    samples: Dataset,
    reference: Dataset,
    *,
    batch_size: int,
    cache_name: str,
) -> dict[str, float]:
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=samples,
        input2=reference,
        cuda=torch.cuda.is_available(),
        batch_size=int(batch_size),
        isc=True,
        fid=True,
        kid=True,
        kid_subsets=100,
        kid_subset_size=1000,
        rng_seed=SAMPLING_SEED,
        input2_cache_name=cache_name,
        cache=True,
        verbose=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


def adm_fid(sample_npz: Path, reference: Path, output: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "experiments/compute_adm_fid.py"),
        "--reference",
        str(reference),
        "--samples",
        str(sample_npz),
        "--batch-size",
        "64",
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "metrics", "all"], default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--branch-name", default="")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--endpoint", type=int, default=10000)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--devices", default="3")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--per-process-batch", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument(
        "--with-adm",
        action="store_true",
        help="Also run the optional TensorFlow-based ADM evaluator.",
    )
    args = parser.parse_args()

    if args.sample_count % 1000 != 0:
        raise ValueError("equal ImageNet label sampling requires a multiple of 1000 samples")
    selected = branches(args.results, args.branch_name)
    if not selected:
        raise RuntimeError("no tiny branches found")

    sample_folders: dict[str, Path] = {}
    if args.mode in {"sample", "all"}:
        for branch in selected:
            sample_folders[branch.name] = sample_branch(
                branch,
                endpoint=args.endpoint,
                sample_count=args.sample_count,
                steps=args.steps,
                devices=args.devices,
                processes=args.processes,
                per_process_batch=args.per_process_batch,
            )
    else:
        name = sample_folder_name(args.sample_count, args.endpoint, args.steps)
        sample_folders = {branch.name: branch / "generation" / name for branch in selected}

    if args.mode == "sample":
        return

    reference_dataset = NumpyRGBDataset(args.reference)
    rows = []
    for branch in selected:
        sample_folder = sample_folders[branch.name]
        if not sample_folder.exists():
            raise FileNotFoundError(sample_folder)
        sample_npz = sample_folder.with_suffix(".npz")
        if not sample_npz.exists():
            raise FileNotFoundError(sample_npz)
        sample_dataset = NumpyRGBDataset(sample_npz)
        if len(sample_dataset) != int(args.sample_count):
            raise ValueError(
                f"expected exactly {args.sample_count} samples in {sample_npz}, "
                f"got {len(sample_dataset)}"
            )
        print(f"metrics {branch.name}", flush=True)
        fidelity = torch_fidelity_metrics(
            sample_dataset,
            reference_dataset,
            batch_size=args.metric_batch_size,
            cache_name="imagenet256_virtual_reference_10k",
        )
        adm = None
        if args.with_adm:
            adm_output = branch / "generation" / "adm_metrics.json"
            adm = adm_fid(sample_npz, args.reference, adm_output)
        manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        result = {
            "branch": branch.name,
            "seed": int(manifest["global_seed"]),
            "treatment": "baseline" if float(manifest["gamma"]) == 0 else "partial",
            "gamma": float(manifest["gamma"]),
            "endpoint": int(args.endpoint),
            "sample_count": int(args.sample_count),
            "sampling_seed": SAMPLING_SEED,
            "sampling_steps": int(args.steps),
            "sampling_processes": int(args.processes),
            "per_process_batch": int(args.per_process_batch),
            "global_sampling_batch": int(args.processes) * int(args.per_process_batch),
            "precision": "fp32",
            "tf32": False,
            "label_sampling": "equal",
            "sample_folder": str(sample_folder),
            **fidelity,
        }
        if adm is not None:
            result.update(
                {
                    "adm_fid_proxy": float(adm["fid"]),
                    "adm_sfid_proxy": float(adm["sfid"]),
                    "adm_inception_score": float(adm["inception_score"]),
                }
            )
        branch_metric_name = (
            "generation_metrics.json"
            if int(args.steps) == 50
            else f"generation_metrics_{int(args.steps)}steps.json"
        )
        output = branch / "generation" / branch_metric_name
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(result)

    table = pd.DataFrame(rows).sort_values(["seed", "treatment"])
    if int(args.steps) == 50 and not args.branch_name:
        output = args.results / "generation_metrics.csv"
    else:
        suffix = f"_{args.branch_name}" if args.branch_name else ""
        output = args.results / f"generation_metrics_{int(args.steps)}steps{suffix}.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
