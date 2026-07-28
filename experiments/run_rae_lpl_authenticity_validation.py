"""Run one preregistered RAE-LPL calibration or paired training branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path.home() / "data/eqvae/models/RAE"
DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_lpl_authenticity"
DEFAULT_DATA = Path("/data/shared/imagenet-1k")


@dataclass(frozen=True)
class PriorSpec:
    config: Path
    checkpoint: Path


PRIOR_SPECS = {
    "ditdh_s_ep14": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml",
        MODEL_ROOT
        / "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep14/stage2_model.pt",
    ),
    "ditdh_s_ep20": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml",
        MODEL_ROOT
        / "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep20/stage2_model.pt",
    ),
    "ditdh_xl_ep20": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_ditdh_xl_dinov2.yaml",
        MODEL_ROOT
        / "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL_ep20/stage2_model.pt",
    ),
    "ditdh_xl_ep80": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_ditdh_xl_dinov2.yaml",
        MODEL_ROOT
        / "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL_ep80/stage2_model.pt",
    ),
    "ditdh_xl_final": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_ditdh_xl_dinov2.yaml",
        MODEL_ROOT
        / "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt",
    ),
    "mae_dit_xl_ep80": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_dit_xl_mae.yaml",
        MODEL_ROOT / "DiTs/MAE/b16/ImageNet256/DiT-XL-ep80/stage2_model.pt",
    ),
    "siglip2_dit_xl_ep80": PriorSpec(
        ROOT / "experiments/configs/rae_strict_lpl_dit_xl_siglip2.yaml",
        MODEL_ROOT / "DiTs/SigLIP2/b16/ImageNet256/DiT-XL-ep80/stage2_model.pt",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source_branch(
    *,
    results_dir: Path,
    prior_name: str,
    spec: PriorSpec,
) -> Path:
    branch = results_dir.expanduser().resolve() / f"{prior_name}_official_source"
    branch.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.config, branch / "config.yaml")
    manifest = {
        "experiment_name": branch.name,
        "objective": "official_source",
        "source_checkpoint": str(spec.checkpoint),
        "source_checkpoint_sha256": file_sha256(spec.checkpoint),
        "source_checkpoint_type": "official_model_only",
        "endpoint_step": 0,
        "training_updates": 0,
        "precision": "fp32",
        "tf32": False,
    }
    (branch / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return branch


def build_command(args: argparse.Namespace, spec: PriorSpec, name: str) -> list[str]:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        str(ROOT / "experiments/train_rae_strict_lpl.py"),
        "--config",
        str(spec.config),
        "--data-path",
        str(args.data_path.expanduser().resolve()),
        "--results-dir",
        str(args.results_dir.expanduser().resolve()),
        "--experiment-name",
        name,
        "--model-ckpt",
        str(spec.checkpoint),
        "--objective",
        args.objective,
        "--global-seed",
        str(args.seed),
        "--max-train-steps",
        str(args.endpoint),
    ]
    if args.mode == "calibrate":
        command.extend(
            [
                "--calibration-batches",
                str(args.calibration_batches),
                "--calibration-mode",
                args.calibration_mode,
                "--calibration-target-lpl-over-flow",
                "0.25",
                "--calibration-target-variance-ratio",
                "0.1",
            ]
        )
    elif args.objective == "full":
        if args.lpl_weight is None or args.lpl_weight <= 0:
            raise ValueError("full training requires a positive --lpl-weight")
        command.extend(["--lpl-weight", str(args.lpl_weight)])
    else:
        command.extend(["--lpl-weight", "0"])
    if args.skip_checkpoint_save:
        command.append("--skip-checkpoint-save")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("source", "calibrate", "train"), required=True
    )
    parser.add_argument("--prior", choices=tuple(PRIOR_SPECS), required=True)
    parser.add_argument("--objective", choices=("flow", "full"), default="full")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--calibration-batches", type=int, default=256)
    parser.add_argument(
        "--calibration-mode",
        choices=("mean_contribution", "variance"),
        default="mean_contribution",
    )
    parser.add_argument("--lpl-weight", type=float)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--skip-checkpoint-save", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.mode == "calibrate" and args.objective == "flow":
        raise ValueError("calibration requires objective=full")
    if args.endpoint < 1:
        raise ValueError("endpoint must be positive")
    if len(args.devices.split(",")) != 4:
        raise ValueError("the preregistered protocol requires exactly four devices")

    spec = PRIOR_SPECS[args.prior]
    for path in (spec.config, spec.checkpoint, args.data_path):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)
    if args.mode == "source":
        branch = prepare_source_branch(
            results_dir=args.results_dir,
            prior_name=args.prior,
            spec=spec,
        )
        print(branch)
        return
    suffix = (
        f"calibration_{args.calibration_mode}_seed{args.seed}"
        if args.mode == "calibrate"
        else f"seed{args.seed}_{args.objective}_to_s{args.endpoint}"
    )
    name = f"{args.prior}_{suffix}"
    command = build_command(args, spec, name)
    print(json.dumps({"experiment": name, "command": command}, indent=2))
    if args.print_only:
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.devices
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
