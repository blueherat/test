"""Paired 5k generation evaluation for strict deterministic-decoder LPL."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_rae_spectral_generation import (
    LABEL_SAMPLER_VERSION,
    NumpyRGBDataset,
    RAE_ROOT,
    SAMPLING_SEED,
    SAMPLING_PROVENANCE_FILENAME,
    adm_fid,
    endpoint_checkpoint,
    file_sha256,
    sample_branch,
    sample_folder_name,
    torch_fidelity_metrics,
)

STRICT_PROCESSES = 4
STRICT_PER_PROCESS_BATCH = 4
SAMPLING_AUDIT_KEYS = (
    "protocol",
    "rank",
    "global_seed",
    "initial_cuda_rng_state_sha256",
    "first_noise_sha256",
    "first_label_sha256",
    "first_labels",
    "iterations",
    "final_cuda_rng_state_sha256",
)


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path).expanduser().resolve()


def validate_strict_sampling_protocol(processes: int, per_process_batch: int) -> None:
    """Keep noise/label pairing identical to the existing Flow/full evaluations."""

    if int(processes) != STRICT_PROCESSES or int(per_process_batch) != STRICT_PER_PROCESS_BATCH:
        raise ValueError(
            "strict paired evaluation requires "
            f"processes={STRICT_PROCESSES} and "
            f"per_process_batch={STRICT_PER_PROCESS_BATCH}; got "
            f"{int(processes)} and {int(per_process_batch)}"
        )


def label_balance_metadata(
    sample_count: int,
    processes: int,
    per_process_batch: int,
) -> dict[str, int | bool]:
    """Describe rounded generation with exact pre-tail class balance."""

    global_batch = int(processes) * int(per_process_batch)
    generated = int(math.ceil(int(sample_count) / global_batch) * global_batch)
    return {
        "class_balance_exact": int(sample_count) % 1000 == 0,
        "samples_generated_before_trim": generated,
    }


def load_sampling_audits(folder: Path, processes: int) -> list[dict[str, object]]:
    audits = []
    for rank in range(int(processes)):
        path = folder / f"sampling_audit_rank{rank}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing strict sampling audit: {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in SAMPLING_AUDIT_KEYS if key not in audit]
        if missing:
            raise ValueError(f"{path} is missing sampling audit keys: {missing}")
        if audit["protocol"] != "interleaved-labels-v2":
            raise ValueError(f"unexpected sampling protocol in {path}: {audit['protocol']}")
        if int(audit["rank"]) != rank:
            raise ValueError(f"rank mismatch in {path}: {audit['rank']} != {rank}")
        audits.append(audit)
    return audits


def assert_paired_sampling_audits(
    audits_by_branch: dict[str, list[dict[str, object]]],
) -> None:
    if not audits_by_branch:
        return
    reference_name, reference = next(iter(audits_by_branch.items()))
    for name, audits in audits_by_branch.items():
        if len(audits) != len(reference):
            raise ValueError(
                f"sampling audit rank count differs: {reference_name}={len(reference)}, "
                f"{name}={len(audits)}"
            )
        for rank, (expected, actual) in enumerate(zip(reference, audits)):
            for key in SAMPLING_AUDIT_KEYS:
                if actual[key] != expected[key]:
                    raise ValueError(
                        f"sampling audit mismatch for branch={name}, rank={rank}, key={key}"
                    )


def reject_partial_sampling_folder(
    branch: Path,
    *,
    endpoint: int,
    sample_count: int,
    steps: int,
    sampling_seed: int = SAMPLING_SEED,
    state_key: str = "ema",
) -> None:
    """Reject legacy cursor resume because it does not restore skipped RNG draws."""

    folder = branch / "generation" / sample_folder_name(
        sample_count,
        endpoint,
        steps,
        sampling_seed=sampling_seed,
        state_key=state_key,
    )
    archive = folder.with_suffix(".npz")
    if not folder.exists() or archive.exists():
        return
    png_count = sum(1 for path in folder.glob("*.png") if path.is_file())
    if png_count:
        raise RuntimeError(
            f"{folder} contains {png_count} partial PNG samples but no NPZ. "
            "Strict paired sampling must restart from an empty folder because the "
            "legacy sampler does not restore the skipped CUDA RNG stream."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--endpoint", type=int, default=500)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--per-process-batch", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=SAMPLING_SEED)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--with-adm",
        action="store_true",
        help="Also compute ADM FID/sFID/IS from the official reference statistics.",
    )
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_count % 1000 != 0:
        raise ValueError("equal ImageNet label sampling requires a multiple of 1000")
    validate_strict_sampling_protocol(args.processes, args.per_process_batch)
    reference_path = args.reference.expanduser().resolve()
    reference = NumpyRGBDataset(reference_path)
    evaluation_script_sha256 = file_sha256(Path(__file__).resolve())
    sampling_script_sha256 = file_sha256(RAE_ROOT / "src/sample_ddp.py")
    reference_sha256 = file_sha256(reference_path)
    torch_fidelity_version = importlib.metadata.version("torch-fidelity")
    rows = []
    audits_by_branch = {}
    for name, branch in args.branch:
        endpoint_path = endpoint_checkpoint(branch, args.endpoint)
        if args.skip_sampling:
            folder = branch / "generation" / sample_folder_name(
                args.sample_count,
                args.endpoint,
                args.steps,
                sampling_seed=args.sampling_seed,
                state_key=args.state_key,
            )
            if not folder.exists() or not folder.with_suffix(".npz").exists():
                raise FileNotFoundError(
                    f"--skip-sampling requires a complete archive at {folder}"
                )
            # sample_branch returns without sampling only after validating the
            # archive-to-checkpoint provenance contract.
            folder = sample_branch(
                branch,
                endpoint=args.endpoint,
                sample_count=args.sample_count,
                steps=args.steps,
                devices=args.devices,
                processes=args.processes,
                per_process_batch=args.per_process_batch,
                sampling_seed=args.sampling_seed,
                state_key=args.state_key,
            )
        else:
            reject_partial_sampling_folder(
                branch,
                endpoint=args.endpoint,
                sample_count=args.sample_count,
                steps=args.steps,
                sampling_seed=args.sampling_seed,
                state_key=args.state_key,
            )
            folder = sample_branch(
                branch,
                endpoint=args.endpoint,
                sample_count=args.sample_count,
                steps=args.steps,
                devices=args.devices,
                processes=args.processes,
                per_process_batch=args.per_process_batch,
                sampling_seed=args.sampling_seed,
                state_key=args.state_key,
            )
        sample_npz = folder.with_suffix(".npz")
        if not sample_npz.exists():
            raise FileNotFoundError(sample_npz)
        sampling_config = (
            branch
            / "generation"
            / f"sampling_{args.state_key}_{endpoint_path.stem}_{args.steps}steps.yaml"
        )
        if not sampling_config.exists():
            raise FileNotFoundError(sampling_config)
        resolved_sampling_config = OmegaConf.load(sampling_config)
        sampling_checkpoint = Path(
            str(resolved_sampling_config.stage_2.ckpt)
        ).expanduser().resolve()
        if not sampling_checkpoint.exists():
            raise FileNotFoundError(sampling_checkpoint)
        sampling_provenance_path = folder / SAMPLING_PROVENANCE_FILENAME
        if not sampling_provenance_path.exists():
            raise FileNotFoundError(sampling_provenance_path)
        sampling_provenance = json.loads(
            sampling_provenance_path.read_text(encoding="utf-8")
        )
        samples = NumpyRGBDataset(sample_npz)
        if len(samples) != int(args.sample_count):
            raise ValueError(f"expected {args.sample_count} samples, found {len(samples)}")
        audits = load_sampling_audits(folder, args.processes)
        audits_by_branch[name] = audits
        metrics = torch_fidelity_metrics(
            samples,
            reference,
            batch_size=int(args.metric_batch_size),
            cache_name="imagenet256_virtual_reference_10k",
            rng_seed=args.sampling_seed,
        )
        adm = None
        if args.with_adm:
            adm_output = branch / "generation" / f"adm_{folder.name}.json"
            adm = adm_fid(sample_npz, args.reference, adm_output)
        manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        balance = label_balance_metadata(
            args.sample_count,
            args.processes,
            args.per_process_batch,
        )
        rows.append(
            {
                "branch": name,
                "objective": manifest["objective"],
                "endpoint": int(args.endpoint),
                "sample_count": int(args.sample_count),
                "sampling_seed": int(args.sampling_seed),
                "sampling_steps": int(args.steps),
                "sampling_state_key": args.state_key,
                "sampling_processes": int(args.processes),
                "per_process_batch": int(args.per_process_batch),
                "global_sampling_batch": int(args.processes * args.per_process_batch),
                "label_sampling": "equal",
                "label_sampler_version": LABEL_SAMPLER_VERSION,
                "sampling_audit_passed": True,
                **balance,
                "precision": "fp32",
                "tf32": False,
                "sample_folder": str(folder),
                "evaluation_script_sha256": evaluation_script_sha256,
                "sampling_script_sha256": sampling_script_sha256,
                "sampling_config": str(sampling_config),
                "sampling_config_sha256": file_sha256(sampling_config),
                "reference": str(reference_path),
                "reference_sha256": reference_sha256,
                "reference_sample_count": len(reference),
                "torch_version": str(torch.__version__),
                "torch_fidelity_version": torch_fidelity_version,
                "endpoint_checkpoint": str(endpoint_path),
                "endpoint_checkpoint_sha256": file_sha256(endpoint_path),
                "sampling_checkpoint": str(sampling_checkpoint),
                "sampling_checkpoint_sha256": file_sha256(sampling_checkpoint),
                "sampling_provenance": str(sampling_provenance_path),
                "sampling_provenance_sha256": file_sha256(
                    sampling_provenance_path
                ),
                "sampling_provenance_protocol": sampling_provenance.get(
                    "protocol"
                ),
                "sample_npz_sha256": file_sha256(sample_npz),
                **metrics,
            }
        )
        if adm is not None:
            rows[-1].update(
                {
                    "adm_fid": float(adm["fid"]),
                    "adm_sfid": float(adm["sfid"]),
                    "adm_inception_score": float(adm["inception_score"]),
                }
            )

    assert_paired_sampling_audits(audits_by_branch)
    table = pd.DataFrame(rows)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table.to_csv(output.with_suffix(".csv"), index=False)
    print(table.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
