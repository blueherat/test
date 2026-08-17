#!/usr/bin/env python3
"""Paired FID-1K study of time-varying checkpoint guidance references.

Default v1 conditions:
  static v270 (gamma 3.5)
  static v400 (gamma 4.0; provisional because prior scan hit its upper boundary)
  static v500 (gamma 3.0)
  v270 -> v400 -> v500
  v500 -> v400 -> v270

All conditions are sampled with the same three loaded weak checkpoints and the
same post-load RNG seed, so this five-condition study is internally paired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

try:
    from experiments.run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        fid_environment,
        parse_gpu_indices,
        run_logged,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import (
        DEFAULT_ADM_PYTHON,
        fid_environment,
        parse_gpu_indices,
        run_logged,
    )
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER = REPO_ROOT / "experiments/sample_imagenet100_sit_checkpoint_reference_schedule_v1.py"
FID_SCRIPT = REPO_ROOT / "experiments/compute_adm_fid.py"

DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_CKPT_DIR = DEFAULT_DATA_ROOT / "runs/sit-s-2_seed0/checkpoints"
DEFAULT_V270 = DEFAULT_CKPT_DIR / "step_00270000.pt"
DEFAULT_V400 = DEFAULT_CKPT_DIR / "step_00400000.pt"
DEFAULT_V500 = DEFAULT_CKPT_DIR / "step_00500000.pt"
DEFAULT_V800 = DEFAULT_CKPT_DIR / "step_00800000.pt"
DEFAULT_REFERENCE = DEFAULT_DATA_ROOT / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "checkpoint_reference_schedule_fid1k_v1"


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def checkpoint_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["config"]
    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "step": int(checkpoint["step"]),
        "protocol": str(checkpoint.get("protocol")),
        "model_name": str(config["model_name"]),
        "prediction_target": str(config.get("prediction_target", "velocity")),
        "seed": int(config.get("seed", -1)),
        "global_batch_size": int(config.get("global_batch_size", -1)),
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
        "official_sit": checkpoint.get("official_sit"),
    }
    del checkpoint
    return metadata


def validate_checkpoint_family(strong: dict[str, object], references: dict[str, dict[str, object]]) -> None:
    if strong["prediction_target"] != "velocity":
        raise ValueError("v800 strong must be native velocity")
    for name, metadata in references.items():
        if metadata["prediction_target"] != "velocity":
            raise ValueError(f"{name} must be native velocity")
        for key in (
            "protocol",
            "model_name",
            "seed",
            "global_batch_size",
            "data_manifest_sha256",
            "official_sit",
        ):
            if metadata[key] != strong[key]:
                raise ValueError(
                    f"{name} is incompatible with v800 on {key}: "
                    f"{metadata[key]!r} != {strong[key]!r}"
                )


def condition_payload(name: str, order: tuple[str, ...], gammas: dict[str, float]) -> dict[str, object]:
    return {
        "format": "eqvae_checkpoint_reference_condition_v1",
        "name": name,
        "order": list(order),
        "gammas": {key: float(value) for key, value in gammas.items()},
        "formula": "S + gamma_reference * (S - W_reference)",
        "equal_time_partitions": True,
    }


def default_conditions(gammas: dict[str, float]) -> list[dict[str, object]]:
    # Put the primary causal comparison first so GPU 1/3 run forward and
    # reverse immediately in parallel; static controls fill the lanes after.
    return [
        condition_payload("forward_v270_v400_v500", ("v270", "v400", "v500"), gammas),
        condition_payload("reverse_v500_v400_v270", ("v500", "v400", "v270"), gammas),
        condition_payload("static_v270", ("v270",), gammas),
        condition_payload("static_v400", ("v400",), gammas),
        condition_payload("static_v500", ("v500",), gammas),
    ]


def fingerprint_condition(
    condition: dict[str, object],
    *,
    strong: dict[str, object],
    references: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> str:
    payload = {
        "condition": condition,
        "strong_sha256": strong["sha256"],
        "reference_sha256": {name: references[name]["sha256"] for name in sorted(references)},
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_output_points": args.num_output_points,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_result(result_path: Path, *, expected_fingerprint: str, expected_samples: int) -> bool:
    if not result_path.is_file():
        return False
    try:
        payload = read_json(result_path)
        metrics = payload["metrics"]
        manifest = payload["sampling_manifest"]
        if payload.get("experiment_fingerprint") != expected_fingerprint:
            return False
        if int(manifest["sampling"]["num_samples"]) != expected_samples:
            return False
        if not manifest.get("noise_sha256") or not manifest.get("label_sha256"):
            return False
        return all(isinstance(metrics.get(key), (int, float)) for key in ("fid", "sfid", "inception_score"))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def valid_samples(
    output_dir: Path,
    *,
    condition: dict[str, object],
    expected_samples: int,
    expected_seed: int,
) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_n{expected_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        return (
            manifest.get("condition") == condition
            and int(manifest["sampling"]["num_samples"]) == expected_samples
            and int(manifest["sampling"]["seed"]) == expected_seed
            and bool(manifest.get("noise_sha256"))
            and bool(manifest.get("label_sha256"))
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_condition(
    *,
    gpu: int,
    condition: dict[str, object],
    strong: dict[str, object],
    references: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    name = str(condition["name"])
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = args.output_root / "conditions" / f"{name}.json"
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(condition, condition_path)

    experiment_fingerprint = fingerprint_condition(
        condition, strong=strong, references=references, args=args
    )
    result_path = output_dir / "condition_result.json"
    if valid_result(
        result_path,
        expected_fingerprint=experiment_fingerprint,
        expected_samples=args.num_samples,
    ):
        print(f"[reuse] {name}", flush=True)
        return read_json(result_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sample_path = output_dir / f"samples_n{args.num_samples}.npz"
    manifest_path = output_dir / "sampling_manifest.json"

    if not valid_samples(
        output_dir,
        condition=condition,
        expected_samples=args.num_samples,
        expected_seed=args.seed,
    ):
        sample_cmd = [
            sys.executable,
            str(SAMPLER),
            "--condition-json", str(condition_path),
            "--output-dir", str(output_dir),
            "--strong-checkpoint", str(args.strong_checkpoint),
            "--num-samples", str(args.num_samples),
            "--batch-size", str(args.batch_size),
            "--vae-decode-batch-size", str(args.vae_decode_batch_size),
            "--seed", str(args.seed),
            "--num-output-points", str(args.num_output_points),
            "--atol", repr(float(args.atol)),
            "--rtol", repr(float(args.rtol)),
            "--cuda-allocator-limit-gib", repr(float(args.cuda_allocator_limit_gib)),
            "--device", "cuda:0",
        ]
        for ref_name, ref_path in args.reference_checkpoints.items():
            sample_cmd += ["--reference-checkpoint", f"{ref_name}={ref_path}"]
        for ref_name, gamma in args.gammas.items():
            sample_cmd += ["--reference-gamma", f"{ref_name}={gamma}"]
        run_logged(
            tuple(sample_cmd),
            output_dir / "sampling.log",
            env=env,
            monitored_gpu_indices=[gpu],
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "sampling_resource_audit.json",
        )
        if not valid_samples(
            output_dir,
            condition=condition,
            expected_samples=args.num_samples,
            expected_seed=args.seed,
        ):
            raise RuntimeError(f"invalid sampler output for {name}")
    else:
        print(f"[reuse samples] {name}", flush=True)

    fid_path = output_dir / "adm_metrics.json"
    fid_cmd = (
        str(args.adm_python),
        str(FID_SCRIPT),
        "--reference", str(args.reference),
        "--samples", str(sample_path),
        "--batch-size", str(args.fid_batch_size),
        "--gpu-memory-fraction", str(args.fid_gpu_memory_fraction),
        "--output", str(fid_path),
    )
    run_logged(
        fid_cmd,
        output_dir / "evaluation.log",
        env=fid_environment(env, cuda_visible_devices=str(gpu)),
        monitored_gpu_indices=[gpu],
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
        memory_poll_interval=args.memory_poll_interval,
        resource_audit_path=output_dir / "fid_resource_audit.json",
    )

    manifest = read_json(manifest_path)
    metrics = read_json(fid_path)
    payload = {
        "format": "eqvae_imagenet100_sit_checkpoint_reference_condition_result_v1",
        "experiment_fingerprint": experiment_fingerprint,
        "condition": condition,
        "sampling_manifest": manifest,
        "metrics": metrics,
        "gpu": gpu,
        "sample_retained": bool(args.keep_samples),
    }
    atomic_json_dump(payload, result_path)
    if not valid_result(
        result_path,
        expected_fingerprint=experiment_fingerprint,
        expected_samples=args.num_samples,
    ):
        raise RuntimeError(f"invalid condition result for {name}")
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(f"[complete] {name}: FID={float(metrics['fid']):.4f}", flush=True)
    return payload


def save_summary(results: list[dict[str, object]], *, args: argparse.Namespace) -> None:
    rows = []
    noises = set()
    labels = set()
    for payload in results:
        condition = payload["condition"]
        manifest = payload["sampling_manifest"]
        metrics = payload["metrics"]
        noise = str(manifest["noise_sha256"])
        label = str(manifest["label_sha256"])
        noises.add(noise)
        labels.add(label)
        order = list(condition["order"])
        rows.append({
            "condition": condition["name"],
            "order": "->".join(order),
            "stage_gammas": "->".join(f"{float(condition['gammas'][name]):g}" for name in order),
            "fid": float(metrics["fid"]),
            "sfid": float(metrics["sfid"]),
            "inception_score": float(metrics["inception_score"]),
            "total_nfe": int(manifest["total_nfe"]),
            "total_model_forwards": int(manifest["strong_forwards"]) + sum(int(x) for x in manifest["reference_forwards"].values()),
            "noise_sha256": noise,
            "label_sha256": label,
        })
    if len(noises) != 1 or len(labels) != 1:
        raise RuntimeError(
            f"conditions are not paired: noise fingerprints={len(noises)}, label fingerprints={len(labels)}"
        )
    rows.sort(key=lambda row: str(row["condition"]))
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "checkpoint_reference_fid1k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {str(row["condition"]): row for row in rows}
    best_static = min(
        (by_name[name] for name in ("static_v270", "static_v400", "static_v500")),
        key=lambda x: float(x["fid"]),
    )
    forward = by_name["forward_v270_v400_v500"]
    reverse = by_name["reverse_v500_v400_v270"]
    headline = {
        "best_static_condition": best_static["condition"],
        "best_static_fid": float(best_static["fid"]),
        "forward_fid": float(forward["fid"]),
        "reverse_fid": float(reverse["fid"]),
        "forward_minus_reverse": float(forward["fid"]) - float(reverse["fid"]),
        "forward_minus_best_static": float(forward["fid"]) - float(best_static["fid"]),
        "reverse_minus_best_static": float(reverse["fid"]) - float(best_static["fid"]),
    }
    summary = {
        "protocol": "imagenet100_sit_checkpoint_reference_schedule_fid1k_v1",
        "formula": "S + gamma_reference * (S - W_reference)",
        "gammas": args.gammas,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "pairing_verified": True,
        "noise_sha256": next(iter(noises)),
        "label_sha256": next(iter(labels)),
        "headline": headline,
        "rows": rows,
        "csv": str(csv_path),
        "v400_gamma_is_provisional": True,
    }
    atomic_json_dump(summary, summary_dir / "summary.json")
    print("\n=== CHECKPOINT REFERENCE V1 ===")
    for row in sorted(rows, key=lambda x: float(x["fid"])):
        print(f"{row['condition']:32s} FID={float(row['fid']):8.4f}  NFE={int(row['total_nfe'])}")
    print(
        f"\nforward - reverse = {headline['forward_minus_reverse']:+.4f} FID\n"
        f"forward - best static = {headline['forward_minus_best_static']:+.4f} FID\n"
        f"summary: {summary_dir / 'summary.json'}"
    )


def self_test() -> None:
    gammas = {"v270": 3.5, "v400": 4.0, "v500": 3.0}
    conditions = default_conditions(gammas)
    assert len(conditions) == 5
    assert conditions[0]["order"] == ["v270", "v400", "v500"]
    assert conditions[1]["order"] == ["v500", "v400", "v270"]
    assert conditions[0]["gammas"] == conditions[1]["gammas"]
    print("SELF_TEST_OK")
    print("conditions: 5")
    print("forward: v270 -> v400 -> v500")
    print("reverse: v500 -> v400 -> v270")
    print("gammas: v270=3.5, v400=4.0, v500=3.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,3")
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--v400-checkpoint", type=Path, default=DEFAULT_V400)
    parser.add_argument("--v500-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--gamma-v270", type=float, default=3.5)
    parser.add_argument("--gamma-v400", type=float, default=4.0)
    parser.add_argument("--gamma-v500", type=float, default=3.0)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=15 * 1024)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    args.gpu_indices = parse_gpu_indices(args.gpus)
    if not args.gpu_indices:
        raise ValueError("at least one GPU is required")
    args.strong_checkpoint = args.strong_checkpoint.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = args.adm_python.expanduser().absolute()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference_checkpoints = {
        "v270": args.v270_checkpoint.expanduser().resolve(),
        "v400": args.v400_checkpoint.expanduser().resolve(),
        "v500": args.v500_checkpoint.expanduser().resolve(),
    }
    args.gammas = {
        "v270": float(args.gamma_v270),
        "v400": float(args.gamma_v400),
        "v500": float(args.gamma_v500),
    }
    if any(not math.isfinite(value) or value < 0 for value in args.gammas.values()):
        raise ValueError("all reference gammas must be finite and non-negative")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch size must be positive")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("allocator limit must leave headroom below memory ceiling")

    required = [
        SAMPLER,
        FID_SCRIPT,
        args.strong_checkpoint,
        args.reference,
        args.adm_python,
        *args.reference_checkpoints.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n  " + "\n  ".join(missing))

    strong = checkpoint_metadata(args.strong_checkpoint)
    references = {name: checkpoint_metadata(path) for name, path in args.reference_checkpoints.items()}
    if int(strong["step"]) != 800000:
        raise ValueError(f"strong checkpoint step is {strong['step']}, expected 800000")
    for name, step in {"v270": 270000, "v400": 400000, "v500": 500000}.items():
        if int(references[name]["step"]) != step:
            raise ValueError(f"{name} checkpoint step is {references[name]['step']}, expected {step}")
    validate_checkpoint_family(strong, references)

    conditions = default_conditions(args.gammas)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print("Formula: S + gamma_reference * (S - W_reference)")
    print("Gammas:", ", ".join(f"{k}={v:g}" for k, v in args.gammas.items()))
    print("GPUs:", ",".join(str(x) for x in args.gpu_indices))
    print("Conditions:", len(conditions))
    print("Memory ceiling:", args.gpu_memory_ceiling_mib, "MiB")
    for index, condition in enumerate(conditions):
        gpu = args.gpu_indices[index % len(args.gpu_indices)]
        print(f"  GPU {gpu}: {condition['name']}  [{' -> '.join(condition['order'])}]")
    if args.dry_run:
        return

    # One sequential lane per physical GPU.  Do not submit one future per
    # condition: a generic thread pool could otherwise start two conditions
    # assigned to the same GPU if another GPU finishes early.
    lanes = {gpu: [] for gpu in args.gpu_indices}
    for index, condition in enumerate(conditions):
        lanes[args.gpu_indices[index % len(args.gpu_indices)]].append(condition)

    def run_lane(gpu: int, lane_conditions: list[dict[str, object]]):
        lane_results = []
        for condition in lane_conditions:
            lane_results.append(
                run_condition(
                    gpu=gpu,
                    condition=condition,
                    strong=strong,
                    references=references,
                    args=args,
                )
            )
        return lane_results

    results = []
    with ThreadPoolExecutor(max_workers=len(args.gpu_indices)) as pool:
        futures = {
            pool.submit(run_lane, gpu, lane_conditions): gpu
            for gpu, lane_conditions in lanes.items()
            if lane_conditions
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"GPU lane failed: {gpu}") from exc
    if len(results) != len(conditions):
        raise RuntimeError("not all conditions completed")
    save_summary(results, args=args)


if __name__ == "__main__":
    main()
