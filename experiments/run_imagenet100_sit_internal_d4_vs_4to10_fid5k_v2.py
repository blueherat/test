#!/usr/bin/env python3
"""Run paired FID-5K confirmation for:
  1) static depth-4 native guidance
  2) depth-4 -> depth-10 native schedule

Default conditions use the best FID-1K screen hyperparameters:
  - static d4: gamma=0.25
  - 4->10:    gamma=0.45

Both runs use the repository's existing multiscale evaluator with exactly the
same (num_samples=5000, batch_size=8, seed=0).  Under the current sampler RNG
protocol this makes the two runs exactly paired.  For the default formal
configuration the script additionally verifies that their noise/label
fingerprints match the previous formal 4->8->10 FID-5K run.

The atomic evaluator is reused rather than reimplementing sampling/FID.
This wrapper adds stricter condition and pairing checks around evaluator reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HISTORICAL_V800_FID5K = 61.00164134432981
HISTORICAL_4_8_10_FID5K = 42.6253920448695

# Previous formal 4->8->10, gamma=0.4, n=5000, batch=8, seed=0.
HISTORICAL_4_8_10_NOISE_SHA256 = (
    "0ea1ae6701039845f0596ad3387b7f35480ce9c79f3907437804ef679ffd2636"
)
HISTORICAL_4_8_10_LABEL_SHA256 = (
    "57849a94ad38e74bda68272bd08e273a7de143b43b6c5979dddc9e10bade8feb"
)


def detect_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, here.parent):
        if (
            candidate / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"
        ).is_file():
            return candidate
    script_repo = Path(__file__).resolve().parents[1]
    if (
        script_repo / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"
    ).is_file():
        return script_repo
    raise FileNotFoundError(
        "Run from /home/zhoushunyu/eqvae or place this file under <repo>/experiments/."
    )


def detect_data_root() -> Path:
    candidates = (
        Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow"),
        Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    )
    strong_rel = "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
    for root in candidates:
        if (root / strong_rel).is_file():
            return root
    return candidates[0]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def static_d4_condition(gamma: float) -> dict:
    return {
        "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
        "name": f"confirm_fid5k_static_depth4_native_g{gamma:g}".replace(".", "p"),
        "hypothesis_id": "time_varying_internal_depth_fid5k_compare",
        "evaluation_group": "fid5k_confirmation",
        "kind": "static_depth",
        "depth": 4,
        "rms_matched": False,
        "gamma": float(gamma),
    }


def schedule_4to10_condition(gamma: float) -> dict:
    return {
        "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
        "name": f"confirm_fid5k_depth4_to_depth10_native_g{gamma:g}".replace(".", "p"),
        "hypothesis_id": "time_varying_internal_depth_fid5k_compare",
        "evaluation_group": "fid5k_confirmation",
        "kind": "depth_schedule",
        "depths": [4, 10],
        "order": "coarse_to_fine",
        "rms_matched": False,
        "gamma": float(gamma),
    }


def check_schedule_patch(repo: Path) -> None:
    core_path = repo / "experiments/imagenet100_sit_multiscale_guidance.py"
    sampler_path = repo / "experiments/sample_imagenet100_sit_multiscale_guidance.py"
    core = core_path.read_text(encoding="utf-8")
    sampler = sampler_path.read_text(encoding="utf-8")
    core_ok = (
        "gamma_schedule_sweep_v4_generalized_schedule_depth" in core
        or ("depths: Sequence[int]" in core and "torch.bucketize" in core)
    )
    sampler_ok = (
        "gamma_schedule_sweep_v4_condition_depths" in sampler
        or (
            'self.condition.get("depths", (4, 8, 10))' in sampler
            and "depths=tuple(" in sampler
        )
    )
    if not (core_ok and sampler_ok):
        raise RuntimeError(
            "Current repository does not expose the generalized depth schedule "
            "needed for the 2-stage 4->10 condition. Update/install the v4 schedule patch."
        )


def validate_existing_result(
    result_path: Path,
    *,
    expected_condition: dict,
    expected_samples: int,
    expected_seed: int,
    expected_batch_size: int,
) -> bool:
    """Return True only when an existing evaluator result is safe to reuse."""
    if not result_path.is_file():
        return False
    result = read_json(result_path)
    condition = result.get("condition")
    manifest = result.get("sampling_manifest")
    metrics = result.get("metrics")
    if condition != expected_condition:
        raise RuntimeError(
            f"Stale/mismatched result exists at {result_path}. "
            "Its stored condition does not equal the requested condition. "
            "Use a fresh output directory or remove that stale result."
        )
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise RuntimeError(f"Malformed existing result: {result_path}")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise RuntimeError(f"Malformed sampling manifest: {result_path}")
    observed = (
        int(sampling.get("num_samples", -1)),
        int(sampling.get("seed", -1)),
        int(sampling.get("batch_size", -1)),
    )
    expected = (int(expected_samples), int(expected_seed), int(expected_batch_size))
    if observed != expected:
        raise RuntimeError(
            f"Existing result at {result_path} has sampling={observed}, expected={expected}. "
            "Use a fresh output directory or remove that stale result."
        )
    for key in ("fid", "sfid", "inception_score"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Invalid metric {key!r} in {result_path}: {value!r}")
    if not manifest.get("noise_sha256") or not manifest.get("label_sha256"):
        raise RuntimeError(f"Existing result lacks pairing fingerprints: {result_path}")
    return True


def validate_returned_result(
    result: dict,
    *,
    expected_condition: dict,
    expected_samples: int,
    expected_seed: int,
    expected_batch_size: int,
) -> None:
    if result.get("condition") != expected_condition:
        raise RuntimeError("Evaluator returned a result for a different condition")
    manifest = result.get("sampling_manifest")
    metrics = result.get("metrics")
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise RuntimeError("Evaluator result is missing manifest/metrics")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise RuntimeError("Evaluator result has malformed sampling metadata")
    observed = (
        int(sampling.get("num_samples", -1)),
        int(sampling.get("seed", -1)),
        int(sampling.get("batch_size", -1)),
    )
    expected = (int(expected_samples), int(expected_seed), int(expected_batch_size))
    if observed != expected:
        raise RuntimeError(f"Evaluator returned sampling={observed}, expected={expected}")
    for key in ("fid", "sfid", "inception_score"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Evaluator returned invalid metric {key}={value!r}")
    if not manifest.get("noise_sha256") or not manifest.get("label_sha256"):
        raise RuntimeError("Evaluator returned result without pairing fingerprints")


def run_condition(
    *,
    repo: Path,
    gpu: int,
    evaluator: Path,
    atlas: Path,
    strong: Path,
    external_v500: Path,
    depth4: Path,
    depth10: Path,
    reference: Path,
    adm_python: Path,
    output_dir: Path,
    condition: dict,
    num_samples: int,
    batch_size: int,
    seed: int,
    cuda_allocator_limit_gib: float,
    keep_samples: bool,
) -> dict:
    condition_path = output_dir / "condition.json"
    atomic_write_json(condition_path, condition)
    result_path = output_dir / "condition_result.json"

    # The repository evaluator's reuse test is intentionally lightweight.
    # Guard it here with exact condition/sampling validation before invoking it.
    validate_existing_result(
        result_path,
        expected_condition=condition,
        expected_samples=num_samples,
        expected_seed=seed,
        expected_batch_size=batch_size,
    )

    command = [
        sys.executable,
        str(evaluator),
        "--condition-json", str(condition_path),
        "--atlas-summary", str(atlas),
        "--output-dir", str(output_dir),
        "--strong-checkpoint", str(strong),
        "--external-weak-checkpoint", str(external_v500),
        "--head", f"depth4_v={depth4}",
        "--head", f"depth10_v={depth10}",
        "--reference", str(reference),
        "--adm-python", str(adm_python),
        "--num-samples", str(num_samples),
        "--batch-size", str(batch_size),
        "--vae-decode-batch-size", "2",
        "--seed", str(seed),
        "--atol", "1e-6",
        "--rtol", "1e-3",
        "--cuda-allocator-limit-gib", str(cuda_allocator_limit_gib),
        "--fid-batch-size", "8",
        "--fid-gpu-memory-fraction", "0.25",
        "--device", "cuda:0",
    ]
    if keep_samples:
        command.append("--keep-samples")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=repo, env=env, check=True)

    result = read_json(result_path)
    validate_returned_result(
        result,
        expected_condition=condition,
        expected_samples=num_samples,
        expected_seed=seed,
        expected_batch_size=batch_size,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gamma-d4", type=float, default=0.25)
    parser.add_argument("--gamma-4to10", type=float, default=0.45)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if not math.isfinite(args.gamma_d4) or args.gamma_d4 < 0:
        raise ValueError("--gamma-d4 must be finite and non-negative")
    if not math.isfinite(args.gamma_4to10) or args.gamma_4to10 < 0:
        raise ValueError("--gamma-4to10 must be finite and non-negative")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    if args.cuda_allocator_limit_gib <= 0:
        raise ValueError("--cuda-allocator-limit-gib must be positive")

    repo = detect_repo_root()
    data = detect_data_root()
    check_schedule_patch(repo)

    evaluator = repo / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"
    atlas = data / "multiscale_guidance_study_v1/atlas/atlas_summary.json"
    strong = data / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
    external_v500 = data / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
    depth4 = data / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
    depth10 = data / "multiscale_guidance_study_v1/runs/depth10_v/checkpoints/step_00050000.pt"
    reference = data / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
    adm_python = Path("/data/shared/envs/adm-fid/bin/python")

    required = [
        evaluator,
        atlas,
        strong,
        external_v500,
        depth4,
        depth10,
        reference,
        adm_python,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n  " + "\n  ".join(missing))

    root = (
        data
        / "internal_head_gamma_schedule_sweep_v4"
        / f"fid5k_compare_d4_vs_4to10_seed{args.seed}"
    )
    static_dir = root / f"static_d4_g{args.gamma_d4:g}".replace(".", "p")
    sched_dir = root / f"schedule_4to10_g{args.gamma_4to10:g}".replace(".", "p")
    static_condition = static_d4_condition(args.gamma_d4)
    sched_condition = schedule_4to10_condition(args.gamma_4to10)

    print("=== PAIRED FID-5K COMPARISON ===")
    print(f"repo:      {repo}")
    print(f"data:      {data}")
    print(f"GPU:       physical {args.gpu}")
    print(f"samples:   {args.num_samples}, batch={args.batch_size}, seed={args.seed}")
    print(f"static d4: gamma={args.gamma_d4:g}")
    print(f"4->10:     gamma={args.gamma_4to10:g}")
    print(f"root:      {root}")

    if args.dry_run:
        print("\nConditions:")
        print(json.dumps(static_condition, indent=2))
        print(json.dumps(sched_condition, indent=2))
        print(f"\nstatic output:   {static_dir}")
        print(f"schedule output: {sched_dir}")
        return

    static_result = run_condition(
        repo=repo,
        gpu=args.gpu,
        evaluator=evaluator,
        atlas=atlas,
        strong=strong,
        external_v500=external_v500,
        depth4=depth4,
        depth10=depth10,
        reference=reference,
        adm_python=adm_python,
        output_dir=static_dir,
        condition=static_condition,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        cuda_allocator_limit_gib=args.cuda_allocator_limit_gib,
        keep_samples=args.keep_samples,
    )
    schedule_result = run_condition(
        repo=repo,
        gpu=args.gpu,
        evaluator=evaluator,
        atlas=atlas,
        strong=strong,
        external_v500=external_v500,
        depth4=depth4,
        depth10=depth10,
        reference=reference,
        adm_python=adm_python,
        output_dir=sched_dir,
        condition=sched_condition,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        cuda_allocator_limit_gib=args.cuda_allocator_limit_gib,
        keep_samples=args.keep_samples,
    )

    static_metrics = static_result["metrics"]
    static_manifest = static_result["sampling_manifest"]
    sched_metrics = schedule_result["metrics"]
    sched_manifest = schedule_result["sampling_manifest"]

    noise_static = str(static_manifest["noise_sha256"])
    label_static = str(static_manifest["label_sha256"])
    noise_sched = str(sched_manifest["noise_sha256"])
    label_sched = str(sched_manifest["label_sha256"])

    if noise_static != noise_sched or label_static != label_sched:
        raise RuntimeError(
            "Static d4 and 4->10 are not exactly paired:\n"
            f"static noise={noise_static}\n"
            f"sched  noise={noise_sched}\n"
            f"static label={label_static}\n"
            f"sched  label={label_sched}"
        )

    formal_default = (
        args.num_samples == 5000 and args.batch_size == 8 and args.seed == 0
    )
    if formal_default:
        if noise_static != HISTORICAL_4_8_10_NOISE_SHA256:
            raise RuntimeError(
                "Current formal 5K noise fingerprint does not match the historical "
                f"4->8->10 formal run:\nobserved={noise_static}\n"
                f"expected={HISTORICAL_4_8_10_NOISE_SHA256}"
            )
        if label_static != HISTORICAL_4_8_10_LABEL_SHA256:
            raise RuntimeError(
                "Current formal 5K label fingerprint does not match the historical "
                f"4->8->10 formal run:\nobserved={label_static}\n"
                f"expected={HISTORICAL_4_8_10_LABEL_SHA256}"
            )

    static_fid = float(static_metrics["fid"])
    sched_fid = float(sched_metrics["fid"])

    rows = [
        {
            "method": "unguided_v800_historical",
            "gamma": "",
            "fid": HISTORICAL_V800_FID5K,
            "sfid": "",
            "inception_score": "",
            "notes": "historical formal 5K baseline; informational comparator",
        },
        {
            "method": "static_d4_native",
            "gamma": f"{args.gamma_d4:g}",
            "fid": static_fid,
            "sfid": float(static_metrics["sfid"]),
            "inception_score": float(static_metrics["inception_score"]),
            "notes": "current paired run",
        },
        {
            "method": "schedule_4to10_native",
            "gamma": f"{args.gamma_4to10:g}",
            "fid": sched_fid,
            "sfid": float(sched_metrics["sfid"]),
            "inception_score": float(sched_metrics["inception_score"]),
            "notes": "current paired run",
        },
        {
            "method": "historical_4to8to10_native",
            "gamma": "0.4",
            "fid": HISTORICAL_4_8_10_FID5K,
            "sfid": 69.17208447207872,
            "inception_score": 35.231746673583984,
            "notes": "historical formal 5K; exact same noise/labels under default config",
        },
    ]

    summary = {
        "format": "eqvae_internal_d4_vs_4to10_fid5k_comparison_v2",
        "static_d4": {
            "gamma": float(args.gamma_d4),
            "fid": static_fid,
            "sfid": float(static_metrics["sfid"]),
            "inception_score": float(static_metrics["inception_score"]),
            "result": str(static_dir / "condition_result.json"),
        },
        "schedule_4to10": {
            "gamma": float(args.gamma_4to10),
            "fid": sched_fid,
            "sfid": float(sched_metrics["sfid"]),
            "inception_score": float(sched_metrics["inception_score"]),
            "result": str(sched_dir / "condition_result.json"),
        },
        "pairing": {
            "noise_sha256": noise_static,
            "label_sha256": label_static,
            "static_equals_schedule": True,
            "matches_historical_4to8to10_5k": bool(
                formal_default
                and noise_static == HISTORICAL_4_8_10_NOISE_SHA256
                and label_static == HISTORICAL_4_8_10_LABEL_SHA256
            ),
        },
        "historical_comparators": {
            "v800_fid5k": HISTORICAL_V800_FID5K,
            "depth4_to_8_to_10_native_gamma0p4_fid5k": HISTORICAL_4_8_10_FID5K,
        },
        "deltas": {
            "schedule_minus_static_fid": sched_fid - static_fid,
            "static_minus_historical_4to8to10": static_fid - HISTORICAL_4_8_10_FID5K,
            "schedule_minus_historical_4to8to10": sched_fid - HISTORICAL_4_8_10_FID5K,
            "static_improvement_vs_historical_v800": HISTORICAL_V800_FID5K - static_fid,
            "schedule_improvement_vs_historical_v800": HISTORICAL_V800_FID5K - sched_fid,
        },
    }

    summary_path = root / "comparison_summary.json"
    atomic_write_json(summary_path, summary)

    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "gamma", "fid", "sfid", "inception_score", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== RESULT ===")
    print(f"static d4 FID-5K:       {static_fid:.6f}")
    print(f"schedule 4->10 FID-5K:  {sched_fid:.6f}")
    print(f"schedule - static:      {sched_fid - static_fid:+.6f} FID")
    print(
        "static vs v800 gain:    "
        f"{HISTORICAL_V800_FID5K - static_fid:+.6f} FID"
    )
    print(
        "4->10 vs v800 gain:     "
        f"{HISTORICAL_V800_FID5K - sched_fid:+.6f} FID"
    )
    print("pairing:                exact same 5K noise/labels")
    if formal_default:
        print("historical pairing:     matches prior 4->8->10 formal 5K")
    print(f"summary:                {summary_path}")
    print(f"csv:                    {csv_path}")


if __name__ == "__main__":
    main()