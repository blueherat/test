"""Summarize preregistered RAE-LPL confirmation runs without hand-copied metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import pandas as pd


FID = "frechet_inception_distance"
KID = "kernel_inception_distance_mean"
IS = "inception_score_mean"
SAMPLING_PROVENANCE_PROTOCOL = "strict-sampling-provenance-v1"
LABEL_SAMPLER_VERSION = "interleaved-v3-provenance"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_means(path: Path, lpl_weight: float) -> dict[str, float | int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    numeric_keys = (
        "total_loss",
        "flow_loss",
        "lpl_batch_contribution",
        "eligible_rate",
        "grad_norm",
        "clip_rate",
    )
    for row in rows:
        for key in numeric_keys:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key} in {path}")
    total = mean(float(row["total_loss"]) for row in rows)
    raw_lpl = mean(float(row["lpl_batch_contribution"]) for row in rows)
    weighted_lpl = float(lpl_weight) * raw_lpl
    return {
        "metric_rows": len(rows),
        "last_step": int(rows[-1]["step"]),
        "train_total_mean": total,
        "train_flow_mean": mean(float(row["flow_loss"]) for row in rows),
        "train_raw_lpl_mean": raw_lpl,
        "train_weighted_lpl_mean": weighted_lpl,
        "train_weighted_lpl_over_total": weighted_lpl / max(total, 1e-12),
        "train_eligible_rate": mean(float(row["eligible_rate"]) for row in rows),
        "train_grad_norm_mean": mean(float(row["grad_norm"]) for row in rows),
        "train_clip_rate": mean(float(row["clip_rate"]) for row in rows),
    }


def evaluation_by_branch(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else [payload]
    indexed = {str(row["branch"]): row for row in rows}
    if not {"flow", "lpl"}.issubset(indexed):
        raise ValueError(f"{path} must contain flow and lpl rows")
    return indexed


def sampling_provenance_valid(row: dict) -> bool:
    return (
        row.get("sampling_provenance_protocol") == SAMPLING_PROVENANCE_PROTOCOL
        and row.get("label_sampler_version") == LABEL_SAMPLER_VERSION
        and all(
            isinstance(row.get(key), str) and bool(row[key])
            for key in (
                "endpoint_checkpoint_sha256",
                "sampling_checkpoint_sha256",
                "sampling_provenance_sha256",
                "sample_npz_sha256",
            )
        )
    )


def official_metrics(path: Path) -> dict[str, float | bool]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else [payload]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain one official row")
    return {
        **{key: float(rows[0][key]) for key in (FID, KID, IS)},
        "sampling_provenance_valid": sampling_provenance_valid(rows[0]),
    }


def official_sampling_seed_comparison(
    official_evaluations: list[Path],
    sampling_stability: dict,
) -> dict:
    official_by_seed = {}
    for path in official_evaluations:
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        if len(rows) != 1:
            raise ValueError(f"{path} must contain one official row")
        row = rows[0]
        sampling_seed = int(row["sampling_seed"])
        if sampling_seed in official_by_seed:
            raise ValueError(f"duplicate official sampling seed {sampling_seed}")
        official_by_seed[sampling_seed] = row

    rows = []
    missing = []
    for paired in sampling_stability.get("rows", []):
        sampling_seed = int(paired["sampling_seed"])
        official = official_by_seed.get(sampling_seed)
        if official is None:
            missing.append(sampling_seed)
            continue
        rows.append(
            {
                "sampling_seed": sampling_seed,
                "official_fid": float(official[FID]),
                "flow_fid": float(paired["flow_fid"]),
                "lpl_fid": float(paired["lpl_fid"]),
                "lpl_minus_official_fid": (
                    float(paired["lpl_fid"]) - float(official[FID])
                ),
                "official_kid": float(official[KID]),
                "flow_kid": float(paired["flow_kid"]),
                "lpl_kid": float(paired["lpl_kid"]),
                "lpl_minus_official_kid": (
                    float(paired["lpl_kid"]) - float(official[KID])
                ),
                "official_is": float(official[IS]),
                "flow_is": float(paired["flow_is"]),
                "lpl_is": float(paired["lpl_is"]),
                "lpl_minus_official_is": (
                    float(paired["lpl_is"]) - float(official[IS])
                ),
                "sampling_provenance_valid": (
                    sampling_provenance_valid(official)
                    and bool(paired["sampling_provenance_valid"])
                ),
            }
        )

    requested = {
        int(seed) for seed in sampling_stability.get("requested_sampling_seeds", [])
    }
    evaluated = (
        bool(sampling_stability.get("evaluated"))
        and len(requested) >= 3
        and {row["sampling_seed"] for row in rows} == requested
        and not missing
    )
    return {
        "evaluated": evaluated,
        "missing_sampling_seeds": missing,
        "all_lpl_fid_better_than_official": (
            evaluated and all(row["lpl_minus_official_fid"] < 0 for row in rows)
        ),
        "all_lpl_kid_better_than_official": (
            evaluated and all(row["lpl_minus_official_kid"] < 0 for row in rows)
        ),
        "all_lpl_is_better_than_official": (
            evaluated and all(row["lpl_minus_official_is"] > 0 for row in rows)
        ),
        "all_sampling_provenance_valid": (
            evaluated and all(row["sampling_provenance_valid"] for row in rows)
        ),
        "rows": rows,
    }


def sampling_seed_stability(
    results: Path,
    *,
    prior: str,
    training_seed: int,
    endpoint: int,
    sampling_seeds: list[int],
) -> dict:
    rows = []
    missing = []
    for sampling_seed in sampling_seeds:
        path = (
            results
            / f"eval_{prior}_seed{int(training_seed)}_pair_s{int(endpoint)}"
            f"_n5000_seed{int(sampling_seed)}.json"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        evaluations = evaluation_by_branch(path)
        flow = evaluations["flow"]
        lpl = evaluations["lpl"]
        rows.append(
            {
                "training_seed": int(training_seed),
                "sampling_seed": int(sampling_seed),
                "flow_fid": float(flow[FID]),
                "lpl_fid": float(lpl[FID]),
                "fid_lpl_minus_flow": float(lpl[FID]) - float(flow[FID]),
                "flow_kid": float(flow[KID]),
                "lpl_kid": float(lpl[KID]),
                "flow_is": float(flow[IS]),
                "lpl_is": float(lpl[IS]),
                "sampling_provenance_valid": (
                    sampling_provenance_valid(flow)
                    and sampling_provenance_valid(lpl)
                ),
            }
        )
    evaluated = len(sampling_seeds) >= 3 and len(rows) == len(sampling_seeds)
    passed = (
        evaluated
        and all(row["fid_lpl_minus_flow"] < 0 for row in rows)
        and all(row["sampling_provenance_valid"] for row in rows)
    )
    return {
        "training_seed": int(training_seed),
        "requested_sampling_seeds": [int(seed) for seed in sampling_seeds],
        "evaluated": evaluated,
        "passed": passed,
        "missing": missing,
        "rows": rows,
    }


def summarize(
    results: Path,
    *,
    prior: str,
    seeds: list[int],
    endpoint: int,
    sampling_seed: int,
    official_evaluation: Path,
    stability_training_seed: int | None = None,
    stability_sampling_seeds: list[int] | None = None,
    official_stability_evaluations: list[Path] | None = None,
) -> dict:
    rows = []
    missing = []
    for seed in seeds:
        flow_dir = results / f"{prior}_seed{seed}_flow_to_s{endpoint}"
        lpl_dir = results / f"{prior}_seed{seed}_full_to_s{endpoint}"
        audit_path = results / f"{prior}_seed{seed}_pair_audit_s{endpoint}.json"
        evaluation_path = (
            results
            / f"eval_{prior}_seed{seed}_pair_s{endpoint}_n5000_seed{sampling_seed}.json"
        )
        required = (
            flow_dir / "manifest.json",
            flow_dir / "metrics.jsonl",
            lpl_dir / "manifest.json",
            lpl_dir / "metrics.jsonl",
            audit_path,
            evaluation_path,
        )
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            missing.extend(absent)
            continue

        flow_manifest = read_json(flow_dir / "manifest.json")
        lpl_manifest = read_json(lpl_dir / "manifest.json")
        audit = read_json(audit_path)
        evaluations = evaluation_by_branch(evaluation_path)
        flow_eval = evaluations["flow"]
        lpl_eval = evaluations["lpl"]
        row = {
            "seed": int(seed),
            "audit_passed": bool(audit["passed"]),
            "stream_hashes_match": not any(
                "data-stream SHA256 differs" in error
                for error in audit.get("errors", [])
            ),
            "source_checkpoint_sha256": flow_manifest["source_checkpoint_sha256"],
            "flow_lpl_weight": float(flow_manifest["lpl_weight"]),
            "lpl_weight": float(lpl_manifest["lpl_weight"]),
            **{
                f"flow_{key}": value
                for key, value in metric_means(
                    flow_dir / "metrics.jsonl",
                    float(flow_manifest["lpl_weight"]),
                ).items()
            },
            **{
                f"lpl_{key}": value
                for key, value in metric_means(
                    lpl_dir / "metrics.jsonl",
                    float(lpl_manifest["lpl_weight"]),
                ).items()
            },
            "flow_fid": float(flow_eval[FID]),
            "lpl_fid": float(lpl_eval[FID]),
            "fid_lpl_minus_flow": float(lpl_eval[FID]) - float(flow_eval[FID]),
            "flow_kid": float(flow_eval[KID]),
            "lpl_kid": float(lpl_eval[KID]),
            "kid_lpl_minus_flow": float(lpl_eval[KID]) - float(flow_eval[KID]),
            "flow_is": float(flow_eval[IS]),
            "lpl_is": float(lpl_eval[IS]),
            "is_lpl_minus_flow": float(lpl_eval[IS]) - float(flow_eval[IS]),
            "sampling_provenance_valid": (
                sampling_provenance_valid(flow_eval)
                and sampling_provenance_valid(lpl_eval)
            ),
        }
        rows.append(row)

    official = official_metrics(official_evaluation)
    completed = len(rows)
    fid_improvements = sum(row["fid_lpl_minus_flow"] < 0 for row in rows)
    kid_improvements = sum(row["kid_lpl_minus_flow"] < 0 for row in rows)
    all_requested_complete = completed == len(seeds)
    mean_lpl_fid = mean(row["lpl_fid"] for row in rows) if rows else None
    all_sampling_provenance_valid = (
        all_requested_complete
        and bool(official["sampling_provenance_valid"])
        and all(row["sampling_provenance_valid"] for row in rows)
    )
    if stability_training_seed is None or not stability_sampling_seeds:
        stability = {
            "training_seed": stability_training_seed,
            "requested_sampling_seeds": stability_sampling_seeds or [],
            "evaluated": False,
            "passed": False,
            "missing": [],
            "rows": [],
        }
    else:
        stability = sampling_seed_stability(
            results,
            prior=prior,
            training_seed=stability_training_seed,
            endpoint=endpoint,
            sampling_seeds=stability_sampling_seeds,
        )
        missing.extend(stability["missing"])
    if official_stability_evaluations:
        official_comparison = official_sampling_seed_comparison(
            [official_evaluation, *official_stability_evaluations],
            stability,
        )
    else:
        official_comparison = {
            "evaluated": False,
            "missing_sampling_seeds": [],
            "all_lpl_fid_better_than_official": False,
            "all_lpl_kid_better_than_official": False,
            "all_lpl_is_better_than_official": False,
            "all_sampling_provenance_valid": False,
            "rows": [],
        }
    fixed_seed_training_gate_passed = (
        all_requested_complete
        and fid_improvements == len(seeds)
        and kid_improvements >= 2
        and mean_lpl_fid is not None
        and mean_lpl_fid < official[FID]
        and all(row["audit_passed"] for row in rows)
        and all_sampling_provenance_valid
    )
    summary = {
        "prior": prior,
        "endpoint": int(endpoint),
        "sampling_seed": int(sampling_seed),
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": completed,
        "all_requested_complete": all_requested_complete,
        "missing": missing,
        "official": official,
        "fid_improvements": fid_improvements,
        "kid_improvements": kid_improvements,
        "mean_flow_fid": mean(row["flow_fid"] for row in rows) if rows else None,
        "mean_lpl_fid": mean_lpl_fid,
        "mean_flow_kid": mean(row["flow_kid"] for row in rows) if rows else None,
        "mean_lpl_kid": mean(row["lpl_kid"] for row in rows) if rows else None,
        "all_audits_passed": bool(rows) and all(row["audit_passed"] for row in rows),
        "all_sampling_provenance_valid": all_sampling_provenance_valid,
        "fixed_seed_training_gate_passed": fixed_seed_training_gate_passed,
        "multi_sampling_seed_gate_evaluated": bool(stability["evaluated"]),
        "multi_sampling_seed_gate_passed": bool(stability["passed"]),
        "official_sampling_seed_comparison": official_comparison,
        "strong_reproduction_complete": (
            fixed_seed_training_gate_passed and bool(stability["passed"])
        ),
        "sampling_seed_stability": stability,
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prior", default="ditdh_s_ep20")
    parser.add_argument("--seeds", type=int, nargs="+", default=(4101, 4102, 4103))
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--sampling-seed", type=int, default=20260715)
    parser.add_argument("--stability-training-seed", type=int)
    parser.add_argument("--stability-sampling-seeds", type=int, nargs="*")
    parser.add_argument(
        "--official-stability-evaluations",
        type=Path,
        nargs="*",
        help=(
            "Additional zero-update official evaluations for the stability sampling "
            "seeds. This diagnostic does not change the preregistered strong gate."
        ),
    )
    parser.add_argument("--official-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize(
        args.results.expanduser().resolve(),
        prior=args.prior,
        seeds=list(args.seeds),
        endpoint=args.endpoint,
        sampling_seed=args.sampling_seed,
        official_evaluation=args.official_evaluation.expanduser().resolve(),
        stability_training_seed=args.stability_training_seed,
        stability_sampling_seeds=args.stability_sampling_seeds,
        official_stability_evaluations=(
            [path.expanduser().resolve() for path in args.official_stability_evaluations]
            if args.official_stability_evaluations
            else None
        ),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(payload["rows"]).to_csv(output.with_suffix(".csv"), index=False)
    print(pd.DataFrame(payload["rows"]).to_string(index=False))
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
