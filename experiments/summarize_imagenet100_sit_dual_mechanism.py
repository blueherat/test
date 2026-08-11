"""Create compact, auditable tables for the SiT dual-output mechanism study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_BASELINE_400 = DATA_ROOT / "fid5k/sit-s-2_step400000_seed0"
DEFAULT_DUAL_400 = DATA_ROOT / "fid5k_dual-output_step400000_seed0"
DEFAULT_DUAL_450 = DATA_ROOT / "fid5k_dual-output_step450000_seed0"
DEFAULT_TRAIN_METRICS = (
    DATA_ROOT / "runs/sit-s-2_dual-output_seed0/train_metrics.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/data/imagenet100_sit_dual_endpoint_audit"


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nfe_rows(
    run_dir: Path,
    *,
    model_family: str,
    checkpoint_step: int,
    mode: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = load_json(run_dir / "sampling_manifest.json")
    if int(manifest["checkpoint_step"]) != checkpoint_step:
        raise ValueError(f"checkpoint step mismatch in {run_dir}")
    batch_size = int(manifest["per_rank_batch_size"])
    world_size = int(manifest["world_size"])
    rank_paths = sorted(run_dir.glob("rank_*.json"))
    if len(rank_paths) != world_size:
        raise ValueError(f"expected {world_size} rank files in {run_dir}")

    rows: list[dict[str, object]] = []
    for rank_path in rank_paths:
        rank = load_json(rank_path)
        sample_count = int(rank["sample_count"])
        if sample_count % batch_size:
            raise ValueError(f"rank sample count is not batch-aligned in {rank_path}")
        batch_count = sample_count // batch_size
        total_nfe = int(rank["total_nfe_across_batches"])
        rows.append(
            {
                "model_family": model_family,
                "checkpoint_step": checkpoint_step,
                "mode": mode,
                "rank": int(rank["rank"]),
                "sample_count": sample_count,
                "batch_size": batch_size,
                "batch_count": batch_count,
                "total_nfe_across_batches": total_nfe,
                "mean_nfe_per_batch": total_nfe / batch_count,
                "elapsed_seconds": float(rank["elapsed_seconds"]),
            }
        )

    total_batches = sum(int(row["batch_count"]) for row in rows)
    total_nfe = sum(int(row["total_nfe_across_batches"]) for row in rows)
    rank_means = [float(row["mean_nfe_per_batch"]) for row in rows]
    summary = {
        "model_family": model_family,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "mode": mode,
        "world_size": world_size,
        "requested_sample_count": int(manifest["requested_samples"]),
        "padded_sample_count": sum(int(row["sample_count"]) for row in rows),
        "batch_size_per_rank": batch_size,
        "total_batches_across_ranks": total_batches,
        "mean_nfe_per_batch": total_nfe / total_batches,
        "min_rank_mean_nfe": min(rank_means),
        "max_rank_mean_nfe": max(rank_means),
        "mean_elapsed_seconds_across_ranks": sum(
            float(row["elapsed_seconds"]) for row in rows
        )
        / len(rows),
        "nfe_grain": "one adaptive ODE trajectory for one complete batch",
        "distribution_limitation": (
            "rank JSON stores cumulative NFE; full per-batch distribution is not recoverable"
        ),
    }
    return rows, summary


def fid_row(
    run_dir: Path,
    *,
    model_family: str,
    checkpoint_step: int,
    mode: str,
) -> dict[str, object]:
    result = load_json(run_dir / "fid5k_adm_results.json")
    manifest = load_json(run_dir / "sampling_manifest.json")
    if int(manifest["checkpoint_step"]) != checkpoint_step:
        raise ValueError(f"checkpoint step mismatch in {run_dir}")
    return {
        "model_family": model_family,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "mode": mode,
        "sample_count": int(manifest["requested_samples"]),
        "guidance": bool(manifest["guidance"]),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
    }


def validation_record(metrics_path: Path, step: int) -> dict:
    matches: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) == step and "ema_validation" in row:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one EMA validation record at step {step}, found {len(matches)}"
        )
    return matches[0]


def validation_tables(metrics_path: Path, steps: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    time_bins: list[dict] = []
    for step in steps:
        record = validation_record(metrics_path, step)
        for weights_key in ("raw_validation", "ema_validation"):
            metrics = record[weights_key]
            summaries.append(
                {
                    "checkpoint_step": step,
                    "weights": weights_key.removesuffix("_validation"),
                    **{
                        key: float(metrics[key])
                        for key in (
                            "clean_mse",
                            "epsilon_mse",
                            "gate_loss",
                            "gate_mean",
                            "gate_std",
                            "velocity_x_mse",
                            "velocity_epsilon_mse",
                            "velocity_dynamic_mse",
                        )
                    },
                }
            )
            for time_bin in metrics["time_bins"]:
                gate_variance = max(
                    0.0,
                    float(time_bin["gate_square"]) - float(time_bin["gate_mean"]) ** 2,
                )
                time_bins.append(
                    {
                        "checkpoint_step": step,
                        "weights": weights_key.removesuffix("_validation"),
                        "t_min": float(time_bin["t_min"]),
                        "t_max": float(time_bin["t_max"]),
                        "count": int(time_bin["count"]),
                        "gate_mean": float(time_bin["gate_mean"]),
                        "gate_std_derived": math.sqrt(gate_variance),
                        "velocity_x_mse": float(time_bin["velocity_x_mse"]),
                        "velocity_epsilon_mse": float(time_bin["velocity_epsilon_mse"]),
                        "velocity_dynamic_mse": float(time_bin["velocity_dynamic_mse"]),
                    }
                )
    return summaries, time_bins


def run(args: argparse.Namespace) -> dict[str, object]:
    nfe_specs = (
        (args.baseline_400, "single_velocity", 400_000, "velocity"),
        (args.dual_400 / "dynamic", "dual_output", 400_000, "dynamic"),
        (args.dual_450 / "x", "dual_output", 450_000, "x"),
        (args.dual_450 / "epsilon", "dual_output", 450_000, "epsilon"),
        (args.dual_450 / "dynamic", "dual_output", 450_000, "dynamic"),
    )
    nfe_rank_rows: list[dict[str, object]] = []
    nfe_summaries: list[dict[str, object]] = []
    for run_dir, family, step, mode in nfe_specs:
        ranks, summary = nfe_rows(
            run_dir,
            model_family=family,
            checkpoint_step=step,
            mode=mode,
        )
        nfe_rank_rows.extend(ranks)
        nfe_summaries.append(summary)

    fid_rows = [
        fid_row(
            run_dir,
            model_family=family,
            checkpoint_step=step,
            mode=mode,
        )
        for run_dir, family, step, mode in nfe_specs
    ]
    validation_summaries, validation_bins = validation_tables(
        args.train_metrics, (400_000, 450_000)
    )

    label_hashes: list[str] = []
    references: list[str] = []
    sampler_protocols: list[dict[str, object]] = []
    for run_dir, _, _, _ in nfe_specs:
        label_paths = list(run_dir.glob("sample_labels_unguided_n*.npy"))
        if len(label_paths) != 1:
            raise ValueError(f"expected one label artifact in {run_dir}")
        label_hashes.append(sha256_file(label_paths[0]))
        references.append(str(load_json(run_dir / "fid5k_adm_results.json")["reference"]))
        sampler = load_json(run_dir / "sampling_manifest.json")["sampler"]
        sampler_protocols.append(
            {
                key: sampler[key]
                for key in (
                    "method",
                    "interval",
                    "num_output_points",
                    "atol",
                    "rtol",
                    "precision",
                    "allow_tf32",
                )
            }
        )
    if len(set(label_hashes)) != 1:
        raise ValueError("sampling label arrays differ across compared runs")
    if len(set(references)) != 1:
        raise ValueError("FID reference artifacts differ across compared runs")
    canonical_sampler = json.dumps(sampler_protocols[0], sort_keys=True)
    if any(json.dumps(item, sort_keys=True) != canonical_sampler for item in sampler_protocols):
        raise ValueError("sampler protocols differ across compared runs")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(nfe_rank_rows, output_dir / "nfe_by_rank.csv")
    write_csv(nfe_summaries, output_dir / "nfe_summary.csv")
    write_csv(fid_rows, output_dir / "fid_summary.csv")
    write_csv(validation_summaries, output_dir / "validation_summary.csv")
    write_csv(validation_bins, output_dir / "validation_time_bins.csv")
    payload: dict[str, object] = {
        "protocol": "imagenet100_sit_dual_mechanism_summary_v1",
        "source_scope": {
            "baseline_400": str(args.baseline_400),
            "dual_400": str(args.dual_400),
            "dual_450": str(args.dual_450),
            "train_metrics": str(args.train_metrics),
        },
        "tables": {
            "nfe_by_rank": "nfe_by_rank.csv",
            "nfe_summary": "nfe_summary.csv",
            "fid_summary": "fid_summary.csv",
            "validation_summary": "validation_summary.csv",
            "validation_time_bins": "validation_time_bins.csv",
        },
        "quality_notes": [
            "All FID rows use 5000 requested samples and no guidance.",
            "NFE is one model-vector-field evaluation for a complete batch trajectory.",
            "NFE totals include 5120 padded samples (1280 per rank), not only the retained 5000.",
            (
                "Original rank artifacts contain cumulative NFE only; "
                "no per-batch distribution is claimed."
            ),
            "Validation rows are the unique records embedded in the original training JSONL.",
        ],
        "quality_checks": {
            "same_label_array": True,
            "label_sha256": label_hashes[0],
            "same_fid_reference": True,
            "fid_reference": references[0],
            "same_sampler_protocol": True,
            "sampler": sampler_protocols[0],
        },
        "nfe_summary": nfe_summaries,
        "fid_summary": fid_rows,
        "validation_summary": validation_summaries,
    }
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-400", type=Path, default=DEFAULT_BASELINE_400)
    parser.add_argument("--dual-400", type=Path, default=DEFAULT_DUAL_400)
    parser.add_argument("--dual-450", type=Path, default=DEFAULT_DUAL_450)
    parser.add_argument("--train-metrics", type=Path, default=DEFAULT_TRAIN_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "event": "complete",
                "nfe_rows": len(payload["nfe_summary"]),
                "fid_rows": len(payload["fid_summary"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
