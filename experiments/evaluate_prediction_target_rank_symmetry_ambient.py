#!/usr/bin/env python3
"""Add ambient-space rollout metrics to completed rank-symmetry settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    sample_spiral_2d,
    stable_seed,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    build_matched_models,
    evaluate_generation,
    sample_models,
    save_csv,
)


def parse_path_list(text: str) -> list[Path]:
    paths = [Path(value.strip()) for value in text.split(",") if value.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("expected comma-separated paths")
    return paths


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--roots", type=parse_path_list, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--check-existing-tolerance",
        type=float,
        default=5e-6,
        help="maximum allowed absolute change in the reproduced intrinsic SWD",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate_root(root: Path, device: torch.device, tolerance: float) -> list[dict]:
    root = root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["args"]
    aggregate: list[dict] = []
    summaries = sorted(root.rglob("summary.json"))
    if not summaries:
        raise RuntimeError(f"no completed settings under {root}")

    for summary_path in summaries:
        setting = json.loads(summary_path.read_text(encoding="utf-8"))
        setting_dir = summary_path.parent
        checkpoint = setting_dir / "models.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seed = int(setting["seed"])
        D = int(setting["D"])
        curvature = float(setting["curvature"])
        output_rank = int(setting["output_rank"])
        # v1 manifests predate the field and used rank-dependent randomness.
        rank_dependent = bool(config.get("rank_dependent_randomness", True))
        rank_seed = output_rank if rank_dependent else 0
        setting_seed = stable_seed(seed, D, int(curvature * 10000), rank_seed, 1009)
        embedding = CurvedEmbedding(
            D,
            curvature=curvature,
            frequency_scale=float(config["frequency_scale"]),
            seed=stable_seed(seed, D, int(curvature * 10000), 41),
            device=device,
            scale_mode=str(config["scale_mode"]),
        )
        models = build_matched_models(
            D=D,
            hidden=int(config["hidden"]),
            output_rank=output_rank,
            depth=int(config["depth"]),
            time_dim=int(config["time_dim"]),
            seed=setting_seed,
            device=device,
        )
        checkpoint_state = torch.load(checkpoint, map_location=device, weights_only=True)
        for condition, model in models.items():
            model.load_state_dict(checkpoint_state[condition])
            model.eval()

        reference_generator = torch.Generator(device=device.type)
        reference_generator.manual_seed(stable_seed(setting_seed, 1201))
        reference_intrinsic = sample_spiral_2d(
            max(2 * int(config["sample_count"]), 8192),
            device=device,
            jitter=float(config["data_jitter"]),
            generator=reference_generator,
        ).cpu().numpy()
        generated = sample_models(
            models=models,
            embedding=embedding,
            count=int(config["sample_count"]),
            batch_size=int(config["sample_batch_size"]),
            steps=int(config["sample_steps"]),
            t_max=float(config["sample_t_max"]),
            t_min=float(config["sample_t_min"]),
            conversion_clip=float(config["conversion_clip"]),
            seed=stable_seed(setting_seed, 1213),
            device=device,
        )
        rows = evaluate_generation(
            samples=generated,
            reference_intrinsic=reference_intrinsic,
            embedding=embedding,
            output_rank=output_rank,
            seed=setting_seed,
            device=device,
            metric_max_points=int(config["metric_max_points"]),
            projections=int(config["swd_projections"]),
            rank_dependent_randomness=rank_dependent,
        )
        old_by_condition = {
            row["condition"]: row
            for row in load_csv(setting_dir / "generation_metrics.csv")
        }
        for row in rows:
            old = old_by_condition[row["condition"]]
            delta = abs(float(row["swd_2d"]) - float(old["swd_2d"]))
            if delta > tolerance:
                raise RuntimeError(
                    f"reproduction mismatch at {setting_dir}/{row['condition']}: {delta}"
                )
            row["intrinsic_swd_reproduction_abs_delta"] = delta
            row["setting_path"] = str(setting_dir)
        save_csv(setting_dir / "generation_metrics_ambient.csv", rows)
        aggregate.extend(rows)
        print(
            f"[ambient] D={D} curvature={curvature:g} rank={output_rank} complete",
            flush=True,
        )

    save_csv(root / "generation_metrics_ambient.csv", aggregate)
    return aggregate


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    all_rows: list[dict] = []
    for root in args.roots:
        all_rows.extend(evaluate_root(root, device, args.check_existing_tolerance))
    print(f"[done] wrote {len(all_rows)} ambient metric rows", flush=True)


if __name__ == "__main__":
    main()
