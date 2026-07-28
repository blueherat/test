"""Audit paired streams and summarize post-switch SPC training dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.evaluate_rae_spc_multiseed import (
    DEFAULT_SEEDS,
    branch_name,
)


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
PAIR_FIELDS = (
    "target_energy",
    "semantic_energy",
    "detail_energy",
    "mean_time",
    "lr",
)


def load_pair(
    results: Path, seed: int, endpoint: int, switch_step: int
) -> pd.DataFrame:
    frames = []
    for condition in ("static", "spc"):
        name = branch_name(seed, condition, endpoint, switch_step)
        rows = [
            json.loads(line)
            for line in (results / name / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        frame = pd.DataFrame(rows)
        frame.insert(0, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit_pairing(pair: pd.DataFrame, switch_step: int) -> dict[str, object]:
    post = pair[pair.step > switch_step]
    wide = post.pivot(index="step", columns="condition", values=list(PAIR_FIELDS))
    max_differences = {
        field: float((wide[(field, "spc")] - wide[(field, "static")]).abs().max())
        for field in PAIR_FIELDS
    }
    return {
        "post_switch_row_count": int(len(wide)),
        "max_absolute_differences": max_differences,
        "model_independent_stream_exact": bool(
            all(value == 0.0 for value in max_differences.values())
        ),
    }


def window_rows(
    pair: pd.DataFrame, seed: int, switch_step: int, endpoint: int
) -> list[dict[str, float | int]]:
    rows = []
    boundaries = [switch_step, 3000, 4000, endpoint]
    boundaries = sorted({value for value in boundaries if switch_step <= value <= endpoint})
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        subset = pair[(pair.step > start) & (pair.step <= end)]
        means = subset.groupby("condition")[["loss", "grad_norm"]].mean()
        rows.append(
            {
                "seed": int(seed),
                "start_exclusive": int(start),
                "end_inclusive": int(end),
                "static_loss": float(means.loc["static", "loss"]),
                "spc_loss": float(means.loc["spc", "loss"]),
                "spc_minus_static_loss": float(
                    means.loc["spc", "loss"] - means.loc["static", "loss"]
                ),
                "static_grad_norm": float(means.loc["static", "grad_norm"]),
                "spc_grad_norm": float(means.loc["spc", "grad_norm"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    output = results / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    audits = {}
    rows = []
    for seed in seeds:
        pair = load_pair(results, seed, args.endpoint, args.switch_step)
        audits[str(seed)] = audit_pairing(pair, args.switch_step)
        rows.extend(window_rows(pair, seed, args.switch_step, args.endpoint))
    table = pd.DataFrame(rows)
    table.to_csv(output / "spc_post_switch_training_windows.csv", index=False)
    decision = {
        "all_post_switch_streams_exact": bool(
            all(row["model_independent_stream_exact"] for row in audits.values())
        ),
        "audits": audits,
    }
    (output / "spc_training_pairing_audit.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
