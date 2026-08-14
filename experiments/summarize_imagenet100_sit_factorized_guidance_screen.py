#!/usr/bin/env python3
"""Summarize the paired factorized-guidance and closed-AG screen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/nominal_intervention_fid5k.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root)
        family = relative.parts[0]
        rows.append(
            {
                "family": family,
                "condition": relative.parts[1],
                "mode": payload["mode"],
                "gamma": float(payload["gamma"]),
                "nominal_scale": float(payload.get("nominal_scale", 1.0)),
                "orthogonal_scale": float(payload.get("orthogonal_scale", 1.0)),
                "response_scale": float(payload.get("response_scale", 1.0)),
                "global_seed": int(payload["global_seed"]),
                "num_samples": int(payload["num_samples"]),
                "fid": float(payload["fid"]),
                "sfid": float(payload["sfid"]),
                "inception_score": float(payload["inception_score"]),
                "total_nfe": int(payload["total_nfe"]),
                "noise_fingerprint": payload["noise_fingerprint"],
                "label_fingerprint": payload["label_fingerprint"],
                "result": str(path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no completed screen results under {root}")
    return rows


def validate_pairing(rows: list[dict[str, object]]) -> None:
    fingerprints: dict[tuple[str, int, int], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        key = (str(row["family"]), int(row["global_seed"]), int(row["num_samples"]))
        fingerprints[key].add(
            (str(row["noise_fingerprint"]), str(row["label_fingerprint"]))
        )
    mismatched = {key: values for key, values in fingerprints.items() if len(values) != 1}
    if mismatched:
        raise ValueError(f"screen conditions are not paired: {mismatched}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = load_rows(root)
    validate_pairing(rows)
    rows.sort(key=lambda row: (str(row["family"]), float(row["fid"])))

    columns = list(rows[0])
    with (root / "screen_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "protocol": "imagenet100_sit_factorized_guidance_screen_v1",
        "root": str(root),
        "num_conditions": len(rows),
        "families": {},
    }
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        by_mode = {
            mode: min(
                (row for row in family_rows if row["mode"] == mode),
                key=lambda row: float(row["fid"]),
            )
            for mode in sorted({str(row["mode"]) for row in family_rows})
        }
        summary["families"][family] = {
            "best_overall": min(family_rows, key=lambda row: float(row["fid"])),
            "best_by_mode": by_mode,
            "top_five": family_rows[:5],
        }
    (root / "screen_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    for family, payload in summary["families"].items():
        print(f"\n[{family}]")
        for row in payload["top_five"]:
            print(
                f"FID={row['fid']:.4f} mode={row['mode']} gamma={row['gamma']:g} "
                f"beta={row['nominal_scale']:g} lambda={row['orthogonal_scale']:g} "
                f"rho={row['response_scale']:g}"
            )


if __name__ == "__main__":
    main()
