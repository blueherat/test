from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize adapter finetune ADM-FID JSON files.")
    parser.add_argument("--run-name", default="finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu")
    parser.add_argument("--sample-root", default=os.path.expanduser("~/data/eqvae/stage2_samples"))
    return parser.parse_args()


def parse_result_name(run_name: str, path: Path) -> dict[str, object] | None:
    pattern = re.compile(
        rf"^{re.escape(run_name)}_step(?P<step>\d+)"
        rf"(?:_(?P<weight_before>model|ema|auto))?"
        rf"_n(?P<num_samples>\d+)"
        rf"(?:_(?P<weight_after>model|ema|auto))?"
        rf"_adm(?:_adm)?_fid\.json$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    weight = match.group("weight_before") or match.group("weight_after") or "auto"
    return {
        "step": int(match.group("step")),
        "num_samples": int(match.group("num_samples")),
        "weight": weight,
    }


def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root)
    rows = []
    for path in sorted(sample_root.glob(f"{args.run_name}_step*_adm_fid.json")):
        parsed = parse_result_name(args.run_name, path)
        if parsed is None:
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                **parsed,
                "fid": data.get("fid"),
                "sfid": data.get("sfid"),
                "is": data.get("inception_score"),
                "path": str(path),
            }
        )

    rows.sort(key=lambda row: (row["step"], row["num_samples"], str(row["weight"])))
    if not rows:
        print("No ADM-FID JSON files found.")
        return

    print("| step | weight | n | FID | sFID | IS |")
    print("| ---: | :--- | ---: | ---: | ---: | ---: |")
    for row in rows:
        fid = row["fid"]
        sfid = row["sfid"]
        iscore = row["is"]
        print(
            f"| {row['step']} | {row['weight']} | {row['num_samples']} | "
            f"{fid:.4f} | {sfid:.4f} | {iscore:.4f} |"
        )


if __name__ == "__main__":
    main()
