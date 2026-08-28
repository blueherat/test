#!/usr/bin/env python3
"""Build read-only endpoint-distance ranking sheets for the old expansion pool.

This is a diagnostic visualization helper.  It does not alter or participate in
any frozen protocol, inferential audit, or third-pool workflow.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BASE = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "bad_good_metric_confirmation_expansion_v1"
)
DEFAULT_FROZEN_SOURCE = (
    DEFAULT_BASE / "endpoint_representation_distance_audit_v1/analysis_source.py"
)
DEFAULT_EMBEDDINGS = DEFAULT_BASE / "endpoint_embeddings_label_free_v1"
DEFAULT_ORIGINAL_LABELS = Path(
    "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
DEFAULT_EXPANSION_LABELS = Path(
    "experiments/annotations/dit_expansion_eval360_adjudicated_consensus_lock_v1"
)
DEFAULT_PROTOCOL = Path(
    "experiments/locks/dit_endpoint_representation_distance_protocol_v1/protocol.json"
)
DEFAULT_OUTPUT = DEFAULT_BASE / "endpoint_representation_rank_visual_audit_v1"


SHORT_NAMES = {
    "inception_fid_pool2048__cosine_to_class_clean_centroid": "inc_centroid",
    "inception_fid_pool2048__shared_ledoitwolf_mahalanobis": "inc_mahal",
    "inception_fid_pool2048__knn5_mean_cosine_to_class_clean": "inc_knn5",
    "dinov2_registers_large_cls1024__cosine_to_class_clean_centroid": "dino_centroid",
    "dinov2_registers_large_cls1024__shared_ledoitwolf_mahalanobis": "dino_mahal",
    "dinov2_registers_large_cls1024__knn5_mean_cosine_to_class_clean": "dino_knn5",
}


def load_frozen_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_endpoint_distance_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rank_percentile(values: pd.Series) -> pd.Series:
    n = len(values)
    if n <= 1:
        return pd.Series(np.full(n, 0.5), index=values.index)
    return (values.rank(method="average") - 1.0) / (n - 1.0)


def fit_text(draw: ImageDraw.ImageDraw, text: str, width: int, font) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    suffix = "..."
    while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > width:
        text = text[:-1]
    return text + suffix


def contact_sheet(rows: pd.DataFrame, title: str, output: Path) -> None:
    tile = 256
    caption = 52
    columns = 6
    rows_n = int(np.ceil(len(rows) / columns))
    header = 52
    canvas = Image.new("RGB", (columns * tile, header + rows_n * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="black", font=font)
    for position, (_, row) in enumerate(rows.iterrows()):
        x = (position % columns) * tile
        y = header + (position // columns) * (tile + caption)
        with Image.open(row.endpoint_png_path) as source:
            image = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        line1 = (
            f"{row.selection} c{int(row.class_id)} s{int(row.global_seed)} "
            f"{row.label}"
        )
        line2 = f"d={row.score:.5f} pct={row.class_percentile:.3f}"
        draw.text((x + 3, y + tile + 3), fit_text(draw, line1, tile - 6, font), fill="black", font=font)
        draw.text((x + 3, y + tile + 21), fit_text(draw, line2, tile - 6, font), fill="black", font=font)
    canvas.save(output)


def build(args: argparse.Namespace) -> None:
    frozen = load_frozen_module(args.frozen_source.resolve())
    protocol = json.loads(args.protocol.read_text())
    index, arrays, _ = frozen.validate_embedding_product(args.embeddings, protocol)
    original, _ = frozen.validate_label_lock(
        args.original_labels,
        frozen.DISCOVERY_SEEDS,
        "FINAL_VISUAL_LABELS_LOCKED_BEFORE_ANY_LABEL_SCORE_JOIN",
    )
    expansion, _ = frozen.validate_label_lock(
        args.expansion_labels,
        frozen.EXPANSION_SEEDS,
        "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN",
    )
    labels = pd.concat([original, expansion], ignore_index=True)
    distances, _ = frozen.compute_distances(index, arrays, labels)
    paths = index[["sample_index", "endpoint_png_path"]]
    distances = distances.merge(paths, on="sample_index", validate="one_to_one")
    expansion_rows = distances[distances.global_seed.isin(frozen.EXPANSION_SEEDS)].copy()

    metrics = list(SHORT_NAMES)
    for metric in metrics:
        expansion_rows[f"{metric}__class_percentile"] = expansion_rows.groupby(
            "class_id", group_keys=False
        )[metric].apply(rank_percentile)

    inc = metrics[:3]
    dino = metrics[3:]
    expansion_rows["inception_mean_percentile"] = expansion_rows[
        [f"{metric}__class_percentile" for metric in inc]
    ].mean(axis=1)
    expansion_rows["dino_mean_percentile"] = expansion_rows[
        [f"{metric}__class_percentile" for metric in dino]
    ].mean(axis=1)
    expansion_rows["inception_minus_dino"] = (
        expansion_rows.inception_mean_percentile - expansion_rows.dino_mean_percentile
    )

    args.output.mkdir(parents=True, exist_ok=False)
    expansion_rows.to_csv(args.output / "all_expansion_distances.csv", index=False)

    selected_parts = []
    for metric in metrics:
        percentile = f"{metric}__class_percentile"
        parts = []
        for class_id, group in expansion_rows.groupby("class_id", sort=True):
            top = group.nlargest(args.top_per_class, metric).copy()
            bottom = group.nsmallest(args.bottom_per_class, metric).copy()
            top["selection"] = "TOP"
            bottom["selection"] = "BOTTOM"
            parts.extend([top, bottom])
        chosen = pd.concat(parts, ignore_index=True)
        chosen["metric"] = metric
        chosen["metric_short"] = SHORT_NAMES[metric]
        chosen["score"] = chosen[metric]
        chosen["class_percentile"] = chosen[percentile]
        chosen = chosen[
            [
                "metric",
                "metric_short",
                "selection",
                "global_seed",
                "class_id",
                "label",
                "score",
                "class_percentile",
                "endpoint_png_path",
            ]
        ]
        chosen.to_csv(args.output / f"ranking_{SHORT_NAMES[metric]}.csv", index=False)
        contact_sheet(
            chosen,
            f"{SHORT_NAMES[metric]}: per-class high distance (TOP) and low distance (BOTTOM)",
            args.output / f"contact_{SHORT_NAMES[metric]}.png",
        )
        selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected.to_csv(args.output / "all_metric_rank_selections.csv", index=False)

    discordant_parts = []
    for class_id, group in expansion_rows.groupby("class_id", sort=True):
        inc_high = group.nlargest(args.discordant_per_class, "inception_minus_dino").copy()
        dino_high = group.nsmallest(args.discordant_per_class, "inception_minus_dino").copy()
        inc_high["selection"] = "INC_HIGH_DINO_LOW"
        dino_high["selection"] = "DINO_HIGH_INC_LOW"
        discordant_parts.extend([inc_high, dino_high])
    discordant = pd.concat(discordant_parts, ignore_index=True)
    discordant["score"] = discordant.inception_minus_dino
    discordant["class_percentile"] = discordant.inception_mean_percentile
    discordant[
        [
            "selection",
            "global_seed",
            "class_id",
            "label",
            "inception_mean_percentile",
            "dino_mean_percentile",
            "inception_minus_dino",
            "score",
            "class_percentile",
            "endpoint_png_path",
        ]
    ].to_csv(args.output / "cross_representation_discordant.csv", index=False)
    contact_sheet(
        discordant,
        "Cross-model discordance; pct caption is Inception mean percentile",
        args.output / "contact_cross_representation_discordant.png",
    )

    summary = {
        "status": "READ_ONLY_POSTHOC_VISUAL_DIAGNOSTIC",
        "scope": "old expansion endpoint pool only; no third-pool data",
        "distance_reference": "all discovery clean-good, class conditional",
        "rank_definition": "within-class empirical percentile among 120 expansion samples",
        "top_per_class_per_metric": args.top_per_class,
        "bottom_per_class_per_metric": args.bottom_per_class,
        "discordant_per_direction_per_class": args.discordant_per_class,
        "warning": (
            "Ranked samples were selected after seeing endpoint representations. "
            "These sheets are diagnostic and cannot establish a confirmatory threshold."
        ),
    }
    (args.output / "README.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-source", type=Path, default=DEFAULT_FROZEN_SOURCE)
    result.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    result.add_argument("--original-labels", type=Path, default=DEFAULT_ORIGINAL_LABELS)
    result.add_argument("--expansion-labels", type=Path, default=DEFAULT_EXPANSION_LABELS)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--top-per-class", type=int, default=4)
    result.add_argument("--bottom-per-class", type=int, default=2)
    result.add_argument("--discordant-per-class", type=int, default=3)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
