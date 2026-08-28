#!/usr/bin/env python3
"""Build one physically isolated, unsupervised B-only or C-only score product.

Each invocation emits exactly three identifier columns plus exactly one frozen
candidate score.  It rejects all label/review/consensus fields and all external
Inception/DINO/FID/embedding inputs.  B decodes only the nine retained
pred-xstart latents with the pinned FP32 SD-VAE; C uses only the retained q2
channel-3 latent planes and alpha-bars.  No labels, AUCs, thresholds, ranks, or
candidate combinations are read or computed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
sys.dont_write_bytecode = True

import numpy as np
from scipy import ndimage

try:
    from .dit_event_rich_dynamic_contract import (
        B_CANDIDATE,
        B_CHECKPOINTS,
        B_FEATURE,
        C_CANDIDATE,
        C_CHECKPOINTS,
        C_FEATURE,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        exact_pairs,
        load_json,
        pair_relative_directory,
        publish_artifact,
        reject_forbidden_external_name,
        require_directory,
        require_regular,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        validate_score_columns,
    )
    from .sample_dit_event_rich_dynamic_traces import (
        B_ARRAY,
        C_ALPHA,
        C_ARRAY,
        TRACE_NAME,
        load_pair_arrays,
        load_source_lock,
        validate_pool,
    )
except ImportError:
    from dit_event_rich_dynamic_contract import (  # type: ignore
        B_CANDIDATE,
        B_CHECKPOINTS,
        B_FEATURE,
        C_CANDIDATE,
        C_CHECKPOINTS,
        C_FEATURE,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        candidate_classes,
        canonical_sha256,
        exact_pairs,
        load_json,
        pair_relative_directory,
        publish_artifact,
        reject_forbidden_external_name,
        require_directory,
        require_regular,
        sha256_file,
        validate_anchor_plan,
        validate_event_protocol,
        validate_score_columns,
    )
    from sample_dit_event_rich_dynamic_traces import (  # type: ignore
        B_ARRAY,
        C_ALPHA,
        C_ARRAY,
        TRACE_NAME,
        load_pair_arrays,
        load_source_lock,
        validate_pool,
    )


EXPERIMENT = "dit_event_rich_single_internal_candidate_product"
EPS = 1e-12
GRID = 4
ACTIVE_TILES = 8
VAE_SCALING_FACTOR = 0.18215


def feature_for(candidate: str) -> str:
    if candidate == B_CANDIDATE:
        return B_FEATURE
    if candidate == C_CANDIDATE:
        return C_FEATURE
    raise ValueError(f"unknown candidate: {candidate}")


def local_blur_severity(images: np.ndarray) -> np.ndarray:
    """Exact frozen B checkpoint statistic for float RGB [N,3,256,256]."""

    value = np.asarray(images)
    if value.ndim != 4 or value.shape[1:] != (3, 256, 256):
        raise ValueError(f"B images must be [N,3,256,256], got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("B images must be finite floating point")
    rgb = np.clip(value.astype(np.float64, copy=False), 0.0, 1.0)
    gray = 0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]
    tile_h = gray.shape[-2] // GRID
    tile_w = gray.shape[-1] // GRID
    tiles = [
        (
            slice(row * tile_h, (row + 1) * tile_h),
            slice(column * tile_w, (column + 1) * tile_w),
        )
        for row in range(GRID)
        for column in range(GRID)
    ]
    result = np.empty(len(gray), dtype=np.float64)
    for index, source in enumerate(gray):
        smooth = ndimage.gaussian_filter(source, sigma=0.7, mode="reflect")
        gx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
        magnitude = np.hypot(gx, gy)
        laplacian = ndimage.laplace(smooth, mode="reflect")
        variances = np.asarray(
            [float(np.var(source[ys, xs], dtype=np.float64)) for ys, xs in tiles]
        )
        active = np.argsort(-variances, kind="stable")[:ACTIVE_TILES]
        q = np.asarray(
            [
                float(np.mean(laplacian[ys, xs] ** 2, dtype=np.float64))
                / (float(np.mean(magnitude[ys, xs] ** 2, dtype=np.float64)) + EPS)
                for ys, xs in tiles
            ],
            dtype=np.float64,
        )
        result[index] = -math.log(float(np.percentile(q[active], 25)) + EPS)
    return result


def c3_low_jump(pred_c3: np.ndarray, alpha_bar: np.ndarray) -> float:
    """Exact frozen C q2 maximum positive alpha-compensated energy jump."""

    pred = np.asarray(pred_c3)
    alpha = np.asarray(alpha_bar)
    if pred.shape != (len(C_CHECKPOINTS), 32, 32) or alpha.shape != (len(C_CHECKPOINTS),):
        raise ValueError("C inputs must be [50,32,32] and [50]")
    if not np.isfinite(pred).all() or not np.isfinite(alpha).all():
        raise ValueError("C inputs must be finite")
    work = pred.astype(np.float64, copy=False)
    vertical = np.diff(work, axis=-2)
    horizontal = np.diff(work, axis=-1)
    energy = np.mean(vertical * vertical, axis=(-2, -1)) + np.mean(
        horizontal * horizontal, axis=(-2, -1)
    )
    track = alpha.astype(np.float64, copy=False) * energy
    return float(max(0.0, float(np.max(np.diff(track)))))


def load_vae(snapshot: Path, source_contract: Mapping[str, Any], device: str) -> Any:
    import torch
    from diffusers.models import AutoencoderKL

    snapshot = require_directory(snapshot, "pinned SD-VAE snapshot")
    strict_path = Path(source_contract["source_lock_path"]) / "sources/reproduce_dit_imagenet256.py"
    # The sampler already bound the exact snapshot file hashes.  The product
    # additionally verifies the runtime path with that frozen strict helper.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_event_product_strict", strict_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen strict asset validator")
    strict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strict)
    if strict.validate_vae_snapshot(snapshot) != source_contract["assets"]["vae_snapshot"]:
        raise RuntimeError("B extractor VAE differs from the frozen sampler VAE")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA for B extraction but CUDA is unavailable")
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    vae = AutoencoderKL.from_pretrained(
        str(snapshot),
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def decode_b_score(vae: Any, pred: np.ndarray, *, device: str, batch_size: int) -> float:
    import torch

    if pred.shape != (len(B_CHECKPOINTS), 4, 32, 32) or pred.dtype != np.float32:
        raise RuntimeError("B retained pred-xstart array schema changed")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pred), batch_size):
            latent = torch.from_numpy(pred[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            decoded = vae.decode(latent / VAE_SCALING_FACTOR).sample
            normalized = (decoded.float() + 1.0) / 2.0
            outputs.append(
                np.ascontiguousarray(normalized.cpu().numpy(), dtype=np.float32)
            )
    images = np.concatenate(outputs, axis=0)
    values = local_blur_severity(images)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise RuntimeError("B checkpoint statistic extraction failed")
    return float(np.mean(values, dtype=np.float64))


def score_rows(
    *,
    candidate: str,
    pool_root: Path,
    plan: Mapping[str, Any],
    vae: Any | None,
    device: str,
    decode_batch_size: int,
) -> list[dict[str, Any]]:
    feature = feature_for(candidate)
    rows: list[dict[str, Any]] = []
    for phase, seed, class_id in exact_pairs(plan, candidate=candidate):
        pair_root = pool_root / pair_relative_directory(phase, seed, class_id)
        pair_manifest = load_json(require_regular(pair_root / "manifest.json", "trace pair manifest"))
        trace_record = pair_manifest.get("trace")
        if not isinstance(trace_record, dict):
            raise RuntimeError("trace pair manifest lacks trace record")
        arrays = load_pair_arrays(
            require_regular(pair_root / TRACE_NAME, "minimum internal trace"),
            trace_record.get("arrays", {}),
            plan,
            class_id,
        )
        if candidate == B_CANDIDATE:
            if vae is None:
                raise RuntimeError("B extraction requires the pinned VAE")
            score = decode_b_score(
                vae, arrays[B_ARRAY], device=device, batch_size=decode_batch_size
            )
        else:
            score = c3_low_jump(arrays[C_ARRAY], arrays[C_ALPHA])
        if not math.isfinite(score):
            raise RuntimeError("candidate computation produced a non-finite score")
        rows.append(
            {
                "phase": phase,
                "global_seed": seed,
                "class_id": class_id,
                feature: score,
            }
        )
    return rows


def rows_to_csv(rows: Sequence[Mapping[str, Any]], candidate: str) -> str:
    columns = validate_score_columns(
        ("phase", "global_seed", "class_id", feature_for(candidate)), candidate
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise RuntimeError("candidate row contains an extra or missing column")
        writer.writerow(
            {
                "phase": row["phase"],
                "global_seed": int(row["global_seed"]),
                "class_id": int(row["class_id"]),
                columns[-1]: format(float(row[columns[-1]]), ".17g"),
            }
        )
    return buffer.getvalue()


def build_catalog(candidate: str) -> dict[str, Any]:
    feature = feature_for(candidate)
    if candidate == B_CANDIDATE:
        formula = (
            "mean over sampling k=[69,79,...,149] of -log(percentile25 over the "
            "eight highest-RGB-gray-variance tiles of mean(Laplacian(Gaussian_0.7(Y))^2) "
            "/ (mean(SobelMagnitude(Gaussian_0.7(Y))^2)+1e-12)) + 1e-12), where "
            "Y=0.2989R+0.5870G+0.1140B and RGB=clip((VAE(pred_xstart/0.18215)+1)/2,0,1)"
        )
        source_arrays = [B_ARRAY]
        timing = "available after the model output at sampling step 149, before that innovation"
        orientation = "bad_high"
    else:
        formula = (
            "max(0,max diff(alpha_bar[k]*(mean(diff_vertical(pred_xstart_c3)^2)+"
            "mean(diff_horizontal(pred_xstart_c3)^2)))) for sampling k=100..149"
        )
        source_arrays = [C_ARRAY, C_ALPHA]
        timing = "available after the model output at sampling step 149, before that innovation"
        orientation = "bad_low"
    catalog = {
        "schema_version": 1,
        "candidate": candidate,
        "score_file": "scores.csv",
        "columns": [
            {"name": "phase", "role": "identifier", "allowed": ["calibration", "confirmation"]},
            {"name": "global_seed", "role": "identifier"},
            {"name": "class_id", "role": "identifier"},
            {
                "name": feature,
                "role": "single_internal_candidate_score",
                "formula": formula,
                "orientation": orientation,
                "observation_timing": timing,
                "source_trace_arrays": source_arrays,
            },
        ],
        "contains_labels_reviews_consensus_or_endpoint_judgments": False,
        "contains_inception_dino_fid_embeddings_or_external_distances": False,
        "contains_other_candidate_score": False,
        "auc_rank_threshold_or_selection_computed": False,
    }
    catalog["identity_sha256"] = canonical_sha256(catalog)
    return catalog


def run_extract(args: argparse.Namespace) -> None:
    candidate = args.candidate
    source_lock = require_directory(args.source_lock, "dynamic source lock")
    source_contract, source_manifest, _ = load_source_lock(source_lock)
    expected_source_hash = source_contract.get("source_snapshots", {}).get(
        "extract_dit_event_rich_candidate_product.py", {}
    ).get("sha256")
    if expected_source_hash != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running candidate extractor differs from frozen source snapshot")
    # Make the source-lock path an explicit part of the runtime asset check
    # without changing its frozen canonical identity.
    runtime_contract = dict(source_contract)
    runtime_contract["source_lock_path"] = str(source_lock)
    protocol = validate_event_protocol(Path(source_contract["event_protocol"]["path"]))
    plan = validate_anchor_plan(args.anchor_plan, protocol)
    classes = candidate_classes(plan, candidate)
    if not classes:
        raise RuntimeError(f"{candidate} anchor decision is STOP; no product may be built")
    pool_root = require_directory(args.trace_pool, "dynamic minimum-trace pool")
    reject_forbidden_external_name(pool_root.name, "trace pool name")
    pool_manifest = validate_pool(
        pool_root,
        source_lock=source_lock,
        source_contract=source_contract,
        plan=plan,
    )
    vae = None
    if candidate == B_CANDIDATE:
        if args.vae_snapshot is None:
            raise RuntimeError("B product requires --vae-snapshot")
        vae = load_vae(args.vae_snapshot, runtime_contract, args.device)
    elif args.vae_snapshot is not None:
        raise RuntimeError("C product must not receive or load a VAE argument")
    rows = score_rows(
        candidate=candidate,
        pool_root=pool_root,
        plan=plan,
        vae=vae,
        device=args.device,
        decode_batch_size=args.decode_batch_size,
    )
    expected = exact_pairs(plan, candidate=candidate)
    observed = tuple((row["phase"], row["global_seed"], row["class_id"]) for row in rows)
    if observed != expected:
        raise RuntimeError("candidate product row axis is not exact")
    catalog = build_catalog(candidate)
    csv_text = rows_to_csv(rows, candidate)
    record = {
        "schema_version": 1,
        "status": "SINGLE_INTERNAL_CANDIDATE_PRODUCT_COMPLETE",
        "experiment": EXPERIMENT,
        "candidate": candidate,
        "feature": feature_for(candidate),
        "orientation": "bad_high" if candidate == B_CANDIDATE else "bad_low",
        "event_protocol_identity_sha256": protocol["identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "dynamic_source_contract_identity_sha256": source_contract["identity_sha256"],
        "dynamic_source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "trace_pool_identity_sha256": pool_manifest["identity_sha256"],
        "classes_ordered": list(classes),
        "calibration_seeds": list(range(1100, 1120)),
        "confirmation_seeds": list(range(1200, 1328)),
        "row_count": len(rows),
        "column_order": ["phase", "global_seed", "class_id", feature_for(candidate)],
        "candidate_products_physically_separate": True,
        "labels_reviews_consensus_or_endpoint_judgments_opened": False,
        "external_representation_or_distance_opened": False,
        "other_candidate_score_computed_or_emitted": False,
        "scores_file_sha256": __import__("hashlib").sha256(csv_text.encode("utf-8")).hexdigest(),
        "column_catalog_identity_sha256": catalog["identity_sha256"],
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    payloads = {
        "scores.csv": csv_text,
        "column_catalog.json": json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        "product_record.json": json.dumps(record, indent=2, sort_keys=True) + "\n",
        "extractor_source.py": Path(__file__).read_text(encoding="utf-8"),
    }
    publish_artifact(
        args.output,
        artifact_kind=f"dit_event_rich_{candidate}_single_score_product_v1",
        payloads=payloads,
        manifest_fields={
            "candidate": candidate,
            "product_record_identity_sha256": record["identity_sha256"],
            "anchor_plan_identity_sha256": plan["identity_sha256"],
            "dynamic_source_contract_identity_sha256": source_contract["identity_sha256"],
        },
    )
    print(
        json.dumps(
            {"status": "complete", "candidate": candidate, "rows": len(rows), "output": str(args.output)},
            sort_keys=True,
        )
    )


def run_self_test() -> None:
    # C agrees with a direct hand calculation and is invariant to float32 input copies.
    y, x = np.mgrid[:32, :32]
    base = (y + 2.0 * x).astype(np.float32)
    pred = np.stack([(1.0 + index / 100.0) * base for index in range(50)]).astype(np.float32)
    alpha = np.linspace(0.2, 0.8, 50, dtype=np.float64)
    observed = c3_low_jump(pred, alpha)
    work = pred.astype(np.float64)
    energy = np.mean(np.diff(work, axis=-2) ** 2, axis=(-2, -1)) + np.mean(
        np.diff(work, axis=-1) ** 2, axis=(-2, -1)
    )
    expected = float(max(0.0, np.max(np.diff(alpha * energy))))
    if observed != expected:
        raise AssertionError("C frozen formula changed")
    image = np.zeros((2, 3, 256, 256), dtype=np.float32)
    image[:, :, :, 128:] = 1.0
    b = local_blur_severity(image)
    if b.shape != (2,) or not np.isfinite(b).all() or b[0] != b[1]:
        raise AssertionError("B pixel formula is not deterministic")
    for candidate in (B_CANDIDATE, C_CANDIDATE):
        catalog = build_catalog(candidate)
        if catalog["contains_labels_reviews_consensus_or_endpoint_judgments"] is not False:
            raise AssertionError("catalog supervision boundary changed")
        row = {
            "phase": "calibration",
            "global_seed": 1100,
            "class_id": 1,
            feature_for(candidate): 0.25,
        }
        text = rows_to_csv([row], candidate)
        header = next(csv.reader(io.StringIO(text)))
        validate_score_columns(header, candidate)
        if any(token in header for token in ("label", "raw_consensus_label")):
            raise AssertionError("legacy placeholder label column reappeared")
    poison = {
        "phase": "confirmation",
        "global_seed": 1200,
        "class_id": 1,
        B_FEATURE: 0.0,
        "label": "unlabeled",
    }
    try:
        rows_to_csv([poison], B_CANDIDATE)
    except RuntimeError:
        pass
    else:
        raise AssertionError("placeholder label poison escaped product writer")
    print("self-test passed: exact B/C formulas and single-column no-label products")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    parser.add_argument("--candidate", choices=(B_CANDIDATE, C_CANDIDATE))
    parser.add_argument("--anchor-plan", type=Path)
    parser.add_argument("--trace-pool", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vae-snapshot", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-batch-size", type=int, default=9)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    missing = [name for name in ("candidate", "anchor_plan", "trace_pool", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    if args.decode_batch_size <= 0:
        raise SystemExit("--decode-batch-size must be positive")
    run_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
