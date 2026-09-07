#!/usr/bin/env python3
"""Compute new4K FID from saved official pooled5K features, entirely on CPU.

First reproduce each official 5K FID with its evaluator's own accumulator and
Frechet function. Then use rows 1000:5000, exactly the four appended blocks.
This is pooled feature evaluation, not an average of per-block FIDs. It does
not re-extract features and therefore is not an independent extractor audit.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.merge_raev2_1k_extension import artifact, checked_file, require, write_json  # noqa: E402

EVALUATOR_ROOT = Path("/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals")
EVALUATOR_COMMIT = "19dfb4c2705333eb8b97e454fb354d47d1fe135b"
REFERENCE = Path("/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz")
REFERENCE_SHA = "925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac"


def load_official_frechet(evaluator_root):
    evaluator_root = Path(evaluator_root).resolve()
    commit = subprocess.run(["git", "-C", str(evaluator_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    require(commit == EVALUATOR_COMMIT, "official evaluator commit changed")
    dirty = subprocess.run(["git", "-C", str(evaluator_root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, "official evaluator tracked sources are dirty")
    source = evaluator_root / "fd_evaluator/fd_evaluator/frechet.py"
    spec = importlib.util.spec_from_file_location("raev2_official_cpu_frechet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, artifact(source)


def fid_from_features(features, mean, covariance, official):
    accumulator = official.FrechetAccumulator(features.shape[1])
    accumulator.update(features)
    mu, sigma = accumulator.finalize()
    return official.frechet_distance(mu, sigma, mean, covariance)


def evaluate(merge_summary_path, evaluation_path, feature_cache, output_dir, *, evaluator_root=EVALUATOR_ROOT,
             reference_path=REFERENCE):
    started, cpu_started = time.perf_counter(), time.process_time()
    merge_summary_path, evaluation_path = Path(merge_summary_path).resolve(), Path(evaluation_path).resolve()
    feature_cache, output_dir = Path(feature_cache).resolve(), Path(output_dir).resolve()
    merged = json.loads(merge_summary_path.read_text())
    require(merged.get("protocol") == "raev2_paired_1k_extension_merge_v1" and merged.get("complete") is True,
            "all pooled blocks must be merged successfully before evaluation")
    require(not output_dir.exists(), "output exists; no overwrite")
    reference_record = {"path": str(reference_path), "sha256": REFERENCE_SHA}
    checked_file(reference_record)
    with np.load(reference_path, allow_pickle=False) as ref:
        mean, covariance = ref["mu"], ref["sigma"]
    require(mean.shape == (2048,) and covariance.shape == (2048, 2048), "wrong FID reference shape")
    official, official_source = load_official_frechet(evaluator_root)
    official_rows = json.loads(evaluation_path.read_text())
    require(isinstance(official_rows, list), "official evaluation must be the JSON list")
    require(len({row["branch"] for row in official_rows}) == len(official_rows), "duplicate official result branch")
    by_name = {row["branch"]: row for row in official_rows}
    output_dir.mkdir(parents=True)
    write_json(output_dir / "request.json", {"protocol": "raev2_extension_official_feature_subsets_v1",
               "merge_summary": artifact(merge_summary_path), "official_evaluation": artifact(evaluation_path),
               "feature_cache": str(feature_cache), "reference": artifact(reference_path),
               "official_frechet_source": official_source, "source": artifact(__file__),
               "subsets": {"pooled5k": [0, 5000], "new4k": [1000, 5000]},
               "pooled5k_reproduction_absolute_tolerance": 1e-6,
               "independent_feature_extraction": False, "inception_score_for_new4k": False})
    rows, identities = [], []
    try:
        for arm, outputs in merged["arms"].items():
            name = f"{merged['family']}_{arm}_pooled5k"
            require(name in by_name, f"missing completed official result: {name}")
            row = by_name[name]
            for subset in ("pooled5k", "new4k"):
                checked_file(outputs[subset]["samples"])
                checked_file(outputs[subset]["summary"])
            archive = outputs["pooled5k"]["samples"]
            require(Path(row["sample_path"]).resolve() == Path(archive["path"]).resolve()
                    and row["sample_sha256"] == archive["sha256"], "official result bound to different samples")
            require(row["evaluator_commit"] == EVALUATOR_COMMIT
                    and Path(row["evaluator_root"]).resolve() == Path(evaluator_root).resolve()
                    and row["fid_reference"] == "imagenet_256_fid_stats", "official evaluation identity mismatch")
            feature_path = feature_cache / f"{name}-{archive['sha256'][:16]}-inception.features.pt"
            features = torch.load(feature_path, map_location="cpu", weights_only=True)
            require(isinstance(features, torch.Tensor) and features.shape == (5000, 2048)
                    and torch.isfinite(features).all().item(), "invalid official feature tensor")
            identities.append(artifact(feature_path))
            for subset, offset in (("pooled5k", 0), ("new4k", 1000)):
                fid = fid_from_features(features[offset:], mean, covariance, official)
                require(np.isfinite(fid) and fid >= 0, "invalid recomputed FID")
                if offset == 0:
                    require(abs(fid - float(row["fid"])) <= 1e-6, "official pooled5K FID not reproduced")
                rows.append({"family": merged["family"], "arm": arm, "subset": subset,
                             "samples": 5000-offset, "fid": fid,
                             "official_pooled5k_fid": row["fid"] if offset == 0 else None,
                             "sample_sha256": outputs[subset]["samples"]["sha256"],
                             "feature_row_start": offset, "feature_row_stop": 5000})
        with (output_dir / "fid.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output_dir / "fid.json", rows)
        summary = {"protocol": "raev2_extension_official_feature_subsets_v1", "complete": True,
                   "results": rows, "features": identities, "gpu_model_calls": 0,
                   "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                   "quality_claim": "new4K is the preselected-method follow-up; pooled5K retains screened observations; neither alone is a population FID guarantee",
                   "independent_feature_extraction": False, "cost_match_claim": False}
        write_json(output_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        write_json(output_dir / "failure.json", {"complete": False, "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-summary", type=Path, required=True)
    parser.add_argument("--official-evaluation", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, default=EVALUATOR_ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.merge_summary, args.official_evaluation, args.feature_cache_dir,
                             args.output_dir, evaluator_root=args.evaluator_root, reference_path=args.reference), indent=2))


if __name__ == "__main__":
    main()
