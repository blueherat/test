#!/usr/bin/env python3
"""Audit saved extension features using independent FP64 symmetric Bures algebra.

CPU only: no evaluator import, model load, feature extraction, or sample mutation.
All numerical comparisons (including failures) are retained in the JSON report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

# Bound this independent audit's BLAS concurrency before importing NumPy/Torch.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np
import scipy
from scipy.linalg import eigh
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz")
REFERENCE_SHA = "925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac"
EVALUATOR = Path("/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals")
EVALUATOR_COMMIT = "19dfb4c2705333eb8b97e454fb354d47d1fe135b"
EXPECTED_ARMS = {"legacy": ["official100", "global_proximal100", "energy100"],
                 "native_global": ["official100", "global100"],
                 "reflection": ["official100", "reflection100", "official201"]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": digest(p), "size_bytes": p.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def check(report, name, passed, **details):
    required = details.pop("required", True)
    report["checks"].append({"name": name, "passed": bool(passed), "required": required, **details})
    return bool(passed)


def bound_file(report, record, name):
    observed = artifact(record["path"])
    ok = observed["sha256"] == record["sha256"]
    if "size_bytes" in record:
        ok = ok and observed["size_bytes"] == record["size_bytes"]
    check(report, name, ok, expected=record, observed=observed)
    if not ok:
        raise ValueError(f"artifact identity changed: {name}")
    return Path(observed["path"])


def spectrum_record(values):
    neg = values[values < 0]
    return {"dimension": len(values), "minimum": float(values.min()),
            "maximum": float(values.max()), "negative_count": len(neg),
            "negative_mass": float(-neg.sum()),
            "clipping": "negative eigenvalues only are replaced by zero; no positive-eigenvalue threshold"}


def bures(features, real_mean, real_cov, real_root):
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0, dtype=np.float64)
    xc = x - mean
    trace_sample = float(np.einsum("ij,ij->", xc, xc) / (len(x) - 1))
    if len(x) < x.shape[1]:
        # Same nonzero spectrum as C_real^(1/2) C_sample C_real^(1/2).
        # This avoids manufacturing D-N extra near-zero eigenvalues for old1K.
        m = (xc @ real_cov @ xc.T) / (len(x) - 1)
        formula = "symmetric rank-N Gram: centered_X @ C_real @ centered_X.T / (N-1)"
    else:
        covariance = (xc.T @ xc) / (len(x) - 1)
        m = real_root @ covariance @ real_root
        formula = "symmetric D-by-D Bures: sqrt(C_real) @ C_sample @ sqrt(C_real)"
    asymmetry = float(np.max(np.abs(m - m.T)))
    m = (m + m.T) * 0.5
    values = eigh(m, eigvals_only=True, check_finite=False, driver="evr")
    affinity = float(np.sqrt(np.maximum(values, 0)).sum())
    mean_term = float(np.dot(mean - real_mean, mean - real_mean))
    covariance_term = trace_sample + float(np.trace(real_cov)) - 2 * affinity
    return {"fid": mean_term + covariance_term, "mean_term": mean_term,
            "covariance_term": covariance_term, "sample_covariance_trace": trace_sample,
            "reference_covariance_trace": float(np.trace(real_cov)),
            "cross_root_trace": affinity, "spectrum": spectrum_record(values),
            "pre_symmetrization_max_asymmetry": asymmetry,
            "formula": formula, "sample_covariance_denominator": len(x) - 1}


def metrics_for_old(old_samples, sample_sha):
    matches = []
    for path in sorted(old_samples.parent.parent.glob("*.json")):
        try:
            obj = read(path)
        except (OSError, ValueError):
            continue
        if not isinstance(obj, list):
            continue
        for row in obj:
            if isinstance(row, dict) and "fid" in row and row.get("sample_sha256") == sample_sha:
                matches.append({"artifact": artifact(path), "row": row})
    return matches


def array_comparison(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"shape_equal": a.shape == b.shape, "bitwise_equal": bool(np.array_equal(a, b)),
            "max_abs_difference": float(np.max(np.abs(delta))),
            "rms_difference": float(np.sqrt(np.mean(delta * delta))),
            "close_at_atol_1e_4_rtol_1e_5": bool(np.allclose(a, b, atol=1e-4, rtol=1e-5))}


def audit(evaluation_root, output):
    started = time.perf_counter()
    cpu_started = time.process_time()
    report = {"protocol": "raev2_extension_independent_bures_audit_v1", "complete": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "rows": [],
              "errors": [], "source": artifact(__file__), "gpu_model_calls": 0,
              "feature_extraction_calls": 0, "independent_feature_extraction": False,
              "new4k_and_pooled5k_absolute_tolerance": 1e-6,
              "old1k_historical_absolute_tolerance": 1e-3,
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "torch_version": torch.__version__, "blas_threads": 4,
              "limits": ["This verifies arithmetic and stored provenance, not an independent Inception extractor.",
                         "Feature row order follows the checked official sequential extraction contract; old1K has an additional row-wise historical-feature comparison.",
                         "Old1K is rank deficient; historical scipy sqrtm and the independent rank-N PSD formula may differ numerically.",
                         "Pooled5K includes the screened old1K; new4K is separate confirmation data.",
                         "No population FID, significance, cost-match, or 5-percent improvement claim is made."]}
    try:
        ref_path = bound_file(report, {"path": str(REFERENCE), "sha256": REFERENCE_SHA}, "reference_identity")
        with np.load(ref_path, allow_pickle=False) as f:
            mr, sr = f["mu"].astype(np.float64), f["sigma"].astype(np.float64)
        check(report, "reference_shape_and_finite", mr.shape == (2048,) and sr.shape == (2048, 2048)
              and np.isfinite(mr).all() and np.isfinite(sr).all())
        sr = (sr + sr.T) * 0.5
        values, vectors = eigh(sr, check_finite=False, driver="evr")
        report["reference_spectrum"] = spectrum_record(values)
        check(report, "reference_psd", values.min() >= -1e-10, minimum=float(values.min()))
        real_root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
        api_path = EVALUATOR / "fd_evaluator/fd_evaluator/api.py"
        commit = subprocess.check_output(["git", "-C", str(EVALUATOR), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(EVALUATOR), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        check(report, "evaluator_committed_identity", commit == EVALUATOR_COMMIT and not dirty,
              commit=commit, tracked_status=dirty)
        report["official_order_contract_source"] = artifact(api_path)
        api = api_path.read_text()
        check(report, "official_sequential_order_contract", all(s in api for s in (
            'key = "arr_0" if "arr_0" in data', 'iterator = _batched(arr, batch_size, total=True)',
            'feats = torch.cat(feats_list, dim=0)', 'yield arr[i : i + bs]')))
        tail_controls = {}
        for family, expected_arms in EXPECTED_ARMS.items():
            base = evaluation_root / f"{family}_merged"
            execution_path = evaluation_root / f"{family}_evaluation_execution.json"
            execution = read(execution_path)
            check(report, f"{family}:evaluation_complete", execution.get("complete") is True
                  and all(j.get("exit_code") == 0 for j in execution["jobs"]), artifact=artifact(execution_path))
            merged = read(base / "summary.json")
            request = read(base / "request.json")
            official = read(base / "official_evaluation.json")
            subsets = read(base / "feature_subsets/summary.json")
            subset_request = read(base / "feature_subsets/request.json")
            check(report, f"{family}:merge_and_subsets_complete", merged.get("complete") is True and subsets.get("complete") is True)
            check(report, f"{family}:arm_set", set(merged["arms"]) == set(expected_arms), observed=list(merged["arms"]))
            check(report, f"{family}:official_branch_set", {x["branch"] for x in official} == {f"{family}_{a}_pooled5k" for a in expected_arms}
                  and len(official) == len(expected_arms))
            for key in ("merge_summary", "official_evaluation", "reference", "official_frechet_source"):
                bound_file(report, subset_request[key], f"{family}:subset_request:{key}")
            bound_file(report, request["plan"], f"{family}:merge_plan")
            check(report, f"{family}:plan_contents_equal", request["plan_contents"] == read(request["plan"]["path"]))
            feature_records = {Path(x["path"]).resolve(): x for x in subsets["features"]}
            by_branch = {x["branch"]: x for x in official}
            by_subset = {(x["arm"], x["subset"]): x for x in subsets["results"]}
            for arm in expected_arms:
                label = f"{family}:{arm}"
                outputs = merged["arms"][arm]
                pooled_path = bound_file(report, outputs["pooled5k"]["samples"], f"{label}:pooled_archive")
                new_path = bound_file(report, outputs["new4k"]["samples"], f"{label}:new4k_archive")
                for subset in ("pooled5k", "new4k"):
                    bound_file(report, outputs[subset]["summary"], f"{label}:{subset}:summary")
                pool_summary = read(outputs["pooled5k"]["summary"]["path"])
                with np.load(pooled_path, allow_pickle=False) as z:
                    pixels = z["arr_0"]
                    check(report, f"{label}:pooled_image_shape", pixels.dtype == np.uint8 and pixels.shape == (5000, 256, 256, 3))
                    check(report, f"{label}:pooled_ids_labels_blocks", np.array_equal(z["ids"], np.arange(5000))
                          and np.array_equal(z["labels"], np.tile(np.arange(1000), 5))
                          and np.array_equal(z["local_ids"], np.tile(np.arange(1000), 5))
                          and np.array_equal(z["block_index"], np.repeat(np.arange(5), 1000)))
                    check(report, f"{label}:pooled_raw_pixel_hash", raw_digest(pixels) == pool_summary["pixels_sha256"])
                with np.load(new_path, allow_pickle=False) as z:
                    check(report, f"{label}:new4k_is_exact_pooled_tail", np.array_equal(z["arr_0"], pixels[1000:]))
                    check(report, f"{label}:new4k_ids_labels_blocks", np.array_equal(z["ids"], np.arange(4000))
                          and np.array_equal(z["labels"], np.tile(np.arange(1000), 4))
                          and np.array_equal(z["local_ids"], np.tile(np.arange(1000), 4))
                          and np.array_equal(z["block_index"], np.repeat(np.arange(1, 5), 1000)))
                old_source = request["plan_contents"]["blocks"][0]["arms"][arm]
                for index, block in enumerate(request["plan_contents"]["blocks"]):
                    source_record = block["arms"][arm]["samples"]
                    source = bound_file(report, source_record, f"{label}:source_block_{index}")
                    with np.load(source, allow_pickle=False) as z:
                        check(report, f"{label}:block_{index}_pixels_exact", np.array_equal(z["arr_0"], pixels[index*1000:(index+1)*1000]),
                              block_id=block["id"], seed=block["seed"])
                del pixels
                official_row = by_branch[f"{family}_{arm}_pooled5k"]
                check(report, f"{label}:official_samples_binding", Path(official_row["sample_path"]).resolve() == pooled_path
                      and official_row["sample_sha256"] == outputs["pooled5k"]["samples"]["sha256"]
                      and official_row["evaluator_commit"] == EVALUATOR_COMMIT and official_row["fid_reference"] == "imagenet_256_fid_stats")
                feature_path = (base / "official_feature_cache" / f"{official_row['branch']}-{official_row['sample_sha256'][:16]}-inception.features.pt").resolve()
                bound_file(report, feature_records[feature_path], f"{label}:official_feature_hash")
                tensor = torch.load(feature_path, map_location="cpu", weights_only=True)
                check(report, f"{label}:feature_shape_finite", isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (5000, 2048)
                      and bool(torch.isfinite(tensor).all()), observed_shape=list(tensor.shape), dtype=str(tensor.dtype))
                x = tensor.numpy()
                for subset, start, stop in (("pooled5k", 0, 5000), ("new4k", 1000, 5000), ("old1k", 0, 1000)):
                    value = bures(x[start:stop], mr, sr, real_root)
                    row = {"family": family, "arm": arm, "subset": subset, "rows": [start, stop],
                           "feature": feature_records[feature_path], **value}
                    check(report, f"{label}:{subset}:numerically_valid", np.isfinite(value["fid"]) and value["fid"] >= 0
                          and value["spectrum"]["minimum"] >= -1e-8, fid=value["fid"], spectrum=value["spectrum"])
                    if subset != "old1k":
                        target = by_subset[(arm, subset)]
                        error = value["fid"] - target["fid"]
                        row.update({"feature_subsets_fid": target["fid"], "difference_from_feature_subsets": error})
                        check(report, f"{label}:{subset}:fid_matches", abs(error) <= 1e-6, difference=error, tolerance=1e-6)
                        check(report, f"{label}:{subset}:row_range_binding", target["feature_row_start"] == start
                              and target["feature_row_stop"] == stop and target["samples"] == stop-start
                              and target["sample_sha256"] == outputs[subset]["samples"]["sha256"])
                        if subset == "pooled5k":
                            err_official = value["fid"] - official_row["fid"]
                            row.update({"official_fid": official_row["fid"], "difference_from_official": err_official})
                            check(report, f"{label}:official_fid_matches", abs(err_official) <= 1e-6, difference=err_official, tolerance=1e-6)
                    else:
                        old_samples = Path(old_source["samples"]["path"]).resolve()
                        old_sha = old_source["samples"]["sha256"]
                        historical = metrics_for_old(old_samples, old_sha)
                        row["historical_metrics"] = historical
                        check(report, f"{label}:historical_metrics_found", len(historical) >= 1, count=len(historical))
                        row["differences_from_historical"] = [value["fid"] - m["row"]["fid"] for m in historical]
                        check(report, f"{label}:historical_fid_close", bool(historical) and all(abs(e) <= 1e-3 for e in row["differences_from_historical"]),
                              differences=row["differences_from_historical"], tolerance=1e-3, required=False)
                        candidates = list((old_samples.parent.parent / "official_feature_cache").glob(f"*-{old_sha[:16]}-inception.features.pt"))
                        check(report, f"{label}:historical_feature_unique", len(candidates) == 1, paths=[str(p) for p in candidates])
                        if len(candidates) == 1:
                            old_path = candidates[0]
                            old_x = torch.load(old_path, map_location="cpu", weights_only=True).numpy()
                            compare = array_comparison(x[:1000], old_x)
                            row["historical_feature_comparison"] = {"artifact": artifact(old_path), **compare}
                            check(report, f"{label}:old1k_feature_value_diagnostic", compare["close_at_atol_1e_4_rtol_1e_5"], required=False, **compare)
                            old_value = bures(old_x, mr, sr, real_root)
                            row["historical_features_independent_fid"] = old_value
                            row["prefix_vs_historical_features_fid_difference"] = value["fid"] - old_value["fid"]
                    report["rows"].append(row)
                    print(json.dumps({"family": family, "arm": arm, "subset": subset, "fid": value["fid"],
                                      "difference": row.get("difference_from_feature_subsets", row.get("differences_from_historical"))}), flush=True)
                if arm == "official100" and family in ("native_global", "reflection"):
                    tail_controls[family] = x[1000:].copy()
        compare = array_comparison(tail_controls["native_global"], tail_controls["reflection"])
        check(report, "shared_native_official_new4k_feature_rows", compare["close_at_atol_1e_4_rtol_1e_5"], **compare)
        check(report, "all_24_subset_results_present", len(report["rows"]) == 24, observed=len(report["rows"]))
        report["complete"] = True
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    report["wall_seconds"] = time.perf_counter() - started
    report["cpu_seconds"] = time.process_time() - cpu_started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_checks_passed"] = report["complete"] and all(c["passed"] for c in report["checks"])
    report["failed_checks"] = [c for c in report["checks"] if not c["passed"]]
    report["failed_required_checks"] = [c for c in report["failed_checks"] if c.get("required", True)]
    report["all_required_checks_passed"] = report["complete"] and not report["failed_required_checks"]
    errors = [abs(r["difference_from_feature_subsets"]) for r in report["rows"] if "difference_from_feature_subsets" in r]
    report["max_5k_4k_absolute_fid_error"] = max(errors) if errors else None
    historical_errors = [abs(e) for r in report["rows"] for e in r.get("differences_from_historical", [])]
    report["max_old1k_historical_absolute_fid_error"] = max(historical_errors) if historical_errors else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("complete", "all_checks_passed", "failed_checks", "errors", "wall_seconds",
                                           "max_5k_4k_absolute_fid_error", "max_old1k_historical_absolute_fid_error")}), flush=True)
    return report["all_required_checks_passed"]


def revise_without_recomputing(previous_path, previous_source, output):
    """Fix audit bookkeeping only; rebind every checked artifact, reuse all FIDs.

The untouched v1 JSON and exact v1 source snapshot retain the initial failed
checks. New4K's IDs are local to its own archive (0..3999), while block_index
keeps original cohort indices 1..4. Historical floating point comparisons
remain visible diagnostics; they are not equality requirements for new FIDs.
"""
    started, cpu_started = time.perf_counter(), time.process_time()
    previous = read(previous_path)
    snapshot = artifact(previous_source)
    if snapshot["sha256"] != previous["source"]["sha256"]:
        raise ValueError("Previous source snapshot does not bind the initial review")
    if not previous["complete"] or previous["errors"] or len(previous["rows"]) != 24:
        raise ValueError("Only a complete numerical review can be reused")
    report = dict(previous)
    report.update({"protocol": "raev2_extension_independent_bures_audit_v2",
                   "source": artifact(__file__), "checks": [], "errors": [], "complete": False,
                   "started_utc": datetime.now(timezone.utc).isoformat(),
                   "previous_review": artifact(previous_path), "previous_source_snapshot": snapshot,
                   "revision": {"changes": ["Correct new4K local IDs from the reviewer's incorrect 1000..4999 expectation to the frozen merger's 0..3999 definition.",
                                            "Keep all historical feature/FID discrepancies as diagnostics, separate from hard new4K/5K numerical requirements."],
                                "samples_or_features_changed": False,
                                "eigendecompositions_repeated": 0,
                                "reused_subset_fids": 24, "reused_historical_feature_fids": 8,
                                "previous_wall_seconds": previous["wall_seconds"],
                                "previous_cpu_seconds": previous["cpu_seconds"],
                                "original_failed_check_count": len(previous["failed_checks"])}})
    try:
        # Rebind all original checked archives, feature tensors, metadata,
        # reference, and source files before reusing a single numerical value.
        records = {}
        def collect(obj):
            if isinstance(obj, dict):
                if {"path", "sha256"} <= obj.keys():
                    p = str(Path(obj["path"]).resolve())
                    # The live reviewer source was intentionally revised; its
                    # previous exact bytes are already bound by the snapshot.
                    if p != str(Path(previous["source"]["path"]).resolve()):
                        records[(p, obj["sha256"])] = obj
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, list):
                for v in obj:
                    collect(v)
        collect(previous)
        for index, record in enumerate(records.values()):
            bound_file(report, record, f"revision_artifact_recheck:{index}")
        report["revision"]["rechecked_distinct_artifacts"] = len(records)
        evaluation_root = Path(previous_path).resolve().parent
        for original in previous["checks"]:
            c = dict(original)
            c["required"] = True
            name = c["name"]
            if name.endswith(":new4k_ids_labels_blocks"):
                family, arm, _ = name.split(":")
                path = evaluation_root / f"{family}_merged" / arm / "new4k/samples.npz"
                with np.load(path, allow_pickle=False) as z:
                    comparisons = {"local_ids_0_to_3999": bool(np.array_equal(z["ids"], np.arange(4000))),
                                   "labels_each_1000_classes": bool(np.array_equal(z["labels"], np.tile(np.arange(1000), 4))),
                                   "local_ids_each_cohort": bool(np.array_equal(z["local_ids"], np.tile(np.arange(1000), 4))),
                                   "original_block_indices_1_to_4": bool(np.array_equal(z["block_index"], np.repeat(np.arange(1, 5), 1000)))}
                    observed = {k: {"shape": list(z[k].shape), "first": int(z[k][0]), "last": int(z[k][-1])}
                                for k in ("ids", "labels", "local_ids", "block_index")}
                c.update({"passed": all(comparisons.values()), "comparisons": comparisons,
                          "observed": observed, "superseded_initial_check": original})
            elif name.endswith(":old1k_feature_row_order"):
                c["name"] = name.replace(":old1k_feature_row_order", ":old1k_feature_value_diagnostic")
                c["required"] = False
                c["superseded_initial_check_name"] = name
                c["interpretation"] = "Stored old and pooled-prefix features are not numerically equal at this tight threshold. Pixel identity is checked independently. This diagnostic alone does not identify the cause of forward differences or establish a row permutation."
            elif name.endswith(":historical_fid_close"):
                c["required"] = False
            report["checks"].append(c)
        report["limits"] = [*previous["limits"],
                            "Historical official 1K FIDs remain authoritative historical reports and are not replaced by this independent formula.",
                            "Observed historical feature-forward and FID differences are recorded separately; no GPU re-extraction or unique causal explanation was attempted."]
        check(report, "revision_reused_all_numerical_rows_unchanged", report["rows"] == previous["rows"])
        report["complete"] = True
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["wall_seconds"] = time.perf_counter() - started
    report["cpu_seconds"] = time.process_time() - cpu_started
    report["all_checks_passed"] = report["complete"] and all(c["passed"] for c in report["checks"])
    report["failed_checks"] = [c for c in report["checks"] if not c["passed"]]
    report["failed_required_checks"] = [c for c in report["failed_checks"] if c["required"]]
    report["all_required_checks_passed"] = report["complete"] and not report["failed_required_checks"]
    report["diagnostic_comparisons_outside_threshold"] = [c for c in report["failed_checks"] if not c["required"]]
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"complete": report["complete"], "all_required_checks_passed": report["all_required_checks_passed"],
                      "failed_required_checks": report["failed_required_checks"],
                      "diagnostics_outside_threshold": len(report["diagnostic_comparisons_outside_threshold"]),
                      "errors": report["errors"], "wall_seconds": report["wall_seconds"]}), flush=True)
    return report["all_required_checks_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-previous-review", type=Path)
    parser.add_argument("--previous-source-snapshot", type=Path)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    output = args.output or root / "final_fid_independent_review.json"
    if output.exists():
        raise FileExistsError(f"No overwrite: {output}")
    if args.reuse_previous_review:
        if args.previous_source_snapshot is None:
            parser.error("--previous-source-snapshot is required for numerical reuse")
        ok = revise_without_recomputing(args.reuse_previous_review, args.previous_source_snapshot, output)
    else:
        ok = audit(root, output)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
