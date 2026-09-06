"""CPU class-mean interaction audit of frozen predicted-clean four-cell features.

This is a retrospective Inception-feature diagnostic. Decoding a head at an
intermediate state is not applying an ODE step or evaluating its causal suffix.
All three times, both complete 5K banks, and the original class splits are fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SEEDS = (20260801, 20260802)
TIMES = (.4, .2, .14)
CONDITIONS = ("full_on_full", "ig_on_full", "full_on_ig", "ig_on_ig")
RESPONSES = ("H", "C", "I")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    result = {"count": len(values), "finite_count": len(valid), "undefined_count": len(values)-len(valid)}
    if not len(valid):
        return result
    mean = float(valid.mean())
    se = float(valid.std(ddof=1)/np.sqrt(len(valid))) if len(valid) > 1 else 0.
    return {**result, "mean": mean, "median": float(np.median(valid)),
            "standard_error": se, "normal_95_diagnostic": [mean-1.96*se, mean+1.96*se],
            "fraction_negative": float(np.mean(valid < 0)),
            "q05": float(np.quantile(valid, .05)), "q95": float(np.quantile(valid, .95)),
            "min": float(valid.min()), "max": float(valid.max())}


def safe_ratio(numerator, denominator):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def geometry(fields):
    norms = {name: np.linalg.norm(value, axis=1) for name, value in fields.items()}
    result = {f"norm_{name}": value for name, value in norms.items()}
    for first, second in (("I", "H"), ("I", "C")):
        result[f"cos_{first}_{second}"] = safe_ratio(
            np.einsum("ij,ij->i", fields[first], fields[second]), norms[first]*norms[second])
        result[f"norm_{first}_over_{second}"] = safe_ratio(norms[first], norms[second])
    return result


def energy_decomposition(baseline, real, guided, responses):
    """Use only 2048-D dot products, never a 2048x2048 covariance or FID."""
    count = len(baseline)
    names = ("baseline", "real", *RESPONSES)
    arrays = np.stack((baseline, real, *(responses[name] for name in RESPONSES)))
    means = arrays.mean(axis=1)
    mean_gram = means @ means.T
    centered = arrays-means[:, None, :]
    covariance_trace_gram = np.einsum("ind,jnd->ij", centered, centered, optimize=True)/(count-1)
    second_gram = np.einsum("ind,jnd->ij", arrays, arrays, optimize=True)/count
    index = {name: i for i, name in enumerate(names)}
    moments = {}
    for name in names:
        i = index[name]
        moments[name] = {"mean_squared_norm": float(mean_gram[i, i]),
                         "covariance_trace_ddof1": float(covariance_trace_gram[i, i]),
                         "mean_per_class_squared_norm": float(second_gram[i, i])}
    pairs = {}
    for first, second in itertools.combinations(names, 2):
        i, j = index[first], index[second]
        pairs[f"{first}__{second}"] = {
            "mean_dot": float(mean_gram[i, j]),
            "covariance_cross_trace_ddof1": float(covariance_trace_gram[i, j]),
            "mean_per_class_dot": float(second_gram[i, j]),
        }
    total = sum(responses.values())
    mu_total = total.mean(axis=0)
    centered_total = total-mu_total
    total_terms = {
        "mean_squared_norm": float(mu_total @ mu_total),
        "covariance_trace_ddof1": float(np.square(centered_total).sum()/(count-1)),
        "mean_per_class_squared_norm": float(np.square(total).sum()/count),
        "mean_self_terms": {name: moments[name]["mean_squared_norm"] for name in RESPONSES},
        "mean_pair_cross_terms": {f"{a}__{b}": 2*pairs[f"{a}__{b}"]["mean_dot"] for a, b in itertools.combinations(RESPONSES, 2)},
        "covariance_self_terms": {name: moments[name]["covariance_trace_ddof1"] for name in RESPONSES},
        "covariance_pair_cross_terms": {f"{a}__{b}": 2*pairs[f"{a}__{b}"]["covariance_cross_trace_ddof1"] for a, b in itertools.combinations(RESPONSES, 2)},
    }
    error = means[0]-means[1]
    after_error = guided.mean(0)-means[1]
    error_terms = {
        "before": float(error @ error), "after": float(after_error @ after_error),
        "baseline_error_response_cross": {name: float(2*error@responses[name].mean(0)) for name in RESPONSES},
        "response_self": total_terms["mean_self_terms"],
        "response_pair_cross": total_terms["mean_pair_cross_terms"],
    }
    error_terms["actual_change"] = error_terms["after"]-error_terms["before"]
    error_terms["decomposition_residual"] = error_terms["actual_change"]-sum(
        sum(error_terms[k].values()) for k in ("baseline_error_response_cross", "response_self", "response_pair_cross"))
    guided_centered = guided-guided.mean(0)
    before_cov = moments["baseline"]["covariance_trace_ddof1"]
    after_cov = float(np.square(guided_centered).sum()/(count-1))
    real_cov = moments["real"]["covariance_trace_ddof1"]
    covariance_terms = {
        "before": before_cov, "after": after_cov, "original_real_reference": real_cov,
        "before_minus_real": before_cov-real_cov, "after_minus_real": after_cov-real_cov,
        "actual_change": after_cov-before_cov,
        "baseline_response_cross": {name: 2*pairs[f"baseline__{name}"]["covariance_cross_trace_ddof1"] for name in RESPONSES},
        "response_self": total_terms["covariance_self_terms"],
        "response_pair_cross": total_terms["covariance_pair_cross_terms"],
    }
    covariance_terms["decomposition_residual"] = covariance_terms["actual_change"]-sum(
        sum(covariance_terms[k].values()) for k in ("baseline_response_cross", "response_self", "response_pair_cross"))
    residuals = {
        "four_cell_reconstruction_maxabs": float(np.abs(baseline+total-guided).max()),
        "second_moment_identity_maxabs": float(np.abs(second_gram-mean_gram-(count-1)/count*covariance_trace_gram).max()),
        "total_mean_energy": total_terms["mean_squared_norm"]-sum(total_terms["mean_self_terms"].values())-sum(total_terms["mean_pair_cross_terms"].values()),
        "total_covariance_trace": total_terms["covariance_trace_ddof1"]-sum(total_terms["covariance_self_terms"].values())-sum(total_terms["covariance_pair_cross_terms"].values()),
        "squared_mean_error": error_terms["decomposition_residual"],
        "guided_covariance_trace": covariance_terms["decomposition_residual"],
    }
    if max(abs(x) for x in residuals.values()) > 1e-9:
        raise ValueError(f"energy identity failed: {residuals}")
    return {"classes": count, "moments": moments, "pair_terms": pairs,
            "total_response_energy": total_terms, "squared_mean_error_to_original_real": error_terms,
            "guided_covariance_trace": covariance_terms, "algebra_residuals": residuals}


def verify_real_rows(protocol, manifest):
    """Reproduce the original seed+31 selector using order-preserving Packed labels."""
    packed_root = Path(manifest["packed_data_path"]).resolve()
    packed_manifest = packed_root/"manifest.json"
    packed = json.loads(packed_manifest.read_text())
    parquet_root = Path(manifest["parquet_data_path"]).resolve()
    data_dir = parquet_root/"data" if (parquet_root/"data").is_dir() else parquet_root
    files = sorted(data_dir.glob("train-*.parquet"))
    if Path(packed["source_root"]).resolve() != data_dir or [p.name for p in files] != [s["source_file"] for s in packed["shards"]]:
        raise ValueError("Packed and parquet source ordering differs")
    all_labels = np.concatenate([np.load(packed_root/s["labels_file"]) for s in packed["shards"]])
    if len(all_labels) != packed["total_rows"]:
        raise ValueError("Packed label row count mismatch")
    order = np.argsort(all_labels, kind="stable")
    counts = np.bincount(all_labels)
    offsets = np.r_[0, np.cumsum(counts)]
    labels = protocol["labels"]
    expected_rows = np.empty(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        candidates = order[offsets[label]:offsets[label+1]].copy()
        np.random.default_rng(int(manifest["seed"])+31+104729*int(label)).shuffle(candidates)
        selected = np.flatnonzero(labels == label)
        expected_rows[selected] = candidates[:len(selected)]
    if not np.array_equal(expected_rows, protocol["real_source_rows"]):
        raise ValueError("original seed+31 source-row selector does not reproduce the saved protocol")
    if not np.array_equal(all_labels[expected_rows], labels):
        raise ValueError("original source labels mismatch")
    return {"packed_manifest": file_record(packed_manifest), "shards": len(packed["shards"]),
            "all_5000_source_rows_reproduced": True, "source_labels_match": True,
            "selection": "original stable-sort class rows; shuffle seed+31+104729*class, without replacement",
            "qualification": "Uses Packed labels in verified original sorted-parquet order; original encoded image bytes are not re-read or rehashed. Predicted-clean generation itself loads no real image: reference matching is by the shared class/ID/seed protocol, not paired reconstruction targets."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("/home/zhoushunyu/data/eqvae/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/predicted_clean_interaction_n5000x2_v1"))
    args = parser.parse_args()
    started = time.perf_counter()
    root, output = args.experiment_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing audit: {output}")
    request = {
        "protocol": "raev2_predicted_clean_class_interaction_v1", "seeds": SEEDS, "requested_times": TIMES,
        "samples_per_bank": 5000, "classes": 1000, "samples_per_class": 5,
        "training_classes": 800, "heldout_classes": 200,
        "feature": "existing original 2048-D Inception coordinates, float32 cache promoted to float64; no whitening, normalization, feature replacement or eigendecomposition",
        "aggregation": "arithmetic mean of all 5 samples within each class before every analysis; covariance is empirical covariance across class means (ddof=1), not full pooled image covariance",
        "definitions": {"X": "F|F", "H": "F|IG - F|F", "C": "IG|F - F|F", "I": "(IG|IG-F|IG)-(IG|F-F|F)", "T": "H+C+I = IG|IG-F|F"},
        "axes": "training mean(H), training mean(C), training mean(F|F-original_real), each normalized only to unit Euclidean norm; no axis is selected using heldout data",
        "zero_vectors": "undefined cosines/ratios retained as missing with explicit counts; zero training axis is an error, no epsilon or substitute direction",
        "intervals": "descriptive class-level mean +/- 1.96*sample_std/sqrt(classes); conditional on fitted train axes, not simultaneous or exact-population coverage",
        "causal_scope": "four cells are directly decoded predicted-clean heads observed by a read-only hook, not actual one-step updates followed by common suffix integration; I can include head/state response, decoder, clamp, uint8 conversion and Inception nonlinearities",
        "no_gpu": True, "no_sampling": True, "no_training": True, "no_new_fid": True,
        "source_records": [file_record(REPO/"experiments"/name) for name in (
            Path(__file__).name, "run_raev2_predicted_clean_audit.py", "run_raev2_decoded_distribution_audit.py",
            "run_raev2_scale_response_study.py", "run_raev2_distribution_auc.py", "build_imagenet_random_access.py")],
        "runs": [],
    }
    write_json(output/"request.json", request)
    (output/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    rows, projection_rows, results, axes_archive = [], [], [], {}
    for seed in SEEDS:
        seed_started = time.perf_counter()
        prediction_dir = root/"raev2_predicted_clean_audit"/f"n5000_seed{seed}_v1"
        real_dir = root/"raev2_ig_scale_response"/f"n5000_seed{seed}_scales7_v1"
        prediction_manifest = json.loads((prediction_dir/"manifest.json").read_text())
        real_manifest = json.loads((real_dir/"manifest.json").read_text())
        for manifest in (prediction_manifest, real_manifest):
            if manifest["seed"] != seed or manifest["samples"] != 5000 or manifest["world_size"] != 4:
                raise ValueError("unexpected bank identity or size")
        with np.load(real_dir/"sample_protocol.npz") as payload:
            protocol = {key: payload[key] for key in payload.files}
        if not np.array_equal(protocol["sample_ids"], np.arange(5000)) or not np.array_equal(protocol["labels"], np.arange(5000)%1000):
            raise ValueError("expected complete original ID/class protocol")
        expected_test = np.isin(protocol["labels"], np.random.default_rng(seed+17).permutation(1000)[:200])
        if not np.array_equal(protocol["test_mask"], expected_test):
            raise ValueError("original class split mismatch")
        class_test = expected_test[:1000]
        source_info = {"seed": seed, "prediction_manifest": file_record(prediction_dir/"manifest.json"),
                       "real_manifest": file_record(real_dir/"manifest.json"),
                       "real_protocol": file_record(real_dir/"sample_protocol.npz"),
                       "real_row_identity": verify_real_rows(protocol, real_manifest), "feature_sources": []}
        classes = {t: {name: np.zeros((1000, 2048), dtype=np.float64) for name in CONDITIONS} for t in TIMES}
        real_class = np.zeros((1000, 2048), dtype=np.float64)
        seen = np.zeros(5000, dtype=bool)
        for rank in range(4):
            pred_path = prediction_dir/f"predicted_clean_features_rank{rank:02d}.npz"
            real_path = real_dir/"inception"/f"source_rank{rank:02d}.npy"
            source_info["feature_sources"].extend((file_record(pred_path), file_record(real_path)))
            with np.load(pred_path) as bank:
                ids, labels, mask = bank["ids"], bank["labels"], bank["test_mask"]
                if not np.array_equal(ids, np.arange(rank, 5000, 4)):
                    raise ValueError("rank-global sample ID alignment differs")
                if not np.array_equal(labels, protocol["labels"][ids]) or not np.array_equal(mask, expected_test[ids]) or seen[ids].any():
                    raise ValueError("prediction/reference identity mismatch")
                seen[ids] = True
                real_features = np.load(real_path, mmap_mode="r")
                if real_features.shape != (1250, 2048) or real_features.dtype != np.float32 or not np.isfinite(real_features).all():
                    raise ValueError("invalid original-source features")
                np.add.at(real_class, labels, real_features)
                for t in TIMES:
                    suffix = f"t{t:.6f}".replace(".", "p")
                    for condition in CONDITIONS:
                        values = bank[f"feat_{condition}_{suffix}"]
                        if values.shape != (1250, 2048) or values.dtype != np.float32 or not np.isfinite(values).all():
                            raise ValueError("invalid predicted-clean features")
                        np.add.at(classes[t][condition], labels, values)
            print(f"seed={seed} loaded aligned rank {rank}/3", flush=True)
        if not seen.all():
            raise ValueError("missing global sample IDs")
        counts = np.bincount(protocol["labels"], minlength=1000)
        if not np.array_equal(counts, np.full(1000, 5)):
            raise ValueError("each class must contain all 5 samples")
        real_class /= counts[:, None]
        source_info["identity_checks"] = {"samples": 5000, "per_class_count": 5, "classes": 1000,
                                           "training_classes": int((~class_test).sum()), "heldout_classes": int(class_test.sum()),
                                           "all_rank_ids_labels_split_exact": True}
        request["runs"].append(source_info)
        write_json(output/"request.json", request)
        for t in TIMES:
            cells = {name: values/counts[:, None] for name, values in classes[t].items()}
            baseline, guided = cells["full_on_full"], cells["ig_on_ig"]
            fields = {"H": cells["full_on_ig"]-baseline, "C": cells["ig_on_full"]-baseline}
            fields["I"] = guided-cells["full_on_ig"]-fields["C"]
            fields["T"] = fields["H"]+fields["C"]+fields["I"]
            g = geometry(fields)
            actual = next(item["actual_time"] for item in prediction_manifest["matched_times"] if item["requested_time"] == t)
            result = {"seed": seed, "requested_time": t, "actual_time": actual, "geometry": {}, "energy": {}, "axes": {}}
            for split, select in (("train", ~class_test), ("heldout", class_test), ("all", np.ones(1000, dtype=bool))):
                result["geometry"][split] = {key: summary(value[select]) for key, value in g.items()}
                result["energy"][split] = energy_decomposition(baseline[select], real_class[select], guided[select], {name: fields[name][select] for name in RESPONSES})
            for label in range(1000):
                rows.append({"seed": seed, "requested_time": t, "actual_time": actual, "class": label,
                             "test_mask": bool(class_test[label]), "samples": 5,
                             **{key: float(value[label]) if np.isfinite(value[label]) else "" for key, value in g.items()}})
            axis_vectors = {"train_H": fields["H"][~class_test].mean(0),
                            "train_C": fields["C"][~class_test].mean(0),
                            "train_original_real_error": (baseline[~class_test]-real_class[~class_test]).mean(0)}
            for name, vector in axis_vectors.items():
                norm = float(np.linalg.norm(vector))
                if norm == 0 or not np.isfinite(norm):
                    raise ValueError("zero/nonfinite training axis")
                axis = vector/norm
                key = f"seed{seed}_t{t:.6f}_{name}".replace(".", "p")
                axes_archive[key] = axis
                projections = {field: value[class_test]@axis for field, value in fields.items()}
                result["axes"][name] = {"fit_classes": 800, "unscaled_axis_norm": norm,
                                       "axis_npz_key": key, "heldout": {field: summary(value) for field, value in projections.items()}}
                for j, label in enumerate(np.flatnonzero(class_test)):
                    projection_rows.append({"seed": seed, "requested_time": t, "actual_time": actual, "class": int(label), "axis": name,
                                            **{field: float(value[j]) for field, value in projections.items()}})
            results.append(result)
            print(json.dumps({"seed": seed, "time": actual,
                              "heldout_cos_I_H": result["geometry"]["heldout"]["cos_I_H"]["mean"],
                              "heldout_cos_I_C": result["geometry"]["heldout"]["cos_I_C"]["mean"],
                              "I_on_train_H": result["axes"]["train_H"]["heldout"]["I"],
                              "I_on_train_original_real_error": result["axes"]["train_original_real_error"]["heldout"]["I"]}), flush=True)
        print(f"seed={seed} completed in {time.perf_counter()-seed_started:.3f} CPU seconds", flush=True)
    for name, values in (("per_class_geometry.csv", rows), ("heldout_projections.csv", projection_rows)):
        with (output/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    np.savez(output/"training_axes.npz", **axes_archive)
    write_json(output/"summary.json", {
        "protocol": request["protocol"], "status": "complete", "runs": results,
        "sources": {"request": file_record(output/"request.json"), "runner": file_record(output/"runner_source.py")},
        "artifacts": [file_record(output/name) for name in ("per_class_geometry.csv", "heldout_projections.csv", "training_axes.npz")],
        "cost": {"device": "cpu", "elapsed_seconds": time.perf_counter()-started, "model_queries": 0, "new_samples": 0},
        "limitations": [
            "Retrospective reuse of two existing banks and their previous class splits; not an independent method confirmation.",
            "Class means contain only five observations, and heldout contains 200 classes: covariance and mean-error estimates include finite-sample variability.",
            "Covariance trace has no covariance-shape information and is not a covariance distance or FID.",
            "No actual solver step or shared suffix is applied to counterfactual heads; causal sampling-quality conclusions do not follow.",
            "Interactions are measured after decoder, clamp, uint8 conversion and nonlinear Inception. They cannot be called latent vector-field interactions.",
            "Negative error-axis projection and reductions of squared feature-mean error do not imply improved complete generated-image distributions.",
            "No feature selection, bandwidth, whitening, direction selection on heldout data, sampling or training is performed.",
        ],
    })
    print(f"complete: {output}", flush=True)


if __name__ == "__main__":
    main()
