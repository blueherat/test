#!/usr/bin/env python3
"""Aggregate frozen heldout potential results, preserving image pairing in time.

Reported errors and witnesses concern the real Gaussian bridge. They are not
an image quality result or a certificate for the full weighted Poisson solve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file, write_csv


def record(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "standard_error_across_fixed_images_and_classes": float(values.std(ddof=1)/np.sqrt(len(values))),
            "minimum": float(values.min()), "maximum": float(values.max()), "positive_fraction": float((values > 0).mean())}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "request.json").exists():
        raise FileExistsError("refusing to overwrite a prior analysis")
    output.mkdir(parents=True, exist_ok=True)
    paths = [path.resolve() for path in args.validation_dirs]
    if len(set(paths)) != len(paths):
        raise ValueError("repeated validation directory")
    requests = [json.loads((path / "request.json").read_text()) for path in paths]
    summaries = [json.loads((path / "summary.json").read_text()) for path in paths]
    reference = requests[0]
    shared = ("protocol", "mode", "bank", "config", "baseline_checkpoint", "potential_checkpoint",
              "time_grid", "time_probability", "time_weight_normalizer", "batch_size", "seeds", "precision")
    steps = {}
    for directory, request, summary in zip(paths, requests, summaries):
        if request["mode"] != "validate" or not summary.get("complete"):
            raise ValueError("incomplete/nonvalidation input")
        if summary["request"]["sha256"] != sha256_file(directory / "request.json"):
            raise ValueError("validation request hash mismatch")
        if any(request[key] != reference[key] for key in shared):
            raise ValueError("incompatible validation shards")
        if request["sources"].keys() != reference["sources"].keys():
            raise ValueError("inconsistent validation sources")
        if any(request["sources"][key]["sha256"] != reference["sources"][key]["sha256"] for key in reference["sources"]):
            raise ValueError("validation source hash differs")
        for k in summary["time_indices"]:
            if k in steps:
                raise ValueError("duplicate validation time")
            steps[k] = directory / f"step{k:03d}.npz"
    if sorted(steps) != list(range(100)):
        raise ValueError("full 100-step validation is required")
    bank_path = Path(reference["bank"]["path"])
    if sha256_file(bank_path) != reference["bank"]["sha256"]:
        raise ValueError("clean bank summary changed")
    bank = json.loads(bank_path.read_text())
    metadata_record = bank["banks"]["validation"]["metadata"]
    metadata_path = Path(metadata_record["path"])
    if sha256_file(metadata_path) != metadata_record["sha256"]:
        raise ValueError("heldout class metadata changed")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        expected_labels = metadata["labels"].copy()
    probabilities = np.asarray(reference["time_probability"], dtype=np.float64)
    grid = np.asarray(reference["time_grid"], dtype=np.float64)
    if (probabilities.shape != (100,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or abs(probabilities.sum()-1) > 1e-12
            or grid.shape != (101,) or not np.all(grid[:-1] > grid[1:]) or grid[-1] != 0):
        raise ValueError("invalid time grid or probabilities")
    weights = probabilities*reference["time_weight_normalizer"]
    if not np.allclose(weights, (grid[:-1]-grid[1:])/grid[:-1]**2, rtol=1e-12, atol=1e-12):
        raise ValueError("time weights do not implement the declared velocity objective")
    shutil.copy2(Path(__file__), output / "runner_source.py")
    atomic_json(output / "request.json", {
        "protocol": "raev2_observable_potential_validation_summary_v1",
        "runner": record(output / "runner_source.py"),
        "inputs": [{"request": record(path / "request.json"), "summary": record(path / "summary.json")} for path in paths],
        "step_artifacts": [record(steps[k]) for k in range(100)],
        "statistics": "pair all 100 times within each heldout image before computing across-image/class standard error",
        "boundary": "One image per fixed class; SEM is descriptive, not an exact class-conditional repeated-noise confidence interval.",
    })
    arrays = {key: [] for key in ("residual_mse", "correction_energy", "residual_dot_correction")}
    labels = None
    time_rows = []
    for k in range(100):
        with np.load(steps[k], allow_pickle=False) as values:
            if not np.array_equal(values["ids"], np.arange(1000)):
                raise ValueError("heldout image order changed")
            if labels is None:
                labels = values["labels"].copy()
                if not np.array_equal(labels, expected_labels):
                    raise ValueError("validation class order differs from frozen bank")
            elif not np.array_equal(labels, values["labels"]):
                raise ValueError("class order changed across time")
            for key in arrays:
                value = values[key].astype(np.float64)
                if value.shape != (1000,) or not np.isfinite(value).all():
                    raise ValueError("invalid validation values")
                arrays[key].append(value)
            gain = 2*arrays["residual_dot_correction"][-1]-arrays["correction_energy"][-1]
            if not np.array_equal(gain, values["paired_clean_mse_gain"]):
                raise ValueError("paired gain identity failed")
            witness = arrays["residual_dot_correction"][-1]-arrays["correction_energy"][-1]
            if not np.array_equal(witness, values["learned_potential_weak_residual_clean_units"]):
                raise ValueError("weak residual identity failed")
        time_rows.append({"step": k, "time": reference["time_grid"][k],
                          "probability": reference["time_probability"][k],
                          "paired_clean_mse_gain": float(gain.mean()),
                          "gain_standard_error_across_images_and_classes": float(gain.std(ddof=1)/np.sqrt(1000)),
                          "weak_residual_clean_units": float(witness.mean()),
                          "correction_energy": float(arrays["correction_energy"][-1].mean()),
                          "residual_dot_correction": float(arrays["residual_dot_correction"][-1].mean())})
    arrays = {key: np.stack(values) for key, values in arrays.items()}
    gain = 2*arrays["residual_dot_correction"]-arrays["correction_energy"]
    witness = arrays["residual_dot_correction"]-arrays["correction_energy"]
    weighted_gain = probabilities @ gain
    velocity_gain = weights @ gain
    weighted_baseline_mse = probabilities @ arrays["residual_mse"]
    remaining_velocity_witness = weights @ witness
    summary = {"complete": True, "samples": 1000, "times": 100,
               "paired_weighted_clean_mse_gain": describe(weighted_gain),
               "integrated_velocity_mse_gain_per_dimension": describe(velocity_gain),
               "integrated_remaining_weak_residual_for_phi_equals_Phi_over_t_per_dimension": describe(remaining_velocity_witness),
               "weighted_baseline_coupling_mse_per_dimension": float(weighted_baseline_mse.mean()),
               "relative_reduction_of_coupling_mse_proxy": float(weighted_gain.mean()/weighted_baseline_mse.mean()),
               "weighted_mean_gain_positive": bool(weighted_gain.mean() > 0),
               "positive_gain_time_count": int((gain.mean(1) > 0).sum()),
               "negative_mean_gain_probability_mass": float(probabilities[gain.mean(1) < 0].sum()),
               "stage2_sample_forwards": sum(item["stage2_sample_forwards"] for item in summaries),
               "stage2_forward_calls": sum(item["stage2_forward_calls"] for item in summaries),
               "potential_forward_input_gradient_calls": sum(item["potential_forward_input_gradient_calls"] for item in summaries),
               "sum_validation_phase_worker_wall_seconds_including_cpu_and_io": sum(item["elapsed_seconds"] for item in summaries),
               "sum_validation_process_wall_seconds": sum(item["total_wall_seconds_excluding_imports_and_final_summary_write"] for item in summaries),
               "validation_peak_gpu_allocated_bytes": max(item["peak_gpu_memory_allocated_bytes"] for item in summaries),
               "elapsed_summary_seconds": time.perf_counter()-started,
               "fid_performed": False, "image_sampling_performed": False,
               "boundary": "Positive bridge proxy gain does not guarantee rollout, full residual solution, decoder distribution, or FID improvement."}
    np.savez(output / "paired_by_image.npz", ids=np.arange(1000), labels=labels,
             weighted_clean_mse_gain=weighted_gain, integrated_velocity_mse_gain=velocity_gain,
             integrated_remaining_velocity_weak_residual=remaining_velocity_witness)
    write_csv(output / "by_time.csv", time_rows)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
