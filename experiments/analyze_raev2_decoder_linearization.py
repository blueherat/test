"""Separate measured decoder linear/nonlinear response on class-heldout images.

This is a fixed-endpoint diagnostic, not a sampler, FID evaluator, or a
certificate that decoder curvature can improve guidance. The quality axis and
diagonal scale are fitted only on the existing 800 training classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    se = values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "standard_error": float(se),
            "normal_95_diagnostic": [float(values.mean()-1.96*se), float(values.mean()+1.96*se)],
            "fraction_negative": float(np.mean(values < 0)),
            "range": [float(values.min()), float(values.max())]}


def analyze_space(real, full, guided, linear, mask):
    arrays = [np.asarray(x, dtype=np.float64) for x in (real, full, guided, linear)]
    real, full, guided, linear = arrays
    if len({x.shape for x in arrays}) != 1 or real.shape[1] != 768:
        raise ValueError("expected matching [samples,768] NCHW block means")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("nonfinite block feature")
    train = ~mask
    if not train.any() or not mask.any():
        raise ValueError("both pre-existing class splits are required")
    actual = guided-full
    nonlinear = actual-linear
    variance = real[train].var(axis=0, ddof=1)
    if np.any(variance <= 0):
        raise ValueError("training real block variance must be positive; no added ridge")
    output = {}
    for scaling, scale in (("unwhitened", np.ones(768)), ("train_real_diagonal_whitened", np.sqrt(variance))):
        # Fit the observed Full distribution-error axis exclusively on train.
        error_train = (full[train].mean(0)-real[train].mean(0))/scale
        norm = np.linalg.norm(error_train)
        if norm == 0:
            raise ValueError("zero training error axis has no signed projection")
        axis = error_train/norm
        projections = {name: value[mask]/scale @ axis
                       for name, value in (("actual", actual), ("linear", linear), ("nonlinear", nonlinear))}
        if not np.allclose(projections["actual"], projections["linear"]+projections["nonlinear"], atol=1e-11, rtol=1e-11):
            raise AssertionError("projection decomposition lost alignment")
        split_results = {}
        for split, selected in (("train", train), ("heldout", mask)):
            error = (full[selected].mean(0)-real[selected].mean(0))/scale
            l = linear[selected].mean(0)/scale
            r = nonlinear[selected].mean(0)/scale
            a = actual[selected].mean(0)/scale
            before = float(error @ error)
            after_linear = float((error+l) @ (error+l))
            after_actual = float((error+a) @ (error+a))
            cross = float(l @ r)
            # Symmetric attribution of the quadratic cross term; algebra only.
            credit_l = float(2*error@l+l@l+cross)
            credit_r = float(2*error@r+r@r+cross)
            if not np.isclose(credit_l+credit_r, after_actual-before, atol=1e-11, rtol=1e-10):
                raise AssertionError("squared mean error decomposition failed")
            split_results[split] = {
                "samples": int(selected.sum()), "squared_mean_error_full": before,
                "squared_mean_error_linear_counterfactual": after_linear,
                "squared_mean_error_actual": after_actual,
                "linear_directional_term": float(2*error@l),
                "nonlinear_directional_term": float(2*error@r),
                "linear_quadratic_attribution": credit_l,
                "nonlinear_quadratic_attribution": credit_r,
                "linear_nonlinear_mean_dot": cross,
                "actual_mean_shift_norm": float(np.linalg.norm(a)),
                "linear_mean_shift_norm": float(np.linalg.norm(l)),
                "nonlinear_mean_shift_norm": float(np.linalg.norm(r)),
            }
        output[scaling] = {
            "training_error_axis_norm": float(norm),
            "heldout_projections_onto_train_error_axis": {k: scalar_summary(v) for k, v in projections.items()},
            "empirical_squared_mean_error": split_results,
        }
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoder-features", type=Path, required=True)
    p.add_argument("--real-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.decoder_features) as bank:
        decoder = {key: bank[key] for key in bank.files}
    with np.load(args.real_features) as bank:
        real = {key: bank[key] for key in bank.files}
    for key in ("sample_ids", "labels", "source_rows", "test_mask"):
        if not np.array_equal(decoder[key], real[key]):
            raise ValueError(f"identity mismatch: {key}")
    labels = decoder["labels"]
    if len(labels) != 1000 or not np.array_equal(np.sort(labels), np.arange(1000)):
        raise ValueError("formal diagnostic requires exactly one image per class")
    mask = decoder["test_mask"].astype(bool)
    if mask.sum() != 200:
        raise ValueError("expected existing 200 heldout classes")
    result = {
        "protocol": "raev2_decoder_linearization_analysis_v1",
        "sources": [file_record(x) for x in (args.decoder_features, args.real_features, Path(__file__))],
        "samples": 1000, "training_classes": 800, "heldout_classes": 200,
        "feature": "linear 16x16-grid RGB block means; NCHW flatten; 768 coordinates",
        "spaces": {},
        "limitations": [
            "Fixed endpoint displacement from full-only to ordinary IG; no new guidance was sampled.",
            "Linear counterfactual pixels need not be valid images and were not clipped for mean-error algebra.",
            "Signed projections and squared feature-mean error are not FID or full distribution distances.",
            "Normal intervals are descriptive class-level approximations, not simultaneous or finite-sample certificates.",
            "These banks and their original class split were used in earlier research; this retrospective diagnostic is not fresh independent confirmation of a method.",
            "Large finite endpoint curvature does not establish local guidance error or justify a new correction.",
            "FP32 derivatives are evaluated at promoted FP16 saved endpoints; not BF16 production derivatives.",
        ],
    }
    for space in ("raw", "clamped"):
        result["spaces"][space] = analyze_space(
            real["real_blockmean"], decoder[f"{space}_full_blockmean"],
            decoder[f"{space}_guided_blockmean"], decoder[f"{space}_linear_blockmean"], mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
