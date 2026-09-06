#!/usr/bin/env python3
"""CPU-only necessary-condition audit of Characteristic Guidance's Gaussian case.

Reads frozen directional Jacobian scalars; performs no model calls or FID.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_case(sf, sb, muf, mub, z, t):
    w, g, alpha = 1.78, 0.78, 1.0 - t
    eye = np.eye(len(z))
    precision = w * np.linalg.inv(sf) - g * np.linalg.inv(sb)
    jf = alpha * np.linalg.solve(alpha * alpha * sf + t * t * eye, sf)
    jb = alpha * np.linalg.solve(alpha * alpha * sb + t * t * eye, sb)
    cf, cb = (eye - alpha * jf) @ muf, (eye - alpha * jb) @ mub
    a = w * jb - g * jf
    result = {
        "t": t, "power_precision_eigenvalues": np.linalg.eigvalsh(precision).tolist(),
        "consensus_A_eigenvalues": np.linalg.eigvalsh(a).tolist(),
        "iteration_eigenvalues": np.linalg.eigvalsh(eye - alpha * a).tolist(),
    }
    if np.linalg.eigvalsh(precision).min() <= 0:
        result["proper_power_target"] = False
        return result
    st = np.linalg.inv(precision)
    mt = st @ (w * np.linalg.solve(sf, muf) - g * np.linalg.solve(sb, mub))
    delta = np.linalg.solve(a, (jf - jb) @ z + cf - cb)
    f = jf @ (z + g * delta) + cf
    b = jb @ (z + w * delta) + cb
    jt = alpha * np.linalg.solve(alpha * alpha * st + t * t * eye, st)
    exact = jt @ z + (eye - alpha * jt) @ mt
    epsf, epsb = (z + g * delta - alpha * f) / t, (z + w * delta - alpha * b) / t
    result.update(
        proper_power_target=True,
        delta=delta.tolist(), delta_norm=float(np.linalg.norm(delta)),
        consensus_max_abs=float(np.max(np.abs(f - b))),
        clean_vs_exact_max_abs=float(np.max(np.abs(w * f - g * b - exact))),
        original_noise_fixed_point_residual_max_abs=float(np.max(np.abs(delta - t * (epsb - epsf)))),
        inverse_identity_max_abs=float(np.max(np.abs(
            w * np.linalg.inv(jf) - g * np.linalg.inv(jb)
            - alpha * eye - (t * t / alpha) * precision))),
    )
    assert result["consensus_max_abs"] < 1e-10
    assert result["clean_vs_exact_max_abs"] < 1e-10
    assert result["original_noise_fixed_point_residual_max_abs"] < 1e-10
    assert result["inverse_identity_max_abs"] < 1e-10
    assert np.linalg.eigvalsh(a).min() > 0
    assert np.max(np.abs(np.linalg.eigvalsh(eye - alpha * a))) < 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.perf_counter(), time.process_time()
    request = json.loads(args.request.read_text())
    root = args.request.resolve().parent
    output = root / "results.json"
    if output.exists():
        raise FileExistsError(output)
    for rec in request["fixed_files"]:
        assert sha(Path(rec["path"])) == rec["sha256"], rec["path"]
    assert request["w"] == "1.78" and request["g"] == "0.78"
    sf = np.array([[1.0, 0.25], [0.25, 0.4]])
    sb = np.array([[1.8, -0.2], [-0.2, 0.9]])
    commutator = float(np.linalg.norm(sf @ sb - sb @ sf))
    assert commutator > 0
    cases = [gaussian_case(sf, sb, np.array([0.3, -0.4]), np.array([-0.2, 0.1]),
                           np.array([0.6, -0.7]), t) for t in [0.1, 0.5, 0.9, 0.99]]
    counterexamples = [gaussian_case(4 * np.eye(2), np.eye(2), np.zeros(2), np.zeros(2),
                                    np.ones(2), t) for t in [0.1, 0.95]]
    assert min(counterexamples[0]["consensus_A_eigenvalues"]) > 0
    assert max(counterexamples[1]["consensus_A_eigenvalues"]) < 0
    with Path(request["scalar_csv"]).open() as stream:
        inputs = list(csv.DictReader(stream))
    assert len(inputs) == 800
    assert {(int(r["step_index"]), int(r["sample_id"])) for r in inputs} == {
        (step, sample) for step in range(100) for sample in range(8)
    }
    rows, boundary = [], []
    with localcontext() as context:
        context.prec = 50
        for source in inputs:
            t = Decimal(source["t"])
            qf, qb = Decimal(source["q_full_gap_direction"]), Decimal(source["q_base_gap_direction"])
            norm2 = Decimal(source["direction_squared_norm"])
            a = Decimal("1.78") * qb - Decimal("0.78") * qf
            assert qf.is_finite() and qb.is_finite() and norm2.is_finite() and norm2 > 0
            row = {"step_index": int(source["step_index"]), "sample_id": int(source["sample_id"]),
                   "t": float(t), "q_full": float(qf), "q_base": float(qb),
                   "direction_squared_norm": float(norm2), "A_quadratic_decimal": str(a),
                   "A_unit_rayleigh": float(a / norm2), "strictly_negative": a < 0}
            if 0 < t < 1:
                rows.append(row)
            else:
                assert t == 1
                boundary.append(row)
    values = np.array([r["A_unit_rayleigh"] for r in rows])
    negative = [r for r in rows if r["strictly_negative"]]
    per_id = {str(i): sum(r["sample_id"] == i for r in negative) for i in range(8)}
    (root / "directional_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    result = {
        "complete": True, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_sha256": sha(args.request), "source_script_sha256": sha(Path(__file__)),
        "gaussian_noncommuting_commutator_norm": commutator,
        "exact_gaussian_cases": cases, "positive_A_is_not_sufficient_examples": counterexamples,
        "source_rows": len(inputs), "interior_rows": len(rows), "excluded_t1_rows": len(boundary),
        "negative_A_rows": len(negative), "negative_A_rows_per_id": per_id,
        "negative_step_indices": sorted({r["step_index"] for r in negative}),
        "unit_rayleigh_min": float(values.min()), "unit_rayleigh_max": float(values.max()),
        "smallest_absolute_unit_rayleigh": float(np.min(np.abs(values))),
        "largest_negative_unit_rayleigh": max((r["A_unit_rayleigh"] for r in negative), default=None),
        "rows_sha256": sha(root / "directional_rows.json"),
        "guarantee_scope": "Necessary condition for exact Gaussian clean heads with a proper Gaussian power target, tested at delta=0 where the Gaussian Jacobian is constant.",
        "limitations": ["A positive directional value does not establish matrix positivity or target normalizability.",
                        "Strict Gaussian convergence theorem assumes full rank; for a common affine support, use tangent inverses and ambient PSD. Negative ambient quadratic values still contradict this necessary PSD condition.",
                        "Negative values do not rule out every non-Gaussian approximation or empirical projected CH implementation.",
                        "Stored FP32 VJP values are evaluated as recorded, not newly differentiated or certified exact.",
                        "800 correlated rollout rows come from 8 sample IDs; no independent-trial probability or FID inference.",
                        "The t=1 fixed-point equation is degenerate and was excluded a priori."],
        "new_model_forwards": 0, "new_vjps": 0, "new_images": 0, "fid_performed": False,
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["complete", "interior_rows", "negative_A_rows",
        "negative_A_rows_per_id", "unit_rayleigh_min", "unit_rayleigh_max", "largest_negative_unit_rayleigh",
        "wall_seconds", "cpu_seconds"]}))


if __name__ == "__main__":
    main()
