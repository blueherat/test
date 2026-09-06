"""Independent review: exact rational CSV arithmetic and precision-form Gaussians.

Does not import or call the producer, evaluate a model, or compute FID.
"""
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_REQUEST = "db4cdaf373fe7e8c8d69b8b645045e25fff466424ec24e060a0cd02a22807058"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def close(actual, expected, atol=1e-11, rtol=2e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)
    return float(np.max(np.abs(actual - expected)))

def gaussian_review(stored, sf, sb, mf, mb, z):
    """Start from power-target posterior precision, not the producer's A solve."""
    w, g, t = 1.78, 0.78, stored["t"]
    alpha, eye = 1.0 - t, np.eye(len(z))
    pf, pb = np.linalg.solve(sf, eye), np.linalg.solve(sb, eye)
    power_precision = w * pf - g * pb
    # Posterior precision representation independently gives both affine heads.
    posterior_f, posterior_b = pf + alpha**2 / t**2 * eye, pb + alpha**2 / t**2 * eye
    jf = np.linalg.solve(posterior_f, alpha / t**2 * eye)
    jb = np.linalg.solve(posterior_b, alpha / t**2 * eye)
    cf = np.linalg.solve(posterior_f, pf @ mf)
    cb = np.linalg.solve(posterior_b, pb @ mb)
    a = w * jb - g * jf
    p_eigs = np.linalg.eigvalsh(power_precision)
    a_eigs = np.linalg.eigvalsh(a)
    it_eigs = np.linalg.eigvalsh(eye - alpha * a)
    differences = [close(p_eigs, stored["power_precision_eigenvalues"]),
                   close(a_eigs, stored["consensus_A_eigenvalues"]),
                   close(it_eigs, stored["iteration_eigenvalues"])]
    proper = bool(p_eigs.min() > 0)
    assert proper == stored["proper_power_target"]
    result = {"t": t, "proper_power_target": proper,
              "power_precision_eigenvalues": p_eigs.tolist(),
              "A_eigenvalues": a_eigs.tolist(), "iteration_eigenvalues": it_eigs.tolist()}
    if proper:
        bstar = w * pf @ mf - g * pb @ mb
        # Exact conditional mean of the normalized Gaussian power prior.
        common = np.linalg.solve(power_precision + alpha**2 / t**2 * eye,
                                 bstar + alpha / t**2 * z)
        # Invert each denoiser at this target value, avoiding solving A*delta=gap.
        zf = alpha * common + t**2 / alpha * pf @ (common - mf)
        zb = alpha * common + t**2 / alpha * pb @ (common - mb)
        delta = zb - zf
        f = jf @ (z + g * delta) + cf
        b = jb @ (z + w * delta) + cb
        guided = w * f - g * b
        epsf = (z + g * delta - alpha * f) / t
        epsb = (z + w * delta - alpha * b) / t
        h = w * np.linalg.solve(jf, eye) - g * np.linalg.solve(jb, eye)
        residuals = {
            "affine_shift_f": close(zf, z + g * delta),
            "affine_shift_b": close(zb, z + w * delta),
            "consensus_f_vs_b": close(f, b),
            "full_vs_power_posterior": close(f, common),
            "guided_vs_power_posterior": close(guided, common),
            "original_noise_fixed_point": close(delta, t * (epsb - epsf)),
            "inverse_identity": close(h, alpha * eye + t**2 / alpha * power_precision),
            "delta_vs_stored": close(delta, stored["delta"]),
            "delta_norm_vs_stored": close(np.linalg.norm(delta), stored["delta_norm"]),
        }
        assert np.linalg.eigvalsh(h).min() > 0 and a_eigs.min() > 0
        assert np.linalg.eigvalsh(w * eye / alpha - a).min() > 0
        assert max(abs(it_eigs)) < 1
        for key in ["consensus_max_abs", "clean_vs_exact_max_abs",
                    "original_noise_fixed_point_residual_max_abs", "inverse_identity_max_abs"]:
            assert np.isfinite(stored[key]) and 0 <= stored[key] < 1e-10
        result.update(residuals=residuals, delta=delta.tolist(),
                      common_clean=common.tolist(), euclidean_contraction_factor=float(max(abs(it_eigs))))
    result["max_spectrum_difference_vs_producer"] = max(differences)
    return result

def main():
    started, cpu = time.perf_counter(), time.process_time()
    output = ROOT / "review.json"
    assert not output.exists()
    request_path, result_path = ROOT / "request.json", ROOT / "results.json"
    rows_path = ROOT / "directional_rows.json"
    assert sha(request_path) == EXPECTED_REQUEST
    request, result, saved_rows = read(request_path), read(result_path), read(rows_path)
    assert result["complete"] and result["request_sha256"] == EXPECTED_REQUEST
    fixed_checks = []
    for record in request["fixed_files"]:
        actual = sha(record["path"])
        assert actual == record["sha256"], record["path"]
        fixed_checks.append({"path": record["path"], "sha256": actual})
    assert result["source_script_sha256"] == fixed_checks[0]["sha256"]
    assert sha(rows_path) == result["rows_sha256"]
    assert datetime.fromisoformat(request["created_utc"]) < datetime.fromisoformat(result["created_utc"])
    input_hashes = {str(p): sha(p) for p in [request_path, result_path, rows_path]}
    source_path = Path(request["scalar_csv"])
    historical_request = read(source_path.parent / "request.json")
    historical_summary = read(source_path.parent / "summary.json")
    assert historical_summary["complete"] and historical_summary["rows"] == 800
    assert sha(source_path) == historical_summary["per_sample_step_sha256"]
    assert historical_request["samples"] == 8 and historical_request["num_steps"] == 100
    with source_path.open() as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(source_rows) == 800
    expected_ids = {(k, i) for k in range(100) for i in range(8)}
    identities = [(int(r["step_index"]), int(r["sample_id"])) for r in source_rows]
    assert set(identities) == expected_ids and len(set(identities)) == 800
    w, g = Fraction(request["w"]), Fraction(request["g"])
    assert w == Fraction(178, 100) and g == Fraction(78, 100)
    interior, boundary, values = [], [], []
    signs = Counter()
    for original in source_rows:
        k, i = int(original["step_index"]), int(original["sample_id"])
        assert int(original["label"]) == historical_request["classes"][i]
        t = Fraction(original["t"])
        assert float(t) == historical_request["time_grid"][k]
        assert float(original["next_t"]) == historical_request["time_grid"][k + 1]
        qf, qb, norm2 = (Fraction(original[key]) for key in
                         ["q_full_gap_direction", "q_base_gap_direction", "direction_squared_norm"])
        assert norm2 > 0
        a = w * qb - g * qf
        if t == 1:
            assert k == 0
            boundary.append((k, i))
            continue
        assert 0 < t < 1 and 1 <= k <= 99
        saved = saved_rows[len(interior)]
        assert (saved["step_index"], saved["sample_id"]) == (k, i)
        for key, expected in [("t", t), ("q_full", qf), ("q_base", qb), ("direction_squared_norm", norm2)]:
            assert saved[key] == float(expected)
        assert Fraction(saved["A_quadratic_decimal"]) == a
        assert saved["A_unit_rayleigh"] == float(a / norm2)
        assert saved["strictly_negative"] == (a < 0)
        interior.append((k, i)); values.append(float(a / norm2))
        signs["positive" if a > 0 else "negative" if a < 0 else "zero"] += 1
    assert len(interior) == len(saved_rows) == result["interior_rows"] == 792
    assert len(boundary) == result["excluded_t1_rows"] == 8
    assert result["source_rows"] == 800
    assert Counter(i for _, i in interior) == Counter({i: 99 for i in range(8)})
    assert Counter(k for k, _ in interior) == Counter({k: 8 for k in range(1, 100)})
    assert signs["negative"] == result["negative_A_rows"] == 0
    assert result["negative_A_rows_per_id"] == {str(i): 0 for i in range(8)}
    assert result["negative_step_indices"] == [] and result["largest_negative_unit_rayleigh"] is None
    assert min(values) == result["unit_rayleigh_min"]
    assert max(values) == result["unit_rayleigh_max"]
    assert min(abs(v) for v in values) == result["smallest_absolute_unit_rayleigh"]
    sf, sb = np.array([[1.0, .25], [.25, .4]]), np.array([[1.8, -.2], [-.2, .9]])
    mf, mb, z = np.array([.3, -.4]), np.array([-.2, .1]), np.array([.6, -.7])
    assert len(result["exact_gaussian_cases"]) == 4
    assert [r["t"] for r in result["exact_gaussian_cases"]] == [.1, .5, .9, .99]
    assert np.linalg.norm(sf @ sb - sb @ sf) > 0
    close(np.linalg.norm(sf @ sb - sb @ sf), result["gaussian_noncommuting_commutator_norm"])
    gaussian_cases = [gaussian_review(c, sf, sb, mf, mb, z) for c in result["exact_gaussian_cases"]]
    improper = result["positive_A_is_not_sufficient_examples"]
    assert len(improper) == 2 and [r["t"] for r in improper] == [.1, .95]
    improper_cases = [gaussian_review(c, 4 * np.eye(2), np.eye(2), np.zeros(2), np.zeros(2), np.ones(2)) for c in improper]
    assert min(improper_cases[0]["A_eigenvalues"]) > 0
    assert max(improper_cases[1]["A_eigenvalues"]) < 0
    assert all(max(c["power_precision_eigenvalues"]) < 0 for c in improper_cases)
    assert all(sha(p) == value for p, value in input_hashes.items())
    review = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "/root/fid_analysis", "review_script_sha256": sha(__file__),
        "request_sha256": EXPECTED_REQUEST, "input_sha256": input_hashes,
        "fixed_file_hash_checks": fixed_checks,
        "request_precedes_results_timestamp": True,
        "independent_methods": ["Fraction arithmetic on original CSV decimal strings; no producer import",
                                "Gaussian posterior-precision solve followed by inverse-head shifts, rather than the producer's A*delta=gap solve"],
        "source_rows": 800, "interior_rows": 792, "excluded_t1_rows": 8,
        "interior_steps": [1, 99], "sample_ids": list(range(8)), "interior_rows_per_id": 99,
        "all_row_identities_labels_times_exact": True,
        "all_792_decimal_combinations_exact": True, "all_792_unit_rayleigh_float_outputs_exact": True,
        "sign_counts": {key: signs[key] for key in ["positive", "zero", "negative"]},
        "unit_rayleigh_min": min(values), "unit_rayleigh_max": max(values),
        "gaussian_cases": gaussian_cases, "improper_target_cases": improper_cases,
        "producer_residuals_and_independent_residuals_below_1e_minus10": True,
        "decision": "The sampled necessary direction condition was not refuted. This does not establish the Gaussian model, matrix positive definiteness, power-target normalizability, nonlinear CH convergence, native BF16 validity, or FID benefit.",
        "limitations": ["Historical data already inspected; 99 correlated interior states for each of 8 IDs, not 792 independent trials.",
                        "Finite-direction positivity cannot certify matrix positivity; stored FP32 VJPs are not certified exact derivatives.",
                        "Same-state Jacobians imply the shifted-root result only in the affine Gaussian case.",
                        "Common affine support requires tangent inverses and ambient PSD; t=1 is excluded.",
                        "Near t=1 Gaussian convergence can be arbitrarily slow, as visible in the fixed .99 case.",
                        "Only the frozen four proper and two improper cases were checked; no parameter search or additional dataset."],
        "new_model_forwards": 0, "new_vjps": 0, "new_images": 0, "fid_performed": False,
        "wall_seconds_before_final_write": time.perf_counter() - started,
        "cpu_seconds_before_final_write": time.process_time() - cpu,
    }
    output.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": True, "interior_rows": 792, "sign_counts": review["sign_counts"],
                      "unit_rayleigh_min": min(values), "unit_rayleigh_max": max(values),
                      "review_sha256": sha(output)}, indent=2))

if __name__ == "__main__":
    main()
