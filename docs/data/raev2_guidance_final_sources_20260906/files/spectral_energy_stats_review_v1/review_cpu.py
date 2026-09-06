"""Independent review of frozen energy artifacts; never opens original latent files."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "spectral_energy_audit_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    start, cpu = time.perf_counter(), time.process_time()
    request_hash = sha(SOURCE / "request.json")
    assert request_hash == "d7a4860fe131a5075ccfc3ec9a9294747359f3b6af56f4223e54b69c66a03f17"
    request = json.loads((SOURCE / "request.json").read_text())
    summary_hash = sha(SOURCE / "summary.json")
    summary = json.loads((SOURCE / "summary.json").read_text())
    assert summary["complete"] and summary["request_sha256"] == request_hash
    assert sha(request["runner"]["path"]) == request["runner"]["sha256"]
    checks, comparisons, maximum = 0, 0, 0.0
    global_vectors = []
    bank_checks = []

    def close(a, b):
        nonlocal checks, comparisons, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        maximum = max(maximum, error)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12), error
        checks += 1
        comparisons += a.size

    for cohort, bank in zip(request["cohorts"], summary["banks"]):
        path = Path(bank["artifact"]["path"])
        assert path.stat().st_size == bank["artifact"]["bytes"]
        assert sha(path) == bank["artifact"]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            ce = z["class_energy"]
            cr = z["class_residual"]
            ge = z["global_equal_class_energy"]
            gr = z["global_equal_class_residual"]
            counts = z["class_counts"]
            ids, labels, sources = z["sample_ids"], z["labels"], z["source_rows"]
            keep, excluded = z["deduplicated_keep"], z["excluded_ids"]
            assert ce.dtype == cr.dtype == ge.dtype == gr.dtype == np.float64
            assert ce.shape == (2, 3, 1000, 2, 1024)
            assert cr.shape == (2, 2, 1000, 2, 1024)
            assert np.array_equal(ids, np.arange(5000))
            assert np.array_equal(labels, ids % 1000)
            assert np.array_equal(excluded, cohort["excluded_ids"])
            assert np.array_equal(keep, ~np.isin(ids, excluded))
            assert np.array_equal(sources[excluded], cohort["excluded_source_rows"])
            assert np.array_equal(labels[excluded], cohort["excluded_labels"])
            assert np.array_equal(counts[0], np.bincount(labels, minlength=1000))
            assert np.array_equal(counts[1], np.bincount(labels[keep], minlength=1000))
            assert keep.sum() == 4979 and np.all(counts[0] == 5)
            assert np.count_nonzero(counts[1] == 4) == 21
            assert np.count_nonzero(counts[1] == 5) == 979
            assert np.isfinite(ce).all() and np.all(ce >= 0)
            for arm in (1, 2):
                close(cr[:, arm - 1], ce[:, arm] - ce[:, 0])

            # Kahan sum over explicit class slices, independent of producer mean(axis=2).
            total = np.zeros((2, 3, 2, 1024), dtype=np.float64)
            correction = np.zeros_like(total)
            for c in range(1000):
                delta = ce[:, :, c] - correction
                new_total = total + delta
                correction = (new_total - total) - delta
                total = new_total
            recomputed = total / 1000
            close(recomputed, ge)
            close(gr, recomputed[:, 1:] - recomputed[:, :1])
            close(ce[0, :, counts[1] == 5], ce[1, :, counts[1] == 5])
            # With one exclusion in each affected class, reconstruct its energy;
            # no assertion that generated rows correspond to real image targets.
            affected = np.flatnonzero(counts[1] == 4)
            removed_energy = 5 * ce[0, :, affected] - 4 * ce[1, :, affected]
            assert removed_energy.min() >= -1e-12

            for vi, view in enumerate(request["views"]):
                for ai, arm in enumerate(request["arms"]):
                    rows = bank["results"][view][arm]
                    component_sums = [math.fsum(recomputed[vi, ai, ci]) for ci in range(2)]
                    close(np.array(component_sums), np.array([rows["energy_sum_over_channels"][c] for c in request["components"]]))
                    close(np.array(sum(component_sums)), np.array(rows["energy_DC_plus_AC_sum"]))
                    close(np.array(sum(component_sums) / 1024), np.array(rows["mean_energy_per_latent_coordinate"]))
                    if ai:
                        ref = [math.fsum(recomputed[vi, 0, ci]) for ci in range(2)]
                        assert all(v > 0 for v in ref)
                        for ci, component in enumerate(request["components"]):
                            close(np.array(component_sums[ci] / ref[ci]), np.array(rows["energy_ratio_to_real"][component]))
                            rv = gr[vi, ai - 1, ci]
                            rinfo = rows["residual_vector"][component]
                            close(np.array(math.sqrt(math.fsum(float(x) ** 2 for x in rv))), np.array(rinfo["l2_norm"]))
                            assert rinfo["positive_channels"] == sum(float(x) > 0 for x in rv)
                            assert rinfo["negative_channels"] == sum(float(x) < 0 for x in rv)
                            assert rinfo["zero_channels"] == sum(float(x) == 0 for x in rv)
                        close(np.array(sum(component_sums) / sum(ref)), np.array(rows["energy_ratio_to_real"]["DC_plus_AC"]))
            global_vectors.append(gr.copy())
            bank_checks.append({
                "seed": bank["seed"], "artifact_sha256": bank["artifact"]["sha256"],
                "class_count": 1000, "primary_samples": 5000, "sensitivity_samples": 4979,
                "sensitivity_class_sizes": {"4": 21, "5": 979},
                "unaffected_class_energies_unchanged": True,
                "excluded_energy_nonnegative_with_fp64_tolerance": True,
                "all_real_component_denominators_positive": True,
                "min_reconstructed_excluded_energy": float(removed_energy.min()),
            })
        del ce, cr, ge, gr, total, correction, recomputed

    for vi, view in enumerate(request["views"]):
        for ai, arm in enumerate(request["arms"][1:]):
            for ci, component in enumerate(request["components"]):
                a, b = global_vectors[0][vi, ai, ci], global_vectors[1][vi, ai, ci]
                record = summary["cross_bank_residual_reproducibility"][view][arm][component]
                aa = math.fsum(float(x) ** 2 for x in a)
                bb = math.fsum(float(x) ** 2 for x in b)
                dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
                assert aa > 0 and bb > 0
                close(np.array(dot / math.sqrt(aa * bb)), np.array(record["residual_cosine"]))
                counts = {"both_positive_channels": 0, "both_negative_channels": 0, "both_zero_channels": 0, "either_zero_channels": 0}
                agreement = 0
                for x, y in zip(a.tolist(), b.tolist()):
                    sx, sy = int(x > 0) - int(x < 0), int(y > 0) - int(y < 0)
                    agreement += sx == sy
                    counts["both_positive_channels"] += x > 0 and y > 0
                    counts["both_negative_channels"] += x < 0 and y < 0
                    counts["both_zero_channels"] += x == 0 and y == 0
                    counts["either_zero_channels"] += x == 0 or y == 0
                assert record["channel_count"] == 1024
                assert record["sign_agreement_fraction"] == agreement / 1024
                assert all(record[k] == v for k, v in counts.items())
    assert sha(SOURCE / "request.json") == request_hash
    assert sha(SOURCE / "summary.json") == summary_hash
    result = {
        "complete": True, "review_kind": "Independent frozen-protocol and saved-statistics review",
        "request_sha256": request_hash, "summary_sha256": summary_hash,
        "review_script_sha256": sha(__file__),
        "banks": bank_checks,
        "numerical_checks": checks, "numeric_values_compared": comparisons,
        "maximum_absolute_difference": maximum,
        "cross_bank_metrics_match": True,
        "all_reported_energy_ratios_match_ratio_of_total_energies": True,
        "reviewer_original_latent_payload_bytes_read": 0,
        "artifact_bytes_hashed": sum(b["artifact"]["bytes"] for b in summary["banks"]),
        "scope": "Saved energy statistics and metadata; not independent re-extraction from FP16 latents, independent sample confirmation, CI or FID evidence.",
        "limitations": [
            "Cannot reconstruct discarded per-image values independently without original latents.",
            "AC is all non-DC spatial energy, not only high frequency.",
            "Shared-source exclusion does not establish statistical independence.",
            "Coordinates/classes/banks are not asserted iid inferential replicates.",
        ],
        "cost": {"wall_seconds_before_output_write": time.perf_counter() - start,
                 "cpu_seconds_before_output_write": time.process_time() - cpu,
                 "gpu_seconds": 0, "model_calls": 0},
    }
    (HERE / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
