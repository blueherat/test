"""Read completed FID metadata after root authorization; never recompute FID."""
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

OUT = Path(__file__).resolve().parent
S = OUT.parent / "spatial_energy_balls_v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def stamp(value):
    return datetime.fromisoformat(value)

def main():
    start, cpu = time.perf_counter(), time.process_time()
    prior = read(OUT / "audit.json")
    assert prior["complete"] and not prior["fid_files_opened_or_metrics_computed"]
    freeze = read(S / "cost_match_before_fid.json")
    sampling = read(S / "sample_jobs_execution.json")
    execution = read(S / "fid_jobs_execution.json")
    jobs = read(S / "fid_jobs.json")
    rows = read(S / "fid_results.json")
    csvrows = list(csv.DictReader((S / "fid_results.csv").open()))
    assert len(rows) == len(csvrows) == 3
    assert sampling["complete"] and execution["complete"]
    assert len(execution["jobs"]) == 1 and execution["jobs"][0]["exit_code"] == 0
    assert sha(S / "fid_jobs.json") == execution["request_sha256"]
    fixed = {str(Path(v["path"]).resolve()): v["sha256"] for v in jobs["fixed_files"]}
    assert sha(S / "cost_match_before_fid.json") == fixed[str((S / "cost_match_before_fid.json").resolve())]
    assert freeze["K"] == prior["cost_rule"]["independent_K_from_frozen_rule"] == 100
    assert not freeze["new_fid_computed_or_read_before_this_decision"]
    assert freeze["quality_results"] is None
    freeze_time = stamp(freeze["created_utc"])
    sample_done = max(stamp(j["finished_utc"]) for j in sampling["jobs"])
    fid_start = stamp(execution["jobs"][0]["started_utc"])
    assert sample_done < freeze_time < stamp(execution["started_utc"]) <= fid_start
    assert execution["jobs"][0]["argv"] == jobs["jobs"][0]["argv"]
    commits, references = set(), set()
    samples = {}
    for row, csvrow in zip(rows, csvrows):
        for k, v in row.items():
            assert str(v) == csvrow[k], (k, v, csvrow[k])
        arm = row["branch"]
        summary = read(S / arm / "summary.json")
        assert sha(S / arm / "summary.json") == freeze["summaries"][arm]["sha256"]
        assert sha(S / arm / "summary.json") == prior["input_metadata_sha256"][arm]["summary.json"]
        path = Path(row["sample_path"]).resolve()
        assert path == (S / arm / "samples.npz").resolve()
        assert row["sample_sha256"] == fixed[str(path)]
        artifact = [v for v in summary.values() if isinstance(v, dict) and v.get("path") and Path(v["path"]).resolve() == path]
        assert len(artifact) == 1 and artifact[0]["sha256"] == row["sample_sha256"]
        samples[arm] = {"fid": row["fid"], "sample_sha256": row["sample_sha256"], "same_hash_in_summary_frozen_jobs_evaluator": True}
        commits.add(row["evaluator_commit"])
        references.add(row["fid_reference"])
    assert len(commits) == len(references) == 1
    evaluator_root = rows[0]["evaluator_root"]
    commit = subprocess.check_output(["git", "-C", evaluator_root, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", evaluator_root, "status", "--porcelain", "--untracked-files=no"], text=True)
    assert commit == next(iter(commits)) and not status
    for path, expected in fixed.items():
        if path.endswith("evaluate_raev2_official_samples.py") or path.endswith("driver.py"):
            assert sha(path) == expected
    decimal_rows = {r["branch"]: Decimal(r["fid"]) for r in csvrows}
    comparisons = {}
    with localcontext() as ctx:
        ctx.prec = 40
        for candidate, base in [("spatial100", "official100"), ("spatial100", "global100"), ("global100", "official100")]:
            difference = decimal_rows[candidate] - decimal_rows[base]
            comparisons[candidate + "_vs_" + base] = {
                "fid_difference_decimal": str(difference),
                "relative_increase_percent_decimal": str(100 * difference / decimal_rows[base]),
                "relative_improvement_percent_decimal": str(-100 * difference / decimal_rows[base]),
                "positive_increase_means_worse": True,
            }
    inputs = ["cost_match_before_fid.json", "sample_jobs_execution.json", "fid_jobs.json", "fid_jobs_execution.json", "fid_results.json", "fid_results.csv"]
    result = {
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "After explicit root unblinding authorization: completed evaluation metadata reconciliation and arithmetic on stored FID only.",
        "prior_identity_cost_audit_sha256": sha(OUT / "audit.json"),
        "review_script_sha256": sha(__file__),
        "input_sha256": {name: sha(S / name) for name in inputs},
        "arms": samples,
        "comparisons_from_full_stored_decimal_precision": comparisons,
        "time_order": {
            "all_sampling_finished_utc": sample_done.isoformat(),
            "cost_freeze_created_utc": freeze_time.isoformat(),
            "fid_worker_started_utc": fid_start.isoformat(),
            "cost_freeze_to_fid_seconds": (fid_start - freeze_time).total_seconds(),
            "ordering_verified": True,
            "freeze_hash_in_fid_job_prerequisites": True,
            "driver_checks_all_fixed_file_hashes_before_starting_worker": True,
            "limitation": "This verifies the archived run's timestamp/hash dependency chain; it is not an external proof about unrecorded processes.",
        },
        "evaluator": {"commit": commit, "tracked_worktree_clean_at_review": True, "fid_reference": next(iter(references)), "one_common_invocation": True, "outer_wall_seconds": execution["jobs"][0]["outer_wall_seconds"], "frozen_file_records": jobs["fixed_files"]},
        "boundary": {
            "stored_sample_hashes_reconciled_not_independently_rehashed": True,
            "sample_pixels_or_features_opened": False,
            "fid_recomputed": False,
            "bootstrap_or_influence_function": False,
            "new_gpu_or_experiment": False,
            "finite_1k_point_estimate_not_population_or_all_scales_claim": True,
            "historical_reference_preparation_cost_still_unclosed": True,
            "conclusion": "Frozen spatial implementation is negative at the preregistered 1K gate; no >=5% improvement and no total-cost success. End this implementation; no tuning/extra seeds/larger run justified by these results.",
        },
        "review_cost": {"wall_seconds_before_final_write": time.perf_counter() - start, "cpu_seconds_before_final_write": time.process_time() - cpu},
    }
    target = OUT / "fid_metadata_audit.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": True, "comparisons": comparisons, "audit_sha256": sha(target), "prior_audit_sha256": sha(OUT / "audit.json")}, indent=2))

if __name__ == "__main__":
    main()
