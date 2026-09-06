"""Read-only identity/cost review. Never accesses FID or the image arr_0 member."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parent / "spatial_energy_balls_v1"
ARMS = {"official100": "official", "global100": "global_ball", "spatial100": "spatial_balls"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_record(record):
    path = Path(record["path"])
    assert path.stat().st_size == record["size_bytes"]
    assert sha(path) == record["sha256"]
    return path


def array_sha(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def main():
    started, cpu = time.perf_counter(), time.process_time()
    execution = read_json(S / "sample_jobs_execution.json")
    assert execution["complete"] and all(j["exit_code"] == 0 for j in execution["jobs"])
    assert sha(execution["request"]) == execution["request_sha256"]
    jobs = {j["name"]: j for j in execution["jobs"]}
    assert set(jobs) == set(ARMS)
    plan = read_json(S / "plan.json")
    parity = read_json(S / "parity16/summary.json")
    assert parity["complete"] and parity["endpoint_bitwise"] and parity["pixel_bitwise"]
    assert parity["samples_per_loop"] == 16 and parity["num_steps"] == 100
    parity_request = read_json(check_record(parity["request"]))

    common_request = common_input = common_noise_audit = common_batches = None
    summaries, reports, preserved_hashes = {}, {}, {}
    all_id_values = 0
    input_keys = ["seed", "noise_shape", "global_noise_sha256", "noise_rng_state_sha256", "global_labels_sha256", "cuda_device"]
    for name, mode in ARMS.items():
        arm = S / name
        request_path, input_path, summary_path = arm / "request.json", arm / "sampling_input.json", arm / "summary.json"
        req, inp, summary = read_json(request_path), read_json(input_path), read_json(summary_path)
        preserved_hashes[name] = {p.name: sha(p) for p in (request_path, input_path, summary_path)}
        assert req["mode"] == mode and req["command"] == "sample"
        assert req["sample_count"] == 1000 and req["batch_size"] == 8 and req["num_steps"] == 100
        assert req["seed"] == 202609121 and req["fid_performed"] is False
        assert inp["noise_shape"] == [1000, 1024, 16, 16] and inp["frozen_before_first_model_forward"]
        assert inp["global_labels_sha256"] == array_sha(np.arange(1000, dtype=np.int64))
        assert "TF32 off" in req["precision"] and "native BF16" in req["precision"]
        check_record(inp["request"])
        check_record(summary["request"])
        check_record(summary["sampling_input"])
        assert summary["complete"] and summary["samples"] == summary["global_cohort_size"] == 1000
        assert summary["fid_performed"] is False
        assert summary["loading"]["checkpoint_step"] == 100080
        assert summary["stage2_forward_calls"] == 12500 and summary["stage2_sample_forwards"] == 100000
        assert summary["stage2_nfe_per_sample"] == 100
        assert summary["decoder_forward_calls"] == 125 and summary["decoder_sample_forwards"] == 1000
        assert summary["global_ids"] == list(range(1000))
        for key in ["global_noise_sha256", "noise_rng_state_sha256", "global_labels_sha256"]:
            assert summary[key] == inp[key]
        assert req["identities"] == parity_request["identities"]
        for record in req["parity"].values():
            check_record(record)
        # Actual archived code rehashed. Large checkpoint/decoder/calibration payloads
        # are compared by recorded identity and size, not redundantly loaded.
        for record in req["sources"].values():
            check_record(record)
        for key, record in req["identities"].items():
            assert Path(record["path"]).stat().st_size == record["size_bytes"]
            if key in ("config", "calibration_request", "calibration_summary", "root_plan", "protocol_document"):
                check_record(record)
        signature = {k: v for k, v in req.items() if k not in ("mode", "sources")}
        signature["source_hashes"] = {k: (v["sha256"], v["size_bytes"]) for k, v in req["sources"].items()}
        input_signature = {k: inp[k] for k in input_keys}
        if common_request is None:
            common_request, common_input = signature, input_signature
        else:
            assert signature == common_request
            assert input_signature == common_input
        noise_path = check_record(inp["paired_noise_audit"])
        with np.load(noise_path, allow_pickle=False) as z:
            noise_snapshot = {key: z[key] for key in z.files}
        assert set(noise_snapshot) == {"first_noise", "rng_state"}
        assert noise_snapshot["first_noise"].shape == (1024, 16, 16)
        assert noise_snapshot["first_noise"].dtype == np.float32
        assert array_sha(noise_snapshot["rng_state"]) == inp["noise_rng_state_sha256"]
        if common_noise_audit is None:
            common_noise_audit = noise_snapshot
        else:
            assert all(np.array_equal(common_noise_audit[k], noise_snapshot[k]) for k in noise_snapshot)

        manifest = read_json(check_record(summary["batch_manifest"]))["batches"]
        assert len(manifest) == 125
        batch_signature = []
        for b, batch in enumerate(manifest):
            ids = list(range(8*b, 8*(b+1)))
            assert batch["global_ids"] == ids
            assert batch["labels_sha256"] == array_sha(np.asarray(ids, dtype=np.int64))
            batch_signature.append((ids, batch["noise_sha256"], batch["labels_sha256"]))
        if common_batches is None:
            common_batches = batch_signature
        else:
            assert batch_signature == common_batches
        # Open only the tiny ids and labels ZIP members. arr_0 is never accessed.
        sample_record = summary["sample_archive"]
        sample_path = Path(sample_record["path"])
        assert sample_path.stat().st_size == sample_record["size_bytes"]
        with np.load(sample_path, allow_pickle=False) as z:
            ids, labels = z["ids"], z["labels"]
        assert ids.dtype == labels.dtype == np.int64
        assert np.array_equal(ids, np.arange(1000)) and np.array_equal(labels, ids % 1000)
        all_id_values += ids.size + labels.size
        assert np.array_equal(np.bincount(labels), np.ones(1000, dtype=np.int64))

        if mode == "official":
            assert summary["diagnostic_steps"] == summary["energy_reductions_in_official"] == 0
            assert summary["step_diagnostics"] is None
            assert summary["projection_including_diagnostics_wall_seconds"] == 0
            close(math.fsum(x["trajectory"]["trajectory_wall_seconds"] for x in manifest), summary["trajectory_wall_seconds"])
            close(summary["model_and_euler_wall_seconds"], summary["trajectory_wall_seconds"])
            residual = 0.0
        else:
            rows = [json.loads(line) for line in check_record(summary["step_diagnostics"]).read_text().splitlines()]
            assert summary["diagnostic_steps"] == len(rows) == 100
            assert all(x["trajectory"] is None for x in manifest)
            assert [x["step_index"] for x in rows] == list(range(100))
            for k, row in enumerate(rows):
                assert row["current"] == req["time_grid"][k] and row["following"] == req["time_grid"][k+1]
                assert row["stage2_forward_calls"] == 125
                assert row["projection"]["cohort_count"] == 1000
                assert row["projection"]["coefficients_are_shared_over_all_images_and_channels"]
            forward = math.fsum(x["model_and_euler_wall_seconds"] for x in rows)
            control = math.fsum(x["projection_and_diagnostics_wall_seconds"] for x in rows)
            close(forward, summary["model_and_euler_wall_seconds"])
            close(control, summary["projection_including_diagnostics_wall_seconds"])
            residual = summary["trajectory_wall_seconds"] - forward - control
            assert residual >= -1e-9
        assert summary["sampling_wall_including_output_seconds"] >= summary["trajectory_wall_seconds"]
        assert jobs[name]["outer_wall_seconds"] >= summary["total_wall_seconds_before_summary"]
        summaries[name] = summary
        reports[name] = {
            "mode": mode, "cuda_visible_devices": jobs[name]["env_overrides"]["CUDA_VISIBLE_DEVICES"],
            "device_name": inp["cuda_device"], "completed_samples": 1000, "paired_batches": 125,
            "archived_sources_rehashed": len(req["sources"]),
            "trajectory_wall_seconds": summary["trajectory_wall_seconds"],
            "model_euler_wall_seconds": summary["model_and_euler_wall_seconds"],
            "projection_and_diagnostics_wall_seconds": summary["projection_including_diagnostics_wall_seconds"],
            "trajectory_misc_included_wall_seconds": residual,
            "decode_uint8_wall_seconds": summary["decode_and_uint8_wall_seconds"],
            "output_write_hash_progress_wall_seconds_OVERLAPS_ball_trajectory": summary["output_write_hash_and_progress_wall_seconds"],
            "worker_outer_wall_seconds": jobs[name]["outer_wall_seconds"],
            "peak_sampling": summary["peak_sampling"],
            "source_and_identity_signature": signature["source_hashes"],
        }
    t0, ts = summaries["official100"]["trajectory_wall_seconds"], summaries["spatial100"]["trajectory_wall_seconds"]
    k = max(100, math.ceil(100 * ts / t0))
    calibration_cost = read_json(HERE.parent / "spectral_energy_audit_v1/summary.json")["cost"]
    result = {
        "complete": True, "audit_scope": "Frozen runtime metadata, code snapshots, stored noise/RNG evidence, and IDs/labels only",
        "review_script_sha256": sha(__file__), "execution_sha256": sha(S / "sample_jobs_execution.json"),
        "input_metadata_sha256": preserved_hashes,
        "all_three_complete_exit_zero": True, "all_three_recorded_identities_match": True,
        "common_input": common_input, "common_large_artifact_identities": common_request["identities"],
        "first_noise_and_final_rng_arrays_exact": True,
        "complete_noise_hash_records_exact": True,
        "paired_batch_noise_and_label_hashes": 125,
        "final_archive_id_label_values_checked": all_id_values,
        "image_arr0_read": False, "sample_pixels_hashed_or_scored": False,
        "fid_files_opened_or_metrics_computed": False,
        "full_cuda_noise_regenerated": False,
        "large_checkpoint_decoder_and_reference_payload_rehashed": False,
        "arms": reports,
        "cost_rule": {
            "plan_rule": plan["cost_rule"], "trajectory_ratio_spatial_over_official": ts/t0,
            "independent_K_from_frozen_rule": k, "additional_official_steps_required_by_this_point_estimate": k > 100,
            "same_nfe_does_not_mean_same_cost": True,
            "matching_scope": "Recorded 1K inference trajectory wall on separate same-model 4090 cards; no causal speedup claim, exact time-equality claim, or total-cost success claim.",
        },
        "preparation_cost_limit": {
            "historical_real_encoder_and_original_bank_preparation_full_outer_cost_closed": False,
            "existing_reference_not_zero_acquisition_cost": True,
            "known_current_two_bank_three_arm_spectral_audit_cost": calibration_cost,
            "known_audit_is_not_minimal_reference_only_cost": True,
            "future_total_cost_comparison_requires": "Recover unresolved historical preparation or state its bound and define amortized sample count; do not equate inference K=100 with total-cost completion.",
        },
        "accounting": "Ball trajectory includes control diagnostics and step progress writes; write breakdown must not be added again. CUDA event spans are not aggregate kernel busy time.",
        "review_cost": {"wall_seconds_before_final_write": time.perf_counter()-started,
                        "cpu_seconds_before_final_write": time.process_time()-cpu,
                        "gpu_calls": 0, "experiments_started": 0},
    }
    for name in ARMS:
        for filename, digest in preserved_hashes[name].items():
            assert sha(S / name / filename) == digest
    (HERE / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "K": k, "trajectory_ratio": ts/t0, "review_cost": result["review_cost"]}, indent=2))


if __name__ == "__main__":
    main()
