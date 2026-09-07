#!/usr/bin/env python3
"""Prepare three CPU merge manifests after all four cohorts and cost decisions.

This helper never starts merging, model inference, or FID. Its output provides
explicit argv for the frozen merger. The cost release is written after reviewing
complete sampling/cost evidence; it is not a user-permission mechanism.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.merge_raev2_1k_extension import artifact, checked_file, require, write_json  # noqa: E402

SEEDS = [202609171, 202609172, 202609173, 202609174]
CONFIG_SHA = "3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342"
CHECKPOINT_SHA = "723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a"


def source_record(directory, *, old_records=None):
    directory = Path(directory).resolve()
    record = {}
    for name in ("request", "summary", "batch_manifest", "samples"):
        filename = "samples.npz" if name == "samples" else name + ".json"
        path = directory / filename
        if old_records is not None:
            checked_file(old_records[filename])
            require(Path(old_records[filename]["path"]).resolve() == path, "old artifact path mismatch")
        record[name] = artifact(path)
    request = json.loads((directory / "request.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    require(summary.get("complete") is True and summary.get("samples") == 1000, "source block is incomplete")
    expected = {name: request[name] for name in ("mode", "num_steps", "batch_size")}
    require(expected["batch_size"] == 8 and request["sample_count"] == 1000, "fixed 1K/B8 source required")
    if "config_sha256" in request:
        require(request["config_sha256"] == CONFIG_SHA and request["checkpoint_sha256"] == CHECKPOINT_SHA,
                "legacy configuration/backbone changed")
        expected.update(config_sha256=CONFIG_SHA, checkpoint_sha256=CHECKPOINT_SHA)
    else:
        identities = request["identities"]
        require(identities["config"]["sha256"] == CONFIG_SHA
                and identities["baseline_checkpoint"]["sha256"] == CHECKPOINT_SHA, "native configuration/backbone changed")
        expected.update({"identities.config.sha256": CONFIG_SHA,
                         "identities.baseline_checkpoint.sha256": CHECKPOINT_SHA})
    record["expected_request_fields"] = expected
    return record


def prepare(root_plan_path, root_plan_sha, cost_release_path, cost_release_sha, output_dir):
    root_plan_path = checked_file({"path": str(root_plan_path), "sha256": root_plan_sha})
    cost_release_path = checked_file({"path": str(cost_release_path), "sha256": cost_release_sha})
    plan = json.loads(root_plan_path.read_text())
    release = json.loads(cost_release_path.read_text())
    require(plan.get("protocol") == "raev2_mild_negative_fixed_1k_to_5k_extension_v1"
            and plan.get("source_freeze_complete") is True and plan.get("seeds") == SEEDS, "wrong root plan")
    require(release.get("complete") is True and release.get("frozen_before_fid") is True
            and release.get("fid_started") is False and release.get("root_plan_sha256") == root_plan_sha,
            "cost decision is incomplete, not pre-FID, or bound to a different root plan")
    require(bool(release.get("evidence")), "cost release must bind its actual evidence")
    for record in release["evidence"]:
        checked_file(record)
    workers, jobs = [], {}
    for cohort in plan["cohorts"]:
        index = cohort["index"]
        state_path = Path(plan["output"]) / f"cohort_{index}/execution.json"
        state = json.loads(state_path.read_text())
        require(state.get("complete") is True and state.get("terminal") is True and state.get("fid_started") is False,
                f"cohort {index} is not terminal successful pre-FID")
        require(state["plan_sha256"] == root_plan_sha and state["seed"] == SEEDS[index-1], "worker input identity mismatch")
        require([job["name"] for job in state["jobs"]] == [job["name"] for job in cohort["jobs"]], "worker jobs incomplete")
        for job, expected in zip(state["jobs"], cohort["jobs"]):
            require(job.get("exit_code") == 0 and job["argv"] == expected["argv"]
                    and job["output_dir"] == expected["output_dir"], "worker command changed or failed")
            checked_file(job["summary"])
            summary = json.loads(Path(job["summary"]["path"]).read_text())
            require(summary.get("complete") is True, "sampler did not finish")
        workers.append(artifact(state_path))
        jobs[index] = {job["name"]: job["output_dir"] for job in state["jobs"]}
    require(set(jobs) == {1, 2, 3, 4}, "missing or duplicated cohort")
    baselines = release["reflection_baselines"]
    require([row["cohort_index"] for row in baselines] == list(range(5)), "reflection cost baseline needs all five ordered cohorts")
    steps = {row["num_steps"] for row in baselines}
    require(len(steps) == 1 and next(iter(steps)) >= 201, "reflection baseline must retain one cost-selected K >= 201")
    steps = next(iter(steps))
    layouts = {
        "legacy": {"seed": 202609066, "arms": {
            "official100": ("legacy_official100", "legacy_official100"),
            "global_proximal100": ("global_proximal100", "global_proximal100"),
            "energy100": ("legacy_energy100", "legacy_energy100")}},
        "native_global": {"seed": 202609121, "arms": {
            "official100": ("spatial_official100", "native_official100"),
            "global100": ("native_global100", "native_global100")}},
        "reflection": {"seed": 202609131, "arms": {
            "official100": ("reflection_official100", "native_official100"),
            "reflection100": ("reflection100", "reflection100")}},
    }
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "preparation output exists")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "request.json", {"root_plan": artifact(root_plan_path),
               "cost_release": artifact(cost_release_path), "worker_execution": workers,
               "source": artifact(__file__), "created_at_utc": datetime.now(timezone.utc).isoformat()})
    outputs = {}
    for family, layout in layouts.items():
        blocks = []
        for index in range(5):
            block = {"id": "old" if index == 0 else f"new{index}",
                     "role": "screen" if index == 0 else "confirmation",
                     "seed": layout["seed"] if index == 0 else SEEDS[index-1], "arms": {}}
            for arm, (old_name, new_name) in layout["arms"].items():
                old = plan["historical_1k"][old_name]
                block["arms"][arm] = source_record(old["directory"] if index == 0 else jobs[index][new_name],
                                                   old_records=old["artifacts"] if index == 0 else None)
            if family == "reflection":
                source = source_record(baselines[index]["directory"])
                require(source["expected_request_fields"]["mode"] == "official"
                        and source["expected_request_fields"]["num_steps"] == steps, "selected baseline source differs from cost release")
                block["arms"][f"official{steps}"] = source
            for extra in release.get("extra_baselines", []):
                if extra["family"] == family:
                    require(extra["arm"] not in block["arms"] and len(extra["directories"]) == 5, "invalid extra cost baseline")
                    source = source_record(extra["directories"][index])
                    require(source["expected_request_fields"]["mode"] == "official"
                            and source["expected_request_fields"]["num_steps"] == extra["num_steps"], "extra baseline source differs")
                    block["arms"][extra["arm"]] = source
            blocks.append(block)
        family_plan = {"protocol": "raev2_paired_1k_extension_merge_v1", "family": family,
                       "invariant_request_fields": ["mode", "num_steps", "batch_size"],
                       "root_plan": artifact(root_plan_path), "cost_release": artifact(cost_release_path), "blocks": blocks}
        path = output_dir / f"{family}_merge_plan.json"
        write_json(path, family_plan)
        outputs[family] = {"plan": artifact(path), "merge_argv": [plan["python"],
                           str(ROOT / "experiments/merge_raev2_1k_extension.py"), "--plan", str(path),
                           "--output-dir", str(output_dir / f"{family}_merged")]}
    summary = {"complete": True, "family_count": 3, "method_count": 4, "families": outputs,
               "cost_release": artifact(cost_release_path),
               "reported_cost_match": release.get("matched"),
               "missing_historical_costs": release.get("missing_historical_costs"),
               "cost_limitations": release.get("limitations", []),
               "gpu_model_calls": 0, "merging_started": False, "fid_started": False}
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-plan", type=Path, required=True)
    parser.add_argument("--root-plan-sha256", required=True)
    parser.add_argument("--cost-release", type=Path, required=True)
    parser.add_argument("--cost-release-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root_plan, args.root_plan_sha256, args.cost_release,
                             args.cost_release_sha256, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
