"""Audit pairing, switch boundaries, and EMA resets in the SPC study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.evaluate_rae_spc_multiseed import (
    DEFAULT_SEEDS,
    branch_name,
)


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_spc_multiseed_v1"
FINGERPRINT_FIELDS = ("images_sha256", "labels_sha256", "time_sha256", "noise_sha256", "clean_sha256")
PAIR_MANIFEST_FIELDS = (
    "global_seed",
    "cache_order_seed",
    "global_batch_size",
    "grad_accum_steps",
    "ema_reset_step",
    "subspace_path",
    "subspace_rank",
    "basis_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_metrics(path: Path) -> tuple[list[dict[str, object]], bool]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )
    return rows, finite


def checkpoint_small_state(path: Path, *, check_ema: bool) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    ema_exact = None
    if check_ema:
        ema_exact = bool(
            payload["model"].keys() == payload["ema"].keys()
            and all(
                torch.equal(payload["model"][key], payload["ema"][key])
                for key in payload["model"]
            )
        )
    state = {
        "step": int(payload["step"]),
        "branch_start_step": int(payload["branch_start_step"]),
        "scheduler": payload["scheduler"],
        "rng_cpu": payload["rng_cpu"].clone(),
        "rng_cuda": [value.clone() for value in payload["rng_cuda"]],
        "ema_model_exact": ema_exact,
    }
    del payload
    gc.collect()
    return state


def small_state_equal(left: dict[str, object], right: dict[str, object]) -> dict[str, bool]:
    return {
        "step": left["step"] == right["step"],
        "branch_start_step": left["branch_start_step"] == right["branch_start_step"],
        "scheduler": left["scheduler"] == right["scheduler"],
        "rng_cpu": torch.equal(left["rng_cpu"], right["rng_cpu"]),
        "rng_cuda": len(left["rng_cuda"]) == len(right["rng_cuda"])
        and all(
            torch.equal(a, b) for a, b in zip(left["rng_cuda"], right["rng_cuda"])
        ),
    }


def audit_seed(results: Path, seed: int, endpoint: int, switch_step: int) -> dict[str, object]:
    branches = {
        condition: results / branch_name(seed, condition, endpoint, switch_step)
        for condition in ("static", "spc")
    }
    manifests = {
        condition: json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
        for condition, branch in branches.items()
    }
    fingerprints = {
        condition: json.loads((branch / "pair_fingerprint.json").read_text(encoding="utf-8"))
        for condition, branch in branches.items()
    }
    metrics = {}
    finite = {}
    for condition, branch in branches.items():
        metrics[condition], finite[condition] = _finite_metrics(branch / "metrics.jsonl")
    pair_manifest = {
        field: manifests["static"].get(field) == manifests["spc"].get(field)
        for field in PAIR_MANIFEST_FIELDS
    }
    pair_fingerprint = {
        field: fingerprints["static"].get(field) == fingerprints["spc"].get(field)
        for field in FINGERPRINT_FIELDS
    }
    static_modes = [row.get("path_mode") for row in metrics["static"]]
    spc_modes = [row.get("path_mode") for row in metrics["spc"]]
    switch_counts = {
        "static_rows": len(static_modes),
        "static_mode_rows": static_modes.count("static"),
        "spc_rows": len(spc_modes),
        "spc_annealed_rows": spc_modes.count("annealed"),
        "spc_static_rows": spc_modes.count("static"),
    }
    switch_exact = switch_counts == {
        "static_rows": endpoint // 10,
        "static_mode_rows": endpoint // 10,
        "spc_rows": endpoint // 10,
        "spc_annealed_rows": switch_step // 10,
        "spc_static_rows": (endpoint - switch_step) // 10,
    }
    checkpoint_checks = {}
    for step in (switch_step, endpoint):
        states = {
            condition: checkpoint_small_state(
                branch / "checkpoints" / f"step-{step:07d}.pt",
                check_ema=step == switch_step,
            )
            for condition, branch in branches.items()
        }
        checkpoint_checks[str(step)] = {
            "paired_small_state": small_state_equal(states["static"], states["spc"]),
            "static_ema_reset_exact": states["static"]["ema_model_exact"],
            "spc_ema_reset_exact": states["spc"]["ema_model_exact"],
        }
    source_hashes = {
        condition: file_sha256(branch / "train_rae_layerwise_path.py")
        for condition, branch in branches.items()
    }
    checks = {
        "manifests_paired": all(pair_manifest.values()),
        "first_batch_paired": all(pair_fingerprint.values()),
        "metrics_finite": all(finite.values()),
        "switch_counts_exact": switch_exact,
        "training_source_equal": len(set(source_hashes.values())) == 1,
        "checkpoint_pairing": all(
            all(row["paired_small_state"].values()) for row in checkpoint_checks.values()
        ),
        "ema_reset_exact": all(
            checkpoint_checks[str(switch_step)][key]
            for key in ("static_ema_reset_exact", "spc_ema_reset_exact")
        ),
    }
    return {
        "seed": seed,
        "pass": bool(all(checks.values())),
        "checks": checks,
        "manifest_fields": pair_manifest,
        "fingerprint_fields": pair_fingerprint,
        "switch_counts": switch_counts,
        "checkpoint_checks": checkpoint_checks,
        "source_hashes": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--endpoint", type=int, default=5000)
    parser.add_argument("--switch-step", type=int, default=2000)
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    rows = [audit_seed(results, seed, args.endpoint, args.switch_step) for seed in seeds]
    report = {"pass": bool(all(row["pass"] for row in rows)), "seeds": rows}
    output = results / "evaluation" / "spc_training_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
