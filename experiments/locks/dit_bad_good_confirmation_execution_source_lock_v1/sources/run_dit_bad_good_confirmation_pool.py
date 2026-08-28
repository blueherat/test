#!/usr/bin/env python3
"""Generate the frozen 300-trajectory DiT bad/good confirmation cohort.

The runner assigns one seed at a time to each selected GPU.  Every seed emits
the three frozen classes in the fixed order 207,602,795.  Existing completed
seed directories are revalidated by the trace program; partial directories are
never overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE_SOURCE = ROOT / "experiments/trace_dit_imagenet256_custom_batch.py"
LOCK_ROOT = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v3"
EXPECTED_TRACE_SOURCE_SHA256 = "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
EXPECTED_PROTOCOL_IDENTITY = "8b7e1c66b106f1bf6862d11f803ec48863004720104e10b5808adaf9d2d9b345"
CLASSES = (207, 602, 795)
SEEDS = tuple(range(30, 130))
DEFAULT_DATA_ROOT = Path(
    os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae")
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT
    / "cross_scale_evidence/dit_bad_good_confirmation_v1_custom_traces_cfg_locked"
)
DEFAULT_DIT_ROOT = DEFAULT_DATA_ROOT / "baselines/DiT"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
DEFAULT_VAE = (
    Path.home()
    / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
    / VAE_REVISION
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_frozen_inputs() -> dict[str, Any]:
    if sha256_file(TRACE_SOURCE) != EXPECTED_TRACE_SOURCE_SHA256:
        raise RuntimeError("trace source changed after confirmation protocol freeze")
    protocol = load_json(LOCK_ROOT / "candidate_protocol.json")
    completion = load_json(LOCK_ROOT / "completion.json")
    if (
        completion.get("complete") is not True
        or protocol.get("identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
        or completion.get("protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("candidate confirmation lock identity is invalid")
    frozen = protocol["fresh_confirmation"]
    frozen_seeds = tuple(
        range(frozen["seeds"]["start_inclusive"], frozen["seeds"]["stop_inclusive"] + 1)
    )
    if tuple(frozen["classes"]) != CLASSES or frozen_seeds != SEEDS:
        raise RuntimeError("runner cohort differs from the candidate confirmation lock")
    return protocol


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--gpus must list unique device identifiers")
    return result


def run_seed(
    seed: int,
    gpu: str,
    output_root: Path,
    dit_root: Path,
    vae_snapshot: Path,
    log_lock: threading.Lock,
) -> dict[str, Any]:
    outdir = output_root / f"confirmation_v1_seed{seed:03d}"
    logdir = output_root / "_runner_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"seed{seed:03d}.log"
    command = [
        sys.executable,
        str(TRACE_SOURCE),
        "--classes",
        ",".join(str(value) for value in CLASSES),
        "--seed",
        str(seed),
        "--dit-root",
        str(dit_root),
        "--vae-snapshot",
        str(vae_snapshot),
        "--outdir",
        str(outdir),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"started_unix": started, "gpu": gpu, "command": command}) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        finished = time.time()
        log.write(
            json.dumps(
                {
                    "finished_unix": finished,
                    "elapsed_seconds": finished - started,
                    "returncode": completed.returncode,
                }
            )
            + "\n"
        )
    if completed.returncode != 0:
        raise RuntimeError(f"seed {seed} failed on GPU {gpu}; inspect {log_path}")
    completion_path = outdir / "completion.json"
    if not completion_path.is_file() or load_json(completion_path).get("output_count") != 8:
        # Three endpoint PNGs plus trace, protocol and metadata yield eight bound outputs.
        raise RuntimeError(f"seed {seed} output did not pass completion audit: {outdir}")
    with log_lock:
        print(
            json.dumps(
                {
                    "seed": seed,
                    "gpu": gpu,
                    "elapsed_seconds": round(finished - started, 3),
                    "completed": True,
                }
            ),
            flush=True,
        )
    return {
        "seed": seed,
        "gpu": gpu,
        "elapsed_seconds": finished - started,
        "output": str(outdir),
    }


def worker(
    gpu: str,
    seeds: tuple[int, ...],
    output_root: Path,
    dit_root: Path,
    vae_snapshot: Path,
    log_lock: threading.Lock,
) -> list[dict[str, Any]]:
    return [
        run_seed(seed, gpu, output_root, dit_root, vae_snapshot, log_lock)
        for seed in seeds
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--vae-snapshot", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol = validate_frozen_inputs()
    output_root = args.output_root.expanduser().absolute()
    dit_root = args.dit_root.expanduser().absolute().resolve()
    vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    if os.path.lexists(output_root) and (not output_root.is_dir() or output_root.is_symlink()):
        raise RuntimeError(f"output root must be a real directory: {output_root}")
    if output_root in (ROOT, ROOT.parent, Path("/")):
        raise RuntimeError("refusing broad output root")
    assignment = {gpu: SEEDS[index :: len(args.gpus)] for index, gpu in enumerate(args.gpus)}
    plan = {
        "status": "FROZEN_EXECUTION_PLAN",
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "trace_source_sha256": sha256_file(TRACE_SOURCE),
        "classes": list(CLASSES),
        "seeds": list(SEEDS),
        "gpus": list(args.gpus),
        "assignment": {gpu: list(seeds) for gpu, seeds in assignment.items()},
        "output_root": str(output_root),
        "dit_root": str(dit_root),
        "vae_snapshot": str(vae_snapshot),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "execution_plan.json"
    if plan_path.exists():
        if load_json(plan_path) != plan:
            raise RuntimeError("existing execution plan differs from this invocation")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_lock = threading.Lock()
    all_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(
                worker,
                gpu,
                seeds,
                output_root,
                dit_root,
                vae_snapshot,
                log_lock,
            ): gpu
            for gpu, seeds in assignment.items()
        }
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
    all_results.sort(key=lambda item: item["seed"])
    completion = {
        "complete": True,
        "candidate_protocol_identity_sha256": protocol["identity_sha256"],
        "trace_source_sha256": sha256_file(TRACE_SOURCE),
        "seed_count": len(all_results),
        "trajectory_count": len(all_results) * len(CLASSES),
        "execution_plan_sha256": sha256_file(plan_path),
        "results": all_results,
    }
    completion_path = output_root / "pool_completion.json"
    if completion_path.exists():
        raise RuntimeError(f"refusing to overwrite pool completion: {completion_path}")
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: completion[key] for key in ("complete", "seed_count", "trajectory_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
