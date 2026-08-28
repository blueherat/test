#!/usr/bin/env python3
"""Strict orchestration wrapper for the released FKC EDM2 image sampler.

This file intentionally does *not* reimplement or repair the sampler.  It
executes the frozen upstream ``generate_images.py`` through ``runpy`` and calls
its original ``generate_images``/``edm_sampler`` objects.  In particular, FKC
mode retains all three unusual behaviours of the released code:

* the first per-particle ``randint`` selects one class shared by the batch;
* a second per-particle ``randint`` is evaluated and then overwritten;
* with 64 steps, systematic resampling runs 63 times and only particle slot 0
  is written to disk after each batch.

The wrapper supports random-class smoke sampling only.  ``--class`` is
deliberately rejected: the upstream function's fixed-class argument is not
honoured by its released implementation, so exposing it here would claim a
reproduction guarantee that we do not have.

Completed output directories are immutable.  Repeating the exact command
validates the manifest, completion record, sources, checkpoints, and every PNG
hash, then exits without sampling again.  A non-empty incomplete directory is
refused rather than silently resumed or overwritten.

Upstream source:
https://github.com/martaskrt/fkc-diffusion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import runpy
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import PIL
import torch
from PIL import Image


FKC_REVISION = "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6"
UPSTREAM_ENTRY_RELATIVE = Path("applications/images/edm2/generate_images.py")
UPSTREAM_ENTRY_SHA256 = "7bcc0b762ec3d47c15c556765678aca3212294ad80f2e136980fdd386ce16745"
CHECKPOINT_RELATIVE = Path("applications/images/edm2/checkpoints")

PRESET = "edm2-img512-xs-guid-fid"
PARTICLE_BATCH_SIZE = 8
CFG_BATCH_SIZE = 32
DEFAULT_STEPS = 64
DEFAULT_CHURN = 40.0
DEFAULT_GUIDANCE = 1.4
IMAGE_SIZE = (512, 512)

MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
MANIFEST_SCHEMA = 1
COMPLETION_SCHEMA = 1


@dataclass(frozen=True)
class CheckpointSpec:
    role: str
    filename: str
    byte_count: int
    sha256: str


CONDITIONAL_CHECKPOINT = CheckpointSpec(
    role="conditional",
    filename="edm2-img512-xs-2147483-0.045.pkl",
    byte_count=249_566_482,
    sha256="27a6c6eaf697b68a74f9c7b72e82f91c2e898d22f629b4546c053865cfe3da68",
)
UNCONDITIONAL_CHECKPOINT = CheckpointSpec(
    role="unconditional",
    filename="edm2-img512-xs-uncond-2147483-0.045.pkl",
    byte_count=248_541_796,
    sha256="2ea8fffdf0e32d68da3b4050e77c3f9defb1a50a2c9a4a845eb8f927355dea08",
)


def parse_int_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges, preserving order."""

    result: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending range is not allowed: {part}")
            result.extend(range(start, stop + 1))
        else:
            try:
                result.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid seed specification: {part}") from exc
    if not result:
        raise argparse.ArgumentTypeError("seed specification is empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seed specification contains duplicates")
    if any(seed < 0 or seed >= 1 << 32 for seed in result):
        raise argparse.ArgumentTypeError("seeds must be in [0, 2^32 - 1]")
    return tuple(result)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def git_output(root: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        stderr=subprocess.PIPE,
        text=not binary,
    )


def tracked_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash the path and current bytes of every Git-tracked file."""

    raw = git_output(root, "ls-files", "-z", binary=True)
    assert isinstance(raw, bytes)
    relative_paths = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    if not relative_paths:
        raise RuntimeError(f"no tracked files found in {root}")
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        relative_bytes = relative.encode("utf-8")
        contents = (root / relative).read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest(), len(relative_paths)


def validate_upstream_contract(source: str) -> None:
    """Fail closed if the pinned entry no longer exposes the audited quirks."""

    required_fragments = (
        "if fkc and i != num_steps - 1:",
        "choice, _ = sample_cat_sys(x_next.shape[0], a_next)",
        "class_idx = rnd.randint(",
        "r.labels = torch.eye(net.label_dim, device=device)[",
        "r.labels[:, :] = 0",
        "r.labels[:, class_idx[0]] = 1",
        "if i > 0 and fkc:",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"pinned upstream behaviour check failed; missing fragments: {missing}")


def validate_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"not an FKC Git checkout: {root}")
    revision = str(git_output(root, "rev-parse", "HEAD")).strip()
    if revision != FKC_REVISION:
        raise RuntimeError(f"wrong FKC revision: {revision} != {FKC_REVISION}")

    dirty = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if dirty.returncode != 0:
        raise RuntimeError(
            "FKC checkout has modified tracked files; refusing an unfrozen source tree"
        )

    untracked_raw = git_output(
        root, "ls-files", "--others", "--exclude-standard", "-z", binary=True
    )
    assert isinstance(untracked_raw, bytes)
    untracked = [item.decode("utf-8") for item in untracked_raw.split(b"\0") if item]
    checkpoint_prefix = CHECKPOINT_RELATIVE.as_posix() + "/"
    unexpected = [path for path in untracked if not path.startswith(checkpoint_prefix)]
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise RuntimeError(f"unexpected untracked files can shadow frozen source: {preview}")

    entry = root / UPSTREAM_ENTRY_RELATIVE
    if not entry.is_file():
        raise FileNotFoundError(f"missing upstream entry point: {entry}")
    entry_digest = sha256_file(entry)
    if entry_digest != UPSTREAM_ENTRY_SHA256:
        raise RuntimeError(
            f"wrong upstream entry SHA-256: {entry_digest} != {UPSTREAM_ENTRY_SHA256}"
        )
    validate_upstream_contract(entry.read_text(encoding="utf-8"))
    tracked_digest, tracked_count = tracked_tree_sha256(root)
    tree = str(git_output(root, "rev-parse", "HEAD^{tree}")).strip()
    return {
        "root": str(root),
        "revision": revision,
        "git_tree": tree,
        "tracked_file_count": tracked_count,
        "tracked_files_sha256": tracked_digest,
        "entry": str(entry.resolve()),
        "entry_sha256": entry_digest,
    }


def validate_checkpoint(path: Path, spec: CheckpointSpec) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing official {spec.role} checkpoint: {path}\n"
            "Run experiments/download_cross_scale_baselines.sh first."
        )
    byte_count = path.stat().st_size
    if byte_count != spec.byte_count:
        raise RuntimeError(
            f"wrong {spec.role} checkpoint size: {byte_count:,} != {spec.byte_count:,}"
        )
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise RuntimeError(
            f"wrong {spec.role} checkpoint SHA-256: {digest} != {spec.sha256}"
        )
    return {
        "role": spec.role,
        "path": str(path),
        "filename": spec.filename,
        "bytes": byte_count,
        "sha256": digest,
    }


def seed_spec_for_upstream(seeds: Sequence[int]) -> str:
    return ",".join(str(seed) for seed in seeds)


def image_relative_path(seed: int) -> str:
    group = seed // 1_000 * 1_000
    return f"{group:06d}/{seed:06d}.png"


def saved_seeds(mode: str, seeds: Sequence[int]) -> tuple[int, ...]:
    if mode == "cfg":
        return tuple(seeds)
    if mode == "fkc":
        return tuple(seeds[offset] for offset in range(0, len(seeds), PARTICLE_BATCH_SIZE))
    raise AssertionError(mode)


def expected_image_paths(mode: str, seeds: Sequence[int]) -> tuple[str, ...]:
    return tuple(image_relative_path(seed) for seed in saved_seeds(mode, seeds))


def execution_batch_size(mode: str) -> int:
    if mode == "fkc":
        return PARTICLE_BATCH_SIZE
    if mode == "cfg":
        return CFG_BATCH_SIZE
    raise AssertionError(mode)


def canonical_wrapper_command(args: argparse.Namespace, script: Path) -> list[str]:
    command = [
        sys.executable,
        str(script.resolve()),
        "--mode",
        args.mode,
        "--seeds",
        seed_spec_for_upstream(args.seeds),
        "--resample-seed",
        str(args.resample_seed),
        "--steps",
        str(args.steps),
        "--churn",
        format(args.churn, ".17g"),
        "--guidance",
        format(args.guidance, ".17g"),
        "--fkc-root",
        str(args.fkc_root.resolve()),
        "--checkpoint-dir",
        str(args.checkpoint_dir.resolve()),
        "--outdir",
        str(args.outdir.resolve()),
    ]
    return command


def upstream_cli_argument_projection(
    args: argparse.Namespace,
    upstream: dict[str, Any],
    checkpoints: Sequence[dict[str, Any]],
) -> list[str]:
    by_role = {item["role"]: item for item in checkpoints}
    command = [
        sys.executable,
        upstream["entry"],
        f"--net={by_role['conditional']['path']}",
        f"--gnet={by_role['unconditional']['path']}",
        f"--outdir={args.outdir.resolve()}",
        "--subdirs",
        f"--seeds={seed_spec_for_upstream(args.seeds)}",
        f"--batch={execution_batch_size(args.mode)}",
        f"--steps={args.steps}",
        f"--guidance={format(args.guidance, '.17g')}",
        f"--S_churn={format(args.churn, '.17g')}",
    ]
    if args.mode == "fkc":
        command.append("--fkc")
    return command


def build_identity(args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve()
    upstream = validate_repository(args.fkc_root)
    checkpoints = [
        validate_checkpoint(
            args.checkpoint_dir / CONDITIONAL_CHECKPOINT.filename,
            CONDITIONAL_CHECKPOINT,
        ),
        validate_checkpoint(
            args.checkpoint_dir / UNCONDITIONAL_CHECKPOINT.filename,
            UNCONDITIONAL_CHECKPOINT,
        ),
    ]
    wrapper_command = canonical_wrapper_command(args, script)
    upstream_arguments = upstream_cli_argument_projection(args, upstream, checkpoints)
    return {
        "schema": MANIFEST_SCHEMA,
        "runner": "reproduce_fkc_edm2",
        "wrapper_source": {"path": str(script), "sha256": sha256_file(script)},
        "upstream": upstream,
        "checkpoints": checkpoints,
        "protocol": {
            "preset": PRESET,
            "mode": args.mode,
            "random_classes": True,
            "fixed_class_supported": False,
            "seeds": list(args.seeds),
            "resample_seed": args.resample_seed,
            "execution_batch_size": execution_batch_size(args.mode),
            "fkc_particle_batch_size": PARTICLE_BATCH_SIZE,
            "num_steps": args.steps,
            "S_churn": args.churn,
            "guidance": args.guidance,
            "subdirs": True,
            "expected_resampling_events_per_batch": args.steps - 1 if args.mode == "fkc" else 0,
            "save_policy": "particle_slot_0_only" if args.mode == "fkc" else "all_slots",
            "expected_saved_seeds": list(saved_seeds(args.mode, args.seeds)),
        },
        "wrapper_command": wrapper_command,
        "wrapper_command_sha256": sha256_json(wrapper_command),
        "upstream_cli_argument_projection": upstream_arguments,
        "upstream_cli_argument_projection_sha256": sha256_json(upstream_arguments),
        "rng_contract": {
            "exact_reproduction_entrypoint": "wrapper_command",
            "upstream_cli_is_rng_equivalent": args.mode == "cfg",
            "reason": (
                None
                if args.mode == "cfg"
                else (
                    "the released CLI has no option for the global CPU torch RNG used by "
                    "systematic resampling"
                )
            ),
        },
        "audited_upstream_behaviour": {
            "fkc_random_label_first_randint_shared_across_batch": True,
            "fkc_second_randint_evaluated_then_overwritten": True,
            "fkc_resampling_before_every_nonterminal_step": True,
            "fkc_only_first_particle_saved": True,
        },
    }


def list_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def inspect_png(path: Path) -> tuple[str, list[int]]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = list(image.size)
    if mode != "RGB" or tuple(size) != IMAGE_SIZE:
        raise RuntimeError(f"unexpected PNG properties for {path}: mode={mode}, size={size}")
    return mode, size


def collect_output_records(
    outdir: Path,
    expected_relative_paths: Sequence[str],
    *,
    allow_metadata: bool,
) -> list[dict[str, Any]]:
    metadata_names = {MANIFEST_NAME, COMPLETION_NAME} if allow_metadata else set()
    files = list_files(outdir)
    actual_images: dict[str, Path] = {}
    unexpected: list[str] = []
    for path in files:
        relative = path.relative_to(outdir).as_posix()
        if relative in metadata_names:
            continue
        if path.suffix.lower() != ".png":
            unexpected.append(relative)
        else:
            actual_images[relative] = path
    if unexpected:
        raise RuntimeError(f"unexpected non-PNG outputs: {unexpected[:5]}")

    expected = set(expected_relative_paths)
    actual = set(actual_images)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"output path mismatch; missing={missing[:5]}, extra={extra[:5]}")

    records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = actual_images[relative]
        mode, size = inspect_png(path)
        records.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mode": mode,
                "size": size,
            }
        )
    return records


def validate_completed_output(outdir: Path, identity: dict[str, Any]) -> None:
    manifest_path = outdir / MANIFEST_NAME
    completion_path = outdir / COMPLETION_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError(
            f"non-empty output directory is incomplete; refusing to overwrite or resume: {outdir}"
        )
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported manifest schema")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"manifest is not complete: {manifest_path}")
    if manifest.get("identity") != identity:
        raise RuntimeError("existing output identity differs from the current locked invocation")
    identity_digest = sha256_json(identity)
    if manifest.get("identity_sha256") != identity_digest:
        raise RuntimeError("manifest identity hash is invalid")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise RuntimeError("unsupported completion schema")
    if completion.get("identity_sha256") != identity_digest:
        raise RuntimeError("completion identity hash is invalid")
    manifest_digest = sha256_file(manifest_path)
    if completion.get("manifest_sha256") != manifest_digest:
        raise RuntimeError("completion record does not match manifest bytes")

    expected = expected_image_paths(
        identity["protocol"]["mode"], identity["protocol"]["seeds"]
    )
    records = collect_output_records(outdir, expected, allow_metadata=True)
    if manifest.get("outputs") != records:
        raise RuntimeError("one or more output hashes or PNG properties have changed")
    records_digest = sha256_json(records)
    if manifest.get("outputs_sha256") != records_digest:
        raise RuntimeError("manifest output aggregate hash is invalid")
    if completion.get("outputs_sha256") != records_digest:
        raise RuntimeError("completion output aggregate hash is invalid")
    if completion.get("output_count") != len(records):
        raise RuntimeError("completion output count is invalid")
    print(f"validated completed output: {outdir} ({len(records)} PNG files); no sampling run")


def ensure_single_process() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 1 or rank != 0 or local_rank != 0:
        raise RuntimeError(
            "this strict wrapper requires one process (WORLD_SIZE=1, RANK=0, LOCAL_RANK=0)"
        )
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1 or torch.distributed.get_rank() != 0:
            raise RuntimeError("an incompatible distributed process group is already initialized")


def run_upstream(
    args: argparse.Namespace,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute the pinned upstream objects and return observed batch metadata."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the released FKC EDM2 image sampler")
    ensure_single_process()

    entry = Path(identity["upstream"]["entry"])
    source_dir = entry.parent
    checkpoint_by_role = {item["role"]: item for item in identity["checkpoints"]}
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    namespace: dict[str, Any] | None = None
    initialized_here = False
    batch_records: list[dict[str, Any]] = []
    try:
        os.chdir(source_dir)
        sys.path.insert(0, str(source_dir))

        # This global CPU RNG is what upstream sample_cat_sys() uses.  Seeding
        # before runpy and calling the original objects in this same process is
        # intentional; subprocess seeding would not establish this contract.
        torch.manual_seed(args.resample_seed)
        namespace = runpy.run_path(str(entry), run_name="eqvae_locked_fkc_generate_images")
        upstream_dist = namespace["dist"]
        if not torch.distributed.is_initialized():
            upstream_dist.init()
            initialized_here = True
        device = torch.device("cuda")

        batch_size = execution_batch_size(args.mode)
        image_iter = namespace["generate_images"](
            net=checkpoint_by_role["conditional"]["path"],
            gnet=checkpoint_by_role["unconditional"]["path"],
            encoder=None,
            outdir=str(args.outdir.resolve()),
            subdirs=True,
            seeds=list(args.seeds),
            class_idx=None,
            fkc=args.mode == "fkc",
            max_batch_size=batch_size,
            encoder_batch_size=4,
            verbose=True,
            device=device,
            sampler_fn=namespace["edm_sampler"],
            num_steps=args.steps,
            guidance=args.guidance,
            S_churn=args.churn,
        )
        expected_batches = math.ceil(len(args.seeds) / batch_size)
        if len(image_iter) != expected_batches:
            raise RuntimeError(
                f"upstream produced {len(image_iter)} batches; expected {expected_batches}"
            )

        expected_indices = np.array_split(np.arange(len(args.seeds)), expected_batches)
        expected_seed_batches = [
            [args.seeds[int(index)] for index in indices] for indices in expected_indices
        ]
        for expected_batch_idx, result in enumerate(image_iter):
            observed_seeds = [int(seed) for seed in result.seeds]
            if int(result.batch_idx) != expected_batch_idx:
                raise RuntimeError(
                    f"upstream batch index changed: {int(result.batch_idx)} != {expected_batch_idx}"
                )
            if observed_seeds != expected_seed_batches[expected_batch_idx]:
                raise RuntimeError(
                    "upstream seed batching or order changed: "
                    f"{observed_seeds} != {expected_seed_batches[expected_batch_idx]}"
                )
            if args.mode == "fkc" and len(observed_seeds) != PARTICLE_BATCH_SIZE:
                raise RuntimeError(
                    "FKC particle batch changed size: "
                    f"{len(observed_seeds)} != {PARTICLE_BATCH_SIZE}"
                )
            if result.labels is None:
                raise RuntimeError("expected the class-conditional EDM2 model to return labels")
            class_ids = [
                int(value)
                for value in result.labels.argmax(dim=1).detach().cpu().tolist()
            ]
            if len(class_ids) != len(observed_seeds):
                raise RuntimeError("upstream label count does not match the observed seed count")
            if args.mode == "fkc" and len(set(class_ids)) != 1:
                raise RuntimeError("upstream FKC batch no longer shares its first random class")
            batch_records.append(
                {
                    "batch_idx": int(result.batch_idx),
                    "seeds": observed_seeds,
                    "class_ids": class_ids,
                    "saved_seeds": observed_seeds[:1] if args.mode == "fkc" else observed_seeds,
                    "expected_resampling_events": args.steps - 1 if args.mode == "fkc" else 0,
                }
            )
            del result
        torch.cuda.synchronize()
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        if initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return batch_records


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.fixed_class is not None:
        parser.error(
            "--class is deliberately unsupported: released FKC random-label code ignores a "
            "fixed class and also performs a second overwritten randint"
        )
    if args.resample_seed is None:
        parser.error("--resample-seed is required for every real or dry-run invocation")
    if args.resample_seed < 0 or args.resample_seed >= 1 << 63:
        parser.error("--resample-seed must be in [0, 2^63 - 1]")
    if args.steps < 2:
        parser.error("--steps must be at least 2 for the released Heun sampler")
    if not math.isfinite(args.churn) or args.churn < 0:
        parser.error("--churn must be finite and non-negative")
    if not math.isfinite(args.guidance):
        parser.error("--guidance must be finite")
    if args.mode == "fkc" and len(args.seeds) % PARTICLE_BATCH_SIZE != 0:
        parser.error(
            f"FKC seed count must be divisible by particle batch {PARTICLE_BATCH_SIZE}; "
            f"got {len(args.seeds)}"
        )


def self_test() -> None:
    assert parse_int_spec("0,2,4-6") == (0, 2, 4, 5, 6)
    assert saved_seeds("cfg", tuple(range(9))) == tuple(range(9))
    assert saved_seeds("fkc", tuple(range(16))) == (0, 8)
    assert execution_batch_size("cfg") == 32
    assert execution_batch_size("fkc") == 8
    assert expected_image_paths("fkc", tuple(range(8))) == ("000000/000000.png",)
    payload = {"b": [2, 1], "a": "x"}
    assert sha256_json(payload) == sha256_json({"a": "x", "b": [2, 1]})
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    default_root = Path("/home/zhoushunyu/data/eqvae/baselines/fkc-diffusion")
    parser = argparse.ArgumentParser(
        description="Strict random-class CFG/FKC EDM2 smoke orchestration wrapper."
    )
    parser.add_argument("--mode", choices=("cfg", "fkc"), default="cfg")
    parser.add_argument("--seeds", type=parse_int_spec, default=parse_int_spec("0-7"))
    parser.add_argument(
        "--resample-seed",
        type=int,
        default=None,
        help="Required global torch seed; in FKC mode it controls systematic resampling.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--churn", type=float, default=DEFAULT_CHURN)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--fkc-root", type=Path, default=default_root)
    parser.add_argument("--checkpoint-dir", type=Path, default=default_root / CHECKPOINT_RELATIVE)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--class",
        dest="fixed_class",
        type=int,
        default=None,
        help="Deliberately unsupported; random classes are part of this locked reproduction.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.fixed_class is not None:
            parser.error("--class is deliberately unsupported")
        self_test()
        return 0
    if args.outdir is None:
        parser.error("--outdir is required unless --self-test is used")
    validate_args(args, parser)

    requested_outdir = args.outdir.expanduser().absolute()
    if os.path.lexists(requested_outdir) and requested_outdir.is_symlink():
        raise RuntimeError(f"output directory must not be a symlink: {requested_outdir}")
    args.fkc_root = args.fkc_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.outdir = requested_outdir.resolve()
    identity = build_identity(args)
    identity_digest = sha256_json(identity)

    if args.outdir.exists() and not args.outdir.is_dir():
        raise RuntimeError(f"output path is not a directory: {args.outdir}")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        validate_completed_output(args.outdir, identity)
        return 0

    if args.dry_run:
        summary = {
            "status": "dry-run",
            "identity_sha256": identity_digest,
            "mode": args.mode,
            "seed_count": len(args.seeds),
            "execution_batch_size": execution_batch_size(args.mode),
            "fkc_particle_batch_size": PARTICLE_BATCH_SIZE,
            "expected_output_count": len(saved_seeds(args.mode, args.seeds)),
            "expected_resampling_events_total": (
                len(args.seeds) // PARTICLE_BATCH_SIZE * (args.steps - 1)
                if args.mode == "fkc"
                else 0
            ),
            "outdir": str(args.outdir),
            "wrapper_command_sha256": identity["wrapper_command_sha256"],
            "upstream_cli_argument_projection_sha256": identity[
                "upstream_cli_argument_projection_sha256"
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    if any(args.outdir.iterdir()):
        raise RuntimeError(f"output directory ceased to be empty: {args.outdir}")

    started_at = time.time()
    running_manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "identity": identity,
        "identity_sha256": identity_digest,
        "started_unix": started_at,
    }
    atomic_json_dump(running_manifest, args.outdir / MANIFEST_NAME)
    try:
        batches = run_upstream(args, identity)
        expected = expected_image_paths(args.mode, args.seeds)
        outputs = collect_output_records(args.outdir, expected, allow_metadata=True)
        outputs_digest = sha256_json(outputs)
        finished_at = time.time()
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_digest,
            "started_unix": started_at,
            "finished_unix": finished_at,
            "elapsed_seconds": finished_at - started_at,
            "batches": batches,
            "outputs": outputs,
            "outputs_sha256": outputs_digest,
            "platform": {
                "hostname": socket.gethostname(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "click": distribution_version("click"),
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cudnn": torch.backends.cudnn.version(),
            },
        }
        manifest_path = args.outdir / MANIFEST_NAME
        atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "identity_sha256": identity_digest,
            "manifest_sha256": sha256_file(manifest_path),
            "outputs_sha256": outputs_digest,
            "output_count": len(outputs),
        }
        atomic_json_dump(completion, args.outdir / COMPLETION_NAME)
        validate_completed_output(args.outdir, identity)
    except BaseException as exc:
        failed_manifest = dict(running_manifest)
        failed_manifest.update(
            {
                "status": "failed",
                "failed_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_json_dump(failed_manifest, args.outdir / MANIFEST_NAME)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
