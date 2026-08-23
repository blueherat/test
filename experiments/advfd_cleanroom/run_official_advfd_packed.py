"""Launch the public AdvFD trainer against the local packed ImageNet copy.

The public trainer is imported after the independent implementation was frozen.
This adapter changes only the real-image storage reader and, when explicitly
requested, the post-all-gather parameter-gradient reduction.  All FD losses,
feature models, update ordering, moment tracking, optimizers, checkpoints and
sampling remain in the public implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


EQVAE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_adapter_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--eqvae-official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT
    )
    parser.add_argument("--eqvae-packed-data", type=Path, required=True)
    parser.add_argument(
        "--eqvae-gradient-reduction",
        choices=("official_avg", "paper_sum"),
        default="official_avg",
    )
    parser.add_argument(
        "--eqvae-lr-schedule-total-steps",
        type=int,
        default=None,
        help=(
            "Optional LR-schedule horizon independent of the run stop step, "
            "for exact prefixes of a longer official schedule."
        ),
    )
    parser.add_argument("--eqvae-adapter-manifest", type=Path, required=True)
    return parser.parse_known_args(argv)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_packed_batch_factory(packed_root: Path):
    def build_real_image_batch_fn(args):
        from experiments.raev2_training_core import DeterministicImageNetPacked

        dataset = DeterministicImageNetPacked(
            packed_root,
            split="train",
            image_size=args.img_size,
            augmentation_seed=args.seed,
            horizontal_flip=False,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                drop_last=True,
            )
            if args.world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        state = {"epoch": 0, "iterator": iter(loader)}

        def next_batch():
            try:
                images, labels, _ = next(state["iterator"])
            except StopIteration:
                state["epoch"] += 1
                if sampler is not None:
                    sampler.set_epoch(state["epoch"])
                state["iterator"] = iter(loader)
                images, labels, _ = next(state["iterator"])
            return (
                images.cuda(non_blocking=True),
                labels.cuda(non_blocking=True),
            )

        return next_batch

    return build_real_image_batch_fn


def sum_parameter_gradients(module: torch.nn.Module) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    calls = 0
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            calls += 1
    return calls


@contextlib.contextmanager
def temporary_attribute(instance, name: str, value):
    original = getattr(instance, name)
    setattr(instance, name, value)
    try:
        yield
    finally:
        setattr(instance, name, original)


def build_schedule_horizon_adapter(adjust_learning_rate_fn, total_steps: int):
    """Use a longer LR horizon without changing the trainer's stopping step."""

    if total_steps <= 0:
        raise ValueError("LR schedule total steps must be positive")

    def adjust_learning_rate(optimizer, step, args):
        run_total_steps = args.total_steps
        if total_steps < run_total_steps:
            raise ValueError(
                "LR schedule total steps cannot be shorter than the run: "
                f"schedule={total_steps}, run={run_total_steps}"
            )
        with temporary_attribute(args, "total_steps", total_steps):
            return adjust_learning_rate_fn(optimizer, step, args)

    return adjust_learning_rate


def main() -> None:
    adapter, official_argv = parse_adapter_args(sys.argv[1:])
    official_root = adapter.eqvae_official_root.expanduser().resolve()
    packed_root = adapter.eqvae_packed_data.expanduser().resolve()
    if not (official_root / "main_fd.py").is_file():
        raise FileNotFoundError(f"Official AdvFD checkout not found: {official_root}")
    if not (packed_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Packed ImageNet not found: {packed_root}")

    sys.path.insert(0, str(EQVAE_ROOT))
    sys.path.insert(0, str(official_root))
    import main_fd  # noqa: PLC0415

    main_fd.build_real_image_batch_fn = build_packed_batch_factory(packed_root)
    if adapter.eqvae_gradient_reduction == "paper_sum":
        main_fd.all_reduce_grads = sum_parameter_gradients
    if adapter.eqvae_lr_schedule_total_steps is not None:
        main_fd.adjust_learning_rate = build_schedule_horizon_adapter(
            main_fd.adjust_learning_rate,
            adapter.eqvae_lr_schedule_total_steps,
        )

    sys.argv = [sys.argv[0], *official_argv]
    args = main_fd.get_args_parser().parse_args()
    adapter_path = Path(__file__).resolve()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    manifest = {
        "protocol": "official_advfd_with_packed_imagenet_adapter_v1",
        "official_advfd_root": str(official_root),
        "official_advfd_commit": git_head(official_root),
        "eqvae_root": str(EQVAE_ROOT),
        "eqvae_commit_at_launch": git_head(EQVAE_ROOT),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "packed_imagenet_root": str(packed_root),
        "horizontal_flip": False,
        "crop_implementation": "ADM center crop, byte-identical algorithm",
        "gradient_reduction": adapter.eqvae_gradient_reduction,
        "lr_schedule_total_steps": adapter.eqvae_lr_schedule_total_steps,
        "resource_scaling": {
            "world_size": world_size,
            "local_batch_size": int(args.batch_size),
            "global_batch_size": world_size * int(args.batch_size),
            "feature_queue_size": int(args.queue_size),
            "official_pmf_reference_global_batch_size": 1024,
            "official_pmf_reference_feature_queue_size": 50000,
        },
        "official_arguments": vars(args),
        "adapter_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(adapter).items()
        },
        "command": sys.argv,
    }
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        adapter.eqvae_adapter_manifest.parent.mkdir(parents=True, exist_ok=True)
        adapter.eqvae_adapter_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)

    try:
        exit_code = main_fd.train_and_evaluate(args)
    finally:
        main_fd._cleanup_distributed()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
