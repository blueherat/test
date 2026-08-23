"""Run the public AdvFD evaluator with an absolute Inception stats path."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_adapter_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eqvae-official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--eqvae-inception-stats", type=Path, required=True)
    parser.add_argument("--eqvae-eval-manifest", type=Path, required=True)
    parser.add_argument(
        "--eqvae-preserve-generated-images",
        choices=("auto", "always", "never"),
        default="auto",
    )
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


def checkpoint_has_adaptive_state(path: str | None) -> bool:
    if not path:
        return False
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        return False
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    states = checkpoint.get("fd_adv_states") if isinstance(checkpoint, dict) else None
    return isinstance(states, list) and bool(states)


def main() -> None:
    adapter, official_argv = parse_adapter_args(sys.argv[1:])
    official_root = adapter.eqvae_official_root.expanduser().resolve()
    stats_path = adapter.eqvae_inception_stats.expanduser().resolve()
    if not (official_root / "eval_all_fds.py").is_file():
        raise FileNotFoundError(f"Official AdvFD checkout not found: {official_root}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Inception reference stats not found: {stats_path}")

    sys.path.insert(0, str(official_root))
    import eval_all_fds  # noqa: PLC0415

    eval_all_fds.INCEPTION_STATS = [("FID(ADM)", str(stats_path))]
    sys.argv = [sys.argv[0], *official_argv]
    args = eval_all_fds.get_args_parser().parse_args()

    # Keep the exact generated samples for post-hoc adaptive-critic auditing.
    # The baseline has no resume checkpoint and therefore keeps the original
    # no-artifact behavior. These flags only affect image persistence; feature
    # extraction, FID accumulation, and generation are unchanged.
    has_adaptive_state = checkpoint_has_adaptive_state(args.resume_from)
    preserve_for_critic_audit = (
        has_adaptive_state
        if adapter.eqvae_preserve_generated_images == "auto"
        else adapter.eqvae_preserve_generated_images == "always"
    )
    if preserve_for_critic_audit:
        args.save_eval_images = True
        args.keep_eval_folder = True

    if preserve_for_critic_audit or args.gen_only:
        # Both preserved evaluation images and gen-only mode serialize PNGs.
        # The public helper defaults to OpenCV solely for that serialization;
        # use its lossless PIL backend when OpenCV is unavailable.
        try:
            import cv2  # noqa: F401, PLC0415
        except ModuleNotFoundError:
            import utils.data_util as official_data_util  # noqa: PLC0415

            official_data_util.save_image = functools.partial(
                official_data_util.save_image, backend="pil"
            )

    script_path = Path(__file__).resolve()
    manifest = {
        "protocol": (
            "official_advfd_pmf_generation_v1"
            if args.gen_only
            else "official_advfd_pmf_inception_evaluation_v1"
        ),
        "official_advfd_root": str(official_root),
        "official_advfd_commit": git_head(official_root),
        "adapter_path": str(script_path),
        "adapter_sha256": sha256_file(script_path),
        "inception_stats_path": str(stats_path),
        "inception_stats_sha256": sha256_file(stats_path),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "checkpoint_has_adaptive_state": has_adaptive_state,
        "image_preservation_mode": adapter.eqvae_preserve_generated_images,
        "preserve_generated_images_for_critic_audit": preserve_for_critic_audit,
        "official_arguments": vars(args),
        "command": sys.argv,
    }
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        adapter.eqvae_eval_manifest.parent.mkdir(parents=True, exist_ok=True)
        adapter.eqvae_eval_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)

    if args.gen_only:
        eval_all_fds.main_gen_only(args)
    else:
        eval_all_fds.main_generate(args)


if __name__ == "__main__":
    main()
