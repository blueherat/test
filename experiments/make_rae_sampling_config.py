from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an RAE stage2 sampling config from a training config and checkpoint.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--checkpoint-weight-key",
        choices=["auto", "ema", "model"],
        default="auto",
        help="For training checkpoints, materialize this state dict before sampling. "
        "'auto' keeps the original checkpoint behavior.",
    )
    parser.add_argument(
        "--materialized-checkpoint",
        default=None,
        help="Optional path for the materialized raw state dict when --checkpoint-weight-key is not auto.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.base_config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if args.checkpoint_weight_key != "auto":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if args.checkpoint_weight_key in checkpoint:
            output_path = Path(args.output).expanduser().resolve()
            materialized = (
                Path(args.materialized_checkpoint).expanduser().resolve()
                if args.materialized_checkpoint
                else output_path.with_name(
                    f"{output_path.stem}_{args.checkpoint_weight_key}_stage2.pt"
                )
            )
            materialized.parent.mkdir(parents=True, exist_ok=True)
            if not materialized.exists():
                torch.save(checkpoint[args.checkpoint_weight_key], materialized)
            checkpoint_path = materialized
        elif "ema" in checkpoint or "model" in checkpoint:
            available = [key for key in ("ema", "model") if key in checkpoint]
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain "
                f"{args.checkpoint_weight_key!r}; available keys: {available}"
            )
    cfg.stage_2.ckpt = str(checkpoint_path)
    cfg.guidance.method = "cfg"
    cfg.guidance.scale = float(args.guidance_scale)
    cfg.guidance.t_min = 0.0
    cfg.guidance.t_max = 1.0
    if "eval" in cfg:
        del cfg["eval"]
    if "training" in cfg:
        del cfg["training"]
    cfg.sampler.params.num_steps = int(args.num_steps)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    print(output)


if __name__ == "__main__":
    main()
