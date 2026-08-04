"""Finite-difference conservativeness audit for the RAEv2 IG head gap.

At states from the unguided sampling trajectory, this script estimates

    u^T J_g v - v^T J_g u,  g(x,t) = full(x,t) - base(x,t).

A gradient field has a symmetric Jacobian and therefore zero antisymmetric
bilinear form.  Stable nonzero energy across finite-difference scales is a
cheap falsification of a CFG-like scalar-potential interpretation.  The model
is frozen throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.raev2_training_core import (  # noqa: E402
    file_sha256,
    split_internal_guidance_output,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import build_requested_labels, load_config  # noqa: E402
from experiments.run_raev2_ig_impulse_response import (  # noqa: E402
    _atomic_json,
    autocast_context,
    bootstrap_mean_interval,
    deterministic_noise,
    euler_x_prediction_step,
    official_shifted_solver_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_ig_curl_audit_v1"


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("finite positive epsilons are required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--steps", type=parse_int_list, default=(10, 30, 50, 70, 90, 98))
    parser.add_argument(
        "--epsilons", type=parse_float_list, default=(1e-2, 3e-3, 1e-3)
    )
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo"),
    )
    return parser.parse_args()


def normalize_rms(value: torch.Tensor) -> torch.Tensor:
    return value / value.square().mean().sqrt().clamp_min(1e-30)


def antisymmetric_bilinear(
    u: torch.Tensor,
    v: torch.Tensor,
    j_u: torch.Tensor,
    j_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len({tensor.shape for tensor in (u, v, j_u, j_v)}) != 1:
        raise ValueError("directions and directional derivatives must align")
    dims = tuple(range(1, u.ndim))
    u_jv = (u * j_v).mean(dim=dims)
    v_ju = (v * j_u).mean(dim=dims)
    return u_jv - v_ju, u_jv, v_ju


def random_probe_pair(
    shape: tuple[int, ...],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u = normalize_rms(torch.randn(shape, generator=generator, dtype=torch.float32))
    v = normalize_rms(torch.randn(shape, generator=generator, dtype=torch.float32))
    return u, v


def _head_gap(
    model: torch.nn.Module,
    state: torch.Tensor,
    time: float,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.full((len(state),), float(time), device=state.device)
    with torch.inference_mode(), autocast_context("fp32"):
        output = model(state, times, context=labels, attn_mask=None)
    full, base = split_internal_guidance_output(output)
    if base is None:
        raise RuntimeError("configured checkpoint has no IG base head")
    return full.float(), full.float() - base.float()


def finite_difference_probe(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    label: torch.Tensor,
    time: float,
    u: torch.Tensor,
    v: torch.Tensor,
    epsilon: float,
) -> dict[str, float]:
    if len(state) != 1 or u.shape != state.shape or v.shape != state.shape:
        raise ValueError("curl probes currently require one aligned sample")
    perturbed = torch.cat(
        (state + epsilon * u, state - epsilon * u, state + epsilon * v, state - epsilon * v),
        dim=0,
    )
    labels = label.expand(4)
    _, gaps = _head_gap(model, perturbed, time, labels)
    j_u = (gaps[0:1] - gaps[1:2]) / (2.0 * epsilon)
    j_v = (gaps[2:3] - gaps[3:4]) / (2.0 * epsilon)
    anti, u_jv, v_ju = antisymmetric_bilinear(u, v, j_u, j_v)
    cross_energy = 0.5 * (u_jv.square() + v_ju.square())
    return {
        "antisymmetric": float(anti.item()),
        "u_jv": float(u_jv.item()),
        "v_ju": float(v_ju.item()),
        "antisymmetric_square": float(anti.square().item()),
        "cross_energy": float(cross_energy.item()),
        "j_u_rms": float(j_u.square().mean().sqrt().item()),
        "j_v_rms": float(j_v.square().mean().sqrt().item()),
    }


def analyze_rows(
    raw: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    per_sample = (
        raw.groupby(["step", "time", "epsilon", "sample_id"], as_index=False)
        .agg(
            antisymmetric_square=("antisymmetric_square", "mean"),
            cross_energy=("cross_energy", "mean"),
            u_jv_square=("u_jv", lambda value: float(np.mean(np.square(value)))),
            v_ju_square=("v_ju", lambda value: float(np.mean(np.square(value)))),
            u_jv_v_ju=("u_jv", "size"),
            j_u_rms=("j_u_rms", "mean"),
            j_v_rms=("j_v_rms", "mean"),
        )
    )
    # Recompute the cross product without relying on a multi-column groupby lambda.
    products = (
        raw.assign(product=raw["u_jv"] * raw["v_ju"])
        .groupby(["step", "time", "epsilon", "sample_id"], as_index=False)["product"]
        .mean()
    )
    per_sample = per_sample.drop(columns="u_jv_v_ju").merge(
        products, on=["step", "time", "epsilon", "sample_id"], validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    for group_index, ((step, time, epsilon), frame) in enumerate(
        per_sample.groupby(["step", "time", "epsilon"], sort=True)
    ):
        anti = frame["antisymmetric_square"].to_numpy(dtype=np.float64)
        cross = frame["cross_energy"].to_numpy(dtype=np.float64)
        ratio_samples = np.sqrt(anti / np.maximum(cross, 1e-30))
        low, high = bootstrap_mean_interval(
            ratio_samples, repeats=repeats, seed=seed + 1009 * group_index
        )
        anti_rms = float(np.sqrt(anti.mean()))
        cross_rms = float(np.sqrt(cross.mean()))
        symmetry_cosine = float(
            frame["product"].mean()
            / max(
                math.sqrt(frame["u_jv_square"].mean() * frame["v_ju_square"].mean()),
                1e-30,
            )
        )
        rows.append(
            {
                "step": int(step),
                "time": float(time),
                "epsilon": float(epsilon),
                "samples": int(len(frame)),
                "antisymmetric_rms": anti_rms,
                "cross_rms": cross_rms,
                "antisymmetric_over_cross": anti_rms / max(cross_rms, 1e-30),
                "sample_ratio_mean": float(ratio_samples.mean()),
                "sample_ratio_ci_low": low,
                "sample_ratio_ci_high": high,
                "symmetry_cosine": symmetry_cosine,
                "j_u_rms_mean": float(frame["j_u_rms"].mean()),
                "j_v_rms_mean": float(frame["j_v_rms"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "epsilon"])


def plot_summary(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for epsilon, local in frame.groupby("epsilon"):
        local = local.sort_values("time")
        axis.plot(
            local["time"],
            local["antisymmetric_over_cross"],
            "o-",
            label=f"epsilon={epsilon:g}",
        )
    axis.invert_xaxis()
    axis.set(
        title="RAEv2 IG gap: normalized antisymmetric Jacobian energy",
        xlabel="solver time t",
        ylabel="anti RMS / cross RMS",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.probe_count <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("samples, probes and bootstrap repeats must be positive")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = official_shifted_solver_grid(int(config.sampler.num_steps), shift)
    steps = tuple(int(step) for step in args.steps)
    if len(set(steps)) != len(steps) or any(step < 0 or step >= len(grid) - 1 for step in steps):
        raise ValueError("curl steps must be unique valid solver indices")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else ""
    objects = [checkpoint_hash]
    dist.broadcast_object_list(objects, src=0)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "training": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": objects[0],
        "state_key": args.state_key,
        "samples": args.samples,
        "probe_count": args.probe_count,
        "steps": list(steps),
        "times": [float(grid[step]) for step in steps],
        "epsilons": list(args.epsilons),
        "seed": args.seed,
        "world_size": world_size,
        "precision": "fp32",
        "tf32": False,
        "state_source": "unguided sampling trajectory",
        "gap_definition": "full clean prediction minus base clean prediction",
    }
    manifest_path = output_dir / "manifest.json"
    if rank == 0:
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = tuple(key for key in manifest if key != "status")
            changed = [key for key in keys if current.get(key) != manifest.get(key)]
            if changed:
                raise RuntimeError(f"cannot resume changed curl protocol: {changed}")
        else:
            _atomic_json(manifest_path, manifest)
            labels = build_requested_labels(args.samples, int(config.misc.num_classes))
            np.savez_compressed(output_dir / "sample_protocol.npz", labels=labels)
    dist.barrier()
    labels = np.load(output_dir / "sample_protocol.npz")["labels"].astype(np.int64)
    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    sample_dir = output_dir / "sample_rows"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for local_index, sample_id in enumerate(local_ids.tolist()):
        sample_path = sample_dir / f"sample_{sample_id:06d}.csv"
        if sample_path.is_file():
            continue
        state = deterministic_noise(
            np.asarray([sample_id]), latent_size, seed=args.seed
        ).to(device)
        label = torch.tensor([labels[sample_id]], device=device, dtype=torch.long)
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for step in range(len(grid) - 1):
                time = float(grid[step])
                h = time - float(grid[step + 1])
                full, _ = _head_gap(model, state, time, label)
                if step in steps:
                    for probe in range(args.probe_count):
                        probe_seed = (
                            args.seed
                            + 10_000_019 * sample_id
                            + 100_003 * step
                            + 1_009 * probe
                        )
                        u_cpu, v_cpu = random_probe_pair(state.shape, seed=probe_seed)
                        u = u_cpu.to(device)
                        v = v_cpu.to(device)
                        for epsilon in args.epsilons:
                            metrics = finite_difference_probe(
                                model=model,
                                state=state,
                                label=label,
                                time=time,
                                u=u,
                                v=v,
                                epsilon=float(epsilon),
                            )
                            rows.append(
                                {
                                    "sample_id": sample_id,
                                    "label": int(labels[sample_id]),
                                    "step": step,
                                    "time": time,
                                    "probe": probe,
                                    "epsilon": float(epsilon),
                                    **metrics,
                                }
                            )
                state = euler_x_prediction_step(
                    state,
                    full,
                    time=time,
                    step_size=h,
                    t_eps=float(config.transport.t_eps),
                )
        temporary = sample_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, sample_path)
        if rank == 0:
            print(f"[rank 0] local curl samples {local_index + 1}/{len(local_ids)}", flush=True)
    dist.barrier()
    if rank == 0:
        paths = sorted(sample_dir.glob("sample_*.csv"))
        if len(paths) != args.samples:
            raise RuntimeError(f"expected {args.samples} curl sample files, found {len(paths)}")
        raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
        summary = analyze_rows(raw, repeats=args.bootstrap_repeats, seed=args.seed + 29)
        raw.to_csv(output_dir / "curl_raw.csv", index=False)
        summary.to_csv(output_dir / "curl_summary.csv", index=False)
        plot_summary(summary, output_dir / "curl_summary.png")
        manifest["status"] = "complete"
        _atomic_json(manifest_path, manifest)
        print(summary.to_string(index=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
