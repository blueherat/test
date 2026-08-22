#!/usr/bin/env python3
"""Train the operator-valued x/v target on completed linear rank baselines.

For an orthogonal projector P onto the clean linear data subspace, define the
native hybrid output

    u_P = P v + (I - P) x.

It is converted back to velocity by

    v_hat = P u_hat + (I - P) z_t / t.

Because (I-P)x is analytically zero in this controlled experiment, the model
output is explicitly projected into range(P); it is not asked to relearn that
known zero.  The construction predicts data-subspace velocity while normal
noise removal is handled analytically.  This script reuses completed
native-x/native-v checkpoints and trains only the hybrid model with their exact
initialization/data protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    parse_int_list,
    sample_spiral_2d,
    stable_seed,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    RankOutputMLP,
    build_matched_models,
    condition_predictions,
    evaluate_generation,
    save_csv,
    set_seed,
)


def parse_path_list(text: str) -> list[Path]:
    paths = [Path(value.strip()) for value in text.split(",") if value.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("expected comma-separated paths")
    return paths


def project_data_subspace(value: torch.Tensor, embedding: CurvedEmbedding) -> torch.Tensor:
    basis = embedding.Q[:, :2].to(dtype=value.dtype)
    return (value @ basis) @ basis.T


def hybrid_velocity(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    embedding: CurvedEmbedding,
    clip: float,
) -> torch.Tensor:
    projected_output = project_data_subspace(output, embedding)
    normal_state = state - project_data_subspace(state, embedding)
    return projected_output + normal_state / time[:, None].clamp_min(clip)


def hybrid_clean(
    output: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    embedding: CurvedEmbedding,
    clip: float,
) -> torch.Tensor:
    return state - time[:, None] * hybrid_velocity(
        output, state, time, embedding, clip
    )


def build_hybrid_model(
    *,
    D: int,
    hidden: int,
    output_rank: int,
    depth: int,
    time_dim: int,
    seed: int,
    device: torch.device,
) -> RankOutputMLP:
    # build_matched_models creates every condition from this exact base state.
    return build_matched_models(
        D=D,
        hidden=hidden,
        output_rank=output_rank,
        depth=depth,
        time_dim=time_dim,
        seed=seed,
        device=device,
    )["native_x"]


def train_hybrid(
    *,
    model: RankOutputMLP,
    embedding: CurvedEmbedding,
    output_rank: int,
    config: dict,
    setting_seed: int,
    device: torch.device,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device=device.type)
    rank_dependent = bool(config.get("rank_dependent_randomness", True))
    rank_seed = output_rank if rank_dependent else 0
    generator.manual_seed(stable_seed(setting_seed, embedding.D, rank_seed, 701))
    use_amp = str(config["amp_dtype"]) == "bf16" and device.type == "cuda"
    history: list[dict] = []
    steps = int(config["train_steps"])
    batch_size = int(config["batch_size"])
    log_every = int(config["log_every"])

    for step in range(1, steps + 1):
        intrinsic = sample_spiral_2d(
            batch_size,
            device=device,
            jitter=float(config["data_jitter"]),
            generator=generator,
        )
        clean = embedding.embed(intrinsic)
        eps = torch.randn(clean.shape, device=device, generator=generator)
        time = torch.empty(batch_size, device=device).uniform_(
            float(config["t_min"]),
            float(config["t_max"]),
            generator=generator,
        )
        state = (1.0 - time[:, None]) * clean + time[:, None] * eps
        true_velocity = eps - clean
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(state, time)
            velocity = hybrid_velocity(
                output,
                state,
                time,
                embedding,
                float(config["conversion_clip"]),
            )
            loss = F.mse_loss(velocity.float(), true_velocity.float())
        loss.backward()
        grad_clip = float(config["grad_clip"])
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            history.append({"step": step, "loss_operator_hybrid": float(loss.detach().cpu())})
            print(
                f"[hybrid D={embedding.D} R={output_rank}] {step}/{steps} "
                f"loss={float(loss.detach().cpu()):.6g}",
                flush=True,
            )
    return history


@torch.inference_mode()
def evaluate_teacher(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    output_rank: int,
    config: dict,
    setting_seed: int,
    device: torch.device,
) -> list[dict]:
    conditions = ("native_x", "native_v", "operator_hybrid")
    rows = []
    for time_index, time_value in enumerate(config["eval_times"]):
        sums = {
            condition: {"total": 0.0, "data": 0.0, "normal": 0.0}
            for condition in conditions
        }
        generator = torch.Generator(device=device.type)
        generator.manual_seed(
            stable_seed(setting_seed, embedding.D, output_rank, time_index, 809)
        )
        samples = int(config["eval_samples"])
        batch_size = int(config["eval_batch_size"])
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            intrinsic = sample_spiral_2d(
                n,
                device=device,
                jitter=float(config["data_jitter"]),
                generator=generator,
            )
            clean = embedding.embed(intrinsic)
            eps = torch.randn(clean.shape, device=device, generator=generator)
            time = torch.full((n,), float(time_value), device=device)
            state = (1.0 - time[:, None]) * clean + time[:, None] * eps
            truth = eps - clean
            for condition in conditions:
                if condition == "operator_hybrid":
                    output = models[condition](state, time)
                    velocity = hybrid_velocity(
                        output,
                        state,
                        time,
                        embedding,
                        float(config["conversion_clip"]),
                    )
                else:
                    _native, velocity, _clean = condition_predictions(
                        model=models[condition],
                        condition=condition,
                        state=state,
                        time=time,
                        clip=float(config["conversion_clip"]),
                    )
                error = velocity - truth
                data_error = project_data_subspace(error, embedding)
                normal_error = error - data_error
                sums[condition]["total"] += float(error.square().sum().cpu())
                sums[condition]["data"] += float(data_error.square().sum().cpu())
                sums[condition]["normal"] += float(normal_error.square().sum().cpu())
        denominator = samples * embedding.D
        for condition in conditions:
            rows.append(
                {
                    "D": embedding.D,
                    "curvature": embedding.curvature,
                    "output_rank": output_rank,
                    "time": float(time_value),
                    "condition": condition,
                    "velocity_mse": sums[condition]["total"] / denominator,
                    "velocity_data_subspace_mse": sums[condition]["data"] / denominator,
                    "velocity_normal_mse": sums[condition]["normal"] / denominator,
                }
            )
    return rows


@torch.inference_mode()
def sample_all(
    *,
    models: dict[str, RankOutputMLP],
    embedding: CurvedEmbedding,
    config: dict,
    setting_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    conditions = ("native_x", "native_v", "operator_hybrid")
    collected = {condition: [] for condition in conditions}
    count = int(config["sample_count"])
    batch_size = int(config["sample_batch_size"])
    grid = torch.linspace(
        float(config["sample_t_max"]),
        float(config["sample_t_min"]),
        int(config["sample_steps"]) + 1,
        device=device,
    )
    sample_seed = stable_seed(setting_seed, 1213)
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(sample_seed + start)
        initial = float(config["sample_t_max"]) * torch.randn(
            (n, embedding.D), device=device, generator=generator
        )
        states = {condition: initial.clone() for condition in conditions}
        for index in range(len(grid) - 1):
            t_now, t_next = grid[index], grid[index + 1]
            time = t_now.expand(n)
            for condition in conditions:
                if condition == "operator_hybrid":
                    output = models[condition](states[condition], time)
                    velocity = hybrid_velocity(
                        output,
                        states[condition],
                        time,
                        embedding,
                        float(config["conversion_clip"]),
                    )
                else:
                    _native, velocity, _clean = condition_predictions(
                        model=models[condition],
                        condition=condition,
                        state=states[condition],
                        time=time,
                        clip=float(config["conversion_clip"]),
                    )
                states[condition] = states[condition] + (t_next - t_now) * velocity

        time = grid[-1].expand(n)
        for condition in conditions:
            if condition == "operator_hybrid":
                output = models[condition](states[condition], time)
                clean = hybrid_clean(
                    output,
                    states[condition],
                    time,
                    embedding,
                    float(config["conversion_clip"]),
                )
            else:
                _native, _velocity, clean = condition_predictions(
                    model=models[condition],
                    condition=condition,
                    state=states[condition],
                    time=time,
                    clip=float(config["conversion_clip"]),
                )
            collected[condition].append(clean.cpu().numpy())
    return {condition: np.concatenate(parts) for condition, parts in collected.items()}


def discover_settings(roots: list[Path], dims: set[int], ranks: set[int]) -> list[tuple[Path, dict, dict]]:
    found: dict[tuple[int, int, int], tuple[Path, dict, dict]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["args"]
        for summary_path in root.rglob("summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = (int(summary["seed"]), int(summary["D"]), int(summary["output_rank"]))
            if float(summary["curvature"]) != 0.0:
                continue
            if int(summary["D"]) not in dims or int(summary["output_rank"]) not in ranks:
                continue
            found[key] = (summary_path.parent, summary, config)
    return [found[key] for key in sorted(found)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline-roots", type=parse_path_list, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("64,512"))
    parser.add_argument("--output-ranks", type=parse_int_list, default=parse_int_list("4,16,64,512"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    settings = discover_settings(
        args.baseline_roots, set(args.dims), set(args.output_ranks)
    )
    if not settings:
        raise RuntimeError("no matching completed linear baseline settings")
    all_teacher: list[dict] = []
    all_generation: list[dict] = []
    manifest_rows = []

    for baseline_dir, summary, config in settings:
        seed = int(summary["seed"])
        D = int(summary["D"])
        output_rank = int(summary["output_rank"])
        rank_dependent = bool(config.get("rank_dependent_randomness", True))
        rank_seed = output_rank if rank_dependent else 0
        setting_seed = stable_seed(seed, D, 0, rank_seed, 1009)
        out = output_root / f"seed{seed}" / f"D{D}" / f"rank{output_rank}"
        if args.resume and (out / "summary.json").is_file():
            print(f"[resume] {out}", flush=True)
            with (out / "teacher_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                teacher = list(csv.DictReader(handle))
            with (out / "generation_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                generation = list(csv.DictReader(handle))
            for row in teacher:
                row["seed"] = seed
                row["setting_seed"] = setting_seed
            for row in generation:
                row["seed"] = seed
                row["setting_seed"] = setting_seed
            all_teacher.extend(teacher)
            all_generation.extend(generation)
            manifest_rows.append(
                json.loads((out / "summary.json").read_text(encoding="utf-8"))
            )
            continue
        out.mkdir(parents=True, exist_ok=True)
        set_seed(setting_seed)
        embedding = CurvedEmbedding(
            D,
            curvature=0.0,
            frequency_scale=float(config["frequency_scale"]),
            seed=stable_seed(seed, D, 0, 41),
            device=device,
            scale_mode=str(config["scale_mode"]),
        )
        baseline_models = build_matched_models(
            D=D,
            hidden=int(config["hidden"]),
            output_rank=output_rank,
            depth=int(config["depth"]),
            time_dim=int(config["time_dim"]),
            seed=setting_seed,
            device=device,
        )
        checkpoint = torch.load(
            baseline_dir / "models.pt", map_location=device, weights_only=True
        )
        for condition in ("native_x", "native_v"):
            baseline_models[condition].load_state_dict(checkpoint[condition])
            baseline_models[condition].eval()
        hybrid = build_hybrid_model(
            D=D,
            hidden=int(config["hidden"]),
            output_rank=output_rank,
            depth=int(config["depth"]),
            time_dim=int(config["time_dim"]),
            seed=setting_seed,
            device=device,
        )
        history = train_hybrid(
            model=hybrid,
            embedding=embedding,
            output_rank=output_rank,
            config=config,
            setting_seed=setting_seed,
            device=device,
        )
        hybrid.eval()
        models = {
            "native_x": baseline_models["native_x"],
            "native_v": baseline_models["native_v"],
            "operator_hybrid": hybrid,
        }
        teacher = evaluate_teacher(
            models=models,
            embedding=embedding,
            output_rank=output_rank,
            config=config,
            setting_seed=setting_seed,
            device=device,
        )
        reference_generator = torch.Generator(device=device.type)
        reference_generator.manual_seed(stable_seed(setting_seed, 1201))
        reference_intrinsic = sample_spiral_2d(
            max(2 * int(config["sample_count"]), 8192),
            device=device,
            jitter=float(config["data_jitter"]),
            generator=reference_generator,
        ).cpu().numpy()
        generated = sample_all(
            models=models,
            embedding=embedding,
            config=config,
            setting_seed=setting_seed,
            device=device,
        )
        generation = evaluate_generation(
            samples=generated,
            reference_intrinsic=reference_intrinsic,
            embedding=embedding,
            output_rank=output_rank,
            seed=setting_seed,
            device=device,
            metric_max_points=int(config["metric_max_points"]),
            projections=int(config["swd_projections"]),
            rank_dependent_randomness=rank_dependent,
        )
        for row in teacher:
            row["seed"] = seed
            row["setting_seed"] = setting_seed
        for row in generation:
            row["seed"] = seed
            row["setting_seed"] = setting_seed
        save_csv(out / "train_history.csv", history)
        save_csv(out / "teacher_metrics.csv", teacher)
        save_csv(out / "generation_metrics.csv", generation)
        if args.save_checkpoints:
            torch.save(hybrid.state_dict(), out / "operator_hybrid.pt")
        result_summary = {
            "seed": seed,
            "D": D,
            "output_rank": output_rank,
            "baseline_dir": str(baseline_dir),
            "generation": {
                row["condition"]: {
                    "swd_2d": row["swd_2d"],
                    "swd_ambient": row["swd_ambient"],
                    "manifold_consistency_rms": row["manifold_consistency_rms"],
                }
                for row in generation
            },
        }
        (out / "summary.json").write_text(
            json.dumps(result_summary, indent=2), encoding="utf-8"
        )
        all_teacher.extend(teacher)
        all_generation.extend(generation)
        manifest_rows.append(result_summary)

    if all_teacher:
        save_csv(output_root / "teacher_metrics.csv", all_teacher)
        save_csv(output_root / "generation_metrics.csv", all_generation)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "definition": "operator target u_P=P v+(I-P)x on the exact linear data subspace",
                "baseline_roots": [str(path) for path in args.baseline_roots],
                "completed": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] completed {len(manifest_rows)} operator settings", flush=True)


if __name__ == "__main__":
    main()
