#!/usr/bin/env python3
"""Fixed-budget scalar-potential solver; numerical pilot, fit, heldout audit.

The baseline is the official native-head-dtype B + 1.78*(F-B), with its
existing [.1,1] interval. This runner never decodes or ranks generated images.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "external/RAEv2/src", ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.audit_raev2_proximal_calibration import atomic_json, atomic_torch_save, sha256_file, write_csv
from experiments.raev2_observable_potential import ScalarGuidancePotential, observable_error_loss
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, shifted_time_grid
from utils.model_utils import instantiate_from_config
from utils.guidance_utils import forward_with_internalguidance

DATA = Path("/home/zhoushunyu/data/eqvae")
RESTART = DATA / "experiments/raev2_guidance_restart_20260906"
PROTOCOL = "raev2_observable_potential_solver_v1"
BATCH_SIZE = 32
UPDATES = 2048
LEARNING_RATE = 1e-4
MODEL_SEED = 202609091
DATA_SEED = 202609092
NOISE_SEED = 202609093
VALIDATION_NOISE_SEED = 202609094


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def load_banks(directory: Path):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("complete") is not True:
        raise ValueError("clean bank is not complete")
    banks = {}
    for name, count, per_class in (("train", 5000, 5), ("validation", 1000, 1)):
        record = summary["banks"][name]
        for kind in ("latents", "metadata"):
            path = directory / name / ("latents.npy" if kind == "latents" else "metadata.npz")
            if sha256_file(path) != record[kind]["sha256"]:
                raise ValueError(f"bank artifact hash mismatch: {path}")
        with np.load(directory / name / "metadata.npz", allow_pickle=False) as values:
            metadata = {key: values[key].copy() for key in values.files}
        latents = np.load(directory / name / "latents.npy", mmap_mode="r", allow_pickle=False)
        if latents.shape != (count, 1024, 16, 16) or latents.dtype != np.float16:
            raise ValueError("unexpected latent array")
        if not np.array_equal(metadata["ids"], np.arange(count)) or len(np.unique(metadata["rows"])) != count:
            raise ValueError("invalid clean bank identities")
        classes, counts = np.unique(metadata["labels"], return_counts=True)
        if not np.array_equal(classes, np.arange(1000)) or not np.all(counts == per_class):
            raise ValueError("invalid class counts")
        banks[name] = (latents, metadata)
    if np.intersect1d(banks["train"][1]["rows"], banks["validation"][1]["rows"]).size:
        raise ValueError("training/validation real source overlap")
    return banks, artifact(summary_path)


def official_clean_from_heads(full, base, times):
    """Preserve official native output dtype during all IG arithmetic."""
    mask = ((times >= .1) & (times <= 1)).reshape(-1, 1, 1, 1)
    return torch.where(mask, base + 1.78 * (full - base), full)


def clean_forward(model, z, times, labels, *, heads=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full, base = model(z, times, context=labels, attn_mask=None)
        guided = official_clean_from_heads(full, base, times)
    result = guided.float().detach()
    return (result, full, base) if heads else result


def batch_from_bank(bank, indices, device):
    latents, metadata = bank
    clean = torch.from_numpy(np.array(latents[indices], copy=True)).to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(metadata["labels"][indices].copy()).to(device=device, dtype=torch.long)
    return clean, labels


def terms(c, clean, guided):
    residual = clean - guided
    c2 = c.square().flatten(1).mean(1)
    rc = (residual * c).flatten(1).mean(1)
    r2 = residual.square().flatten(1).mean(1)
    return r2, c2, rc


def checked_gradients(potential):
    norm2 = 0.0
    for parameter in potential.parameters():
        if parameter.grad is None:
            raise RuntimeError("unexpected unused potential parameter")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("nonfinite potential gradient")
        norm2 += float(parameter.grad.square().sum())
    if norm2 <= 0:
        raise FloatingPointError("zero full parameter gradient")
    return norm2 ** .5


def source_records(output):
    records = {}
    for path in (Path(__file__), ROOT / "experiments/raev2_observable_potential.py",
                 ROOT / "external/RAEv2/src/utils/guidance_utils.py"):
        destination = output / path.name
        shutil.copy2(path, destination)
        records[path.name] = artifact(destination)
    return records


def pilot(model, potential, banks, grid, output, device):
    clean, labels = batch_from_bank(banks["train"], np.arange(BATCH_SIZE), device)
    generator = torch.Generator(device=device).manual_seed(NOISE_SEED + 100)
    indices = torch.tensor([0, 47, 67, 84, 92, 97, 98, 99] * 4, device=device)
    times = grid[indices]
    noise = torch.randn(clean.shape, generator=generator, device=device)
    z = (1-times[:, None, None, None])*clean + times[:, None, None, None]*noise
    started = time.perf_counter()
    guided, full, base = clean_forward(model, z, times, labels, heads=True)

    class CachedHeads:
        in_channels = 1024
        def __call__(self, x, t, **kwargs):
            assert x.shape == z.shape and torch.equal(t, times)
            return full, base

    reference = forward_with_internalguidance(
        CachedHeads(), torch.cat([z, z]), torch.cat([times, times]),
        1.78, (.1, 1), context=torch.cat([labels, labels]), attn_mask=None,
    )[:len(z)].float()
    if not torch.equal(reference, guided):
        raise AssertionError("native-head official wrapper parity failed")
    c = potential.clean_correction(z, times, labels, create_graph=True)
    if torch.count_nonzero(c) or not torch.equal(guided + c.detach(), reference):
        raise AssertionError("zero potential changed official prediction")
    next_times = grid[indices+1]
    h = (times-next_times)[:, None, None, None]
    denominator = times[:, None, None, None]
    baseline_step = z - h*((z-reference)/denominator)
    corrected_step = z - h*((z-guided-c.detach())/denominator)
    if not torch.equal(baseline_step, corrected_step):
        raise AssertionError("zero correction Euler parity failed")
    loss = observable_error_loss(c, clean, guided)
    loss.backward()
    zero_initial_gradient_norm = checked_gradients(potential)
    readout = potential.readout.weight
    direction = readout.grad.detach().clone()
    direction /= direction.norm()
    analytic = float((readout.grad * direction).sum())
    original = readout.detach().clone()
    fd_rows = []
    for epsilon in (1e-3, 3e-4):
        losses = []
        for sign in (1, -1):
            with torch.no_grad():
                readout.copy_(original + sign*epsilon*direction)
            trial = potential.clean_correction(z, times, labels)
            # Float64 summation only for checking the FP32 network derivative.
            losses.append(float(observable_error_loss(trial.double(), clean.double(), guided.double())))
        finite_difference = (losses[0]-losses[1])/(2*epsilon)
        relative_error = abs(finite_difference-analytic)/max(abs(analytic), 1e-20)
        fd_rows.append({"epsilon": epsilon, "analytic": analytic, "finite_difference": finite_difference,
                        "relative_error": relative_error})
        if relative_error > 1e-2:
            raise AssertionError("real-target parameter derivative finite difference failed")
    with torch.no_grad():
        readout.copy_(original)
    potential.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(potential.parameters(), lr=LEARNING_RATE, weight_decay=0)
    fit_rows = []
    for update in range(8):
        optimizer.zero_grad(set_to_none=True)
        c = potential.clean_correction(z, times, labels, create_graph=True)
        loss = observable_error_loss(c, clean, guided)
        loss.backward()
        norm = checked_gradients(potential)
        optimizer.step()
        fit_rows.append({"update": update+1, "loss": float(loss.detach()), "gradient_norm": norm,
                         "correction_rms": float(c.detach().square().mean().sqrt())})
    if not fit_rows[-1]["loss"] < fit_rows[0]["loss"]:
        raise AssertionError("eight-step numerical pilot did not descend on its fixed training batch")
    timings = {}
    for name in ("baseline_forward", "potential_inference", "potential_train_step"):
        torch.cuda.synchronize()
        timer = time.perf_counter()
        for _ in range(3):
            if name == "baseline_forward":
                clean_forward(model, z, times, labels)
            elif name == "potential_inference":
                potential.clean_correction(z, times, labels)
            else:
                optimizer.zero_grad(set_to_none=True)
                correction = potential.clean_correction(z, times, labels, create_graph=True)
                observable_error_loss(correction, clean, guided).backward()
        torch.cuda.synchronize()
        timings[name+"_seconds_per_batch32"] = (time.perf_counter()-timer)/3
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("baseline was not frozen")
    write_csv(output / "pilot_updates.csv", fit_rows)
    write_csv(output / "parameter_derivative.csv", fd_rows)
    return {"complete": True, "pilot_passed": True, "training_checkpoint_written": False,
            "numerical_pilot_is_not_heldout_evidence": True, "head_dtype": str(full.dtype),
            "official_wrapper_zero_parity": True, "zero_initial_gradient_norm": zero_initial_gradient_norm,
            "finite_difference": fd_rows, "timings": timings,
            "stage2_forward_calls": 4, "stage2_sample_forwards": 4*BATCH_SIZE,
            "potential_forward_input_gradient_calls": 19,
            "potential_parameter_backward_calls": 12,
            "elapsed_seconds": time.perf_counter()-started}


def fit(model, potential, banks, grid, probabilities, output, device):
    optimizer = torch.optim.Adam(potential.parameters(), lr=LEARNING_RATE, weight_decay=0)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in potential.parameters()}:
        raise AssertionError("optimizer boundary mismatch")
    rng = np.random.default_rng(DATA_SEED)
    generator = torch.Generator(device=device).manual_seed(NOISE_SEED)
    histogram = np.zeros(100, dtype=np.int64)
    image_histogram = np.zeros(5000, dtype=np.int64)
    rows = []
    started = time.perf_counter()
    for update in range(UPDATES):
        image_ids = rng.integers(0, 5000, size=BATCH_SIZE)
        step_ids = rng.choice(100, size=BATCH_SIZE, p=probabilities)
        np.add.at(histogram, step_ids, 1)
        np.add.at(image_histogram, image_ids, 1)
        clean, labels = batch_from_bank(banks["train"], image_ids, device)
        times = grid[torch.from_numpy(step_ids).to(device)]
        noise = torch.randn(clean.shape, generator=generator, device=device)
        z = (1-times[:, None, None, None])*clean + times[:, None, None, None]*noise
        guided = clean_forward(model, z, times, labels)
        optimizer.zero_grad(set_to_none=True)
        c = potential.clean_correction(z, times, labels, create_graph=True)
        loss = observable_error_loss(c, clean, guided)
        loss.backward()
        gradient_norm = checked_gradients(potential)
        optimizer.step()
        with torch.no_grad():
            r2, c2, rc = terms(c, clean, guided)
            row = {"update": update+1, "loss": float(loss), "residual_mse": float(r2.mean()),
                   "correction_energy": float(c2.mean()), "residual_dot_correction": float(rc.mean()),
                   "paired_clean_mse_gain": float((2*rc-c2).mean()), "gradient_norm": gradient_norm,
                   "elapsed_seconds": time.perf_counter()-started}
        rows.append(row)
        if update == 0 or (update+1) % 32 == 0:
            atomic_json(output / "progress.json", {"complete": False, **row})
            print(json.dumps(row), flush=True)
        if (update+1) % 256 == 0:
            write_csv(output / "training.csv", rows)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("baseline changed optimizer boundary")
    checkpoint = output / "potential_final.pt"
    atomic_torch_save(checkpoint, {"protocol": PROTOCOL, "potential": potential.cpu().state_dict(),
                                 "updates": UPDATES, "request": str(output / "request.json"),
                                 "request_sha256": sha256_file(output / "request.json"),
                                 "noise_rng_state": generator.get_state(), "numpy_rng_state": rng.bit_generator.state})
    write_csv(output / "training.csv", rows)
    np.savez(output / "sampling_counts.npz", step_counts=histogram, image_counts=image_histogram)
    return {"complete": True, "updates": UPDATES, "checkpoint": artifact(checkpoint),
            "checkpoint_selection": "fixed final update only", "actual_time_histogram": histogram.tolist(),
            "unique_training_images_drawn": int(np.count_nonzero(image_histogram)),
            "image_reuse_min": int(image_histogram.min()), "image_reuse_max": int(image_histogram.max()),
            "stage2_forward_calls": UPDATES, "stage2_sample_forwards": UPDATES*BATCH_SIZE,
            "potential_forward_input_gradient_calls": UPDATES, "potential_parameter_backward_calls": UPDATES,
            "elapsed_seconds": elapsed, "last_training_row": rows[-1]}


def validate(model, potential, banks, grid, output, device, shard_index, num_shards):
    """All 100 times x 1000 heldout images; shard time, identical noise per image."""
    rows = []
    batch_calls = 0
    started = time.perf_counter()
    for k in range(shard_index, 100, num_shards):
        generator = torch.Generator(device=device).manual_seed(VALIDATION_NOISE_SEED)
        all_terms = []
        for start in range(0, 1000, BATCH_SIZE):
            clean, labels = batch_from_bank(banks["validation"], np.arange(start, min(start+BATCH_SIZE, 1000)), device)
            times = grid[k].expand(len(clean))
            noise = torch.randn(clean.shape, generator=generator, device=device)
            z = (1-times[:, None, None, None])*clean + times[:, None, None, None]*noise
            guided = clean_forward(model, z, times, labels)
            c = potential.clean_correction(z, times, labels)
            if not bool(torch.isfinite(c).all()):
                raise FloatingPointError("nonfinite heldout correction")
            all_terms.append(torch.stack(terms(c, clean, guided), 1).double().cpu().numpy())
            batch_calls += 1
        values = np.concatenate(all_terms)
        r2, c2, rc = values.T
        gain = 2*rc-c2
        residual_witness = rc-c2
        np.savez(output / f"step{k:03d}.npz", ids=banks["validation"][1]["ids"],
                 labels=banks["validation"][1]["labels"], residual_mse=r2, correction_energy=c2,
                 residual_dot_correction=rc, paired_clean_mse_gain=gain,
                 learned_potential_weak_residual=residual_witness)
        row = {"step": k, "time": float(grid[k]), "next_time": float(grid[k+1]), "samples": 1000,
               "residual_mse": float(r2.mean()), "correction_energy": float(c2.mean()),
               "residual_dot_correction": float(rc.mean()), "paired_clean_mse_gain": float(gain.mean()),
               "gain_standard_error_across_images": float(gain.std(ddof=1)/np.sqrt(1000)),
               "learned_potential_weak_residual": float(residual_witness.mean()),
               "witness_standard_error_across_images": float(residual_witness.std(ddof=1)/np.sqrt(1000)),
               "elapsed_seconds": time.perf_counter()-started}
        rows.append(row)
        atomic_json(output / "progress.json", {"complete": False, **row})
        print(json.dumps(row), flush=True)
    write_csv(output / "validation.csv", rows)
    return {"complete": True, "time_indices": [row["step"] for row in rows], "unique_validation_images": 1000,
            "stage2_forward_calls": batch_calls, "stage2_sample_forwards": 1000*len(rows),
            "potential_forward_input_gradient_calls": batch_calls, "potential_parameter_backward_calls": 0,
            "elapsed_seconds": time.perf_counter()-started,
            "boundary": "Frozen learned potential is one heldout witness, not a complete Poisson residual certificate; no FID."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "train", "validate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=RESTART / "potential_clean_bank_fp32_v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--potential-checkpoint", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if (output / "request.json").exists():
        raise FileExistsError("refusing to overwrite prior run")
    if not (1 <= args.num_shards <= 100 and 0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid validation shard")
    if args.mode != "validate" and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("only heldout validation may shard time")
    output.mkdir(parents=True, exist_ok=True)
    banks, bank_identity = load_banks(args.bank)
    grid_cpu = shifted_time_grid(100, 8, torch.device("cpu"))
    weights = ((grid_cpu[:-1]-grid_cpu[1:]).double()/grid_cpu[:-1].double().square()).numpy()
    probabilities = weights/weights.sum()
    sources = source_records(output)
    pilot_identity = None
    if args.mode == "train":
        if args.pilot is None:
            raise ValueError("training requires a completed numerical pilot")
        prior = json.loads((args.pilot / "summary.json").read_text())
        prior_request = json.loads((args.pilot / "request.json").read_text())
        if not prior.get("pilot_passed") or not prior.get("complete"):
            raise ValueError("pilot has not passed")
        for name in sources:
            if sources[name]["sha256"] != prior_request["sources"][name]["sha256"]:
                raise ValueError("code differs from the passed pilot")
        if prior_request["bank"] != bank_identity:
            raise ValueError("bank differs from pilot")
        pilot_identity = artifact(args.pilot / "summary.json")
    if args.mode == "validate" and args.potential_checkpoint is None:
        raise ValueError("validation requires final potential checkpoint")
    request = {"protocol": PROTOCOL, "mode": args.mode, "sources": sources, "bank": bank_identity,
               "config": artifact(args.config), "baseline_checkpoint": artifact(args.checkpoint),
               "baseline_formula": "native head dtype B + 1.78*(F-B), official interval [.1,1], then float32",
               "precision": "baseline BF16 autocast; FP32 state and potential; TF32 disabled globally",
               "structure": "608000-parameter scalar potential, direct z/t/class, c=grad_z Phi, clean=G+c, gain=1",
               "guidance_window": "potential at all 100 positive official query times, no added window",
               "optimizer": {"type": "Adam", "lr": LEARNING_RATE, "weight_decay": 0, "betas": [.9,.999], "eps": 1e-8},
               "batch_size": BATCH_SIZE, "updates": UPDATES, "ema": False, "lr_schedule": None,
               "checkpoint_selection": "fixed final update, no validation selection", "pilot": pilot_identity,
               "seeds": {"model": MODEL_SEED, "data_and_time": DATA_SEED, "training_noise": NOISE_SEED,
                         "validation_noise": VALIDATION_NOISE_SEED},
               "time_grid": grid_cpu.tolist(), "time_probability": probabilities.tolist(),
               "time_weight_normalizer": float(weights.sum()),
               "objective": "mean_dimensions then batch mean of .5*c^2-(X-G)*c; t sampled independently per image proportional to h/t^2",
               "unique_training_images": 5000, "unique_validation_images": 1000,
               "validation": "all 100 times, same independent noise per heldout image across times; current-fit heldout only",
               "shard_index": args.shard_index, "num_shards": args.num_shards,
               "potential_checkpoint": artifact(args.potential_checkpoint) if args.potential_checkpoint else None,
               "torch_version": torch.__version__, "fid_performed": False, "image_sampling_performed": False}
    atomic_json(output / "request.json", request)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(MODEL_SEED)
    potential = ScalarGuidancePotential().to(device).train(args.mode != "validate")
    if sum(p.numel() for p in potential.parameters()) != 608000:
        raise AssertionError("production structure changed")
    if args.potential_checkpoint:
        checkpoint = torch.load(args.potential_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != PROTOCOL or checkpoint["updates"] != UPDATES:
            raise ValueError("not the frozen final potential")
        potential.load_state_dict(checkpoint["potential"], strict=True)
        del checkpoint
    config = load_config(args.config)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    gc.collect()
    grid = grid_cpu.to(device)
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "pilot":
        summary = pilot(model, potential, banks, grid, output, device)
    elif args.mode == "train":
        summary = fit(model, potential, banks, grid, probabilities, output, device)
    else:
        summary = validate(model, potential, banks, grid, output, device, args.shard_index, args.num_shards)
    summary.update(protocol=PROTOCOL, mode=args.mode, baseline_checkpoint_step=checkpoint_step,
                   peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                   request=artifact(output / "request.json"), fid_performed=False, image_sampling_performed=False)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"complete": True, "summary": artifact(output / "summary.json")})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
