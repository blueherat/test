#!/usr/bin/env python3
"""Frozen observable-potential sampling and separately accounted B8 timing.

Potential mode uses all 100 official query times with clean correction grad Phi.
Official mode accepts an explicitly supplied Euler step count for cost matching.
Benchmark mode generates no images and never selects a step count or checkpoint.
All modes keep the same final potential, stage-2 model and decoder resident.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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

from experiments.train_raev2_observable_potential import (
    PROTOCOL as TRAINING_PROTOCOL, UPDATES, artifact, atomic_json, clean_forward,
    official_clean_from_heads, sha256_file, write_csv,
)
from experiments.raev2_observable_potential import ScalarGuidancePotential
from experiments.sample_raev2_pfr_retiming import (
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, shifted_time_grid,
)

PROTOCOL = "raev2_observable_potential_sampling_v1"
BATCH_SIZE = 8
COHORT_SIZE = 1000
LATENT_SHAPE = (1024, 16, 16)
T_EPS = 0.05
CONFIG_SHA256 = "3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342"
BASELINE_SHA256 = "723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a"
BENCHMARK_WARMUP = 1
BENCHMARK_REPEATS = 3
SOURCE_FILES = (
    "experiments/sample_raev2_observable_potential.py",
    "experiments/train_raev2_observable_potential.py",
    "experiments/raev2_observable_potential.py",
    "experiments/sample_raev2_pfr_retiming.py",
    "experiments/audit_raev2_proximal_calibration.py",
    "experiments/raev2_stage1_compat.py",
    "external/RAEv2/src/utils/guidance_utils.py",
    "external/RAEv2/src/utils/model_utils.py",
    "external/RAEv2/src/stage2/models/DDT.py",
    "external/RAEv2/src/stage2/models/model_utils.py",
    "external/RAEv2/src/stage1/rae.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official", "potential", "benchmark"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--potential-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=COHORT_SIZE)
    parser.add_argument("--seed", type=int, default=202609095)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_steps <= 0 or (args.mode != "official" and args.num_steps != 100):
        parser.error("positive --num-steps may differ from 100 only in official mode")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must satisfy 0 <= seed < 2**63")
    try:
        validate_sample_count(args.num_samples)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.num_shards <= args.num_samples // BATCH_SIZE or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard: partition complete global B8 batches")
    if args.mode == "benchmark" and args.num_samples != COHORT_SIZE:
        parser.error("benchmark requires --num-samples 1000")
    if args.mode == "benchmark" and (args.num_shards != 1 or args.shard_index != 0):
        parser.error("benchmark is one process; it does not produce a sample shard")
    return args


def validate_sample_count(count):
    if type(count) is not int or count <= 0 or count % 1000 or count % BATCH_SIZE:
        raise ValueError("--num-samples must be a positive multiple of 1000 and 8")


def assigned_batches(shard_index: int, num_shards: int, *, count, batch_size=BATCH_SIZE):
    """Partition whole global batches without changing membership or order."""
    if count <= 0 or batch_size <= 0 or count % batch_size:
        raise ValueError("cohort must contain a positive integral number of full batches")
    if not 1 <= num_shards <= count // batch_size or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard")
    return [np.arange(k*batch_size, (k+1)*batch_size, dtype=np.int64)
            for k in range(shard_index, count // batch_size, num_shards)]


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def global_labels_sha256(*, count):
    return tensor_sha256(torch.arange(count, dtype=torch.long) % 1000)


def paired_noise(seed, device, *, count, latent_shape=LATENT_SHAPE):
    """One fixed-shape device draw before sharding; return the cohort on CPU.

    Production always uses CUDA. CPU/device/shape arguments permit small tests.
    Neither rank, number of shards, step count nor batch size enters the RNG.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((count, *latent_shape), generator=generator,
                        device=device, dtype=torch.float32).cpu()
    return noise, generator.get_state().cpu()


def euler_update(state, guided, correction, current, following):
    """Official FP32 Euler, including t_eps for higher-cost baseline grids."""
    if not 0 <= following < current <= 1:
        raise ValueError("require 0 <= next < current <= 1")
    if state.dtype != torch.float32 or guided.dtype != torch.float32 or guided.shape != state.shape:
        raise ValueError("state and clean prediction must have matching FP32 shapes")
    residual = state - guided
    if correction is not None:
        if correction.dtype != torch.float32 or correction.shape != state.shape:
            raise ValueError("correction must have the matching FP32 shape")
        residual = residual - correction
    return state - (current-following)*(residual/max(current, T_EPS))


def sampling_step(model, potential, state, labels, current, following, *, use_potential):
    """No backbone gradients; potential input gradients remain enabled locally."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("use torch.no_grad, not inference_mode")
    with torch.no_grad():
        times = torch.full((len(state),), current, device=state.device, dtype=torch.float32)
        guided = clean_forward(model, state, times, labels)
        correction = potential.clean_correction(state, times, labels) if use_potential else None
        return euler_update(state, guided, correction, current, following)


def native_uint8(decoded):
    """Keep BF16 through clamp and multiply, matching official image export."""
    if decoded.dtype != torch.bfloat16:
        raise ValueError("official decoder must return BF16 before pixel arithmetic")
    if decoded.ndim != 4 or decoded.shape[1:] != (3, 256, 256):
        raise ValueError("expected decoded [B,3,256,256]")
    if not bool(torch.isfinite(decoded).all()):
        raise FloatingPointError("nonfinite decoded image")
    return decoded.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()


def save_npz(path, *arrays, **named):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, *arrays, **named)
    temporary.replace(path)


def verified_json(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return json.loads(path.read_text())


def verify_training_chain(checkpoint, checkpoint_identity, *, config_identity, baseline_identity,
                          source_hashes, decoder_identity, stats_identity):
    """Bind final weights to the completed fixed-budget fit and its data units."""
    if checkpoint.get("protocol") != TRAINING_PROTOCOL or checkpoint.get("updates") != UPDATES:
        raise ValueError("requires the fixed final potential checkpoint")
    training_request_record = {"path": checkpoint["request"], "sha256": checkpoint["request_sha256"]}
    training = verified_json(training_request_record)
    if (training.get("mode") != "train" or training.get("protocol") != TRAINING_PROTOCOL
            or training.get("updates") != UPDATES or training.get("config") != config_identity
            or training.get("baseline_checkpoint") != baseline_identity):
        raise ValueError("potential training protocol or baseline differs")
    for name, old in training["sources"].items():
        if source_hashes.get(name) != old["sha256"]:
            raise ValueError(f"training source changed: {name}")
    summary_path = Path(checkpoint["request"]).parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("complete") is not True or summary.get("mode") != "train"
            or summary.get("updates") != UPDATES or summary.get("checkpoint") != checkpoint_identity
            or summary.get("request", {}).get("sha256") != checkpoint["request_sha256"]):
        raise ValueError("checkpoint is not the completed training run's final artifact")
    bank = verified_json(training["bank"])
    if bank.get("complete") is not True:
        raise ValueError("training clean bank is incomplete")
    bank_request = verified_json(bank["request"])
    identity = bank_request["identity"]
    if identity["normalization_stats"] != stats_identity:
        raise ValueError("latent normalization differs from the training clean bank")
    if identity["decoder_weights_instantiated_but_not_used"] != decoder_identity:
        raise ValueError("decoder differs from the training clean bank's frozen RAE")
    return {"training_request": artifact(Path(checkpoint["request"])),
            "training_summary": artifact(summary_path), "training_bank": training["bank"],
            "training_bank_request": bank["request"], "training_cost": {
                key: summary[key] for key in (
                    "stage2_forward_calls", "stage2_sample_forwards", "potential_forward_input_gradient_calls",
                    "potential_parameter_backward_calls", "total_wall_seconds_excluding_imports_and_final_summary_write",
                    "total_cpu_seconds_excluding_imports_and_final_summary_write") if key in summary}}


def benchmark(model, potential, noise, grid, output, device, *, count):
    """Complete deployed rollouts, both arms reset to the same B8 initial noise.

    One complete warmup per arm, then three complete timed rollouts per arm.
    Arm order alternates by repetition. Decoder stays resident but is not called.
    Timings include all Euler steps and Python loop, not transfer/reset or hashes.
    """
    if count != COHORT_SIZE or len(noise) != count:
        raise ValueError("benchmark requires the complete 1000-sample noise cohort")
    labels = torch.arange(BATCH_SIZE, device=device, dtype=torch.long)
    initial = noise[:BATCH_SIZE].to(device)
    rows = []
    for repeat in range(BENCHMARK_WARMUP+BENCHMARK_REPEATS):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for order_index, enabled in enumerate(order):
            state = initial.clone()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for current, following in zip(grid[:-1], grid[1:]):
                state = sampling_step(model, potential, state, labels, current, following,
                                      use_potential=enabled)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError("nonfinite benchmark rollout")
            rows.append({"mode": "potential" if enabled else "official",
                         "warmup": repeat < BENCHMARK_WARMUP,
                         "repeat": repeat-BENCHMARK_WARMUP, "order_index": order_index,
                         "seconds": elapsed, "endpoint_sha256": tensor_sha256(state)})
    write_csv(output / "benchmark.csv", rows)
    estimates = {mode: float(np.mean([row["seconds"] for row in rows
                 if row["mode"] == mode and not row["warmup"]]))
                 for mode in ("official", "potential")}
    per_arm_calls = 100*(BENCHMARK_WARMUP+BENCHMARK_REPEATS)
    return {"complete": True, "image_sampling_performed": False, "decoder_forward_calls": 0,
            "initial_noise_sha256": tensor_sha256(initial),
            "warmup_complete_rollouts_per_arm": BENCHMARK_WARMUP,
            "timed_complete_rollouts_per_arm": BENCHMARK_REPEATS,
            "mean_100step_rollout_seconds_per_batch8": estimates,
            "official_mean_seconds_per_100_steps_batch8": estimates["official"],
            "potential_mean_seconds_per_100_steps_batch8": estimates["potential"],
            "candidate_to_official_100step_wall_ratio": estimates["potential"]/estimates["official"],
            "timing_boundary": "Full deployed 100-step trajectories, reset to identical initial noise per arm/repetition; arithmetic mean after warmup. Decoder not called; no step count selected.",
            "stage2_forward_calls": 2*per_arm_calls,
            "stage2_sample_forwards": 2*BATCH_SIZE*per_arm_calls,
            "potential_forward_input_gradient_calls": per_arm_calls,
            "potential_sample_input_gradients": BATCH_SIZE*per_arm_calls,
            "potential_parameter_backward_calls": 0, "artifact": artifact(output / "benchmark.csv"),
            "cost_scope": "All warmup and measurement rollouts are additional research cost; no amortization implied."}


def sample(model, potential, decoder, noise, grid, args, output, device, *, count):
    validate_sample_count(count)
    if len(noise) != count:
        raise ValueError("noise must contain the complete requested sample cohort")
    batches = assigned_batches(args.shard_index, args.num_shards, count=count)
    manifest = []
    completed = 0
    trajectory_wall = 0.0
    decoder_wall = 0.0
    sampling_started = time.perf_counter()
    (output / "batches").mkdir()
    for ids in batches:
        state = noise[torch.from_numpy(ids)].to(device)
        labels = torch.from_numpy(ids % 1000).to(device=device, dtype=torch.long)
        initial_hash = tensor_sha256(state)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for current, following in zip(grid[:-1], grid[1:]):
            state = sampling_step(model, potential, state, labels, current, following,
                                  use_potential=args.mode == "potential")
        torch.cuda.synchronize()
        batch_trajectory_wall = time.perf_counter()-started
        trajectory_wall += batch_trajectory_wall
        if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
            raise FloatingPointError("nonfinite or non-FP32 endpoint")
        endpoint_hash = tensor_sha256(state)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = decoder.decode(state)
            images = native_uint8(decoded)
        torch.cuda.synchronize()
        batch_decoder_wall = time.perf_counter()-started
        decoder_wall += batch_decoder_wall
        path = output / "batches" / f"{int(ids[0]):06d}_{int(ids[-1])+1:06d}.npz"
        save_npz(path, images, ids=ids, labels=ids % 1000)
        completed += len(ids)
        manifest.append({"global_ids": ids.tolist(), "noise_sha256": initial_hash,
                         "endpoint_sha256": endpoint_hash, "archive": artifact(path),
                         "trajectory_wall_seconds": batch_trajectory_wall,
                         "decode_and_uint8_wall_seconds": batch_decoder_wall})
        atomic_json(output / "batch_manifest.json", {"batches": manifest})
        progress = {"complete": False, "completed": completed, "assigned_samples": BATCH_SIZE*len(batches),
                    "stage2_forward_calls": len(manifest)*args.num_steps,
                    "elapsed_seconds": time.perf_counter()-sampling_started}
        atomic_json(output / "progress.json", progress)
        if len(manifest) == 1 or len(manifest) % 8 == 0 or len(manifest) == len(batches):
            print(json.dumps(progress), flush=True)
    image_parts, id_parts, label_parts = [], [], []
    for batch in manifest:
        with np.load(batch["archive"]["path"], allow_pickle=False) as archive:
            image_parts.append(archive["arr_0"])
            id_parts.append(archive["ids"])
            label_parts.append(archive["labels"])
    ids = np.concatenate(id_parts)
    if not np.array_equal(ids, np.concatenate(batches)):
        raise AssertionError("sample archive order differs from the assigned global IDs")
    path = output / "samples.npz"
    save_npz(path, np.concatenate(image_parts), ids=ids, labels=np.concatenate(label_parts))
    enabled = args.mode == "potential"
    return {"complete": True, "samples": completed, "global_cohort_size": count,
            "global_ids": ids.tolist(), "sample_archive": artifact(path),
            "archive_sha256": sha256_file(path), "batch_manifest": artifact(output / "batch_manifest.json"),
            "image_sampling_performed": True, "stage2_forward_calls": len(batches)*args.num_steps,
            "stage2_sample_forwards": completed*args.num_steps,
            "stage2_nfe_per_sample": args.num_steps,
            "potential_forward_input_gradient_calls": len(batches)*args.num_steps if enabled else 0,
            "potential_sample_input_gradients": completed*args.num_steps if enabled else 0,
            "potential_parameter_backward_calls": 0, "decoder_forward_calls": len(batches),
            "decoder_sample_forwards": completed, "trajectory_wall_seconds": trajectory_wall,
            "decode_and_uint8_wall_seconds": decoder_wall,
            "sampling_wall_including_output_seconds": time.perf_counter()-sampling_started,
            "base_head_note": "Base is returned in the same full-stage-2 forward; not a separate NFE.",
            "output_order": "Ascending global IDs within this shard; arr_0 is uint8 NHWC for ADM evaluator.",
            "cost_scope": "Sampling cost only; training, validation, preparation and benchmark costs remain separate."}


def main():
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    args = parse_args()
    for key in ("output_dir", "potential_checkpoint", "config", "checkpoint"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing run")
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative in SOURCE_FILES:
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        sources[relative] = artifact(destination)
    config_identity, baseline_identity = artifact(args.config), artifact(args.checkpoint)
    if config_identity["sha256"] != CONFIG_SHA256 or baseline_identity["sha256"] != BASELINE_SHA256:
        raise ValueError("requires the frozen official config and EMA checkpoint")
    config = load_config(args.config)
    if (tuple(config.misc.latent_size) != LATENT_SHAPE or config.misc.num_classes != 1000
            or config.transport.prediction != "x" or float(config.transport.t_eps) != T_EPS
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != .1
            or config.guidance.ig.t_max != 1 or config.guidance.cfg.scale != 1):
        raise ValueError("configuration is not the frozen official latent/transport/guidance protocol")
    decoder_identity = artifact(Path(config.stage_1.params["pretrained_decoder_path"]))
    stats_identity = artifact(Path(config.stage_1.params["normalization_stat_path"]))
    potential_identity = artifact(args.potential_checkpoint)
    load_started = time.perf_counter()
    checkpoint = torch.load(args.potential_checkpoint, map_location="cpu", weights_only=False)
    potential_checkpoint_read_seconds = time.perf_counter()-load_started
    chain = verify_training_chain(checkpoint, potential_identity,
        config_identity=config_identity, baseline_identity=baseline_identity,
        source_hashes={Path(name).name: record["sha256"] for name, record in sources.items()},
        decoder_identity=decoder_identity, stats_identity=stats_identity)
    grid = shifted_time_grid(args.num_steps, 8, torch.device("cpu")).tolist()
    if not all(0 <= following < current <= 1 for current, following in zip(grid[:-1], grid[1:])):
        raise ValueError("FP32 grid is not strictly decreasing")
    request = {"protocol": PROTOCOL, "mode": args.mode, "seed": args.seed, "num_steps": args.num_steps,
               "batch_size": BATCH_SIZE, "sample_count": args.num_samples,
               "shard_index": args.shard_index, "num_shards": args.num_shards,
               "config": config_identity, "baseline_checkpoint": baseline_identity,
               "potential_checkpoint": potential_identity, "decoder_checkpoint": decoder_identity,
               "normalization_stats": stats_identity, "training_chain": chain, "sources": sources,
               "time_grid": grid, "time_shift": 8, "transport_t_eps": T_EPS,
               "denominator": "max(original noise time t, 0.05), including baseline K > 100",
               "precision": "native BF16 heads and B+1.78*(F-B) then FP32; FP32 state, potential and Euler; TF32 false",
               "baseline_interval": [.1, 1.0], "cfg_scale": 1.0, "ig_scale": 1.78,
               "correction": "c=grad_z Phi, Gnew=G+c, all 100 queries, coefficient 1, no added window",
               "pixel_arithmetic": "native BF16 decoder -> clamp(0,1) -> BF16 multiply255 -> uint8 NHWC",
               "noise_schema": f"one CUDA Generator(seed), one randn([{args.num_samples},1024,16,16]); CPU cohort sliced by global IDs; no rank/batch seed offsets",
               "label_schema": "global_id % 1000", "partition": "complete global B8 batches assigned modulo num_shards",
               "all_modes_resident": ["same frozen stage2", "same frozen final potential", "same frozen decoder"],
               "torch_version": str(torch.__version__), "fid_performed": False,
               "benchmark": {"full_rollout_steps": 100, "batch_size": BATCH_SIZE,
                             "warmup_rollouts_per_arm": BENCHMARK_WARMUP,
                             "timed_rollouts_per_arm": BENCHMARK_REPEATS,
                             "reset": "identical fixed B8 initial noise each arm/repetition",
                             "order": "alternate arm order per repetition",
                             "aggregation": "arithmetic mean of complete rollout wall time",
                             "decoder": "resident, never called",
                             "automatic_step_selection": False} if args.mode == "benchmark" else None}
    atomic_json(output / "request.json", request)
    atomic_json(output / "progress.json", {"complete": False, "status": "loading"})
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from utils.model_utils import instantiate_from_config
    os.environ.setdefault("DINOV3_CKPT_DIR", "/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3")
    install_raev2_decoder_config_compat()
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    potential = ScalarGuidancePotential().to(device).eval().requires_grad_(False)
    potential.load_state_dict(checkpoint["potential"], strict=True)
    torch.cuda.synchronize()
    potential_construct_and_copy_seconds = time.perf_counter()-load_started
    potential_load_seconds = potential_checkpoint_read_seconds+potential_construct_and_copy_seconds
    del checkpoint
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    del decoder.encoder
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(checkpoint["ema"], strict=True)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    torch.cuda.synchronize()
    common_models_load_seconds = time.perf_counter()-load_started
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    noise_started = time.perf_counter()
    noise, rng_state = paired_noise(args.seed, device, count=args.num_samples)
    global_noise_hash = tensor_sha256(noise)
    save_npz(output / "paired_noise_audit.npz", first_noise=noise[0].numpy(), rng_state=rng_state.numpy())
    noise_wall = time.perf_counter()-noise_started
    request.update(cuda_device=torch.cuda.get_device_name(device),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   baseline_checkpoint_step=checkpoint_step, global_noise_sha256=global_noise_hash,
                   global_labels_sha256=global_labels_sha256(count=args.num_samples),
                   paired_noise_audit=artifact(output / "paired_noise_audit.npz"))
    atomic_json(output / "request.json", request)
    try:
        if args.mode == "benchmark":
            summary = benchmark(model, potential, noise, grid, output, device, count=args.num_samples)
        else:
            summary = sample(model, potential, decoder, noise, grid, args, output, device, count=args.num_samples)
        summary.update(protocol=PROTOCOL, mode=args.mode, seed=args.seed, num_steps=args.num_steps,
                       request=artifact(output / "request.json"), global_noise_sha256=global_noise_hash,
                       potential_checkpoint_load_seconds=potential_load_seconds,
                       potential_load_components_seconds={"checkpoint_cpu_read": potential_checkpoint_read_seconds,
                                                          "construction_and_device_copy": potential_construct_and_copy_seconds},
                       common_backbone_and_decoder_load_seconds=common_models_load_seconds,
                       noise_generation_copy_and_audit_wall_seconds=noise_wall,
                       peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                       total_wall_seconds_excluding_imports_and_final_summary_write=time.perf_counter()-wall_started,
                       total_cpu_seconds_excluding_imports_and_final_summary_write=time.process_time()-cpu_started,
                       fid_performed=False, training_performed=False)
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", {"complete": True, "summary": artifact(output / "summary.json")})
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(output / "failure.json", {"complete": False, "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter()-wall_started,
                    "completed_batches_remain_in": str(output / "batch_manifest.json")})
        raise


if __name__ == "__main__":
    main()
