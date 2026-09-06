#!/usr/bin/env python3
"""Numerical feasibility of a 2049-parameter, full-rollout guidance gate.

Eight images (four fixed classes, two independent noises each) only verify
gradients, zero-gate parity, and cost. No optimizer, FID or image selection.
All backbone weights are frozen; derivatives through their inputs remain on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file
from experiments.audit_raev2_decoder_linearization import load_decoder_only
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG, DEFAULT_CHECKPOINT
from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
from experiments.raev2_distribution_guidance import AffineTokenGuidance, conditional_energy_training_objective

DATA = Path("/home/zhoushunyu/data/eqvae")
RESTART = DATA / "experiments/raev2_guidance_restart_20260906"


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=202609080)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--real-features", type=Path, default=RESTART / "posterior_cls_features_v1/real_train.pt")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = load_config(args.config)
    if (config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != .1
            or config.guidance.ig.t_max != 1 or config.sampler.num_steps != 100
            or config.transport.prediction != "x"):
        raise ValueError("expected the frozen official IG configuration")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["DINOV3_REPO_DIR"] = str(DATA / "models/RAEv2/dinov3_repo")
    os.environ["DINOV3_CKPT_DIR"] = str(DATA / "models/RAEv2/encoders/dinov3")
    real_payload = torch.load(args.real_features, map_location="cpu", weights_only=False)
    if real_payload["split"] != "real_train" or real_payload["features"].shape != (1000, 1024):
        raise ValueError("expected frozen genuine last-CLS real A training features")
    real_labels = torch.arange(4, device=device)
    real_parts = []
    for label in range(4):
        take = torch.where(real_payload["labels"] == label)[0]
        if len(take) != 1:
            raise ValueError("expected one training reference per fixed pilot class")
        real_parts.append(real_payload["features"][take[0]])
    real_features = torch.stack(real_parts).to(device=device, dtype=torch.float64)
    labels = torch.arange(4, device=device).repeat_interleave(2)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn((8, *config.misc.latent_size), device=device, generator=rng)
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
                      / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    sources = [Path(__file__), ROOT / "experiments/raev2_distribution_guidance.py",
               ROOT / "experiments/audit_raev2_decoder_linearization.py",
               ROOT / "external/RAEv2/src/stage2/models/DDT.py",
               ROOT / "external/RAEv2/src/encoders/vision_encoder.py"]
    request = {"protocol": "raev2_full_rollout_affine_gate_gradient_v1", "seed": args.seed,
               "samples": 8, "labels": labels.cpu().tolist(), "microbatch": 2, "steps": 100,
               "feature": "genuine DINOv3 final CLS; FP64 unit L2; frozen weights, input derivatives retained",
               "pixels": "continuous FP32 decoder -> clamp(0,1) -> *255 -> official preprocess; no uint8",
               "objective": "uniform mean over 4 classes of 2*gen-real mean distance minus offdiagonal gen-gen mean distance; real-real constant omitted",
               "reference": "one fixed A training image per class; numerical pilot, not population evidence",
               "precision": "FP32, no autocast/TF32; FP64 unit features and scalar loss",
               "gate": "raw F/B tokenwise affine, 2049 parameters, zero initialized, shared across all 100 steps",
               "time_grid": grid, "initial_noise_sha256": tensor_hash(initial),
               "real_features_sha256": sha256_file(args.real_features),
               "config_sha256": sha256_file(args.config), "checkpoint_sha256": sha256_file(args.checkpoint),
               "decoder_sha256": sha256_file(Path(config.stage_1.params["pretrained_decoder_path"])),
               "normalization_sha256": sha256_file(Path(config.stage_1.params["normalization_stat_path"])),
               "dinov3_sha256": sha256_file(DATA / "models/RAEv2/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
               "sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
               "fd_epsilons": [.001, .0003, .0001],
               "fd_direction": "normalized exact numerical parameter gradient at zero; used only for derivative verification",
               "no_optimizer": True, "no_fid": True, "no_sample_selection": True,
               "zero_gate_parity": "every step compares guided clean tensor to independent official combination from identical F/B outputs",
               "torch_version": str(torch.__version__), "cuda_device": torch.cuda.get_device_name(device)}
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing"})
    from utils.model_utils import instantiate_from_config
    from encoders import create_encoder
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(weights["ema"], strict=True)
    request["checkpoint_step"] = int(weights.get("step", 0))
    del weights
    decoder = load_decoder_only(config, device)
    encoder = create_encoder(config.stage_1.params["encoder_name"], device=device, resolution=256)
    encoder.eval().requires_grad_(False)
    if encoder.model.norm.elementwise_affine:
        raise ValueError("feature norm differs from the frozen real-feature protocol")
    gate = AffineTokenGuidance(1024).to(device)
    if sum(p.numel() for p in gate.parameters()) != 2049:
        raise ValueError("unexpected guidance parameter count")
    atomic_json(out / "request.json", request)
    phase, ledgers = "", {}
    parity = {"comparisons": 0, "all_equal": True}

    def count(name, samples):
        ledger = ledgers.setdefault(phase, {})
        ledger[name+"_calls"] = ledger.get(name+"_calls", 0)+1
        ledger[name+"_samples"] = ledger.get(name+"_samples", 0)+samples

    def observe(state):
        count("decoder", len(state))
        pixels = decoder.decode(state).clamp(0, 1)
        count("encoder", len(state))
        raw = encoder.model.forward_features(encoder.preprocess(pixels*255))["x_norm_clstoken"]
        raw = raw.double()
        return raw/torch.linalg.vector_norm(raw, dim=1, keepdim=True)

    def rollout(differentiable, check_zero=False):
        observations, endpoints = [], []
        for start in range(0, 8, 2):
            state, target_labels = initial[start:start+2].clone(), labels[start:start+2]
            for step, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                times = torch.full((2,), current, device=device, dtype=torch.float32)
                # Bind every time/label in defaults: backward checkpoint
                # recomputation must not capture the loop's final values.
                def transition(z, times=times, ys=target_labels, t=current, s=following):
                    count("stage2", len(z))
                    full, base = model(z, times, context=ys, attn_mask=None)
                    official = full+.78*(full-base) if .1 <= t <= 1 else full
                    guided = gate(full, base, official)
                    if check_zero:
                        parity["comparisons"] += 1
                        parity["all_equal"] &= bool(torch.equal(guided, official))
                        if not parity["all_equal"]:
                            raise AssertionError("zero guidance changed official clean output")
                    return z-(t-s)*((z-guided)/t)
                state = (activation_checkpoint(transition, state, use_reentrant=False, preserve_rng_state=False)
                         if differentiable else transition(state))
            endpoints.append(state.detach().cpu())
            observations.append(activation_checkpoint(observe, state, use_reentrant=False, preserve_rng_state=False)
                                if differentiable else observe(state))
            atomic_json(out / "progress.json", {"status": phase, "completed_paths": start+2,
                        "ledgers": ledgers, "wall_seconds": time.perf_counter()-started})
            print(json.dumps({"phase": phase, "completed_paths": start+2}), flush=True)
        features = torch.cat(observations)
        loss = conditional_energy_training_objective(features, labels, real_features, real_labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite terminal training objective")
        return loss, features, torch.cat(endpoints)

    try:
        phase = "autograd_forward"
        begin = time.perf_counter()
        loss, features, endpoint = rollout(True, check_zero=True)
        phase = "autograd_backward"
        loss.backward()
        torch.cuda.synchronize(device)
        gradient = parameters_to_vector([p.grad for p in gate.parameters()]).detach().clone()
        if not bool(torch.isfinite(gradient).all()) or gradient.norm() == 0:
            raise FloatingPointError("gate gradient is zero or nonfinite")
        if any(p.grad is not None for module in (model, decoder, encoder) for p in module.parameters()):
            raise AssertionError("frozen backbone accumulated parameter gradients")
        gradient_seconds = time.perf_counter()-begin
        zero = parameters_to_vector(gate.parameters()).detach().clone()
        direction = (gradient.double()/gradient.double().norm()).float()
        predicted = float(gradient.double() @ direction.double())
        baseline_loss = float(loss.detach())
        torch.save({"initial_noise": initial.cpu(), "labels": labels.cpu(), "endpoint": endpoint,
                    "unit_features": features.detach().cpu(), "gradient": gradient.cpu(),
                    "fd_direction": direction.cpu()}, out / "zero_gate_and_gradient.pt")
        del loss, features, endpoint
        records = []
        for epsilon in request["fd_epsilons"]:
            values = {}
            for sign in (1, -1):
                phase = f"fd_{epsilon:g}_{sign:+d}"
                tick = time.perf_counter()
                with torch.no_grad():
                    vector_to_parameters(zero+sign*epsilon*direction, gate.parameters())
                    value, _, _ = rollout(False)
                torch.cuda.synchronize(device)
                values[sign] = float(value)
                ledgers[phase]["wall_seconds"] = time.perf_counter()-tick
            fd = (values[1]-values[-1])/(2*epsilon)
            records.append({"epsilon": epsilon, "plus_loss": values[1], "minus_loss": values[-1],
                            "finite_difference": fd, "autograd_directional": predicted,
                            "relative_error": abs(fd-predicted)/abs(predicted)})
            atomic_json(out / "finite_difference.json", {"records": records})
        with torch.no_grad():
            vector_to_parameters(zero, gate.parameters())
        summary = {"protocol": request["protocol"], "complete": True, "samples": 8,
                   "baseline_training_objective_without_real_constant": baseline_loss,
                   "parameter_count": 2049, "gradient_norm": float(gradient.double().norm()),
                   "gradient_max_abs": float(gradient.abs().max()), "zero_gate_parity": parity,
                   "gradient_forward_backward_seconds": gradient_seconds, "ledgers": ledgers,
                   "finite_difference": records, "total_wall_seconds": time.perf_counter()-started,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                   "gradient_artifact_sha256": sha256_file(out / "zero_gate_and_gradient.pt"),
                   "no_optimizer": True, "no_fid": True, "weights_left_at_zero": bool((parameters_to_vector(gate.parameters()) == 0).all()),
                   "boundary": "Numerical gradient/cost feasibility only. Fixed four-class empirical objective, no population improvement or image-quality claim."}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {"status": "complete", "complete": True})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "error": repr(exc),
                    "phase": phase, "ledgers": ledgers, "wall_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
