#!/usr/bin/env python3
"""Decompose the frozen RAEv2 full/base gap by depth and readout crossings.

Uses existing, unmodified teacher/rollout states. The native corners must be
bitwise equal to model.forward and the saved original predictions. Crossed
readouts are off-training-path diagnostics, not validated negative models.
There is no target MSE, fitting, decoding, FID, intervention, or candidate gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import (
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, atomic_json, atomic_torch_save,
    sha256_file, write_csv,
)
from experiments.sample_raev2_pfr_retiming import load_config

PROTOCOL = "raev2_frozen_depth_readout_crossing_audit_v1"
DEFAULT_STATES = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states")
CORNERS = ("Y8B", "Y28F", "Y8F", "Y28B")


def full_readout(model, state, features, t_emb_base):
    # seq is per-image-token modulation conditioning; decoder tokens still
    # originate from the original noisy input through x_embedder.
    seq = model.s_projector(F.silu(t_emb_base + features))
    x = model.x_embedder(state)
    for index in range(model.num_dec_blocks):
        x = model.blocks[model.num_enc_blocks + index](x, seq, model.dec_rope)
    x = model.final_layer(x, seq)
    return model.unpatchify(x, model.x_patch_size)


def base_readout(model, features, t_emb_base):
    x_base = F.silu(t_emb_base + features)
    x_base = model.base_final_layer(x_base, x_base)
    return model.unpatchify(x_base, model.s_patch_size)


@torch.no_grad()
def four_corners(model, state, times, labels):
    """One shared native encoder pass, native readouts first in native order."""
    if model.training or (model.base_model_depth, model.num_enc_blocks) != (8, 28):
        raise ValueError("this audit requires the frozen eval-mode 8/28-depth DDT")
    if getattr(model, "use_cfg_conds", False):
        raise ValueError("CFG condition tokens are outside the frozen protocol")
    conditions = {"context": labels, "attn_mask": None}
    seq, t_emb_base = model._build_sequence(state, times, conditions)
    attn_mask = model._build_attn_mask(seq, conditions)
    h8 = None
    for index in range(model.num_enc_blocks):
        seq = model.blocks[index](seq, model.enc_rope, attn_mask)
        if index + 1 == model.base_model_depth:
            h8 = seq[:, :model.s_embedder.num_patches, :]
    h28 = seq[:, :model.s_embedder.num_patches, :]
    # Keep exact DiTwDDTHeadIG.forward order for both native corners.
    y28f = full_readout(model, state, h28, t_emb_base)
    y8b = base_readout(model, h8, t_emb_base)
    y8f = full_readout(model, state, h8, t_emb_base)
    y28b = base_readout(model, h28, t_emb_base)
    return {"Y8B": y8b, "Y28F": y28f, "Y8F": y8f, "Y28B": y28b}


def norms(value):
    return value.flatten(1).square().sum(1).sqrt()


def ratios(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), 0)


def crossing_metrics(corners):
    values = {name: value.float().double() for name, value in corners.items()}
    a, b, c, d = (values[name] for name in ("Y28F", "Y8F", "Y28B", "Y8B"))
    gap = a-d
    depth = .5*((a-b)+(c-d))
    readout = .5*((a-c)+(b-d))
    interaction = a-b-c+d
    gap_norm = norms(gap)
    result = {f"{name}_norm": norms(value) for name, value in values.items()}
    result.update(raw_gap_norm=gap_norm, zero_gap=(gap_norm == 0).double(),
                  decomposition_residual_norm=norms(depth+readout-gap))
    for name, value in (("depth", depth), ("readout", readout), ("interaction", interaction)):
        length = norms(value)
        result[f"{name}_norm"] = length
        result[f"{name}_norm_over_gap_norm"] = ratios(length, gap_norm)
        result[f"{name}_cos_gap"] = ratios((value*gap).flatten(1).sum(1), length*gap_norm)
    result["depth_cos_readout"] = ratios((depth*readout).flatten(1).sum(1), norms(depth)*norms(readout))
    for first, second in itertools.combinations(CORNERS, 2):
        length = norms(values[first]-values[second])
        result[f"pair_{first}_{second}_norm"] = length
        result[f"pair_{first}_{second}_norm_over_gap_norm"] = ratios(length, gap_norm)
    if not bool((result["decomposition_residual_norm"] <= 1e-12*(1+gap_norm)).all()):
        raise RuntimeError("depth + readout failed to reconstruct the original gap")
    return result


def assert_native_parity(corners, native, saved):
    result = {}
    for name, fresh, cached in (("Y28F", native[0], saved["full"]),
                                ("Y8B", native[1], saved["base"])):
        candidate = corners[name]
        same_native = candidate.dtype == fresh.dtype and torch.equal(candidate, fresh)
        # Snapshots intentionally store native BF16 outputs promoted to FP32.
        same_saved = cached.dtype == torch.float32 and torch.equal(candidate.float().cpu(), cached)
        result[f"{name}_native_bitwise"] = same_native
        result[f"{name}_saved_bitwise"] = same_saved
        if not same_native or not same_saved:
            native_error = float((candidate.float()-fresh.float()).abs().max())
            saved_error = float((candidate.float().cpu()-cached).abs().max())
            raise RuntimeError(f"{name} parity failed: native={same_native}, saved={same_saved}, "
                               f"max errors native={native_error}, saved={saved_error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save-first-snapshot", action="store_true")
    args = parser.parse_args()
    for key in ("state_dir", "output_dir", "config", "checkpoint"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    paths = sorted(args.state_dir.glob("step_*.pt"))
    parent = args.state_dir.parent
    source_request = json.loads((parent / "request.json").read_text())
    source_summary = json.loads((parent / "summary.json").read_text())
    if not source_summary.get("complete") or len(paths) != 10:
        raise ValueError("expected the complete 10-snapshot normal-noise audit")
    if [int(path.stem.split("_")[-1]) for path in paths] != source_request["snapshot_indices"]:
        raise ValueError("snapshot indices differ from the source request")
    if source_request["precision"] != "bf16" or source_request["tf32"] is not True:
        raise ValueError("source precision differs from the frozen BF16/TF32 protocol")
    state_artifacts = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    locked_states = {int(row["step_index"]): row["sha256"] for row in source_summary["snapshots"]}
    if any(row["sha256"] != locked_states[int(path.stem.split("_")[-1])]
           for path, row in zip(paths, state_artifacts)):
        raise ValueError("snapshot content changed since the source audit")
    config_hash, checkpoint_hash = sha256_file(args.config), sha256_file(args.checkpoint)
    if (config_hash != source_request["config_sha256"] or
            checkpoint_hash != source_request["checkpoint_sha256"]):
        raise ValueError("config or checkpoint differs from the frozen states")
    source_paths = (Path(__file__), ROOT / "experiments/sample_raev2_pfr_retiming.py",
                    ROOT / "external/RAEv2/src/stage2/models/DDT.py",
                    ROOT / "external/RAEv2/src/stage2/models/model_utils.py")
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    for name, digest in source_hashes.items():
        if name in source_request["source_sha256"] and digest != source_request["source_sha256"][name]:
            raise ValueError(f"model or configuration helper changed since source audit: {name}")
    out.mkdir(parents=True, exist_ok=True)
    request = {"protocol": PROTOCOL,
               **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
               "states": state_artifacts, "source_sha256": source_hashes,
               "state_source_request_sha256": sha256_file(parent / "request.json"),
               "state_source_summary_sha256": sha256_file(parent / "summary.json"),
               "state_seed": source_request["seed"], "state_noise_sha256": source_summary["noise_sha256"],
               "state_key": "ema", "precision": "BF16_autocast_TF32_true_B8",
               "model_parameters_and_states": "fp32", "diagnostic_arithmetic": "fp64 after FP32 promotion",
               "corners": {"Y8B": "native base readout at h8", "Y28F": "native full readout at h28",
                           "Y8F": "unchanged full decoder/readout conditioned by h8; original x_embedder(state)",
                           "Y28B": "unchanged base readout applied to h28"},
               "decomposition": {"depth": "0.5*((Y28F-Y8F)+(Y28B-Y8B))",
                                 "readout": "0.5*((Y28F-Y28B)+(Y8F-Y8B))",
                                 "interaction": "Y28F-Y8F-Y28B+Y8B"},
               "zero_norm_convention": "ratios/cosines with zero denominator defined as zero; zero_gap explicitly recorded",
               "extra_verification": "one separate complete model.forward per B8 state batch",
               "no_clean_target_mse": True, "no_decode": True, "no_fid": True,
               "no_intervention": True, "no_candidate_gate": True,
               "interpretation": "crossed readouts are off-training-path diagnostics; decomposition is algebraic, not a quality or causal-error guarantee",
               "torch_version": str(torch.__version__)}
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing", "rows": 0})
    started = time.perf_counter()
    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from utils.model_utils import instantiate_from_config
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request.update(checkpoint_step=int(checkpoint.get("step", 0)), cuda_device=torch.cuda.get_device_name(device))
    del checkpoint
    atomic_json(out / "request.json", request)
    rows, parity_rows, identities, batches = [], [], None, 0
    first_corners = {}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for position, (path, artifact) in enumerate(zip(paths, state_artifacts)):
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError("snapshot changed during audit")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                ids = torch.as_tensor(payload["sample_ids"]).tolist()
                labels = torch.as_tensor(payload["labels"]).long()
                index, current = int(payload["step_index"]), float(payload["t"])
                if identities is None:
                    identities = (ids, labels.tolist())
                if ((ids, labels.tolist()) != identities or len(ids) != 8
                        or current != source_request["time_grid"][index]):
                    raise ValueError("snapshot identities, batch size, or time changed")
                times = torch.full((8,), current, device=device, dtype=torch.float32)
                for domain in ("teacher", "rollout"):
                    saved = payload[domain]
                    if saved["state"].dtype != torch.float32 or tuple(saved["state"].shape) != (8, *config.misc.latent_size):
                        raise ValueError("snapshot must contain original FP32 B8 states")
                    state = saved["state"].to(device)
                    corners = four_corners(model, state, times, labels.to(device))
                    native = model(state, times, context=labels.to(device), attn_mask=None)
                    parity = assert_native_parity(corners, native, saved)
                    batches += 1
                    parity_rows.append({"step_index": index, "t": current, "domain": domain, **parity})
                    with torch.autocast("cuda", enabled=False):
                        metrics = {key: value.cpu().numpy() for key, value in crossing_metrics(corners).items()}
                    if any(not np.isfinite(value).all() for value in metrics.values()):
                        raise FloatingPointError("nonfinite crossing diagnostic")
                    for slot, sample_id in enumerate(ids):
                        rows.append({"step_index": index, "t": current, "domain": domain,
                                     "sample_id": sample_id, "label": int(labels[slot]),
                                     **{key: float(value[slot]) for key, value in metrics.items()}})
                    if position == 0 and args.save_first_snapshot:
                        first_corners[domain] = {key: value.float().cpu() for key, value in corners.items()}
                write_csv(out / "per_sample_depth_readout.csv", rows)
                write_csv(out / "native_parity.csv", parity_rows)
                progress = {"status": "auditing", "last_step_index": index, "rows": len(rows),
                            "state_batches": batches, "elapsed_seconds": time.perf_counter()-started}
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        groups = []
        for domain in ("teacher", "rollout"):
            for step in source_request["snapshot_indices"]:
                selected = [row for row in rows if row["domain"] == domain and row["step_index"] == step]
                groups.append({"domain": domain, "step_index": step, "t": selected[0]["t"],
                               "samples": len(selected),
                               **{key: float(np.mean([row[key] for row in selected])) for key in metrics}})
        write_csv(out / "per_step_depth_readout.csv", groups)
        if len(rows) != 160 or batches != 20:
            raise RuntimeError("incomplete frozen-state audit")
        if args.save_first_snapshot:
            atomic_torch_save(out / "first_snapshot_corners.pt", {"step_index": 0,
                              "sample_ids": identities[0], "labels": identities[1], **first_corners})
        summary = {"protocol": PROTOCOL, "complete": True, "unique_sample_states": len(rows),
                   "state_batches": batches, "all_native_and_saved_bitwise_parity": True,
                   "shared_encoder_passes": batches, "extra_native_verification_encoder_passes": batches,
                   "full_readout_passes_including_verification": 3*batches,
                   "base_readout_passes_including_verification": 3*batches,
                   "sample_encoder_evaluations_including_verification": 16*batches,
                   "elapsed_seconds": time.perf_counter()-started,
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "csv_sha256": sha256_file(out / "per_sample_depth_readout.csv"),
                   "no_decode": True, "no_fid": True, "no_intervention": True,
                   "conclusion_boundary": request["interpretation"]}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**summary, "status": "complete"})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "rows": len(rows),
                    "state_batches": batches, "error": repr(exc), "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
