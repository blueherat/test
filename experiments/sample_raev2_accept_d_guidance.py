#!/usr/bin/env python3
"""Class-preserving accept-D guidance with a frozen posterior probe.

Each requested class receives independent official RAEv2 proposals until an
independent uniform is below D(image). Every first proposal is retained as the
paired baseline. There is no temperature, acceptance floor, forced acceptance,
attempt limit, or FID access. Short active batches are padded to eight and
dummy images are discarded. Four workers partition the initial global batches.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, atomic_torch_save, sha256_file
from experiments.extract_raev2_posterior_features import unit_l2_cls
from experiments.raev2_posterior_probe import probe_probabilities
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, official_euler_step

PROTOCOL = "raev2_class_preserving_accept_D_guidance_v1"
SEED = 202609068
COHORT = 1000
BATCH = 8
RESTART = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906")
DEFAULT_PROBE = RESTART / "posterior_cls_probe_v1/probe.pt"
DEFAULT_FEATURE_REQUEST = RESTART / "posterior_cls_features_v1/request.json"
DEFAULT_REVIEW = RESTART / "posterior_cls_probe_v1/independent_review.json"


def assigned_class_ids(rank: int, world_size: int) -> list[int]:
    if world_size != 4 or not 0 <= rank < world_size:
        raise ValueError("the frozen protocol uses rank 0..3 and world_size 4")
    return [index for index in range(COHORT) if (index // BATCH) % world_size == rank]


class ClassProposalQueue:
    """One live request per class; rejected requests return to the FIFO tail."""
    def __init__(self, ids):
        ids = list(ids)
        if not ids or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("class requests must be unique, nonnegative and nonempty")
        self.pending = deque(ids)
        self.attempts = {index: 0 for index in ids}
        self.accepted = set()
        self.inflight = None

    @property
    def done(self):
        return not self.pending and self.inflight is None

    def next_batch(self):
        if self.inflight is not None or self.done:
            raise ValueError("finish the active batch before requesting another")
        active = [self.pending.popleft() for _ in range(min(BATCH, len(self.pending)))]
        self.inflight = active
        for index in active:
            self.attempts[index] += 1
        return active + [-1] * (BATCH-len(active))

    def finish_batch(self, class_ids, accept):
        active = [index for index in class_ids if index >= 0]
        if active != self.inflight or len(class_ids) != BATCH or len(accept) != BATCH:
            raise ValueError("completion must match the active padded batch")
        for index, success in zip(class_ids, accept):
            if index < 0:
                continue
            if success:
                self.accepted.add(index)
            else:
                self.pending.append(index)
        self.inflight = None

    def state_dict(self):
        return {"pending": list(self.pending), "attempts": dict(self.attempts),
                "accepted": sorted(self.accepted), "inflight": self.inflight}


def draw_acceptance(probabilities, generator):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ValueError("acceptance probabilities must be finite in [0,1]")
    uniforms = torch.rand(len(probabilities), generator=generator, device="cpu", dtype=torch.float64).numpy()
    return uniforms, uniforms < probabilities


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--noise-bank", type=Path, required=True)
    p.add_argument("--noise-manifest", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, choices=(4,), default=4)
    p.add_argument("--seed", type=int, default=SEED, help="independent noise-bank seed; does not change the frozen method")
    p.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    p.add_argument("--feature-request", type=Path, default=DEFAULT_FEATURE_REQUEST)
    p.add_argument("--independent-review", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--preflight-only", action="store_true", help="verify frozen artifacts and four cached image scores, then exit without proposals")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args(argv)
    if not 0 <= args.rank < 4:
        p.error("rank must be 0..3")
    if not 0 <= args.seed < 2**63-2000004:
        p.error("seed must permit distinct nonnegative retry and acceptance namespaces")
    return args


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_noise_bank(path, manifest_path, expected_seed=SEED):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != "raev2_official_sequential_cuda_noise_bank_v1"
            or manifest.get("complete") is not True
            or manifest.get("seed") != expected_seed or manifest.get("batch_size") != BATCH
            or manifest.get("shape") != [COHORT, 1024, 16, 16]
            or manifest.get("dtype") != "float32"):
        raise ValueError("initial noise manifest differs from the fixed official seed/cohort protocol")
    if manifest.get("labels_sha256") != hashlib.sha256(np.arange(COHORT, dtype=np.int64).tobytes()).hexdigest():
        raise ValueError("initial noise bank label identity mismatch")
    digest = sha256_file(path)
    if digest != manifest.get("file_sha256"):
        raise ValueError("initial noise file hash mismatch")
    values = np.load(path, mmap_mode="r")
    if values.shape != (COHORT, 1024, 16, 16) or values.dtype != np.float32:
        raise ValueError("invalid initial noise tensor")
    raw_digest = hashlib.sha256()
    batches = manifest.get("batches", [])
    if len(batches) != COHORT//BATCH:
        raise ValueError("initial noise manifest must contain all 125 batch hashes")
    for batch_index, start in enumerate(range(0, COHORT, BATCH)):
        chunk = np.ascontiguousarray(values[start:start+BATCH])
        if not np.isfinite(chunk).all():
            raise ValueError("nonfinite initial noise")
        if batches[batch_index] != {"start": start, "stop": start+BATCH,
                                  "noise_sha256": hashlib.sha256(chunk.tobytes()).hexdigest()}:
            raise ValueError("initial noise batch hash/order mismatch")
        raw_digest.update(chunk.tobytes())
    if raw_digest.hexdigest() != manifest.get("noise_sha256"):
        raise ValueError("initial raw noise hash mismatch")
    return values, manifest


def validate_probe_admission(probe_path, review_path):
    """Require the independently reviewed, frozen-before-test posterior."""
    frozen_path = probe_path.parent / "model_freeze.json"
    summary_path = probe_path.parent / "summary.json"
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads(summary_path.read_text())
    review = json.loads(review_path.read_text())
    digest = sha256_file(probe_path)
    if (frozen.get("probe_sha256") != digest or frozen.get("frozen_before_loading_test_features") is not True
            or summary.get("probe", {}).get("probe_sha256") != digest
            or summary.get("complete") is not True or summary.get("formal_pass") is not True
            or not summary.get("empirical_bernstein_upper", float("inf")) < 0):
        raise ValueError("frozen posterior audit did not pass its preselected gate")
    if (review.get("probe_sha256") != digest or review.get("complete") is not True
            or review.get("formal_pass") is not True
            or not review.get("empirical_bernstein_full_target_upper", float("inf")) < 0):
        raise ValueError("independent full-target posterior review did not pass")
    for key, path in (("probe_artifact", probe_path), ("model_freeze_artifact", frozen_path),
                      ("audit_summary_artifact", summary_path)):
        if review.get(key, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"independent review artifact mismatch: {key}")
    return {"model_freeze": artifact(frozen_path), "audit_summary": artifact(summary_path),
            "independent_review": artifact(review_path), "independent_review_contents": review}


def atomic_images(path, images, ids):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, np.stack([images[index] for index in ids]), ids=np.asarray(ids, dtype=np.int64),
                 labels=np.asarray(ids, dtype=np.int64))
    temporary.replace(path)


def score_decoded_images(encoder, pixels, fit, device):
    """Match the locked feature extraction and CPU FP64 posterior exactly."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features, feature_hashes = [], []
    with torch.autocast("cuda", enabled=False):
        for image in pixels:
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
            inputs = encoder.preprocess(tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
            raw = encoder.model.forward_features(inputs)["x_norm_clstoken"].float().cpu()[0].contiguous()
            unit = unit_l2_cls(raw)
            features.append(unit.numpy())
            feature_hashes.append({"raw_cls_fp32_sha256": tensor_hash(raw), "unit_cls_fp64_sha256": tensor_hash(unit)})
    result = probe_probabilities(np.stack(features), fit)
    return result, feature_hashes


def cached_feature_preflight(encoder, fit, feature_request, feature_request_path, device):
    """Exact image/CLS/probability check after switching from proposal precision."""
    selected = [0, 127, 511, 999]
    feature_path = feature_request_path.parent / "fake_train.pt"
    feature_manifest = json.loads((feature_request_path.parent / "split_manifest.json").read_text())
    if sha256_file(feature_path) != feature_manifest["fake_train"]["sha256"]:
        raise ValueError("preflight cached training features changed")
    cached = torch.load(feature_path, map_location="cpu", weights_only=True)
    if cached["request_sha256"] != sha256_file(feature_request_path):
        raise ValueError("preflight feature/request mismatch")
    image_source = feature_request["fake_train"]["samples"]
    if sha256_file(Path(image_source["path"])) != image_source["sha256"]:
        raise ValueError("preflight cached official pixels changed")
    with np.load(image_source["path"]) as archive:
        pixels = archive["arr_0"][selected]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dummy = torch.ones((8, 8), device=device)
        _ = dummy @ dummy
    observed, hashes = score_decoded_images(encoder, pixels, fit, device)
    expected = probe_probabilities(cached["features"][selected].numpy(), fit)
    rows = []
    for slot, sample_id in enumerate(selected):
        record = cached["records"][sample_id]
        if record["sample_id"] != sample_id:
            raise ValueError("preflight cached feature IDs are not ordered")
        rgb = torch.from_numpy(pixels[slot].copy()).permute(2, 0, 1)
        if tensor_hash(rgb) != record["rgb_uint8_sha256"]:
            raise ValueError("preflight RGB identity mismatch")
        for key in ("raw_cls_fp32_sha256", "unit_cls_fp64_sha256"):
            if hashes[slot][key] != record[key]:
                raise ValueError(f"preflight exact feature mismatch: sample {sample_id}, {key}")
        for key in ("logits", "log_D", "D"):
            if observed[key][slot] != expected[key][slot]:
                raise ValueError(f"preflight exact posterior mismatch: sample {sample_id}, {key}")
        rows.append({"sample_id": sample_id, **hashes[slot], "D": float(observed["D"][slot]),
                     "log_D": float(observed["log_D"][slot]), "exact_cached_feature_and_probability_match": True})
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("classifier did not disable TF32")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return {"complete": True, "selected_ids": selected, "records": rows,
            "cached_feature": artifact(feature_path), "cached_pixels": image_source,
            "proposal_tf32_flags_restored": True, "proposal_sampling_performed": False}


def main():
    args = parse_args()
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    ids = assigned_class_ids(args.rank, args.world_size)
    out = args.output_dir / f"shard_{args.rank:02d}_of_{args.world_size:02d}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir()
    (out / "sources").mkdir()
    source_files = (Path(__file__), ROOT / "experiments/sample_raev2_proximal_calibration.py",
                    ROOT / "experiments/extract_raev2_posterior_features.py",
                    ROOT / "experiments/raev2_posterior_probe.py", ROOT / "experiments/raev2_stage1_compat.py",
                    ROOT / "experiments/audit_raev2_proximal_calibration.py",
                    ROOT / "experiments/sample_raev2_pfr_retiming.py", ROOT / "external/RAEv2/src/stage1/rae.py",
                    ROOT / "external/RAEv2/src/stage2/models/DDT.py", ROOT / "external/RAEv2/src/stage2/models/model_utils.py",
                    ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
                    ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py")
    source_artifacts = {}
    for index, path in enumerate(source_files):
        snapshot = out / "sources" / f"{index:02d}_{path.name}"
        snapshot.write_bytes(path.read_bytes())
        source_artifacts[str(path.relative_to(ROOT))] = {**artifact(snapshot), "original_path": str(path)}
    started = time.perf_counter()
    request = {"protocol": PROTOCOL, **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "assigned_class_ids": ids, "assigned_class_count": len(ids), "total_classes": COHORT,
               "partition": "global initial batch8 index modulo4", "queue": "FIFO; every rejection requeued; unique live class requests",
               "batch_size": BATCH, "num_steps": 100, "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
               "state_key": "ema", "sampling_precision": "bf16_autocast_fp32_IG_Euler_tf32_enabled",
               "pixel_arithmetic": "native_decode_clamp_mul255_uint8", "forward_layout": "single_conditional_batch",
               "classifier_precision": "FP32_no_autocast_noTF32_batch1_CLS_then_FP64_unitnorm_and_CPU_probability",
               "acceptance": "uniform_CPU_fp64 < frozen D(image); no multiplication by m, no temperature",
               "first_noise_source": "root official CUDA-generator batch8 bank, addressed by global class ID",
               "retry_noise_seed": args.seed+1000000+args.rank, "acceptance_uniform_seed": args.seed+2000000+args.rank,
               "retry_noise_schema": "continuous CUDA torch.Generator; full batch8 draws including dummy slots; no per-batch reseeding",
               "uniform_schema": "independent continuous CPU torch.Generator; one FP64 uniform per valid proposal, no dummy uniforms",
               "dummy_policy": "model and decoder run all8 slots; class_id=-1 slots ignored for classifier/acceptance/output",
               "attempt_limit": None, "forced_acceptance": False, "fid_read_or_computed": False,
               "source_snapshots": source_artifacts,
               "recovery": "every completed proposal batch, queue, RNG states and evidence durable; no automatic mid-trajectory resume",
               "status": "initializing", "torch_version": str(torch.__version__)}
    atomic_json(out / "request.json", request)
    atomic_json(out / "progress.json", {"status": "initializing", "accepted": 0, "total": len(ids)})
    bank, bank_manifest = load_noise_bank(args.noise_bank, args.noise_manifest, expected_seed=args.seed)
    admission = validate_probe_admission(args.probe, args.independent_review)
    feature_request = json.loads(args.feature_request.read_text())
    fit_payload = torch.load(args.probe, map_location="cpu", weights_only=True)
    if (fit_payload["weight"].dtype != torch.float64 or fit_payload["weight"].shape != (1024,)
            or fit_payload["request"]["features_request_sha256"] != sha256_file(args.feature_request)):
        raise ValueError("probe does not match the frozen FP64 CLS feature protocol")
    for path, expected in fit_payload["request"]["source_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"posterior probe source changed: {path}")
    for item in feature_request["sources"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"feature source changed: {item['path']}")
    if sha256_file(Path(feature_request["dino_weights"]["path"])) != feature_request["dino_weights"]["sha256"]:
        raise ValueError("DINO weights changed since the feature audit")
    fit = {**fit_payload, "weight": fit_payload["weight"].numpy()}
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from utils.model_utils import instantiate_from_config
    from torchvision.utils import save_image
    config = load_config(args.config)
    baseline = feature_request["fake_train"]["request_contents"]
    if (sha256_file(args.config) != baseline["config_sha256"]
            or sha256_file(args.checkpoint) != baseline["checkpoint_sha256"]
            or baseline["num_steps"] != 100 or baseline["ig_scale"] != 1.78
            or baseline["ig_interval"] != [0.1, 1.0]):
        raise ValueError("proposal model/config differ from the q used for posterior fitting")
    for key, expected in baseline["decoder_artifacts"].items():
        if sha256_file(Path(config.stage_1.params[key])) != expected["sha256"]:
            raise ValueError(f"decoder artifact changed: {key}")
    for path, digest in baseline["source_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"official proposal source changed: {path}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    shift = math.sqrt((config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size)) / config.misc.time_dist_shift_base)
    grid = shifted_time_grid(100, shift, device).cpu().tolist()
    if grid != baseline["time_grid"] or float(config.transport.t_eps) >= grid[-2]:
        raise ValueError("proposal Euler grid differs from the q used for posterior fitting")
    local_repo = Path("/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo")
    if not (local_repo / "hubconf.py").is_file():
        raise FileNotFoundError("frozen local DINO repository is required; no download")
    os.environ["DINOV3_CKPT_DIR"] = str(Path(feature_request["dino_weights"]["path"]).parent)
    os.environ["DINOV3_REPO_DIR"] = str(local_repo.resolve())
    install_raev2_decoder_config_compat()
    decoder = instantiate_from_config(config.stage_1).to(device).eval().requires_grad_(False)
    encoder = decoder.encoder
    if encoder.model.norm.elementwise_affine:
        raise ValueError("classifier feature protocol requires the official affine-free final LayerNorm")
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    request.update(noise_bank=artifact(args.noise_bank), noise_manifest=artifact(args.noise_manifest),
                   noise_manifest_contents=bank_manifest, probe=artifact(args.probe),
                   probe_admission=admission,
                   feature_request=artifact(args.feature_request), frozen_train_m=fit["train_m"],
                   train_m_not_used_for_acceptance=True, config=artifact(args.config), checkpoint=artifact(args.checkpoint),
                   decoder_artifacts=baseline["decoder_artifacts"], dino_weights=feature_request["dino_weights"],
                   time_grid=grid, cuda_device=torch.cuda.get_device_name(device), checkpoint_step=int(checkpoint.get("step", 0)),
                   status="sampling")
    del checkpoint
    atomic_json(out / "request.json", request)
    with torch.inference_mode():
        preflight = cached_feature_preflight(encoder, fit, feature_request, args.feature_request, device)
    atomic_json(out / "feature_preflight.json", preflight)
    if args.preflight_only:
        atomic_json(out / "summary.json", {"protocol": PROTOCOL, "complete": True, "preflight_only": True,
                    "feature_preflight": artifact(out / "feature_preflight.json"), "proposal_count": 0})
        atomic_json(out / "progress.json", {"status": "preflight_complete", "accepted": 0, "total": len(ids)})
        return
    retry_rng = torch.Generator(device=device).manual_seed(request["retry_noise_seed"])
    uniform_rng = torch.Generator(device="cpu").manual_seed(request["acceptance_uniform_seed"])
    queue = ClassProposalQueue(ids)
    first_images, accepted_images, first_records, accepted_records = {}, {}, {}, {}
    counters = {"proposal_batches": 0, "valid_proposals": 0, "padding_proposals": 0,
                "full_model_calls": 0, "full_sample_evaluations_including_padding": 0,
                "decoder_sample_evaluations_including_padding": 0, "classifier_evaluations": 0,
                "initial_bank_samples_used": 0, "retry_noise_samples_drawn_including_padding": 0,
                "acceptance_uniforms_drawn": 0}
    first_saved = False
    try:
        with torch.inference_mode():
            while not queue.done:
                class_ids = queue.next_batch()
                valid_count = sum(index >= 0 for index in class_ids)
                valid_ids = class_ids[:valid_count]
                first_flags = [queue.attempts[index] == 1 for index in valid_ids]
                if any(first_flags):
                    if not all(first_flags) or valid_count != BATCH:
                        raise RuntimeError("initial global batches must remain intact before retries")
                    state = torch.from_numpy(np.asarray(bank[valid_ids]).copy()).to(device=device, dtype=torch.float32)
                    noise_source = "initial_bank"
                    counters["initial_bank_samples_used"] += BATCH
                else:
                    state = torch.randn(BATCH, *config.misc.latent_size, device=device, generator=retry_rng, dtype=torch.float32)
                    noise_source = "retry_generator"
                    counters["retry_noise_samples_drawn_including_padding"] += BATCH
                noise_hashes = [tensor_hash(value) for value in state]
                labels = torch.tensor([index if index >= 0 else 0 for index in class_ids], device=device, dtype=torch.int64)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for current, following in zip(grid[:-1], grid[1:]):
                        state = official_euler_step(model, state, labels, current, following)
                    if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                        raise FloatingPointError("invalid official proposal endpoint")
                    endpoint_hashes = [tensor_hash(value) for value in state]
                    decoded = decoder.decode(state)
                    if not bool(torch.isfinite(decoded).all()):
                        raise FloatingPointError("nonfinite official proposal pixels")
                    pixels = decoded.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
                probabilities, feature_hashes = score_decoded_images(encoder, pixels[:valid_count], fit, device)
                uniforms, accepted = draw_acceptance(probabilities["D"], uniform_rng)
                flags = accepted.tolist()+[False]*(BATCH-valid_count)
                batch_index = counters["proposal_batches"]
                batch_path = out / "batches" / f"batch_{batch_index:06d}.npz"
                temporary = batch_path.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    np.savez(file, pixels, class_ids=np.asarray(class_ids, dtype=np.int64), model_labels=labels.cpu().numpy(),
                             valid=np.asarray([index >= 0 for index in class_ids]),
                             noise_sha256=np.asarray(noise_hashes), endpoint_sha256=np.asarray(endpoint_hashes))
                temporary.replace(batch_path)
                archive_sha = sha256_file(batch_path)
                records = []
                for slot, class_id in enumerate(valid_ids):
                    record = {"class_id": class_id, "label": class_id, "attempt": queue.attempts[class_id],
                              "proposal_batch": batch_index, "slot": slot, "noise_source": noise_source,
                              "noise_sha256": noise_hashes[slot], "endpoint_sha256": endpoint_hashes[slot],
                              "pixels_sha256": hashlib.sha256(pixels[slot].tobytes()).hexdigest(),
                              **feature_hashes[slot], "D": float(probabilities["D"][slot]),
                              "log_D": float(probabilities["log_D"][slot]), "uniform": float(uniforms[slot]),
                              "accepted": bool(accepted[slot]), "archive": str(batch_path.relative_to(out)),
                              "archive_sha256": archive_sha}
                    records.append(record)
                    if queue.attempts[class_id] == 1:
                        first_images[class_id] = pixels[slot].copy()
                        first_records[class_id] = record
                    if accepted[slot]:
                        accepted_images[class_id] = pixels[slot].copy()
                        accepted_records[class_id] = record
                with (out / "proposals.jsonl").open("a") as file:
                    for record in records:
                        file.write(json.dumps(record, allow_nan=False)+"\n")
                    file.flush()
                    os.fsync(file.fileno())
                queue.finish_batch(class_ids, flags)
                counters["proposal_batches"] += 1
                counters["valid_proposals"] += valid_count
                counters["padding_proposals"] += BATCH-valid_count
                counters["full_model_calls"] += 100
                counters["full_sample_evaluations_including_padding"] += 100*BATCH
                counters["decoder_sample_evaluations_including_padding"] += BATCH
                counters["classifier_evaluations"] += valid_count
                counters["acceptance_uniforms_drawn"] += valid_count
                atomic_torch_save(out / "rng_and_queue_state.pt", {"retry_rng": retry_rng.get_state(),
                                  "uniform_rng": uniform_rng.get_state(), "queue": queue.state_dict(), "counters": counters})
                if len(first_images) == len(ids) and not first_saved:
                    atomic_images(out / "first_proposals.npz", first_images, ids)
                    atomic_json(out / "first_proposal_records.json", {"records": [first_records[index] for index in ids]})
                    preview = torch.from_numpy(np.stack([first_images[index] for index in ids[:16]])).permute(0, 3, 1, 2).float()/255
                    save_image(preview, out / "first_preview.png", nrow=4)
                    first_saved = True
                progress = {"status": "sampling", "accepted": len(queue.accepted), "total": len(ids),
                            "first_proposals": len(first_images), "first_baseline_saved": first_saved,
                            "pending": len(queue.pending), "maximum_attempts_so_far": max(queue.attempts.values()),
                            **counters, "elapsed_seconds": time.perf_counter()-started}
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if set(accepted_images) != set(ids) or set(first_images) != set(ids):
            raise RuntimeError("completed run must contain every assigned class exactly once")
        atomic_images(out / "accepted_samples.npz", accepted_images, ids)
        atomic_json(out / "accepted_proposal_records.json", {"records": [accepted_records[index] for index in ids]})
        preview = torch.from_numpy(np.stack([accepted_images[index] for index in ids[:16]])).permute(0, 3, 1, 2).float()/255
        save_image(preview, out / "accepted_preview.png", nrow=4)
        summary = {"protocol": PROTOCOL, "complete": True, "rank": args.rank, "world_size": args.world_size,
                   "samples": len(ids), "class_ids": ids, "seed": args.seed, **counters,
                   "first_proposals": artifact(out / "first_proposals.npz"), "accepted_samples": artifact(out / "accepted_samples.npz"),
                   "proposals_jsonl": artifact(out / "proposals.jsonl"), "probe_sha256": request["probe"]["sha256"],
                   "noise_bank_sha256": request["noise_bank"]["sha256"], "noise_manifest_sha256": request["noise_manifest"]["sha256"],
                   "acceptance_rate_valid": len(ids)/counters["valid_proposals"],
                   "attempts_by_class": queue.attempts, "maximum_attempts": max(queue.attempts.values()),
                   "actual_model_nfe_per_accepted_image": counters["full_sample_evaluations_including_padding"]/len(ids),
                   "first_proposal_model_nfe_per_image": 100, "forced_acceptance": False,
                   "fid_read_or_computed": False, "elapsed_seconds": time.perf_counter()-started,
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**progress, "status": "complete", "complete": True})
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(out / "progress.json", {"status": "failed", "accepted": len(queue.accepted),
                    "total": len(ids), "queue": queue.state_dict(), **counters,
                    "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
