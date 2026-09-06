#!/usr/bin/env python3
"""Audit the decoder linearization between two previously sampled endpoints.

Only D(z), J_D(z) delta, and D(z+delta) are evaluated. This script does not
sample, train, evaluate FID, or install a sampler intervention. Pixel block
means are linear 16x16-grid averages, flattened in RGB/NCHW order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file, write_csv
from experiments.sample_raev2_proximal_calibration import DEFAULT_CONFIG
from experiments.sample_raev2_pfr_retiming import load_config
from experiments.raev2_stage1_compat import resolve_decoder_config

PROTOCOL = "raev2_frozen_endpoint_decoder_linearization_v1"
SCALES = ("scale_s1p000000", "scale_s1p780000")
FD_EPSILONS = (.01, .003, .001)


def blockmean(value):
    if value.ndim != 4 or tuple(value.shape[1:]) != (3, 256, 256):
        raise ValueError("expected RGB NCHW 256x256 pixels")
    return value.reshape(len(value), 3, 16, 16, 16, 16).mean(dim=(3, 5)).flatten(1)


def dot(first, second):
    return (first.double()*second.double()).flatten(1).sum(1)


def norm(value):
    return dot(value, value).sqrt()


def ratio(first, second):
    return torch.where(second > 0, first/second.clamp_min(1e-300), 0)


def vector_comparison(actual, predicted):
    na, npred = norm(actual), norm(predicted)
    error = norm(actual-predicted)
    return {"actual_norm": na, "predicted_norm": npred, "error_norm": error,
            "error_rms": error/(actual[0].numel()**.5),
            "relative_error": ratio(error, npred), "cross": dot(actual, predicted),
            "cosine": ratio(dot(actual, predicted), na*npred),
            "prediction_zero": (npred == 0).double()}


def decomposition_stats(d0, d1, linear):
    change, remainder = d1-d0, d1-d0-linear
    result = {"d0_norm": norm(d0), "d1_norm": norm(d1), "change_norm": norm(change),
              "jvp_norm": norm(linear), "remainder_norm": norm(remainder),
              "remainder_over_change_norm": ratio(norm(remainder), norm(change)),
              "change_zero": (norm(change) == 0).double()}
    for first_name, first, second_name, second in (
        ("change", change, "jvp", linear), ("jvp", linear, "remainder", remainder),
        ("change", change, "remainder", remainder), ("d0", d0, "jvp", linear),
        ("d0", d0, "remainder", remainder)):
        result[f"{first_name}_{second_name}_cross"] = dot(first, second)
        result[f"{first_name}_{second_name}_cosine"] = ratio(dot(first, second), norm(first)*norm(second))
    return result


def load_decoder_only(config, device):
    """Build the official RAE decoder without constructing its unused encoder."""
    from stage1.rae import RAE
    from stage1.decoders import GeneralDecoder
    params = config.stage_1.params
    channels, height, width = tuple(config.misc.latent_size)
    patches = height*width
    patch_size = int(params.get("decoder_patch_size", 16))
    decoder_config = resolve_decoder_config(params["decoder_config_path"], hidden_size=channels,
                                            patch_size=patch_size, num_patches=patches)
    if decoder_config is None:
        from transformers import AutoConfig
        decoder_config = AutoConfig.from_pretrained(params["decoder_config_path"])
        decoder_config.hidden_size, decoder_config.patch_size = channels, patch_size
        decoder_config.image_size = int(patch_size*patches**.5)
    rae = RAE.__new__(RAE)
    torch.nn.Module.__init__(rae)
    rae.decoder = GeneralDecoder(decoder_config, num_patches=patches)
    weights = torch.load(params["pretrained_decoder_path"], map_location="cpu", weights_only=True, mmap=True)
    rae.decoder.load_state_dict(weights, strict=True)
    del weights
    stats = torch.load(params["normalization_stat_path"], map_location="cpu", weights_only=True)
    if stats["mean"].shape != (channels, height, width) or stats["var"].shape != stats["mean"].shape:
        raise ValueError("normalization statistics differ from the frozen latent shape")
    rae.register_buffer("latent_mean", stats["mean"].float())
    rae.register_buffer("latent_var", stats["var"].float())
    rae.eps, rae.do_normalization = float(params.get("eps", 1e-5)), True
    return rae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def selected_protocol(root, samples):
    with np.load(root / "sample_protocol.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids" if "sample_ids" in archive else "ids"].astype(np.int64)
        labels = archive["labels"].astype(np.int64)
        source_rows = archive["real_source_rows"].astype(np.int64)
        test_mask = archive["test_mask"].astype(bool)
    if not np.array_equal(ids, np.arange(5000)) or any(value.shape != ids.shape for value in (labels, source_rows, test_mask)):
        raise ValueError("expected the original 5000 ordered global identities")
    if not np.array_equal(np.unique(labels), np.arange(1000)):
        raise ValueError("source protocol must contain every ImageNet class")
    if any(len(np.unique(test_mask[labels == label])) != 1 for label in range(1000)):
        raise ValueError("source split is not class-disjoint")
    selected = np.sort(np.array([ids[labels == label].min() for label in range(1000)]))[:samples]
    if samples == 1000 and (not np.array_equal(selected, np.arange(1000)) or test_mask[selected].sum() != 200):
        raise ValueError("formal selection must be ids 0..999 with the existing 200 held-out classes")
    return {"sample_id": selected, "label": labels[selected],
            "source_row": source_rows[selected], "test_mask": test_mask[selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000 or args.batch_size < 1:
        parser.error("samples must be in [1,1000] and batch-size positive")
    if args.smoke_samples is not None and not 1 <= args.smoke_samples <= args.samples:
        parser.error("smoke-samples must be positive and no larger than samples")
    for key in ("input_root", "output_dir", "config"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    out, root = args.output_dir, args.input_root
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = args.smoke_samples or args.samples
    cohort = "smoke" if args.smoke_samples is not None else ("formal_1k" if count == 1000 else "partial_class_cohort")
    selected = selected_protocol(root, count)
    config = load_config(args.config)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["world_size"] != 4 or manifest["precision"] != "bf16"
            or not manifest["same_noise_and_labels_across_scales"] or manifest["status"] != "complete"
            or manifest["state_key"] != "ema" or manifest["sampler_steps"] != 100
            or manifest["ig_interval"] != [.1, 1.]):
        raise ValueError("source provenance differs from the expected paired BF16 endpoint bank")
    source_files = (Path(__file__), ROOT / "experiments/raev2_stage1_compat.py",
                    ROOT / "experiments/sample_raev2_pfr_retiming.py", ROOT / "external/RAEv2/src/stage1/rae.py",
                    ROOT / "external/RAEv2/src/stage1/decoders/decoder.py",
                    ROOT / "external/RAEv2/src/stage1/decoders/utils.py")
    request = {"protocol": PROTOCOL,
               **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "cohort": cohort, "actual_samples": count, "source_seed": manifest["seed"],
               "selection": "minimum global id per class, sorted by global id; optional smoke/partial prefix only",
               "selected_identity": {key: value.tolist() for key, value in selected.items()},
               "source_manifest_sha256": sha256_file(root / "manifest.json"),
               "original_checkpoint_identity": {key: manifest[key] for key in ("checkpoint", "checkpoint_size", "state_key")},
               "source_protocol_sha256": sha256_file(root / "sample_protocol.npz"),
               "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
               "config_sha256": sha256_file(args.config),
               "decoder_artifacts": {name: {"path": str(config.stage_1.params[name]),
                                              "sha256": sha256_file(Path(config.stage_1.params[name]))}
                    for name in ("pretrained_decoder_path", "normalization_stat_path")},
               "decoder_config_sha256": sha256_file(Path(config.stage_1.params["decoder_config_path"]) / "config.json"),
               "precision": "FP32 no autocast no TF32; torch.func.jvp forward-mode AD; FP64 pixel statistics",
               "decoder_function": "official stage1.RAE.decode, no clamp in differentiated D",
               "decoder_attention": "official explicit matmul/softmax ViTMAESelfAttention; no SDPA backend replacement",
               "delta": "stored z_scale1.78 minus stored z_scale1.0, each promoted from float16 to FP32",
               "blockmean": "[N,768], RGB channel first, 16x16 grid of nonoverlapping 16x16 pixel means; NCHW flatten",
               "clamp_jvp": "raw JVP multiplied by 1{0<=D0<=1}, matching PyTorch clamp boundary derivative; exact boundary counts recorded",
               "fd_epsilons": list(FD_EPSILONS), "fd_selection": "first min(8,N) selected global ids, all epsilon values",
               "cached_decode_comparison": "first min(8,N), stored BF16 decode then clamp then float16 NHWC; drift includes precision/storage effects, no bitwise claim",
               "input_digest_scope": "SHA256 of exact selected row bytes per scale (global-id order); unselected rows are not hashed",
               "no_training": True, "no_sampling": True, "no_fid": True, "no_sampler_integration": True,
               "torch_version": str(torch.__version__)}
    maps, input_files = {}, []
    for group in ("latents", "decoded"):
        for scale in SCALES:
            for rank in range(4):
                path = root / group / f"{scale}_rank{rank:02d}.npy"
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                expected = (1250, 1024, 16, 16) if group == "latents" else (1250, 256, 256, 3)
                if array.shape != expected or array.dtype != np.float16:
                    raise ValueError(f"source shape/dtype mismatch: {path}")
                maps[group, scale, rank] = array
                st = path.stat()
                input_files.append({"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                    "shape": list(array.shape), "dtype": str(array.dtype)})
    request["input_files"] = input_files
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out / "progress.json", {"status": "initializing", "cohort": cohort, "completed": 0, "total": count})
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    decoder = load_decoder_only(config, device)
    request["cuda_device"] = torch.cuda.get_device_name(device)
    atomic_json(out / "request.json", request)
    initialized = time.perf_counter()
    digests = {scale: hashlib.sha256() for scale in SCALES}
    cached_digests = {scale: hashlib.sha256() for scale in SCALES}
    features, rows, fd_rows, cached_rows = {}, [], [], []
    costs = {"jvp_batch_calls": 0, "jvp_sample_evaluations": 0, "jvp_seconds": 0.,
             "endpoint_decoder_batch_calls": 0, "endpoint_decoder_sample_evaluations": 0, "endpoint_decoder_seconds": 0.,
             "fd_decoder_batch_calls": 0, "fd_decoder_sample_evaluations": 0, "fd_decoder_seconds": 0.}

    def timed_decode(value, prefix):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        result = decoder.decode(value)
        torch.cuda.synchronize(device)
        costs[prefix+"_seconds"] += time.perf_counter()-tick
        costs[prefix+"_batch_calls"] += 1
        costs[prefix+"_sample_evaluations"] += len(value)
        return result

    def numpy_rows(group, scale, ids):
        return np.stack([np.asarray(maps[group, scale, int(i)%4][int(i)//4]) for i in ids])

    def finite(values):
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise FloatingPointError("nonfinite decoder/JVP/statistic")

    try:
        # inference_mode would disable forward-mode AD. no_grad preserves JVP.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for start in range(0, count, args.batch_size):
                stop = min(start+args.batch_size, count)
                ids = selected["sample_id"][start:stop]
                endpoints = [numpy_rows("latents", scale, ids) for scale in SCALES]
                for scale, array in zip(SCALES, endpoints):
                    digests[scale].update(array.tobytes())
                z0, z1 = (torch.from_numpy(array.astype(np.float32)).to(device) for array in endpoints)
                delta = z1-z0
                finite((z0, z1, delta))
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                d0, jvp = torch.func.jvp(decoder.decode, (z0,), (delta,))
                torch.cuda.synchronize(device)
                costs["jvp_seconds"] += time.perf_counter()-tick
                costs["jvp_batch_calls"] += 1
                costs["jvp_sample_evaluations"] += len(ids)
                d1 = timed_decode(z1, "endpoint_decoder")
                finite((d0, d1, jvp))
                c0, c1 = d0.clamp(0, 1), d1.clamp(0, 1)
                mask = (d0 >= 0) & (d0 <= 1)
                cjvp = mask*jvp
                raw_r, clamp_r = d1-d0-jvp, c1-c0-cjvp
                pixels = {"raw_full": d0, "raw_guided": d1, "raw_linear": jvp, "raw_remainder": raw_r,
                          "clamped_full": c0, "clamped_guided": c1, "clamped_linear": cjvp, "clamped_remainder": clamp_r,
                          "clamp_change_contribution": (c1-c0)-(d1-d0),
                          "clamp_jvp_contribution": cjvp-jvp, "clamp_remainder_contribution": clamp_r-raw_r}
                for key, value in pixels.items():
                    pooled = blockmean(value).cpu().numpy()
                    if not np.isfinite(pooled).all():
                        raise FloatingPointError("nonfinite blockmean feature")
                    name = key+"_blockmean"
                    if name not in features:
                        features[name] = np.empty((count, 768), dtype=np.float32)
                    features[name][start:stop] = pooled
                stats = {f"{prefix}_{key}": value for prefix, triple in
                         (("raw", (d0, d1, jvp)), ("clamped", (c0, c1, cjvp)))
                         for key, value in decomposition_stats(*triple).items()}
                stats.update(latent_delta_norm=norm(delta),
                             latent_z0_plus_delta_vs_z1_maxabs=(z0+delta-z1).abs().flatten(1).max(1).values.double(),
                             d0_outside_fraction=((d0 < 0) | (d0 > 1)).double().flatten(1).mean(1),
                             d1_outside_fraction=((d1 < 0) | (d1 > 1)).double().flatten(1).mean(1),
                             d0_exact_zero_count=(d0 == 0).flatten(1).sum(1).double(),
                             d0_exact_one_count=(d0 == 1).flatten(1).sum(1).double(),
                             d1_exact_zero_count=(d1 == 0).flatten(1).sum(1).double(),
                             d1_exact_one_count=(d1 == 1).flatten(1).sum(1).double(),
                             d0_min_distance_to_clamp_boundary=torch.minimum(d0.abs(), (d0-1).abs()).flatten(1).min(1).values.double())
                for key in ("clamp_change_contribution", "clamp_jvp_contribution", "clamp_remainder_contribution"):
                    stats[key+"_norm"] = norm(pixels[key])
                finite(stats.values())
                arrays = {key: value.cpu().numpy() for key, value in stats.items()}
                rows.extend({**{key: value[index].item() for key, value in selected.items()},
                             **{key: float(value[local]) for key, value in arrays.items()}}
                            for local, index in enumerate(range(start, stop)))
                validation_count = max(0, min(stop, 8)-start)
                if validation_count:
                    for epsilon in FD_EPSILONS:
                        plus = timed_decode(z0[:validation_count]+epsilon*delta[:validation_count], "fd_decoder")
                        minus = timed_decode(z0[:validation_count]-epsilon*delta[:validation_count], "fd_decoder")
                        finite((plus, minus))
                        for domain, actual, predicted in (
                            ("raw", (plus-minus)/(2*epsilon), jvp[:validation_count]),
                            ("clamped", (plus.clamp(0,1)-minus.clamp(0,1))/(2*epsilon), cjvp[:validation_count])):
                            values = vector_comparison(actual, predicted)
                            finite(values.values())
                            data = {key: value.cpu().numpy() for key, value in values.items()}
                            fd_rows.extend({"sample_id": int(ids[i]), "epsilon": epsilon, "domain": domain,
                                            **{key: float(value[i]) for key, value in data.items()}}
                                           for i in range(validation_count))
                    for scale, candidate in zip(SCALES, (c0, c1)):
                        cached = numpy_rows("decoded", scale, ids[:validation_count])
                        cached_digests[scale].update(cached.tobytes())
                        cached_tensor = torch.from_numpy(cached.astype(np.float32)).permute(0,3,1,2).to(device)
                        values = vector_comparison(candidate[:validation_count], cached_tensor)
                        finite(values.values())
                        data = {key: value.cpu().numpy() for key, value in values.items()}
                        cached_rows.extend({"sample_id": int(ids[i]), "scale": scale,
                                            **{key: float(value[i]) for key, value in data.items()}}
                                           for i in range(validation_count))
                if stop % 32 == 0 or stop == count:
                    write_csv(out / "per_sample_statistics.csv", rows)
                    if fd_rows:
                        write_csv(out / "finite_difference.csv", fd_rows)
                        write_csv(out / "stored_decode_drift.csv", cached_rows)
                    progress = {"status": "auditing", "cohort": cohort, "completed": stop, "total": count,
                                "elapsed_seconds": time.perf_counter()-started, "costs": costs}
                    atomic_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        for artifact in input_files:
            current = Path(artifact["path"]).stat()
            if current.st_size != artifact["size"] or current.st_mtime_ns != artifact["mtime_ns"]:
                raise ValueError("a source array changed during the audit")
        np.savez_compressed(out / "features.npz", sample_ids=selected["sample_id"], labels=selected["label"],
                            source_rows=selected["source_row"], test_mask=selected["test_mask"], **features)
        summary = {"protocol": PROTOCOL, "complete": True, "cohort": cohort, "samples": count,
                   "source_seed": manifest["seed"], "formal_1k": cohort == "formal_1k",
                   "selected_latent_row_sha256": {key: value.hexdigest() for key, value in digests.items()},
                   "validation_cached_pixel_row_sha256": {key: value.hexdigest() for key, value in cached_digests.items()},
                   "features_sha256": sha256_file(out / "features.npz"),
                   "statistics_sha256": sha256_file(out / "per_sample_statistics.csv"),
                   "fd_validation_rows": len(fd_rows), "cached_pixel_validation_rows": len(cached_rows),
                   "initialization_seconds": initialized-started, "total_wall_seconds": time.perf_counter()-started,
                   "costs": costs, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "no_training": True, "no_sampling": True, "no_fid": True, "no_sampler_integration": True,
                   "claim_boundary": "decoder local linearization and clamp decomposition only; finite differences and cached drift are reported without an automatic quality gate"}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {**summary, "status": "complete"})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "cohort": cohort, "completed": len(rows),
                    "total": count, "costs": costs, "error": repr(exc), "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
