#!/usr/bin/env python3
"""Run label-free short rollout probes on the existing targeted DiT traces.

The runner restores a saved state and launches four independent *diagnostic*
rollouts for at most 16 ancestral-DDPM transitions.  It records only the
predicted-clean latent at fixed dyadic horizons.  The probes never replace the
baseline trajectory, select an endpoint, decode an image, or read a quality
label.  Their sole purpose is to estimate the frozen denoiser's operational
consistency defect under its own implemented transition kernel.

This is a post-unseal discovery experiment.  Any quality association found by
a later, separate join must be confirmed on new trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import torch

try:
    from . import reproduce_dit_imagenet256 as strict
    from .extract_dit_v22_doob_consistency_probe import consistency_moments
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    from extract_dit_v22_doob_consistency_probe import consistency_moments


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
DEFAULT_DIT_ROOT = DEFAULT_DATA_ROOT / "baselines/DiT"
if not DEFAULT_DIT_ROOT.exists():
    DEFAULT_DIT_ROOT = Path("/data/users/zhoushunyu/eqvae/baselines/DiT")
DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_custom_traces_cfg_locked"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_doob_consistency_discovery_probe_v1"
)
RUNNER_NAME = "run_dit_doob_consistency_discovery_probes"
ARTIFACT_KIND = "DIT_DOOB_CONSISTENCY_DISCOVERY_PROBE_SHARD_V1"
RNG_NAMESPACE = "eqvae-dit-doob-consistency-discovery-probe-v1"
CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
SEEDS = tuple(range(10, 30))
CHECKPOINTS = (99, 149, 199)
HORIZONS = (1, 2, 4, 8, 16)
PROBE_COUNT = 4
EXPECTED_RUNNER_SHA256 = "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
EXPECTED_STRICT_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
EXPECTED_CHECKPOINT_SHA256 = "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError("CSV schema changed between rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def derive_probe_seed(trace_identity: str, seed: int, checkpoint: int, probe: int) -> int:
    payload = (
        f"{RNG_NAMESPACE}\0{trace_identity}\0{seed}\0{checkpoint}\0{probe}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    if value == 0:
        raise RuntimeError("derived an invalid zero probe seed")
    return value


def validate_source_trace(root: Path, expected_seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    trace_path = root / "trace.npz"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity", {})
    protocol = identity.get("protocol", {})
    outputs = manifest.get("outputs", [])
    trace_record = next(
        (
            record
            for record in outputs
            if isinstance(record, dict) and record.get("relative_path") == "trace.npz"
        ),
        None,
    )
    if (
        manifest.get("status") != "complete"
        or identity.get("runner") != "trace_dit_imagenet256_custom_batch"
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or protocol.get("global_torch_seed") != expected_seed
        or tuple(protocol.get("class_ids_ordered", [])) != CLASSES
        or protocol.get("sampling_steps") != 250
        or protocol.get("cfg_scale") != 4.0
        or identity.get("runner_source", {}).get("sha256") != EXPECTED_RUNNER_SHA256
        or identity.get("strict_reproduction_helper", {}).get("sha256")
        != EXPECTED_STRICT_SHA256
        or identity.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or completion.get("identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or not isinstance(trace_record, dict)
        or trace_record.get("sha256") != sha256_file(trace_path)
    ):
        raise RuntimeError(f"source trace validation failed: {root}")
    with np.load(trace_path, allow_pickle=False) as archive:
        required = ("state_before", "pred_xstart", "internal_timestep")
        if any(name not in archive.files for name in required):
            raise RuntimeError(f"source trace arrays missing: {trace_path}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    if (
        arrays["state_before"].shape != (8, 250, 4, 32, 32)
        or arrays["pred_xstart"].shape != (8, 250, 4, 32, 32)
        or arrays["internal_timestep"].shape != (250,)
        or arrays["state_before"].dtype != np.float32
        or arrays["pred_xstart"].dtype != np.float32
        or arrays["internal_timestep"].dtype != np.int16
        or not np.array_equal(arrays["internal_timestep"], np.arange(249, -1, -1, dtype=np.int16))
        or not np.isfinite(arrays["state_before"]).all()
        or not np.isfinite(arrays["pred_xstart"]).all()
    ):
        raise RuntimeError(f"source trace tensor contract changed: {trace_path}")
    records = manifest.get("trace_array_records", {})
    for name in ("state_before", "pred_xstart", "internal_timestep"):
        if records.get(name, {}).get("raw_sha256") != raw_sha256(arrays[name]):
            raise RuntimeError(f"source trace raw hash failed for {name}: {trace_path}")
    return manifest, arrays


def model_query(
    diffusion: Any,
    model: Any,
    first: torch.Tensor,
    *,
    internal_t: int,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(first)
    full = torch.cat([first, first], dim=0)
    null = torch.full_like(labels, strict.NULL_CLASS_ID)
    model_kwargs = {
        "y": torch.cat([labels, null], dim=0),
        "cfg_scale": strict.CFG_SCALE,
    }
    t = torch.full((2 * batch,), internal_t, dtype=torch.long, device=first.device)
    with torch.no_grad():
        output = diffusion.p_mean_variance(
            model.forward_with_cfg,
            full,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
    mean = output["mean"][:batch].contiguous()
    prediction = output["pred_xstart"][:batch].contiguous()
    sigma = torch.exp(0.5 * output["log_variance"][:batch]).contiguous()
    expected = (batch, 4, 32, 32)
    if any(tuple(value.shape) != expected for value in (mean, prediction, sigma)):
        raise RuntimeError("model query tensor shape changed")
    return mean, prediction, sigma


def probe_checkpoint(
    diffusion: Any,
    model: Any,
    *,
    state: np.ndarray,
    expected_prediction: np.ndarray,
    trace_identity: str,
    global_seed: int,
    checkpoint: int,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    labels = torch.tensor(CLASSES, dtype=torch.long, device=device)
    reference = torch.from_numpy(state).to(device)
    expected = torch.from_numpy(expected_prediction).to(device)
    collected: list[np.ndarray] = []
    probe_seeds: list[int] = []
    for probe in range(PROBE_COUNT):
        probe_seed = derive_probe_seed(trace_identity, global_seed, checkpoint, probe)
        probe_seeds.append(probe_seed)
        generator = torch.Generator(device=device).manual_seed(probe_seed)
        first = reference.clone()
        predictions: list[np.ndarray] = []
        for offset in range(max(HORIZONS) + 1):
            internal_t = 249 - checkpoint - offset
            if internal_t < 0:
                raise RuntimeError("probe horizon ran beyond terminal time")
            mean, prediction, sigma = model_query(
                diffusion, model, first, internal_t=internal_t, labels=labels
            )
            if offset == 0 and not torch.equal(prediction, expected):
                error = float((prediction - expected).abs().max())
                raise RuntimeError(
                    f"restored current prediction is not bitwise exact at seed={global_seed}, "
                    f"checkpoint={checkpoint}, max_abs={error}"
                )
            if offset in HORIZONS:
                predictions.append(
                    np.ascontiguousarray(prediction.cpu().numpy(), dtype=np.float32)
                )
            if offset == max(HORIZONS):
                break
            noise = torch.randn(
                mean.shape, dtype=mean.dtype, device=device, generator=generator
            )
            first = mean + sigma * noise
        if len(predictions) != len(HORIZONS):
            raise AssertionError("probe failed to record every horizon")
        collected.append(np.stack(predictions, axis=0))
    if len(set(probe_seeds)) != PROBE_COUNT:
        raise RuntimeError("probe RNG streams collided")
    return np.ascontiguousarray(np.stack(collected, axis=0)), probe_seeds


def score_predictions(
    current: np.ndarray,
    probes: np.ndarray,
    *,
    global_seed: int,
    checkpoint: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # current: [B,C,H,W]; probes: [R,Horizon,B,C,H,W]
    if current.shape != (8, 4, 32, 32) or probes.shape != (
        PROBE_COUNT,
        len(HORIZONS),
        8,
        4,
        32,
        32,
    ):
        raise RuntimeError("scoring tensors have the wrong shape")
    horizon_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for slot, class_id in enumerate(CLASSES):
        cross_sum = 0.0
        energy_sum = 0.0
        for horizon_index, horizon in enumerate(HORIZONS):
            deltas = probes[:, horizon_index, slot].astype(np.float64) - current[slot]
            moments = consistency_moments(deltas)
            cross_sum += moments["pair_cross_u_stat"]
            energy_sum += moments["update_energy"]
            horizon_rows.append(
                {
                    "global_seed": global_seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "checkpoint": checkpoint,
                    "horizon": horizon,
                    **moments,
                }
            )
        coherence = cross_sum / max(energy_sum, 1e-12)
        sample_rows.append(
            {
                "global_seed": global_seed,
                "class_slot": slot,
                "class_id": class_id,
                "checkpoint": checkpoint,
                "dyadic_pair_cross_sum": cross_sum,
                "dyadic_update_energy_sum": energy_sum,
                "dyadic_coherence": coherence,
                "positive_dyadic_coherence_descriptive": max(coherence, 0.0),
            }
        )
    return sample_rows, horizon_rows


def run_source(
    diffusion: Any,
    model: Any,
    *,
    trace_root: Path,
    output_root: Path,
    global_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    source = trace_root / f"targeted_scan_v1_seed{global_seed}"
    manifest, arrays = validate_source_trace(source, global_seed)
    destination = output_root / f"seed{global_seed:02d}"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite an existing seed product: {destination}")
    started = time.time()
    current_predictions = []
    all_probe_predictions = []
    sample_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    stream_rows: list[dict[str, Any]] = []
    trace_identity = str(manifest["identity_sha256"])
    for checkpoint in CHECKPOINTS:
        current = np.ascontiguousarray(arrays["pred_xstart"][:, checkpoint])
        probes, seeds = probe_checkpoint(
            diffusion,
            model,
            state=np.ascontiguousarray(arrays["state_before"][:, checkpoint]),
            expected_prediction=current,
            trace_identity=trace_identity,
            global_seed=global_seed,
            checkpoint=checkpoint,
            device=device,
        )
        current_predictions.append(current)
        all_probe_predictions.append(probes)
        rows, detailed = score_predictions(
            current, probes, global_seed=global_seed, checkpoint=checkpoint
        )
        sample_rows.extend(rows)
        horizon_rows.extend(detailed)
        stream_rows.extend(
            {
                "checkpoint": checkpoint,
                "probe_index": probe,
                "stream_seed": seed,
            }
            for probe, seed in enumerate(seeds)
        )
    arrays_out = {
        "global_seed": np.asarray(global_seed, dtype=np.int64),
        "class_ids": np.asarray(CLASSES, dtype=np.int16),
        "checkpoints": np.asarray(CHECKPOINTS, dtype=np.int16),
        "horizons": np.asarray(HORIZONS, dtype=np.int16),
        "current_pred_xstart": np.ascontiguousarray(
            np.stack(current_predictions, axis=0), dtype=np.float32
        ),
        "probe_pred_xstart": np.ascontiguousarray(
            np.stack(all_probe_predictions, axis=0), dtype=np.float32
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=output_root))
    try:
        atomic_npz(staging / "probes.npz", arrays_out)
        write_csv(staging / "sample_scores.csv", sample_rows)
        write_csv(staging / "horizon_scores.csv", horizon_rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_DOOB_CONSISTENCY_DISCOVERY_PROBE_SEED_V1",
            "status": "complete",
            "runner": RUNNER_NAME,
            "global_seed": global_seed,
            "class_ids": list(CLASSES),
            "checkpoints": list(CHECKPOINTS),
            "horizons": list(HORIZONS),
            "probe_count": PROBE_COUNT,
            "rng_namespace": RNG_NAMESPACE,
            "probe_streams": stream_rows,
            "source_trace": {
                "root": str(source),
                "identity_sha256": trace_identity,
                "manifest_file_sha256": sha256_file(source / "manifest.json"),
                "trace_file_sha256": sha256_file(source / "trace.npz"),
            },
            "firewall": {
                "labels_reviews_pngs_decoded_images_opened": False,
                "endpoint_or_branch_selected": False,
                "baseline_state_changed": False,
                "external_metric_or_embedding_opened": False,
                "source_arrays_read": ["state_before", "pred_xstart", "internal_timestep"],
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("probes.npz", "sample_scores.csv", "horizon_scores.csv")
            },
            "wall_seconds": time.time() - started,
        }
        record["identity_sha256"] = canonical_sha256(record)
        write_json(staging / "record.json", record)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "global_seed": global_seed,
        "output": str(destination),
        "identity_sha256": record["identity_sha256"],
        "wall_seconds": record["wall_seconds"],
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    trace_root = args.trace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dit_root = args.dit_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if sha256_file(Path(strict.__file__).resolve()) != EXPECTED_STRICT_SHA256:
        raise RuntimeError("strict reproduction helper changed")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("DiT checkpoint changed")
    source = strict.validate_repository(dit_root, checkpoint)
    if source.get("pinned_source_sha256", {}).get("models.py") != "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca":
        raise RuntimeError("DiT source validation changed")
    selected = [seed for index, seed in enumerate(SEEDS) if index % args.shard_count == args.shard_index]
    if not selected:
        raise RuntimeError("selected shard has no seeds")
    shard = output_root / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    if shard.exists():
        raise RuntimeError(f"refusing to overwrite shard: {shard}")
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    prior_cwd = Path.cwd()
    prior_path = list(sys.path)
    prior_grad = torch.is_grad_enabled()
    preexisting = {
        name
        for name in sys.modules
        if name == "models" or name == "download" or name == "diffusion" or name.startswith("diffusion.")
    }
    if preexisting:
        raise RuntimeError(f"ambiguous pre-imported DiT modules: {sorted(preexisting)}")
    started = time.time()
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from download import find_model
        from models import DiT_models

        device = torch.device("cuda")
        torch.manual_seed(20260828 + args.shard_index)
        torch.set_grad_enabled(False)
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to(device)
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        records = []
        for ordinal, seed in enumerate(selected, start=1):
            print(f"shard {args.shard_index}: seed {seed} ({ordinal}/{len(selected)})", flush=True)
            records.append(
                run_source(
                    diffusion,
                    model,
                    trace_root=trace_root,
                    output_root=shard,
                    global_seed=seed,
                    device=device,
                )
            )
    finally:
        torch.set_grad_enabled(prior_grad)
        os.chdir(prior_cwd)
        sys.path[:] = prior_path
        for name in list(sys.modules):
            if (name in {"models", "download", "diffusion"} or name.startswith("diffusion.")) and name not in preexisting:
                sys.modules.pop(name, None)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "status": "complete",
        "runner": RUNNER_NAME,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "seeds": selected,
        "records": records,
        "method": {
            "checkpoints": list(CHECKPOINTS),
            "horizons": list(HORIZONS),
            "probe_count": PROBE_COUNT,
            "rng_namespace": RNG_NAMESPACE,
            "quality_direction_selected": False,
            "status": "label_free_mechanics_and_post_unseal_discovery_features_only",
        },
        "firewall": {
            "labels_reviews_pngs_decoded_images_opened": False,
            "endpoint_or_branch_selected": False,
            "baseline_state_changed": False,
        },
        "wall_seconds": time.time() - started,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    write_json(shard / "receipt.json", receipt)
    print(json.dumps({"status": "complete", "shard": str(shard), "identity_sha256": receipt["identity_sha256"]}, indent=2))


def self_test() -> None:
    values = {
        derive_probe_seed("a" * 64, 10, checkpoint, probe)
        for checkpoint in CHECKPOINTS
        for probe in range(PROBE_COUNT)
    }
    assert len(values) == len(CHECKPOINTS) * PROBE_COUNT
    current = np.zeros((8, 4, 32, 32), dtype=np.float32)
    probes = np.ones(
        (PROBE_COUNT, len(HORIZONS), 8, 4, 32, 32), dtype=np.float32
    )
    sample_rows, horizon_rows = score_predictions(
        current, probes, global_seed=10, checkpoint=99
    )
    assert len(sample_rows) == 8 and len(horizon_rows) == 8 * len(HORIZONS)
    assert all(math.isclose(float(row["dyadic_coherence"]), 1.0) for row in sample_rows)
    print("self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=False)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = (
        args.dit_root / "pretrained_models" / strict.CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint
    )
    if args.self_test:
        self_test()
        raise SystemExit(0)
    if args.shard_index is None:
        parser.error("--shard-index is required unless --self-test is used")
    return args


if __name__ == "__main__":
    run(parse_args())
