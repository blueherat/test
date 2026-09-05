#!/usr/bin/env python3
"""Equal-compute fixed-step audit for PFR versus ordinary Internal Guidance.

The comparison uses Heun integration and counts the two distinct model calls:

* a paired strong/depth-4 evaluation runs the complete 12-block backbone;
* a PFR reference query runs only the first four blocks and the depth-4 head.

With the measured prefix/full latency ratio 0.391, ordinary IG with 32 Heun
steps costs 64 full forwards per batch, while PFR with 27 Heun steps costs
54 full forwards plus 27 prefix calls, or 64.557 full-forward equivalents.
The runner also records synchronized wall time, so the analytical accounting
can be checked against the actual hardware rather than treated as a claim.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch_seed_schema import (  # noqa: E402
    BATCH_SEED_SCHEMAS,
    DEFAULT_BATCH_SEED_SCHEMA,
    batch_rng_manifest,
    batch_seed,
    manifest_uses_batch_rng,
)
from experiments.implicit_fixed_point_solvers import integrate_fixed_grid  # noqa: E402
from experiments.pfr_eulerian_decomposition import (  # noqa: E402
    DECOMPOSITION_KINDS,
    EulerianDecompositionField,
)
from experiments.pfr_ou_semigroup_controls import (  # noqa: E402
    OU_SPECTRAL_CONTROL_KINDS,
    OUSpectralControlField,
)
from experiments.pfr_stage_reuse import (  # noqa: E402
    StageReusedProjectedField,
    integrate_stage_reused_heun,
)
from experiments.pfr_retiming_controls import (  # noqa: E402
    MULTIDEPTH_RETIMING_KINDS,
    RETIMING_CONTROL_KINDS,
    MultiDepthRetimingField,
    RetimingControlField,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
    parse_gpus,
    read_json,
)
from experiments.run_imagenet100_sit_pfr_query_controls import (  # noqa: E402
    QueryControlledField,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
    load_runtime,
)


REFERENCE_NAME = "imagenet100_validation_n5000_adm_stats.npz"
PREFIX_FULL_RATIO = 0.391
DEPTH10_PREFIX_FULL_RATIO = 0.817
CONDITION_KINDS = (
    "ordinary_ig",
    "projected",
    "projected_stage_reuse",
    *DECOMPOSITION_KINDS,
    *RETIMING_CONTROL_KINDS,
    *MULTIDEPTH_RETIMING_KINDS,
    *OU_SPECTRAL_CONTROL_KINDS,
)
LABEL_MODES = ("random", "balanced")


def _tag(value: float) -> str:
    return (f"{value:.6f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


@dataclass(frozen=True)
class Condition:
    kind: str
    steps: int
    anchor_horizon: float = HORIZON
    revision_scale: float = 1.0

    def validate(self) -> None:
        if self.kind not in CONDITION_KINDS:
            raise ValueError(f"unsupported condition kind: {self.kind}")
        if self.steps < 4:
            raise ValueError("at least four Heun steps are required")
        if not math.isfinite(self.anchor_horizon) or self.anchor_horizon <= 0.0:
            raise ValueError("anchor_horizon must be positive and finite")
        if not math.isfinite(self.revision_scale):
            raise ValueError("revision_scale must be finite")
        if self.kind != "time_only" and (
            self.anchor_horizon != HORIZON or self.revision_scale != 1.0
        ):
            raise ValueError(
                "custom horizon/scale are currently supported only for time_only"
            )

    @property
    def name(self) -> str:
        self.validate()
        prefixes = {
            "ordinary_ig": "ordinary",
            "projected": "pfr",
            "projected_stage_reuse": "pfr_stage_reuse",
            "time_only": "pfr_time_only",
            "material_guided": "pfr_material_guided",
            "frame_guided": "pfr_frame_guided",
            "weak_time_pair": "pfr_weak_time_pair",
            "strong_time": "pfr_strong_time",
            "strong_time_rms_matched": "pfr_strong_time_rms_matched",
            "weak_common_strong": "pfr_weak_common_strong",
            "weak_unique_strong": "pfr_weak_unique_strong",
            "weak_common_depth10": "pfr_weak_common_depth10",
            "weak_unique_depth10": "pfr_weak_unique_depth10",
            "ou_d1_common": "pfr_ou_d1_common",
            "ou_d1_unique": "pfr_ou_d1_unique",
            "ou_d1_rms_matched": "pfr_ou_d1_rms_matched",
            "ou_d1_common_first": "pfr_ou_d1_common_first",
            "ou_d1_unique_first": "pfr_ou_d1_unique_first",
            "ou_d1_common_norm_raw_direction_first": (
                "pfr_ou_d1_common_norm_raw_direction_first"
            ),
            "ou_d1_common_direction_raw_norm_first": (
                "pfr_ou_d1_common_direction_raw_norm_first"
            ),
            "ou_d1_common_then_projected": "pfr_ou_d1_common_then_projected",
            "ou_d1_common_plus_spatial": "pfr_ou_d1_common_plus_spatial",
            "ou_d1_energy_adaptive": "pfr_ou_d1_energy_adaptive",
            "ou_d1_two_scale_span_first": "pfr_ou_d1_two_scale_span_first",
            "ou_d1_strong_common_first": "pfr_ou_d1_strong_common_first",
            "ou_d1_strong_unique_first": "pfr_ou_d1_strong_unique_first",
            "ou_d1_strong_common_norm_raw_direction_first": (
                "pfr_ou_d1_strong_common_norm_raw_direction_first"
            ),
            "ou_d1_strong_common_direction_raw_norm_first": (
                "pfr_ou_d1_strong_common_direction_raw_norm_first"
            ),
            "ou_d1_strong_anchored_common_direction_raw_norm_first": (
                "pfr_ou_d1_strong_anchored_common_direction_raw_norm_first"
            ),
            "ou_d1_strong_anchored_angular_first": (
                "pfr_ou_d1_strong_anchored_angular_first"
            ),
            "ou_d2_strong_common_first": "pfr_ou_d2_strong_common_first",
            "ou_d2_strong_common_direction_raw_norm_first": (
                "pfr_ou_d2_strong_common_direction_raw_norm_first"
            ),
            "ou_d2_common_first": "pfr_ou_d2_common_first",
            "ou_d2_unique_first": "pfr_ou_d2_unique_first",
        }
        prefix = prefixes[self.kind]
        if self.kind == "time_only" and (
            self.anchor_horizon != HORIZON or self.revision_scale != 1.0
        ):
            prefix += (
                f"_h{_tag(self.anchor_horizon)}"
                f"_r{_tag(self.revision_scale)}"
            )
        return f"{prefix}_heun_n{self.steps}"

    @property
    def cli_spec(self) -> str:
        self.validate()
        if self.kind == "time_only" and (
            self.anchor_horizon != HORIZON or self.revision_scale != 1.0
        ):
            return (
                f"{self.kind}:{self.steps}:{self.anchor_horizon}:"
                f"{self.revision_scale}"
            )
        return f"{self.kind}:{self.steps}"

    @property
    def full_calls_per_batch(self) -> int:
        self.validate()
        return 2 * self.steps

    @property
    def query_calls_per_batch(self) -> int:
        self.validate()
        if self.kind == "ordinary_ig":
            return 0
        first, second, _ = segment_step_counts(self.steps)
        if self.kind == "projected_stage_reuse":
            return first + second
        # Every Heun evaluation before t=.5 queries W(q). The t=.5 endpoint
        # itself uses gamma=0 and therefore has no query.
        active_evaluations = 2 * (first + second) - 1
        if self.kind in {"material_guided", "frame_guided"}:
            return 2 * active_evaluations
        if self.kind in OU_SPECTRAL_CONTROL_KINDS:
            if self.kind in {
                "ou_d1_strong_common_first",
                "ou_d1_strong_unique_first",
                "ou_d1_strong_common_norm_raw_direction_first",
                "ou_d1_strong_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_common_direction_raw_norm_first",
                "ou_d1_strong_anchored_angular_first",
                "ou_d2_strong_common_first",
                "ou_d2_strong_common_direction_raw_norm_first",
            }:
                return active_evaluations
            if self.kind in {
                "ou_d1_common_first",
                "ou_d1_unique_first",
                "ou_d1_common_norm_raw_direction_first",
                "ou_d1_common_direction_raw_norm_first",
                "ou_d1_common_then_projected",
                "ou_d2_common_first",
                "ou_d2_unique_first",
            }:
                first, second, _ = segment_step_counts(self.steps)
                # The first segment's terminal evaluation is exactly t=.25
                # and therefore belongs to the unfiltered second stage.
                return 4 * first + 2 * second - 2
            if self.kind == "ou_d1_common_plus_spatial":
                first, second, _ = segment_step_counts(self.steps)
                return 6 * first + 2 * second - 3
            if self.kind == "ou_d1_two_scale_span_first":
                first, second, _ = segment_step_counts(self.steps)
                return 6 * first + 2 * second - 3
            return 2 * active_evaluations
        if self.kind in RETIMING_CONTROL_KINDS:
            return 0
        if self.kind in MULTIDEPTH_RETIMING_KINDS:
            return active_evaluations
        return active_evaluations

    @property
    def full_query_calls_per_batch(self) -> int:
        self.validate()
        if self.kind in {
            "ou_d1_strong_common_first",
            "ou_d1_strong_unique_first",
            "ou_d1_strong_common_norm_raw_direction_first",
            "ou_d1_strong_common_direction_raw_norm_first",
            "ou_d1_strong_anchored_common_direction_raw_norm_first",
            "ou_d1_strong_anchored_angular_first",
            "ou_d2_strong_common_first",
            "ou_d2_strong_common_direction_raw_norm_first",
        }:
            first, _, _ = segment_step_counts(self.steps)
            return 2 * first - 1
        if self.kind not in RETIMING_CONTROL_KINDS:
            return 0
        first, second, _ = segment_step_counts(self.steps)
        return 2 * (first + second) - 1

    def query_full_ratio(
        self,
        prefix_full_ratio: float,
        depth10_prefix_full_ratio: float = DEPTH10_PREFIX_FULL_RATIO,
    ) -> float:
        if not math.isfinite(prefix_full_ratio) or prefix_full_ratio <= 0.0:
            raise ValueError("prefix/full ratio must be positive and finite")
        if (
            not math.isfinite(depth10_prefix_full_ratio)
            or depth10_prefix_full_ratio <= 0.0
        ):
            raise ValueError("depth-10 prefix/full ratio must be positive and finite")
        if self.kind in MULTIDEPTH_RETIMING_KINDS:
            return depth10_prefix_full_ratio
        return prefix_full_ratio

    def full_forward_equivalents(
        self,
        prefix_full_ratio: float,
        depth10_prefix_full_ratio: float = DEPTH10_PREFIX_FULL_RATIO,
    ) -> float:
        query_ratio = self.query_full_ratio(
            prefix_full_ratio, depth10_prefix_full_ratio
        )
        return (
            self.full_calls_per_batch
            + self.full_query_calls_per_batch
            + query_ratio * self.query_calls_per_batch
        )

    def payload(
        self,
        prefix_full_ratio: float,
        depth10_prefix_full_ratio: float = DEPTH10_PREFIX_FULL_RATIO,
    ) -> dict[str, Any]:
        query_ratio = self.query_full_ratio(
            prefix_full_ratio, depth10_prefix_full_ratio
        )
        payload = {
            "format": "eqvae_pfr_equal_compute_condition_v1",
            "name": self.name,
            "kind": self.kind,
            "integrator": "heun",
            "steps": self.steps,
            "full_calls_per_batch": self.full_calls_per_batch,
            "query_calls_per_batch": self.query_calls_per_batch,
            "prefix_full_ratio": query_ratio,
            "query_prefix_depth": (
                10 if self.kind in MULTIDEPTH_RETIMING_KINDS else 4
            ),
            "full_forward_equivalents_per_batch": self.full_forward_equivalents(
                prefix_full_ratio, depth10_prefix_full_ratio
            ),
        }
        if self.full_query_calls_per_batch:
            payload["full_query_calls_per_batch"] = (
                self.full_query_calls_per_batch
            )
        if self.kind == "time_only" and (
            self.anchor_horizon != HORIZON or self.revision_scale != 1.0
        ):
            payload["anchor_horizon"] = self.anchor_horizon
            payload["revision_scale"] = self.revision_scale
        return payload


def parse_condition(text: str) -> Condition:
    try:
        parts = text.split(":")
        if len(parts) == 2:
            kind, steps_text = parts
            condition = Condition(kind=kind, steps=int(steps_text))
        elif len(parts) == 4:
            kind, steps_text, horizon_text, scale_text = parts
            condition = Condition(
                kind=kind,
                steps=int(steps_text),
                anchor_horizon=float(horizon_text),
                revision_scale=float(scale_text),
            )
        else:
            raise ValueError
        condition.validate()
        return condition
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "condition must be KIND:STEPS or "
            "time_only:STEPS:HORIZON:REVISION_SCALE, where KIND is one of "
            + ",".join(CONDITION_KINDS)
        ) from error


def segment_step_counts(total: int) -> tuple[int, int, int]:
    """Allocate steps to the two IG intervals and the unguided suffix."""

    if total < 4:
        raise ValueError("at least four fixed steps are required")
    first = max(1, int(round(0.25 * total)))
    second = max(1, int(round(0.25 * total)))
    late = total - first - second
    if late <= 0:
        raise ValueError("step allocation left no unguided suffix")
    return first, second, late


def balanced_labels(
    *, num_samples: int, num_classes: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Return a deterministic, exactly balanced and shuffled class bank."""

    if num_samples % num_classes:
        raise ValueError("balanced labels require num_samples divisible by num_classes")
    labels = torch.arange(num_samples, dtype=torch.int64) % num_classes
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    return labels[permutation].to(device)


def reusable(path: Path, condition: Condition, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        metrics = result.get("metrics")
        return (
            result["condition"]
            == condition.payload(
                args.prefix_full_ratio, args.depth10_prefix_full_ratio
            )
            and int(manifest["sampling"]["num_samples"]) == args.num_samples
            and int(manifest["sampling"]["batch_size"]) == args.batch_size
            and int(manifest["sampling"]["seed"]) == args.seed
            and manifest["sampling"]["label_mode"] == args.label_mode
            and manifest_uses_batch_rng(
                manifest, args.seed, schema=args.batch_seed_schema
            )
            and (
                args.skip_fid
                or isinstance(metrics, dict)
                and all(
                    math.isfinite(float(metrics[key]))
                    for key in ("fid", "sfid", "inception_score")
                )
            )
        )
    except Exception:
        return False


def _integrate_condition(
    *,
    field: Any,
    noise: torch.Tensor,
    steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    state = noise.float()
    total_solver_nfe = 0
    counts = segment_step_counts(steps)
    for (start, end), count in zip(
        ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0)), counts, strict=True
    ):
        times = torch.linspace(start, end, count + 1, device=device)
        if isinstance(field, StageReusedProjectedField):
            result = integrate_stage_reused_heun(field, state, times)
        else:
            result = integrate_fixed_grid(
                field,
                state,
                times,
                method="heun",
            )
        state = result.endpoint
        total_solver_nfe += result.nfe
    if total_solver_nfe != field.nfe:
        raise AssertionError("solver and field NFE accounting disagree")
    return state, total_solver_nfe


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image

    condition = parse_condition(args.condition)
    repo = args.repo.expanduser().resolve()
    data = args.data.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        print(json.dumps({"event": "reuse", "condition": condition.name}), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
        adm_python=args.adm_python,
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    deeper_head = None
    if condition.kind in MULTIDEPTH_RETIMING_KINDS:
        depth10_path = (
            data
            / "multiscale_guidance_study_v1/runs/depth10_v/"
            "checkpoints/step_00050000.pt"
        )
        sit_module, source_metadata = runtime.modules["load_official_sit_module"](
            Path(runtime.modules["DEFAULT_OFFICIAL_SIT_REPO"])
            .expanduser()
            .resolve(),
            verify_source=True,
        )
        deeper_head = runtime.modules["load_internal_head_for_source"](
            checkpoint_path=depth10_path,
            name="depth10_v",
            head_weights="ema",
            model=runtime.strong,
            sit_module=sit_module,
            source_checkpoint_path=runtime.paths["strong"],
            source_metadata=source_metadata,
            device=device,
        )
    vae = (
        AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    all_balanced_labels = None
    if args.label_mode == "balanced":
        all_balanced_labels = balanced_labels(
            num_samples=args.num_samples,
            num_classes=runtime.modules["NUM_CLASSES"],
            seed=args.seed ^ 0x5A17,
            device=device,
        )

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    total_nfe = 0
    total_query_nfe = 0
    total_full_query_nfe = 0
    integration_seconds = 0.0
    decode_seconds = 0.0
    cursor = 0
    preview = None

    with torch.inference_mode():
        while cursor < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - cursor)
            batch_index = cursor // args.batch_size
            generator = torch.Generator(device=device).manual_seed(
                batch_seed(
                    args.seed,
                    batch_index,
                    schema=args.batch_seed_schema,
                )
            )
            noise = torch.randn(
                current_batch,
                *runtime.modules["LATENT_SHAPE"],
                generator=generator,
                device=device,
            )
            if all_balanced_labels is None:
                labels = torch.randint(
                    0,
                    runtime.modules["NUM_CLASSES"],
                    (current_batch,),
                    generator=generator,
                    device=device,
                )
            else:
                labels = all_balanced_labels[cursor : cursor + current_batch]

            if condition.kind == "projected_stage_reuse":
                field = StageReusedProjectedField(runtime, labels)
            elif condition.kind in DECOMPOSITION_KINDS:
                field = EulerianDecompositionField(
                    runtime,
                    labels,
                    condition.kind,
                    anchor_horizon=condition.anchor_horizon,
                    revision_scale=condition.revision_scale,
                )
            elif condition.kind in RETIMING_CONTROL_KINDS:
                field = RetimingControlField(runtime, labels, condition.kind)
            elif condition.kind in MULTIDEPTH_RETIMING_KINDS:
                if deeper_head is None:
                    raise AssertionError("multidepth condition did not load its head")
                field = MultiDepthRetimingField(
                    runtime,
                    labels,
                    condition.kind,
                    deeper_head=deeper_head,
                )
            elif condition.kind in OU_SPECTRAL_CONTROL_KINDS:
                field = OUSpectralControlField(
                    runtime,
                    labels,
                    condition.kind,
                )
            else:
                field = QueryControlledField(
                    runtime,
                    labels,
                    condition.kind,
                    record_diagnostics=False,
                )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            endpoint, batch_nfe = _integrate_condition(
                field=field,
                noise=noise,
                steps=condition.steps,
                device=device,
            )
            torch.cuda.synchronize(device)
            integration_seconds += time.perf_counter() - started
            if batch_nfe != condition.full_calls_per_batch:
                raise AssertionError("unexpected full-call count")
            if field.query_nfe != condition.query_calls_per_batch:
                raise AssertionError("unexpected prefix-query count")
            if getattr(field, "full_query_nfe", 0) != (
                condition.full_query_calls_per_batch
            ):
                raise AssertionError("unexpected full-query count")
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(condition.name)

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            decoded = runtime.modules["decode_latents_in_chunks"](
                vae,
                endpoint,
                scaling_factor=runtime.modules["SD_VAE_SCALING_FACTOR"],
                chunk_size=args.vae_decode_batch_size,
            )
            torch.cuda.synchronize(device)
            decode_seconds += time.perf_counter() - started

            stop = cursor + current_batch
            images[cursor:stop] = runtime.modules["official_pixel_quantization"](
                decoded
            )
            labels_array[cursor:stop] = labels.cpu().numpy().astype(
                np.int16, copy=False
            )
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if preview is None:
                preview = decoded[: min(16, len(decoded))].cpu()
            total_nfe += field.nfe
            total_query_nfe += field.query_nfe
            total_full_query_nfe += getattr(field, "full_query_nfe", 0)
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 256 == 0:
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "generated": cursor,
                            "total": args.num_samples,
                            "integration_seconds": integration_seconds,
                        }
                    ),
                    flush=True,
                )

    batches = math.ceil(args.num_samples / args.batch_size)
    if total_nfe != batches * condition.full_calls_per_batch:
        raise AssertionError("aggregate full-call count mismatch")
    if total_query_nfe != batches * condition.query_calls_per_batch:
        raise AssertionError("aggregate prefix-query count mismatch")
    if total_full_query_nfe != batches * condition.full_query_calls_per_batch:
        raise AssertionError("aggregate full-query count mismatch")

    sample_path = output / f"samples_n{args.num_samples}.npz"
    label_path = output / f"labels_n{args.num_samples}.npy"
    activation_path = output / "adm_activations.npz"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    if preview is None:
        raise RuntimeError("sampling produced no preview")
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))

    query_ratio = condition.query_full_ratio(
        args.prefix_full_ratio, args.depth10_prefix_full_ratio
    )
    equivalent_total = (
        total_nfe
        + total_full_query_nfe
        + query_ratio * total_query_nfe
    )
    manifest = {
        "format": "eqvae_pfr_equal_compute_samples_v1",
        "condition": condition.payload(
            args.prefix_full_ratio, args.depth10_prefix_full_ratio
        ),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "label_mode": args.label_mode,
            "integrator": "heun",
        },
        "batch_rng": batch_rng_manifest(
            args.seed, schema=args.batch_seed_schema
        ),
        "strong": runtime.strong_metadata,
        "weak_checkpoint": str(runtime.paths["depth4"]),
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
        "total_query_nfe": total_query_nfe,
        "total_full_query_nfe": total_full_query_nfe,
        "full_forward_equivalents": equivalent_total,
        "timing": {
            "integration_seconds": integration_seconds,
            "decode_seconds": decode_seconds,
            "integration_samples_per_second": args.num_samples / integration_seconds,
        },
        "samples": str(sample_path),
        "labels": str(label_path),
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    runtime.modules["atomic_json_dump"](
        manifest, output / "sampling_manifest.json"
    )

    metrics = None
    if not args.skip_fid:
        del vae, runtime
        gc.collect()
        torch.cuda.empty_cache()
        metric_path = output / "adm_metrics.json"
        environment = os.environ.copy()
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        subprocess.run(
            [
                str(args.adm_python),
                str(repo / "experiments/compute_adm_fid.py"),
                "--reference",
                str(data / "adm_reference_stats" / REFERENCE_NAME),
                "--samples",
                str(sample_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--activations-output",
                str(activation_path),
                "--output",
                str(metric_path),
            ],
            cwd=repo,
            env=environment,
            check=True,
        )
        metrics = read_json(metric_path)

    result = {
        "format": "eqvae_pfr_equal_compute_result_v1",
        "condition": condition.payload(
            args.prefix_full_ratio, args.depth10_prefix_full_ratio
        ),
        "sampling_manifest": manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
        "activations_retained": bool(not args.skip_fid),
    }
    atomic_json(result_path, result)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "condition": condition.name,
                "fid": None if metrics is None else metrics["fid"],
                "full_forward_equivalents": equivalent_total,
                "integration_seconds": integration_seconds,
            }
        ),
        flush=True,
    )


def run_one(
    condition: Condition,
    gpu: int,
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / condition.name
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        return read_json(result_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--repo",
        str(repo),
        "--data",
        str(data),
        "--adm-python",
        str(adm_python),
        "--condition",
        condition.cli_spec,
        "--output-dir",
        str(output),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--batch-seed-schema",
        args.batch_seed_schema,
        "--label-mode",
        args.label_mode,
        "--prefix-full-ratio",
        str(args.prefix_full_ratio),
        "--depth10-prefix-full-ratio",
        str(args.depth10_prefix_full_ratio),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        command.append("--keep-samples")
    if args.skip_fid:
        command.append("--skip-fid")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-100:])
        raise RuntimeError(f"{condition.name} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    metrics = result.get("metrics")
    suffix = "" if metrics is None else f" FID={float(metrics['fid']):.4f}"
    print(f"[GPU {gpu}] {condition.name}:{suffix}", flush=True)
    return result


def write_summary(
    root: Path, results: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        condition = result["condition"]
        manifest = result["sampling_manifest"]
        metrics = result.get("metrics")
        rows.append(
            {
                "condition": condition["name"],
                "kind": condition["kind"],
                "steps": condition["steps"],
                "full_calls": manifest["total_nfe"],
                "prefix_calls": manifest["total_query_nfe"],
                "full_query_calls": manifest.get("total_full_query_nfe", 0),
                "full_forward_equivalents": manifest[
                    "full_forward_equivalents"
                ],
                "integration_seconds": manifest["timing"]["integration_seconds"],
                "integration_samples_per_second": manifest["timing"][
                    "integration_samples_per_second"
                ],
                "fid": None if metrics is None else float(metrics["fid"]),
                "fid_mean_component": (
                    None if metrics is None else float(metrics["fid_mean_component"])
                ),
                "fid_covariance_component": (
                    None
                    if metrics is None
                    else float(metrics["fid_covariance_component"])
                ),
                "sfid": None if metrics is None else float(metrics["sfid"]),
                "inception_score": (
                    None if metrics is None else float(metrics["inception_score"])
                ),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    rows.sort(key=lambda row: row["condition"])
    if len({row["noise_sha256"] for row in rows}) != 1:
        raise RuntimeError("conditions did not use paired noise")
    if len({row["label_sha256"] for row in rows}) != 1:
        raise RuntimeError("conditions did not use paired labels")

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "all_conditions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    measured = [row for row in rows if row["fid"] is not None]
    best = min(measured, key=lambda row: row["fid"]) if measured else None
    atomic_json(
        summary_dir / "summary.json",
        {
            "format": "eqvae_pfr_equal_compute_summary_v1",
            "question": (
                "Does PFR retain its gain when Heun integration is matched in "
                "full-forward-equivalent compute?"
            ),
            "protocol": {
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "batch_rng": batch_rng_manifest(
                    args.seed, schema=args.batch_seed_schema
                ),
                "label_mode": args.label_mode,
                "prefix_full_ratio": args.prefix_full_ratio,
                "depth10_prefix_full_ratio": args.depth10_prefix_full_ratio,
            },
            "best": best,
            "rows": rows,
            "table": str(csv_path),
        },
    )
    print(json.dumps({"event": "summary", "best": best}, indent=2), flush=True)


def sweep(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data / "pfr_equal_compute_v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    conditions = tuple(dict.fromkeys(args.conditions))
    atomic_json(
        root / "request.json",
        {
            "format": "eqvae_pfr_equal_compute_request_v1",
            "conditions": [
                condition.payload(
                    args.prefix_full_ratio, args.depth10_prefix_full_ratio
                )
                for condition in conditions
            ],
            "gpus": list(args.gpus),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "batch_seed_schema": args.batch_seed_schema,
            "label_mode": args.label_mode,
        },
    )
    if args.dry_run:
        print((root / "request.json").read_text(encoding="utf-8"))
        return

    lanes: list[list[Condition]] = [[] for _ in args.gpus]
    for index, condition in enumerate(conditions):
        lanes[index % len(args.gpus)].append(condition)

    def lane(gpu: int, items: list[Condition]) -> list[dict[str, Any]]:
        return [
            run_one(condition, gpu, root, repo, data, adm_python, args)
            for condition in items
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, items)
            for gpu, items in zip(args.gpus, lanes, strict=True)
            if items
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    write_summary(root, results, args)


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-seed-schema",
        choices=BATCH_SEED_SCHEMAS,
        default=DEFAULT_BATCH_SEED_SCHEMA,
    )
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="random")
    parser.add_argument("--prefix-full-ratio", type=float, default=PREFIX_FULL_RATIO)
    parser.add_argument(
        "--depth10-prefix-full-ratio",
        type=float,
        default=DEPTH10_PREFIX_FULL_RATIO,
    )
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep_parser = subparsers.add_parser("sweep")
    add_shared(sweep_parser)
    sweep_parser.add_argument("--gpus", type=parse_gpus, default=(1, 2, 3))
    sweep_parser.add_argument("--output-root", type=Path)
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument(
        "--conditions",
        type=parse_condition,
        nargs="+",
        default=(
            Condition("ordinary_ig", 32),
            Condition("projected", 26),
            Condition("projected", 27),
        ),
    )
    worker_parser = subparsers.add_parser("worker")
    add_shared(worker_parser)
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--condition", required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch sizes must be positive")
    if not math.isfinite(args.prefix_full_ratio) or args.prefix_full_ratio <= 0:
        raise ValueError("prefix/full ratio must be positive and finite")
    if (
        not math.isfinite(args.depth10_prefix_full_ratio)
        or args.depth10_prefix_full_ratio <= 0
    ):
        raise ValueError("depth-10 prefix/full ratio must be positive and finite")
    if args.label_mode == "balanced" and args.num_samples % 100:
        raise ValueError("balanced labels require num_samples divisible by 100")
    batch_seed(
        args.seed,
        (args.num_samples - 1) // args.batch_size,
        schema=args.batch_seed_schema,
    )
    if args.command == "worker":
        worker(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
