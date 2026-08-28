#!/usr/bin/env python3
"""Build CPU-only, seed-labelled review grids from strict DiT baselines.

All input runs must use the same ordered class batch and sampling contract and
must differ only in global seed/runtime metadata.  For every requested class,
this tool emits three contact sheets ordered by numeric seed:

* ``native``: original 256x256 pixels, without resampling;
* ``smooth``: 2x Lanczos enlargement; and
* ``nearest``: 2x nearest-neighbour enlargement.

The sheets are for relative human inspection only.  This tool computes no
quality score, ranking, threshold, bad-case label, selection, or intervention.
Both the frozen official-eight runner and the compatible custom-class runner
are accepted, so existing official runs need not be regenerated.  Input
manifests, completion records, PNG bytes, and pixel hashes are validated before
aggregation.  Outputs are staged, self-hashed, and never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

try:  # Package and direct CLI imports.
    from . import reproduce_dit_imagenet256 as strict
    from . import sample_dit_imagenet256_custom as custom
except ImportError:  # pragma: no cover - direct CLI invocation.
    import reproduce_dit_imagenet256 as strict
    import sample_dit_imagenet256_custom as custom


RUNNER_NAME = "build_dit_custom_review_grids"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
UPSCALE_FACTOR = 2
CELL_GAP = 8
OUTER_MARGIN = 8
LABEL_HEIGHT = 24


@dataclass(frozen=True)
class RunInput:
    root: Path
    producer: str
    seed: int
    classes: tuple[int, ...]
    identity_sha256: str
    manifest_sha256: str
    contract_sha256: str
    images: dict[int, dict[str, Any]]


def _as_classes(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} class list is missing")
    classes = tuple(value)
    if (
        not 1 <= len(classes) <= custom.MAX_CLASSES
        or any(type(item) is not int or not 0 <= item < strict.NUM_CLASSES for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise RuntimeError(f"{context} class list is invalid")
    return classes


def sampling_contract(identity: dict[str, Any], classes: Sequence[int]) -> dict[str, Any]:
    protocol = identity.get("protocol", {})
    rng = identity.get("rng_contract", {})
    checkpoint = identity.get("checkpoint", {})
    vae = identity.get("vae_snapshot", {})
    source = identity.get("source", {})
    vae_files = vae.get("files")
    if not isinstance(vae_files, list):
        raise RuntimeError("input identity lacks VAE file records")
    contract = {
        "ordered_class_batch": list(classes),
        "model": protocol.get("model"),
        "image_size": protocol.get("image_size"),
        "num_classes": protocol.get("num_classes"),
        "null_class_id": protocol.get("null_class_id"),
        "num_sampling_steps": protocol.get("num_sampling_steps"),
        "sampler": protocol.get("sampler"),
        "clip_denoised": protocol.get("clip_denoised"),
        "cfg_scale": protocol.get("cfg_scale"),
        "cfg_epsilon_channels": protocol.get("cfg_epsilon_channels"),
        "vae_kind": protocol.get("vae"),
        "vae_scaling_factor": protocol.get("vae_scaling_factor"),
        "rng_transition_calls": rng.get("transition_randn_like_calls"),
        "rng_transition_shape": rng.get("transition_noise_shape_each_call"),
        "rng_t0_draw": rng.get("terminal_t0_randn_consumed_then_masked"),
        "rng_discarded_half": rng.get(
            "second_half_transition_noises_consumed_then_state_discarded"
        ),
        "source_revision": source.get("revision"),
        "source_tree_sha256": source.get("working_tracked_tree_sha256"),
        "pinned_source_sha256": source.get("pinned_source_sha256"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "vae_model_id": vae.get("model_id"),
        "vae_revision": vae.get("revision"),
        "vae_files": [
            {
                "name": record.get("name"),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }
            for record in vae_files
        ],
    }
    missing = [
        key
        for key, value in contract.items()
        if value is None or (isinstance(value, list) and not value)
    ]
    if missing:
        raise RuntimeError(f"input sampling contract is incomplete: {missing}")
    return contract


def load_run(root: Path) -> RunInput:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"input run must be a non-symlink directory: {root}")
    root = root.resolve()
    manifest_path = root / custom.MANIFEST_NAME
    manifest = strict.load_json(manifest_path)
    if manifest.get("schema") != custom.MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"input custom run is not complete: {root}")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"input manifest lacks an identity: {root}")
    producer = identity.get("runner")
    if producer not in (custom.RUNNER_NAME, "reproduce_dit_imagenet256"):
        raise RuntimeError(f"input was not produced by an accepted strict DiT runner: {root}")
    if identity.get("baseline_only") is not True:
        raise RuntimeError(f"input is not a baseline-only scan: {root}")
    if producer == custom.RUNNER_NAME and identity.get("custom_class_scanner") is not True:
        raise RuntimeError(f"custom-runner input lacks its scanner identity: {root}")
    protocol = identity.get("protocol", {})
    class_field = "class_ids_ordered" if producer == custom.RUNNER_NAME else "class_ids"
    classes = _as_classes(protocol.get(class_field), str(root))
    seed = protocol.get("global_torch_seed")
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise RuntimeError(f"input run has an invalid global seed: {root}")
    if producer == custom.RUNNER_NAME:
        custom.validate_completed_output(root, identity, classes)
        relative_paths = custom.individual_relative_paths(classes)
    else:
        if classes != tuple(strict.CLASS_IDS):
            raise RuntimeError("frozen official runner input changed its eight-class protocol")
        strict.validate_completed_output(root, identity)
        relative_paths = strict.individual_relative_paths()
    identity_sha256 = strict.sha256_json(identity)
    if manifest.get("identity_sha256") != identity_sha256:
        raise RuntimeError(f"input identity self-hash failed: {root}")

    output_records = manifest.get("outputs")
    if not isinstance(output_records, list):
        raise RuntimeError(f"input manifest lacks output records: {root}")
    by_path = {
        record.get("relative_path"): record
        for record in output_records
        if isinstance(record, dict)
    }
    images: dict[int, dict[str, Any]] = {}
    for class_id, relative in zip(
        classes, relative_paths, strict=True
    ):
        record = by_path.get(relative)
        if not isinstance(record, dict):
            raise RuntimeError(f"input native image record is missing: {root / relative}")
        images[class_id] = {
            "path": str((root / relative).resolve()),
            "relative_path": relative,
            "bytes": record.get("bytes"),
            "sha256": record.get("sha256"),
            "pixel_sha256": record.get("pixel_sha256"),
        }
    contract = sampling_contract(identity, classes)
    return RunInput(
        root=root,
        producer=str(producer),
        seed=seed,
        classes=classes,
        identity_sha256=identity_sha256,
        manifest_sha256=strict.sha256_file(manifest_path),
        contract_sha256=strict.sha256_json(contract),
        images=images,
    )


def load_inputs(run_dirs: Sequence[Path]) -> tuple[RunInput, ...]:
    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    resolved = [path.expanduser().absolute().resolve() for path in run_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--run-dir entries must be unique")
    runs = tuple(load_run(path) for path in resolved)
    expected_classes = runs[0].classes
    expected_contract = runs[0].contract_sha256
    if any(run.classes != expected_classes for run in runs):
        raise RuntimeError(
            "all runs must use the same ordered class batch; mixing batch shapes/orders "
            "would mix different RNG contracts"
        )
    if any(run.contract_sha256 != expected_contract for run in runs):
        raise RuntimeError("input runs do not share one frozen sampling contract")
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("input runs contain duplicate global seeds")
    return tuple(sorted(runs, key=lambda run: run.seed))


def sheet_size(count: int, columns: int, tile_size: int) -> tuple[int, int]:
    used_columns = min(columns, count)
    rows = (count + used_columns - 1) // used_columns
    width = 2 * OUTER_MARGIN + used_columns * tile_size + (used_columns - 1) * CELL_GAP
    height = (
        2 * OUTER_MARGIN
        + rows * (tile_size + LABEL_HEIGHT)
        + (rows - 1) * CELL_GAP
    )
    return width, height


def render_sheet(
    runs: Sequence[RunInput],
    class_id: int,
    *,
    columns: int,
    variant: str,
) -> Image.Image:
    if variant == "native":
        tile_size = strict.IMAGE_SIZE
        resample: Image.Resampling | None = None
    elif variant == "smooth":
        tile_size = strict.IMAGE_SIZE * UPSCALE_FACTOR
        resample = Image.Resampling.LANCZOS
    elif variant == "nearest":
        tile_size = strict.IMAGE_SIZE * UPSCALE_FACTOR
        resample = Image.Resampling.NEAREST
    else:
        raise ValueError(f"unknown review variant: {variant}")
    used_columns = min(columns, len(runs))
    canvas = Image.new("RGB", sheet_size(len(runs), columns, tile_size), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    for index, run in enumerate(runs):
        column = index % used_columns
        row = index // used_columns
        x = OUTER_MARGIN + column * (tile_size + CELL_GAP)
        y = OUTER_MARGIN + row * (tile_size + LABEL_HEIGHT + CELL_GAP)
        with Image.open(run.images[class_id]["path"]) as source:
            source.load()
            tile = source.convert("RGB")
        if resample is not None:
            tile = tile.resize((tile_size, tile_size), resample=resample)
        canvas.paste(tile, (x, y))
        draw.text(
            (x + 3, y + tile_size + 4),
            f"class {class_id:04d}   seed {run.seed}",
            fill=(245, 245, 245),
        )
    return canvas


def output_relative_path(variant: str, class_id: int) -> str:
    return f"{variant}/class{class_id:04d}.png"


def expected_output_paths(classes: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        output_relative_path(variant, class_id)
        for variant in ("native", "smooth", "nearest")
        for class_id in classes
    )


def _save_sheet(
    image: Image.Image,
    path: Path,
    *,
    identity_sha256: str,
    class_id: int,
    seeds: Sequence[int],
    variant: str,
) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite review sheet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    fields = {
        "runner": RUNNER_NAME,
        "identity_sha256": identity_sha256,
        "class_id": str(class_id),
        "seeds_ordered": ",".join(str(seed) for seed in seeds),
        "variant": variant,
        "automatic_quality_scoring": "false",
        "automatic_ranking_or_selection": "false",
    }
    for key, value in fields.items():
        metadata.add_text(key, value)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def inspect_outputs(root: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    classes = tuple(identity["classes_reviewed"])
    expected = set(expected_output_paths(classes))
    expected_files = {
        (root / relative).resolve() for relative in expected
    } | {(root / MANIFEST_NAME).resolve(), (root / COMPLETION_NAME).resolve()}
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("review output file set is incomplete or contains extra files")
    records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = root / relative
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise RuntimeError(f"review sheet is not RGB: {path}")
            metadata = dict(image.info)
            pixels = image.tobytes()
            size = list(image.size)
        variant = relative.split("/", 1)[0]
        class_id = int(Path(relative).stem.removeprefix("class"))
        expected_metadata = {
            "runner": RUNNER_NAME,
            "identity_sha256": identity["identity_sha256"],
            "class_id": str(class_id),
            "seeds_ordered": ",".join(str(seed) for seed in identity["seeds_ordered"]),
            "variant": variant,
            "automatic_quality_scoring": "false",
            "automatic_ranking_or_selection": "false",
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"review PNG provenance mismatch: {path}")
        records.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": strict.sha256_file(path),
                "pixel_sha256": strict.sha256_bytes(pixels),
                "mode": "RGB",
                "size": size,
            }
        )
    return records


def validate_completed(root: Path, identity: dict[str, Any]) -> None:
    manifest_path = root / MANIFEST_NAME
    completion_path = root / COMPLETION_NAME
    manifest = strict.load_json(manifest_path)
    completion = strict.load_json(completion_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("review manifest is not complete")
    identity_payload = dict(identity)
    stored_identity_sha256 = identity_payload.pop("identity_sha256", None)
    if (
        not isinstance(stored_identity_sha256, str)
        or stored_identity_sha256 != strict.sha256_json(identity_payload)
    ):
        raise RuntimeError("review identity self-hash is invalid")
    if manifest.get("identity") != identity:
        raise RuntimeError("existing review identity differs")
    records = inspect_outputs(root, identity)
    outputs_sha256 = strict.sha256_json(records)
    if manifest.get("outputs") != records or manifest.get("outputs_sha256") != outputs_sha256:
        raise RuntimeError("review output hashes changed")
    expected_completion = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity["identity_sha256"],
        "manifest_sha256": strict.sha256_file(manifest_path),
        "outputs_sha256": outputs_sha256,
        "output_count": len(records),
    }
    if completion != expected_completion:
        raise RuntimeError("review completion record is invalid")


def build_identity(
    runs: Sequence[RunInput], classes: Sequence[int], columns: int
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "runner_source": {"path": str(runner), "sha256": strict.sha256_file(runner)},
        "role": "HUMAN_REVIEW_CONTACT_SHEETS_ONLY",
        "automatic_quality_scoring": False,
        "automatic_ranking_or_selection": False,
        "classes_reviewed": list(classes),
        "ordered_class_batch": list(runs[0].classes),
        "seeds_ordered": [run.seed for run in runs],
        "sample_count_per_class": len(runs),
        "columns": columns,
        "variants": {
            "native": {"tile_size": 256, "resampling": "none"},
            "smooth": {"tile_size": 512, "resampling": "Pillow LANCZOS"},
            "nearest": {"tile_size": 512, "resampling": "Pillow NEAREST"},
        },
        "common_sampling_contract_sha256": runs[0].contract_sha256,
        "inputs": [
            {
                "root": str(run.root),
                "producer": run.producer,
                "seed": run.seed,
                "identity_sha256": run.identity_sha256,
                "manifest_sha256": run.manifest_sha256,
                "images": {
                    str(class_id): run.images[class_id] for class_id in classes
                },
            }
            for run in runs
        ],
        "expected_outputs": list(expected_output_paths(classes)),
    }
    payload["identity_sha256"] = strict.sha256_json(payload)
    return payload


def run(args: argparse.Namespace) -> None:
    runs = load_inputs(args.run_dirs)
    classes = tuple(args.classes) if args.classes is not None else runs[0].classes
    missing = [class_id for class_id in classes if class_id not in runs[0].classes]
    if missing:
        raise RuntimeError(f"requested review classes are absent from inputs: {missing}")
    if args.expected_per_class is not None and len(runs) != args.expected_per_class:
        raise RuntimeError(
            f"expected {args.expected_per_class} seeds per class, found {len(runs)}"
        )
    identity = build_identity(runs, classes, args.columns)
    outdir = args.outdir
    if outdir.exists():
        if not outdir.is_dir() or outdir.is_symlink():
            raise RuntimeError(f"review output must be a non-symlink directory: {outdir}")
        if any(outdir.iterdir()):
            validate_completed(outdir, identity)
            print(f"validated completed review grids: {outdir}; no files changed")
            return
        raise RuntimeError(f"refusing to replace pre-existing empty directory: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    if any(
        custom._paths_overlap(outdir, run.root)  # noqa: SLF001 - shared safety primitive.
        for run in runs
    ):
        raise RuntimeError("review output directory overlaps an input run")

    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary)
        seeds = [run.seed for run in runs]
        for variant in ("native", "smooth", "nearest"):
            for class_id in classes:
                sheet = render_sheet(
                    runs, class_id, columns=args.columns, variant=variant
                )
                _save_sheet(
                    sheet,
                    staging / output_relative_path(variant, class_id),
                    identity_sha256=identity["identity_sha256"],
                    class_id=class_id,
                    seeds=seeds,
                    variant=variant,
                )
        # Completion is written after the output records are known.  Its path is
        # reserved by inspect_outputs, so create a temporary placeholder first.
        strict.atomic_json_dump({}, staging / COMPLETION_NAME)
        strict.atomic_json_dump({}, staging / MANIFEST_NAME)
        records = inspect_outputs(staging, identity)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "outputs": records,
            "outputs_sha256": strict.sha256_json(records),
        }
        strict.atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": identity["identity_sha256"],
            "manifest_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "outputs_sha256": strict.sha256_json(records),
            "output_count": len(records),
        }
        strict.atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_completed(staging, identity)
        if outdir.exists():
            raise RuntimeError("review output appeared during staging; refusing overwrite")
        os.replace(staging, outdir)
    print(
        json.dumps(
            {
                "complete": True,
                "outdir": str(outdir),
                "classes": list(classes),
                "seeds": [run.seed for run in runs],
                "automatic_quality_scoring": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Completed custom-run directories; all must share one ordered class batch.",
    )
    parser.add_argument(
        "--classes",
        type=custom.parse_classes,
        default=None,
        help="Optional subset of classes to render; defaults to every class in the runs.",
    )
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--expected-per-class",
        type=int,
        default=None,
        help="Fail unless exactly this many distinct seed runs are supplied (for example 20).",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.columns <= 10:
        parser.error("--columns must lie in [1,10]")
    if args.expected_per_class is not None and args.expected_per_class < 1:
        parser.error("--expected-per-class must be positive")
    args.run_dirs = tuple(path.expanduser().absolute().resolve() for path in args.run_dirs)
    requested = args.outdir.expanduser().absolute()
    if os.path.lexists(requested) and requested.is_symlink():
        parser.error(f"--outdir must not be a symlink: {requested}")
    args.outdir = requested.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
