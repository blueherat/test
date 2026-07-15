"""Command-line entry point for the no-training official RAE audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.rae_spectral_gradient_audit import (
    RAEAuditConfig,
    dct2_basis,
    decoder_sensitivity_sweep,
    load_cached_latents,
    radial_band_masks,
    run_rae_spectral_gradient_audit,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_spectral_gradient_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--gradient-microbatches", type=int, default=64)
    parser.add_argument("--decoder-sample-count", type=int, default=16)
    parser.add_argument("--no-decoder", action="store_true")
    args = parser.parse_args()
    config = replace(
        RAEAuditConfig(),
        device=args.device,
        train_count=args.train_count,
        validation_count=args.validation_count,
        gradient_microbatches=args.gradient_microbatches,
        decoder_sample_count=args.decoder_sample_count,
    )
    result = run_rae_spectral_gradient_audit(
        config,
        include_decoder=not args.no_decoder,
        verbose=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "residual_table": result.residual_table,
        "basis_control": result.basis_control_table,
        "gradient_summary": result.gradient_summary,
        "gradient_bands": result.gradient_band_table,
        "cross_band_correlation": result.cross_band_correlation,
        "decoder_sensitivity": result.decoder_sensitivity,
    }
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    if not args.no_decoder:
        payload = load_cached_latents(config)
        size = payload["validation"].shape[-1]
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        sweep = decoder_sensitivity_sweep(
            payload["validation"],
            dct2_basis(size).float(),
            radial_band_masks(size, config.spatial_band_count),
            config,
            device=device,
        )
        sweep.to_csv(args.output_dir / "decoder_sensitivity_sweep.csv", index=False)
    metadata = {"config": asdict(config), **result.metadata}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    print(f"saved audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()
