#!/usr/bin/env python3
"""Freeze the post-gate exploratory third-pool B/C label-sensitivity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import audit_dit_third_pool_bc_label_sensitivity as auditor
except ImportError:
    import audit_dit_third_pool_bc_label_sensitivity as auditor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=auditor.DEFAULT_SOURCE_LOCK)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--expected-manifest-identity")
    args = parser.parse_args()
    if args.validate:
        protocol, manifest = auditor.validate_source_lock(
            args.output, args.expected_manifest_identity
        )
    else:
        auditor.freeze_source_lock(args.output)
        protocol, manifest = auditor.validate_source_lock(args.output)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
