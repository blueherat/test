#!/usr/bin/env python3
"""Mechanical amendment launcher for inventory replication protocol v1.1.

Protocol v1 failed closed before labels were joined to features because the
frozen unique-track count was written as 190.  The three label-free catalogs
contain 79 + 106 + 7 disjoint tracks, hence the correct union is 192.  Scalar
feature count (6,844), cohorts, selection, permutation test, multiplicity
correction, and all other analysis rules are unchanged.

This launcher pins the exact failed-closed implementation, changes only that
one validation constant and the output path, and makes the published
``analysis_source.py`` point back to this amendment launcher so the runtime
change remains visible in the immutable result.
"""

from __future__ import annotations

from pathlib import Path

import explore_dit_bad_good_inventory_replication as implementation


EXPECTED_IMPLEMENTATION_SHA256 = (
    "c23ef50713e3a310bf8a6bb7b0541563717b5979cca035f3b15394f68aaaca90"
)
AMENDED_EXPECTED_TOTAL_TRACKS = 192
AMENDED_DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "bad_good_metric_confirmation_expansion_v1/"
    "inventory_replication_exploratory_amendment_v1_1"
)


def main() -> None:
    implementation_path = Path(implementation.__file__).resolve()
    observed_sha256 = implementation.sha256_file(implementation_path)
    if observed_sha256 != EXPECTED_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "pinned v1 implementation changed: "
            f"expected {EXPECTED_IMPLEMENTATION_SHA256}, got {observed_sha256}"
        )
    implementation.EXPECTED_TOTAL_TRACKS = AMENDED_EXPECTED_TOTAL_TRACKS
    implementation.DEFAULT_OUTPUT = AMENDED_DEFAULT_OUTPUT
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
