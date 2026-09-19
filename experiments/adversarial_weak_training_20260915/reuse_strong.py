"""Preserve complete 1K sampling provenance when reusing the a=0 baseline."""

import os
from pathlib import Path
import shutil
import numpy as np
from . import common as c


def reuse_strong_one_k(point, checkpoint, digest, run, step, weights, noise, labels):
    original = c.original.model_root("sit_small") / "points/strong__c0000"
    old_summary = c.read(original / "n1000/summary.json")
    old_metrics = c.read(original / "n1000/metrics.json")
    records = []
    for start in range(0, 1000, 8):
        old_path = original / "batches" / f"{start:05d}.npz"
        old_record = c.read(old_path.with_suffix(".json"))
        assert c.sha(old_path) == old_record["sha256"]
        assert old_record["counts"]["head"] == 0
        with np.load(old_path) as batch:
            assert int(batch["start"]) == start
            np.testing.assert_array_equal(batch["labels"], labels[start:start + 8])
            assert str(batch["noise_sha256"]) == c.original.array_sha(noise[start:start + 8])
        path = point / "batches" / old_path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                os.link(old_path, path)
            except OSError:
                shutil.copy2(old_path, path)
        assert c.sha(path) == old_record["sha256"]
        c.atomic(path.with_suffix(".json"), dict(sha256=old_record["sha256"],
            checkpoint_sha256=digest, checkpoint=str(checkpoint), counts=old_record["counts"],
            start=start, noise_sha256=c.original.array_sha(noise[start:start + 8]),
            tick=0, coefficient=0., samples=8, reused_from=str(old_path)))
        records.append(dict(file=str(path), sha256=old_record["sha256"]))
    assert [row["sha256"] for row in records] == [row["sha256"] for row in old_summary["records"]]
    provenance = dict(records=records, checkpoint_sha256=digest, tick=0, coefficient=0.,
        run=run, step=step, weights=weights, reused_from=str(original / "n1000"),
        reuse_reason="Unchanged strong-only sampler; identical preserved images, labels and noise",
        reused_utc=c.now())
    stage = point / "n1000"
    c.atomic(stage / "summary.json", dict(old_summary, **provenance))
    c.atomic(stage / "metrics.json", dict(old_metrics, **provenance))
