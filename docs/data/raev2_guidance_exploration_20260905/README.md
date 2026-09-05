# RAEv2 guidance exploration compact archive

This directory is the Git-safe evidence bundle for the 2026-09-05 RAEv2
guidance exploration archive. The reader-facing synthesis is
`docs/RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md`.

## Contents

- `raw/`: copied CSV, JSON, JSONL, and a small number of explanatory PNG files
  from the original experiment directories.
- `archive_manifest.csv`: original absolute source, repository-relative archive
  path, byte count, and SHA256 for every archived file.
- `experiment_inventory.csv`: source-directory size and completion status by
  experiment family.
- `official_fid_records.csv`: normalized rows extracted from saved official
  metric files. Rows from different RNG banks remain separate.
- `final_screen_summary.csv`: compact decision table for the final radius,
  semigroup-value, flow-pullback, and relative-transport screens.

The final additions include the two-bank radius control, paired semigroup-value
screen, finite-interval flow-pullback screen, and relative-transport iteration
screen. Their reports state whether each protocol is paired and which baseline
is valid for comparison.

## Exclusions

The archive intentionally excludes `samples.npz`, checkpoints, training and
held-out banks, evaluator feature caches, per-rank image arrays, and disposable
logs. Those files remain under `/home/zhoushunyu/data/eqvae/experiments/` and are
identified by the requests, sample hashes, and summaries kept here. No result
should be reconstructed by comparing unrelated rows only because their sample
counts or nominal seeds match.
