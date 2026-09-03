# PFR mechanism audit data (2026-09-03)

This directory archives the compact outputs used by
`docs/PFR_MECHANISM_AUDIT_20260903_ZH.md`.

## Fixed model protocol

- Strong model: ImageNet-100 SiT-S/2 velocity checkpoint at 800K steps.
- Weak model: the EMA depth-4 velocity head trained for 50K steps on the
  frozen strong backbone.
- Sampling path: linear flow from Gaussian noise at `t=0` to data at `t=1`.
- Ordinary IG schedule: `gamma=0.6` for `t<0.25`, `gamma=0.7` for
  `0.25<=t<0.5`, and zero afterwards.
- Canonical PFR query horizon: `h=1/32` in raw affine-flow time.
- FID comparisons within each CSV are paired by the recorded noise and label
  SHA-256 hashes. FID-1K and FID-5K values must not be compared as if they
  used the same estimator variance.

## Contents

`candidate_theories/` contains compact generation summaries for hypotheses
that were tested while trying to explain PFR: depth-information
difference-in-differences, telescoping scale space, cross-time velocity
decomposition, posterior-pressure trajectories, affine reference ensembles,
and hierarchy controls.

`query_response/` contains the path-evidence control, the two-seed
information-revision sweep, paired query controls, temporal/spatial response
splits, and matched raw-time versus SNR/log-SNR clock tests.

`diagnostics/` contains the held-out conserved-velocity audit, failed
posterior-projection test, ADM terminal-distribution decomposition,
conservativity probes, finite weak-head secant geometry, and weak-density
line-integral controls.

## Deliberately excluded

Generated PNG samples, decoded sample archives, ADM activation banks,
per-sample multi-megabyte diagnostic tables, checkpoints, and optimizer state
are not copied into Git. The archived CSV/JSON files preserve aggregate data,
protocol fields, seeds, hashes, and original result paths needed to locate the
full local artifacts under:

`/home/zhoushunyu/data/eqvae/imagenet_sit_flow/`
