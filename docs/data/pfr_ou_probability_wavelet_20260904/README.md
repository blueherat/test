# PFR OU probability-wavelet compact results

This directory contains compact, reviewable metrics for the OU degree-control
and equal-compute experiments. Large sample arrays, activations and model
weights remain outside Git under `/data/users/zhoushunyu/eqvae`.

- `paired_fid1k_degree_controls.csv`: two exactly paired balanced 1K banks.
- `paired_fid5k_equal_compute.csv`: the formal balanced 5K comparison.
- `energy_adaptive_fid1k.csv`: fixed-step and equal-compute checks of the
  parameter-free energy-adaptive revision.
- `two_scale_span_fid1k.csv`: paired screening of the one-scale certificate
  against its parameter-free dyadic two-scale span; the gain was below the
  FID-1K resolution, so no 5K run was performed.
- `projection_granularity_seed*.csv`: pointwise, per-class, and global
  projection coefficients and explained-energy fractions on two independent
  100-class banks.
- `multiscale_rank_seed*.csv`: direction, unique-energy, amplitude-ratio, and
  effective-degree diagnostics for the `h` and `2h` OU defects, plus raw-PFR
  cosine and projection-contraction statistics for weak/strong certificates.
- `direction_magnitude_fid1k_seed*.csv`: two paired screening banks for the
  no-parameter direction/norm exchange controls.
- `direction_magnitude_fid5k.csv`: formal paired 5K causal comparison of raw,
  common, raw-direction/common-norm, and common-direction/raw-norm revisions.
- `strong_certificate_fid1k.csv`: paired common/unique causal controls using
  the strong model's OU degree-1 defect as the certificate axis.
- `strong_direction_magnitude_fid1k_seed*.csv`: two paired 1K banks exchanging
  the raw-revision and strong-certificate direction/norm factors.
- `strong_direction_magnitude_fid5k_seed{5,6}.csv`: two independent, balanced,
  exactly paired 5K confirmations of the norm-preserving strong-certificate
  revision; large activations and decoded sample arrays stay outside Git.
- `strong_degree_polar_fid1k.csv`: two paired banks comparing strong degree-1
  and degree-2 certificates after both are given the same raw-revision norm.
- `location_shape_energy_seed*.csv`: 100-class, 128-state-per-class sampled
  class-constant/state-varying energy decompositions with finite-sample bias
  correction.
- `analytic_conditional_score_identity.csv`: exact-score checks on solvable
  one-dimensional Gaussian mixtures; the conditional expectation is evaluated
  independently with Gauss--Hermite quadrature.
- `analytic_score_shape_identity.csv`: finite-scale verification that the
  degree-1 defect equals the integrated OU score-Fokker--Planck shape operator.

The derivation, protocol boundaries and interpretation are documented in
`docs/PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md`.
