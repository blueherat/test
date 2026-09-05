# PFR counterfactual-residual theory compact artifacts

This directory contains the compact, reproducible evidence added for
`docs/PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`.

## Files

- `analytic_counterexample_summary.json`: closed-form local-risk and terminal-risk
  values for the exact one-dimensional PFR counterexample.
- `analytic_counterexample_fields.csv`: dense-grid values used only as a numerical
  cross-check of the closed-form construction.
- `terminal_observable_audit/terminal_mean_witness.csv`: exact ADM feature-mean
  witness, including a post-hoc disjoint split-half stability check.
- `terminal_observable_audit/terminal_distribution_summary.{csv,json}`: FID
  mean/covariance decomposition for temporal/spatial condition comparisons.
- `terminal_observable_audit/paired_feature_response.csv`: paired terminal-feature
  response magnitudes and cosines.
- `fid5k_independent_seed1000003.{csv,json}`: a non-overlapping, partially
  seen 5K RNG-bank confirmation of ordinary IG versus the algebraically
  equivalent projected query-control implementation. The historical filename
  is retained, but the JSON records the narrower claim.
- `sampling_rng_schema_validation.json`: legacy overlap audit, namespaced-v2
  schema, and CUDA stream/hash validation.
- `solver_equal_nfe64.csv`: compact copy of the paired equal-NFE solver
  falsification table used in the report. Its source is
  `/home/zhoushunyu/data/eqvae/imagenet_sit_flow/implicit_future_state_ig_solver_v1/fid1k_equal_nfe64/summary/all_conditions.csv`;
  only columns used by the report are retained.

The terminal audit reads retained full activations from
`/home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_distribution_audit_v1/`.
Those large source artifacts are intentionally not duplicated here.

## Reproduction

```bash
python experiments/pfr_counterfactual_residual_theory.py \
  --output-dir docs/data/pfr_counterfactual_residual_theory_20260903

python experiments/analyze_pfr_terminal_distribution.py \
  --root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_distribution_audit_v1/seed0 \
  --reference-stats /home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz \
  --reference-activations /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_distribution_audit_v1/reference_imagenet100_validation_n5000_adm_activations.npz \
  --output-dir docs/data/pfr_counterfactual_residual_theory_20260903/terminal_observable_audit

python experiments/run_imagenet100_sit_pfr_query_controls.py fid \
  --output-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_counterfactual_residual_theory_v1/fid5k_reproduction_seed1000003 \
  --gpus 1,2 --conditions ordinary_ig,projected \
  --num-samples 5000 --batch-size 8 --seed 1000003 \
  --batch-seed-schema legacy_additive_v1
```

## Scope

The analytic toy is a schedule- and clamp-matched, state-independent logical
counterexample, not a model of Bayes-optimal SiT heads.  The terminal feature
witness is algebraically equivalent to the mean component of FID; it is a
re-expression, not an independent quality metric or causal mechanism test.
Both limitations are recorded in the main report.

The non-overlapping FID-5K confirmation completed immediately before the RNG fix,
so its exact replay uses the explicit legacy schema shown above.  Its batch
seed interval `1000003..1000627` is disjoint from historical `0..625`; the
legacy flag is for byte reproduction, not a claim that adjacent legacy run
seeds are safe.  The reproduction command deliberately uses a fresh output
root; pointing it at the archived `fid5k_independent_seed1000003` directory
would reuse cached condition results and metrics instead of performing a fresh
sampling run.
