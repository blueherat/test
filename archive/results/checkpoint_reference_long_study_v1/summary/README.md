# Checkpoint Reference Long Study v1

## Headline

- best static reference: **v180**, gamma=3.05, FID-1K=54.7064
- factorial forward FID: 65.3068
- factorial baseline FID: 86.8148
- Shapley FID-benefit attribution: early_v270=+11.704, mid_v400=+9.475, late_v500=+0.330
- best dynamic/tuning condition: **beststart_v180_v180_v500**, FID-1K=54.5240, delta vs best static=-0.1824

## Files

- all paired conditions: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/checkpoint_reference_long_study_v1/summary/all_conditions.csv`
- static maturity: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/checkpoint_reference_long_study_v1/summary/static_maturity_best.csv`
- atlas: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/checkpoint_reference_long_study_v1/00_atlas/summary.json`

Interpret FID-1K differences below roughly 0.5 as screening-scale unless later confirmed with larger samples/seeds.
