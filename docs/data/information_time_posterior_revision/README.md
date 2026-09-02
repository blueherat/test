# Information-time query semantics

Paired ImageNet-100 SiT-S/2 FID-1K controls for the query semantics used by
Information-Time Posterior Revision IG. All conditions use the same v800 EMA
strong model, depth-4 EMA internal head, guidance schedule, horizon `1/32`,
noise/labels within each sampling seed, and ADM evaluation pipeline.

Raw artifacts:

- seed 0 controls: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/calibration_split_ig_v1/information_time_query_fid1k`
- seed 0 oblique anchor: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/calibration_split_ig_v1/canonical_ablation_fid1k`
- seed 1 controls: `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/calibration_split_ig_v1/information_time_query_fid1k_seed1`
- rejected Jensen control: `/data/users/zhoushunyu/eqvae/experiments/semigroup_local_jensen_sit_fid1k`

The query ablation was preregistered in
`docs/INFORMATION_TIME_POSTERIOR_REVISION_IG_ZH.md` before reading either
seed's metrics.
