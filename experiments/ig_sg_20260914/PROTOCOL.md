# IG + SG paired pilot, 2026-09-14

Frozen before production results. User requested both paper SG and log-score SG on IG.

For each model compare IG, IG + paper SG (omega 1, time shift .01 toward noise), IG + same-time antithetic log-score SG (omega 1, kappa .2), and an IG-only solver control for each added compute budget. Both residuals use raw strong predictions, added to the fixed native IG field. No CFG. The implementation follows the released flow-velocity SG arithmetic; it does not claim cross-time velocity differences equal exact cross-time score differences.

- SiT-S/2 ImageNet100: EMA800K, depth4 v head EMA50K; native IG alpha .8*6/7 for t<.25, .8 for .25<=t<.5, zero afterwards. Heun64 with IG amount fixed at each left stage. Full evaluations: 128, 255, 382; controls Heun128 (256) and Heun191 (382).
- JiT-B/16: official EMA1, original trained depth4 internal readout EMA50K; IG alpha .3 for t<.5. Euler100. Evaluations: 100, 199, 300; controls Euler199 and Euler300.
- RAEv2 DINOv3-Lk7: native shared strong/weak heads; weak+1.78*(strong-weak) on noise time [.1,1]. Euler100 shifted grid8. Evaluations: 100, 199, 300; controls Euler199 and Euler300.

Each arm uses 1,000 paired images, seed 2026091407, exactly the previous weak_reference pilot input bank. SiT has 10 samples/class over 100 classes; others one/class over 1,000 classes. Batch16 SiT; batch32 others. Preserve prior precision, decoder and metric protocols. No coefficient tuning in this experiment.

RAE baseline, log-SG, and Euler300 are reused from weak_reference_20260914/cross_screen_1k with explicit original paths and hashes, after bitwise equivalence checks of the new wrapper. They are not independent new replications. The other twelve arms (12K images) are newly generated. All fifteen arms (15K images) are audited. An outcome is promising only if it beats both IG baseline and compute control; 1K FID is a screening measurement, not a final quality claim. Do not compare absolute FID across SiT/ImageNet100 ADM and the ImageNet1K Nanogen protocols.

Before production, all three real-model checks passed: native IG baseline bitwise, omega0 bitwise, nonzero residual effective, finite outputs/decode, actual first-block hook traversal counts. Each phase request freezes model/weak-head/decoder/metric assets, relevant source code, check result, paired inputs and reuse provenance. Post-run audit checks full coverage, input/output hashes, finite outputs, costs, sample-to-batch equality, and FP64 FID reconstructed from cached features (not an independent extractor).

Production:
```sh
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.run prepare
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.run controller
```

## Follow-up selected after JiT screen

JiT original SG FID55.2985 beat IG55.5089 by only0.2103 and equal-NFE IG56.8960 by1.5975. This small screen gain motivates an independent confirmation, not a success declaration. Freeze the same omega1/shift.01 and all IG settings; use new PCG64 seed2026091417, 1K per arm, and compare IG, IG+original SG, and Euler199 IG (3K additional new samples). No parameter selection on the confirmation bank; report the result even if it fails. This extension was chosen after seeing the screen results, rather than claimed as part of the original five-arm protocol.
