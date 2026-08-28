# Third-pool endpoint label reliability audit

This is an external-label quality audit, not a proposed detector. It opened only
the three endpoint-only review locks, adjudication/consensus locks, and endpoint
PNGs. It did not open any trajectory, model-derived feature, embedding, score,
threshold, rank, or alert.

## Dataset and grain

- 1,800 frozen-model endpoints: 600 each for ImageNet classes 207, 602, and 795.
- Three independent endpoint-only severity ratings per image on the ordinal scale
  0/1/2/3. For the binary audit, `clear bad = severity >= 2`.
- 29 images had at least two clear-bad votes. Conservative adjudication retained
  6 and downgraded 23; the frozen final consensus matches those decisions exactly.

## Rating prevalence

| Reviewer | Severity counts | Clear bad | Any imperfection (severity > 0) |
|---|---:|---:|---:|
| reviewer 1 | {0: 1722, 1: 30, 2: 44, 3: 4} | 48 (2.67%) | 78 (4.33%) |
| reviewer 2 | {0: 1774, 1: 8, 2: 13, 3: 5} | 18 (1.00%) | 26 (1.44%) |
| reviewer 3 | {0: 1653, 1: 36, 2: 96, 3: 15} | 111 (6.17%) | 147 (8.17%) |

The spread in positive prevalence is material: reviewer 3 calls far more clear
bad cases than reviewer 2. Agreement must therefore be interpreted with
positive agreement and kappa, not overall agreement alone.

## Agreement

| Pair | Four-level exact | Four-level κ | Quadratic κ | Binary exact | Binary κ | Positive agreement | Both / A-only / B-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| reviewer_1_vs_2 | 95.11% | 0.141 | 0.226 | 97.11% | 0.200 | 21.21% | 7 / 41 / 11 |
| reviewer_1_vs_3 | 90.83% | 0.235 | 0.307 | 93.72% | 0.262 | 28.93% | 23 / 25 / 88 |
| reviewer_2_vs_3 | 90.61% | 0.006 | 0.002 | 92.94% | -0.002 | 1.55% | 1 / 17 / 110 |

- All-three exact agreement: 88.83% on four levels and
  91.89% after clear-bad binarization.
- Fleiss κ: 0.130 on four levels;
  0.147 for clear bad.
- Raw-majority clear bad: 29; unanimous clear bad:
  1; exactly two votes:
  28; at least one
  clear-bad vote: 147.

### Agreement by class

| Class | Pair | Four-level κ | Binary κ | Positive agreement | Both positive |
|---|---|---:|---:|---:|---:|
| 207 | reviewer_1_vs_2 | 0.115 | 0.094 | 10.26% | 2 |
| 207 | reviewer_1_vs_3 | 0.407 | 0.529 | 56.10% | 23 |
| 207 | reviewer_2_vs_3 | 0.036 | 0.032 | 4.08% | 1 |
| 602 | reviewer_1_vs_2 | 0.253 | 0.566 | 57.14% | 4 |
| 602 | reviewer_1_vs_3 | -0.005 | -0.007 | 0.00% | 0 |
| 602 | reviewer_2_vs_3 | -0.005 | -0.007 | 0.00% | 0 |
| 795 | reviewer_1_vs_2 | 0.102 | 0.145 | 15.38% | 1 |
| 795 | reviewer_1_vs_3 | -0.016 | -0.016 | 0.00% | 0 |
| 795 | reviewer_2_vs_3 | -0.025 | -0.024 | 0.00% | 0 |

## Component usage

| Component | Reviewer 1 | Reviewer 2 | Reviewer 3 | 2-of-3 majority |
|---|---:|---:|---:|---:|
| global_blur | 40 | 5 | 129 | 29 |
| local_blur | 11 | 6 | 0 | 2 |
| soft_fusion_or_melting | 14 | 2 | 8 | 5 |
| discrete_duplication_or_extra_part | 10 | 6 | 1 | 2 |
| detachment_or_floating_part | 3 | 7 | 2 | 1 |
| topology_or_attachment_error | 6 | 4 | 7 | 4 |
| limb_or_object_misalignment | 11 | 11 | 4 | 3 |
| texture_break | 15 | 1 | 12 | 5 |
| other | 0 | 1 | 0 | 0 |

## Interpretation

- High overall agreement is dominated by the large negative class. The scarce
  positive labels have substantially weaker agreement and reviewer-specific
  thresholds.
- The 29-case majority set is suitable as a high-precision *screening set*, but
  adjudication removed 79.3% of it. All 23 downgrades are class 207; retention was
  1/24 (4.2%) for class 207 versus 4/4 and 1/1 for classes 602 and 795. This is a
  phenotype/class-dependent policy, not a neutral uniform tightening.
- Because adjudication was downgrade-only, false positives were reduced at the
  deliberate cost of unmeasured false negatives. The final six cannot establish
  recall, and a gate requiring 30 final clear-bad cases cannot be evaluated from
  this pool.

## Independent visual re-audit

All 29 raw-majority images were inspected at native 256x256 resolution, with
multiple frozen class-207 reference grids used to calibrate ordinary model
quality. This was a single-auditor QA pass, so it is evidence about label quality,
not a replacement immutable truth set.

- All final six remain clearly bad: no obvious adjudication false positive was
  found.
- Seven of the 23 downgrades still meet the user's intended clear-bad standard.
  This gives 13 visually clear cases in the 29-case screening set, 14 borderline
  cases, and 2 acceptable-for-class cases.
- Therefore the final six are **high precision but over-conservative** for the
  stated target. The missed clear cases are:

- `a_515a4b503c1fd57fd5ee` (sample 201, class 207): strong global smoothing and plastic blur.
- `a_ef44037d7ee862b96b89` (sample 216, class 207): blurred/melted eyes and head contour.
- `a_4858b4716a422c5eb990` (sample 375, class 207): strong global blur with lost facial texture.
- `a_720cc0c5f7c3e89c2a5a` (sample 399, class 207): hind-limb/topology fusion and misalignment.
- `a_ed27d667ce92e5820e7f` (sample 927, class 207): clear global blur below class band.
- `a_a2416447e741903e7dfe` (sample 951, class 207): face and eyes globally blurred below class band.
- `a_d9f66b1812eebd828d92` (sample 1509, class 207): severe global blur and body/background fusion.

The strongest internal warning is `a_ed27d667ce92e5820e7f`: all three reviewers
assigned severity >=2, yet downgrade-only adjudication removed it. The image is
visibly globally blurred relative to the class band. This shows that adjudication
silently changed the estimand from "obvious below-model-average blur/fusion/
misalignment" toward "only catastrophic structural corruption."

## Recommended next-round rubric changes

1. Freeze 8-12 positive and 8-12 negative anchors per class before review. Include
   three boundary pairs for global blur: ordinary softness, clear below-band blur,
   and catastrophic melting.
2. Split the decision into two questions: `(a)` is there a visible defect, and
   `(b)` is it materially below the frozen class band? Severity 2 requires yes/yes.
3. For every severity 2/3, require a component and a one-line localization. For
   every downgrade of a unanimous bad vote, require two adjudicators or a written
   exception tied to a frozen anchor.
4. Run a 60-image calibration round before the next pool. Do not start production
   labeling until each pair reaches positive agreement >=60% and binary kappa
   >=0.50 on the calibration set; reconcile component definitions first.
5. Use stratified adjudication: expose an equal-sized blinded sample of majority
   negatives alongside majority positives. The current downgrade-only design can
   estimate precision but cannot reveal false negatives.
6. Keep endpoint labels strictly external. They evaluate whether an internal
   trajectory statistic works; FID/embeddings and these labels are not themselves
   candidate intervention signals.
