# What Is Gained by Looking Ahead? State Transport and Reference Timing in Diffusion Guidance

**Research working draft, 2026-09-08. Not submission-ready.** Novelty, transfer to a stronger representation model, and the main mechanism remain incomplete. Numerical results below come from saved experiments; unresolved experiments are explicitly marked pending. This draft does not claim an ICLR-level result has been achieved.

## Abstract

Lookahead guidance can change both a latent state and the time at which a reference network is queried. We study these two interventions separately in a controlled Euler formulation. On a fixed ImageNet100 SiT model, a short state step with a more distant reference query improves paired 5K FID over a short-step, short-query control: 28.96 to 28.21 for classifier-free guidance and 39.78 to 38.40 for internal guidance. The same method improves on a compute-proximate 42-step Projected Future Reference baseline, but does not exceed the best quality of the more expensive original PFR sampler. Discovery-bank controls find that omitting the forward state prediction retains similar FID, questioning whether that prediction is necessary for the observed gain. A local decomposition explains why reference-time effects can remain first-order when state-prediction effects are second-order in the calibration step. These results motivate a separation of state and reference clocks, but do not establish a new fixed-point convergence theory, statistical equivalence of the controls, or successful transfer to RAEv2.

## 1. Research question and contribution boundary

Foresight Guidance (FSG) frames guidance as latent calibration toward consistency between conditional and unconditional generation [1]. The repository's PFR instead evaluates a weak reference at a counterfactual state and time [2]. Both motivate looking ahead, but they do not use the same reference field or numerical operator.

Our question is narrower and experimentally testable: **which part of a lookahead intervention accounts for its benefit at a fixed inference budget—moving the state, changing the query time, or introducing an additional contrastive update?** A second question is whether the answer survives a change from VAE latents to representation-autoencoder latents.

The currently defensible contribution is a controlled empirical decomposition. The following broader claims are not established:

- The operator is a new guidance principle rather than a useful combination of known timing and calibration ideas.
- It improves the quality–compute frontier across budgets or models.
- It writes conditional information into latents more effectively.
- A small prediction gap, local contraction, or local error bound guarantees better generated-image quality.

## 2. Separating the two clocks

Use noise-to-data time \(t\). Let \(G(z,t)\) be the ordinary guided velocity and \(R(z,t)\) the calibration reference. A calibration event is

\[
z_1=z+hG(z,t),\qquad
A_{h,H}(z)=z_1-hR(z_1,t+H).
\]

The state-step size \(h\) and query horizon \(H\) are distinct. We compare a long matched-clock operator \(h=H=.125\), a short matched-clock operator \(h=H=.025\), and an asynchronous operator \(h=.025,H=.125\). Each operator is followed by the same ordinary guided Euler step. There are five calibration iterations at three fixed events: two at step 0, two at step 5, and one at step 15 of a 40-step trajectory.

Two controls remove the forward state prediction:

\[
Q_{h,H}(z)=z+h[G(z,t)-R(z,t+H)],\qquad
D_h(z)=z+h[G(z,t)-R(z,t)].
\]

These respectively retain only the distant reference query and only the extra current-time contrast. A further control, **currently pending**, sets \(R=G\) on both calibration legs while retaining the ordinary local guided steps. It tests whether generic numerical transport changes can reproduce the benefit without a distinct calibration reference.

### 2.1 Implemented fields

For CFG, the historical SiT protocol guides the first three of four latent channels at scale 1.5. The reference uses unconditional outputs on those three channels and conditional output on the fourth. This detail matters: it is not a fully unconditional four-channel reference, and each field evaluation uses two model forwards.

For IG, let \(S\) be the full model and \(W\) a post-hoc depth-4 head. We retain

\[
G=S+\gamma(t)(S-W),\qquad R=S-\gamma(t)(S-W),
\]

with \(\gamma=.6\) before .25, .7 between .25 and .5, and zero thereafter. This effective reference differs from the raw weak head. Both \(G\) and \(R\) require one shared full/head backbone evaluation. A query at .25 or .5 can change both network time and \(\gamma\); a separate control freezes reference \(\gamma\) at the current calibration event time.

### 2.2 A local decomposition, not a quality guarantee

If \(R(\cdot,t+H)\) is \(L\)-Lipschitz on the relevant state segment,

\[
\|A_{h,H}(z)-Q_{h,H}(z)\|
\le h^2L\|G(z,t)\|.
\]

This follows directly by subtracting the two operators and applying the Lipschitz inequality. Meanwhile,

\[
Q_{h,H}(z)-D_h(z)=-h[R(z,t+H)-R(z,t)].
\]

For a continuously differentiable time dependence, the bracket is the integral of \(\partial_t R\) over the query interval. With fixed \(H\) and decreasing \(h\), the state-prediction difference is second-order while the query-time difference can be first-order. For the piecewise IG schedule, time-derivative integrals must include jump contributions or be restricted to smooth segments.

This elementary bound motivates the controls; it is not presented as a novel theorem. Its constants have not been bounded along the learned trajectories, and it gives no FID guarantee. Even with \(G=R=a(t)z\), the discrete map has factor \((1+ha(t))(1-ha(t+H))\), whereas an exact forward/inverse flow pair is the identity. Thus, a nontrivial calibration effect does not by itself establish a conditional-consistency mechanism.

## 3. Experimental protocol

The main model is the repository's ImageNet100 SiT-S/2 EMA at step 800,000, with the frozen-backbone depth-4 velocity head trained for 50,000 updates. Images are decoded with the existing SD VAE and its original quantization. All clock comparisons use FP32 with TF32, batch size 8, and the same continuous CUDA noise/label generator within each bank.

The 1K discovery seed is 202609411; the independently fixed 5K seed is 202609412. ADM Inception FID uses the same cached ImageNet100 validation-5K reference. This is a custom ImageNet100 study, not an ImageNet1K leaderboard comparison. FID depends on sample count, so 1K and 5K absolute values are not compared directly.

Closed50 and each calibrated40 method use 100 complete model forwards per CFG sample, or 50 shared backbone forwards per IG sample. The original PFR additionally uses genuinely truncated depth-4 queries; these are counted separately. All image counts, input hashes, model/head identities, and relevant inference settings were checked. Independent float64 covariance calculations using a symmetric positive-semidefinite eigendecomposition reconstruct saved FIDs within 3e-6; this verifies calculations from features, not a second Inception extraction.

## 4. Results

### 4.1 Independent 5K clock comparison

| Method | CFG FID ↓ | IG FID ↓ |
|---|---:|---:|
| Ordinary Euler50 | 34.627031 | 41.533513 |
| Short state, short query | 28.960328 | 39.783258 |
| Short state, distant query | 28.205314 | 38.395560 |

The distant query improves on the short-query control by 2.61% and 3.49%, respectively. The long matched-clock operator was tested at 1K: it improved CFG relative to closed50, but severely worsened IG (122.30 versus 69.15). It was not expanded to 5K. These are fixed-bank point estimates; uncertainty across repeated independent 5K banks remains unmeasured.

### 4.2 Does forward state prediction explain the gain?

| Discovery-bank method | CFG FID-1K ↓ | IG FID-1K ↓ |
|---|---:|---:|
| Current-time contrast \(D_h\) | 55.446518 | 66.505439 |
| Distant query without state prediction \(Q_{h,H}\) | 53.466699 | 64.405892 |
| Asynchronous state/query operator \(A_{h,H}\) | 53.403568 | 64.615465 |

Forward state prediction has no clear additional benefit at this scale. This is not a statistical equivalence test. The controls were proposed after the initial screen and frozen before their own results; they are not independent confirmations. Freezing the IG reference coefficient at the event's current time gives 64.482336, so a coefficient-boundary change alone is not supported as the explanation.

### 4.3 Stronger solver and original PFR comparison

All values below use the same 5K input bank as Section 4.1.

| IG method | FID ↓ | Full forwards/sample | Prefix4 forwards/sample | Sampling seconds |
|---|---:|---:|---:|---:|
| Ordinary DOPRI5 | 41.278966 | 67.9712 | 0 | 304.61 |
| Original PFR DOPRI5 | 37.786383 | 75.5840 | 34.1232 | 390.27 |
| Original PFR Euler42 | 39.591289 | 42 | 21 | 230.36 |
| Asynchronous calibration | 38.395560 | 50 | 0 | 235.95 |

The original PFR DOPRI5 remains better in maximum quality. Euler42 was selected from a block-count budget, before its FID: 42×12+21×4=588 Transformer blocks versus 600 for asynchronous calibration. Their observed sampling times are close, and asynchronous calibration improves FID by 3.02% at this budget point. Block counts omit differences in embedding/head costs; timings were not repeated as controlled hardware benchmarks. This does not establish a complete Pareto frontier.

Ordinary CFG DOPRI5 gives 33.372389, also worse than the asynchronous result. The benefit is therefore not explained solely by replacing the Euler50 baseline with the original adaptive solver.

### 4.4 Representation transfer — pending

A fixed five-arm ImageNet1K RAEv2 screen is running: original100, original110, short, asynchronous, and query-only. It preserves the official 100080 EMA, DINOv3L-k7 representation/decoder, shifted Euler grid, IG1.78 interval, and original evaluator. Calibration events are mapped to the same linear-bridge coefficients, not assumed to represent identical semantic stages. Every candidate has 110 shared full/base forwards, matching original110. Native100 matches the original sampler pixel-for-pixel on eight images. No RAEv2 clock-quality result is available at this draft revision.

## 5. Relationship to prior work

FSG [1] supplies the latent-calibration viewpoint and motivates the event schedule. Our Euler clock intervention does not reproduce its SDXL DDIM pipeline. A separate released-code audit identifies nominal/executed clock differences in a specifically documented software environment; it neither reproduces nor refutes the authors' image results [3].

PFR [2] is an existing repository method, not a new method introduced in this draft. Its future weak query and projected state displacement differ from the effective-reference calibration studied here. Timestep Guidance [4] and Time-Shift Sampler [5] already establish that changing a model's time input can be useful. Consequently, “querying another time” is not sufficient novelty.

Stage-wise guidance dynamics [6] and information-theoretic schedule optimization [7] cover broad temporal-allocation ideas. The condition-erasure probes in this repository suggest different semantic-retention stages across SiT and RAEv2, but they do not establish that those differences cause PFR transfer failure. A new cross-representation explanation must be tested, not inferred from these correlations.

## 6. What is required before this becomes a paper claim?

The current evidence supports a promising low-budget empirical observation. It does not yet justify a submission claiming a new general guidance method. Critical gaps are the completed RAEv2 result, independent confirmation of the mechanism controls, the common-field control, uncertainty estimates, and a sharper distinction from existing time-input guidance and FSG. Stronger model/benchmark coverage is also needed: visible structural artifacts remain in the small SiT model's samples.

Any failed transfer or failed mechanism test must change the central claim rather than be omitted. The manuscript should be rewritten around the resulting evidence; producing this draft is not completion of the research objective.

## References and evidence

1. [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/html/2510.21512v1).
2. [Repository PFR counterfactual-residual theory and evidence](../docs/PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md).
3. [Released FSG clock audit](../docs/FSG_RELEASED_CLOCK_AUDIT_20260908_ZH.md).
4. [No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models](https://arxiv.org/abs/2407.02687).
5. [Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps](https://arxiv.org/abs/2305.15583).
6. [Stage-wise Dynamics of Classifier-Free Guidance](https://arxiv.org/html/2509.22007v1).
7. [Information-Theoretic CFG with Adaptive Schedule Optimization](https://arxiv.org/html/2606.24025v1).

Machine-readable evidence: [clock 5K](../experiments/results/terminal_defect_20260908/fsg_clock_quality_confirmation5k.csv), [mechanism controls](../experiments/results/terminal_defect_20260908/fsg_clock_mechanism_controls.csv), [frozen reference](../experiments/results/terminal_defect_20260908/fsg_clock_frozen_reference.csv), [original PFR comparison](../experiments/results/terminal_defect_20260908/fsg_pfr_matched_bank.csv). Frozen protocols: [clock screen](../docs/FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md), [RAEv2 transfer](../docs/RAEV2_FSG_CLOCK_TRANSFER_PROTOCOL_20260908_ZH.md), [common-field control](../docs/FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md).
