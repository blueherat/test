# Research Status

> **Final round 2/8 — 2026-09-07.** Fixed prefix64K1K is complete: FID38.576569833 versus official38.486773927, 0.233316% worse, inference2.351653×. The unchanged independent5K is also complete: FID6.926132977 versus official6.949768478, 0.340090% better, inference2.353807×. Both independent FID reconstructions and full pixel/identity checks pass; neither meets3%, so no further data/strength expansion. Two previously fixed legacy5K methods are now live. One CPU-only probability calibration is rejected: even-class alpha1.300557 worsens odd-class loss by .210390 (class SE .090984), so no new scaled sampler. The remaining fixed [conditional-variance protocol](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md) has theory and a heldout gate but no training or sampling yet. See [calibration result](RAEV2_PREFIX_RATIO_CALIBRATION_20260907_ZH.md), [research index](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md), and [eight-round ledger](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md). The goal remains unmet.

> **Latest user limit — 2026-09-07, final round 1/8.** At most eight further research turns, then close and commit all theory, ideas and evidence even if the 3% target remains unmet; do not reset the count for debugging or waiting. The fixed 64K prefix head now passes held-out classification (−1.215789 ± .217665 class SE). A real-gradient audit caught precision protection ending before backward; the controlled fix reduced the same-input FP32/FP64 discrepancy from .006861 to .00000413 without changing weights, formula or tolerance. All five fixed gradient checks and original8 pixel parity now pass. Fixed1K is running, unchanged independent5K follows regardless of its FID. Two unchanged legacy5K methods are queued after it if quality is still below target. See [eight-round limit and decisions](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md). The goal is still unmet.

> **New user-authorized restart — 2026-09-07.** The active goal is a theory-grounded guidance method achieving at least **3%** relative FID improvement on paired 1K or 5K RAEv2 sampling, without extensive tuning. The user explicitly reopened research; the older four-round closure below remains historical. Completed fixed 1K screens: official IG 38.486774, historical interval control 38.335024; ancestral 38.894423, partial ancestral 38.506604, calibrated reverse variance 38.541200, velocity projection 38.478746, noise projection 38.483936. None meets the goal. Fixed stochastic weak-reference FID is 38.270118 (+0.5629%); deterministic mean gating is 38.566323. Independent seed202609072 5K is complete: official 6.949768, interval 7.011577, stochastic weak 7.027320 (1.1159% worse than official; 1.723× inference cost). The weak-reference branch failed confirmation and is retired without tuning. All five balanced 1K blocks are reported as diagnostics; none gives this candidate 3%. A decoded-image critic gradient route passed analytic checks, fixed-state finite differences and 8-image trajectories; two frozen critic 1K arms are complete: isotropic 38.535461, exchangeable 38.434218 (+0.1366% vs official, worse than interval; about 5.568× inference cost). A finite-Euler two-spatial-mode moment correction passed the original official8 pixel parity and completed 1K: FID38.518130 (0.0815% worse; approximately baseline inference cost), with four data-estimated moments and no FID-selected coefficients. Before its first 1K FID, one unchanged independent 5K was also frozen to examine sample-count-dependent moment effects, reusing paired official controls; that unchanged 5K completed at FID6.933352 versus official6.949768 (+0.236216%, insufficient). Frozen semantic-add/orthogonal 1K completed at 37.746546 / 37.704792 (+1.9233% / +2.0318%). Unchanged orthogonal .15 completed independent 5K at FID7.189965 (3.456173% worse than official; 1.990× inference cost). Its planned Heun cost control was cancelled by the predeclared quality condition; the patch was never applied. No strength/window search followed. One fixed 2048-update paired-noise ratio critic fit is complete (308.098 GPU seconds); independent held-out scaled logistic loss −10.149506, class-cluster two-SE upper bound −9.455588 versus zero, passes the predetermined sampling entry condition. Actual gradient and original pixel checks passed. The fixed raw-ratio 1K is 38.442135 (+0.115985%); a single global probability temperature 1.504111 fitted on even classes and validated on odd classes improves classification loss but gives FID38.567599 (−0.210008%). No further temperature/window trials. A 64-class, all-99-time diagnostic found this frozen critic loses its classification-risk gain on actual native states (loss +3.566283) despite negative loss on re-noised endpoints (−7.511846); paired class SE of the 11.078128 difference is 1.275236. The 5K/1K actual-state bank completed (283976 main-model calls). One actual-state transformer fit failed its held-out gate (loss −0.029079, two-SE upper +0.259915); two fixed convex heads on the native pretrained depth8 prefix also failed (single pairing +1.321792; exact five-real conditional average +0.937831). No FID images were generated from these failed heads. One fixed 64K/8K expansion is running with the same prefix, dim/N prior and unit guidance strength: real encoding and all72K native states are complete (3590360 main-model calls). The first feature preflight failed before extraction due to a local name shadowing bug; its outputs are retained, that name is fixed, and the same data/fit/quality protocol has resumed. Before the new head or any FID exists, a continuation also freezes paired1K and unchanged independent5K regardless of the 1K score, conditional on the original fit/gradient/pixel checks; both results will be reported, with no automatic goal completion. See [actual-law results](RAEV2_ACTUAL_RATIO_20260907_ZH.md) and [prefix theory and frozen expansion](RAEV2_PREFIX_RATIO_20260907_ZH.md). The 3% goal remains active and unmet. See [current goal](RAEV2_GUIDANCE_GOAL_20260907_ZH.md), [complete screen ledger](../experiments/results/raev2_guidance_20260907/screen_ledger.json), and [weak-reference protocol](RAEV2_STOCHASTIC_WEAK_20260907_ZH.md).

> **Final closeout — 2026-09-06 UTC / 2026-09-07 China.** The user-authorized final four rounds are complete; no fifth round or further method expansion. The ≥5% fair-total-cost FID goal remains **unmet**. Paired bridge 1K FID38.316312 versus inference-matched official107 FID38.372179 is only a 0.145592% point-estimate reduction, without independent quality confirmation. Pressure/innovation, unanchored temporal score consistency and sample-only finite-density transport each failed their stated mechanism entry conditions. 52 primary papers, historical theories, implementations and compact evidence are organized for Git. See [final closeout](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260906_ZH.md), [four-round ledger](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md) and [archive index](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md). Entries below are dated historical records and do not authorize resuming retired work.

> **Authoritative goal reset — 2026-09-06.** Theory must be coherent and
> elegant, explain a concrete guidance mechanism, and derive an intervention
> inside the denoising trajectory. Success requires at least 5% relative FID
> reduction on RAEv2 against a reliable baseline at comparable computational
> cost, with independent confirmation. Post-generation selection is excluded.
> These requirements govern the resumed research. See
> [`RAEV2_GUIDANCE_GOAL_20260906_ZH.md`](RAEV2_GUIDANCE_GOAL_20260906_ZH.md)
> for the complete current scope and cost accounting; it supersedes older goals.

> **User imposed a final iteration limit — 2026-09-06.** The active
> paired-bridge 1K is round 1; research must close by the end of round 5
> (round 4 may be the final round if evidence is sufficient). No coefficient,
> window or seed searches may disguise extra rounds. If the ≥5% fair-cost
> performance target remains unmet, say so and deliver organized theories,
> ideas, experiments, compact results/data manifests and a Git commit.
> Archival work is now running in parallel. See
> [the authoritative final-round ledger](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md).
> Stochastic Interpolants brings the documented reading to 50 primary papers;
> its conditional KL mechanism is retained without granting the current
> deterministic sampler an unproved guarantee.

> **Paired-bridge 1K screen launched — 2026-09-06 14:59 UTC.** The
> unchanged 2048-update candidate/control checkpoint is entering a fixed
> parity → official100 → candidate100 → control100 → inference-cost-matched
> officialK → uniform FID pipeline, all on physical GPU3. New seed202609151,
> 1000 ascending classes and B8 are fixed. Source identities, decoder config
> and evaluator files were frozen before GPU work. The driver runs in an
> independent session; inspect its PID/state before any recovery. No image
> selection, coefficient/window adjustment or FID-driven step selection is
> allowed. This is an inference-matched quality screen, with additional
> training/preparation costs still explicit; no total-cost or ≥5% success claim.
> See [protocol](RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md) and
> [live execution](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1/screen_v1/screen_execution.json).

> **Original Flow Matching reading — 2026-09-06.** The documented primary
> reading now covers 49 papers. Original CFM theorem proofs distinguish the
> population regression objective from finite-network accuracy. A local weak
> continuity argument covers the current non-independent paired endpoints; it
> does not assume their conditional Dirac paths satisfy the original positive
> density conditions. Neither reuse across auxiliary time nor a midpoint solver
> guarantees successful learned transport. The current negative regression
> result remains explicit. See
> [`RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md`](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md).
> All 17 archived identities were independently verified.

> **Paired-bridge fixed mechanism experiment complete — 2026-09-06.** One
> 2048-update joint fit, 10K held-out teacher records and 96 actual trajectories
> are complete. The learned normalized velocity risk is about 0.082% worse than
> zero prediction. In contrast, the predefined finite-map channel-moment gap is
> 7.30% smaller than official; the mean-only control does not share the overall
> improvement. 98.51% of candidate-versus-control net advantage comes from the
> last step. All times remain included. The 32-class actual trajectories are
> finite and their saved endpoint identities pass independent CPU checks. This
> supports preparing one unchanged paired 1K image screen, not a FID claim.
> Original training completed; only unstarted validation/rollout were resumed
> after the parent disappeared. Training was not restarted. Timing gaps and
> preparation costs remain explicit. See
> [`RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md`](RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md).

> **Finite-step correction structure derived — 2026-09-06.** A paired auxiliary
> bridge from a teacher state's native Euler successor to its true next-noise
> state defines a specific conditional velocity. Exact transport removes the
> native step's marginal mismatch and prevents relative KL growth for any actual
> input law; invertible maps preserve the existing KL rather than strictly
> reducing it. A conditional covariance divergence explains finite bridge motion
> even when the initial Bayes residual mean is zero. The derivation and endpoint
> conditions pass independent review and a fixed scalar algebra check. This is
> a design structure, not a trained RAEv2 result. Approximation, transfer to
> actual trajectories and cost remain to be tested. See
> [`RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md`](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md).

> **Actual-distribution feedback reading — 2026-09-06.** Two further primary
> papers bring the documented reading to 48. Sobolev score-difference estimation
> supplies a concrete gradient regularizer and its statistical bias; Discriminator
> Flow supplies an implementation that refreshes negatives from the current
> generated trajectory. Neither fixes a natural finite correction amplitude for
> RAEv2. The image DF critic is an IPM critic, not a calibrated density ratio,
> and its trajectory-refresh and backward costs must be counted. The Sobolev
> paper's ECG code differs from the noisy two-sample CE theory and score-to-epsilon
> conversion; this is a source correspondence finding, not a reproduced failure
> of the paper's table. Both papers' positive results and limits are retained in
> [`RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md`](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md).
> All 102 archived source identities pass an independent hash check. No new
> training, sampling or GPU work was performed for this reading. The ≥5% goal
> remains active and unmet.

> **Affine-reflection guidance: fixed quality screen complete; retired —
> 2026-09-06.** Known RAE affine support supplies a parameter-free two-query
> reflection average followed by clean-support projection. In ideal arithmetic,
> teacher risk contracts; the ideal arithmetic version of the fixed 100-step
> Euler structure also preserves the reflection symmetry and normal Gaussian
> bridge. Native off-branch endpoints and pixels match production bitwise on
> all 16 parity images, but the deployed reflected-field invariance has RMS
> residual 0.006598 in the fixed GPU check; strict finite-precision invariance
> or risk contraction is not established. This does not
> guarantee FID. The paired 1K screen is complete: official100 FID
> 38.2515918863, reflection100 38.2886408205, cost-selected official201
> 38.5581217308. The candidate is 0.096856% worse than the cheaper original
> baseline and 0.698895% better than official201; the latter is not success.
> Cost-selected201 covers candidate T/W by 0.346066% / 0.361610%, and outer
> wall by 0.383736%. The earlier cost-only200 remains archived and charged;
> its T was 1.350953 seconds short, triggering201 before any FID.
> Independent low-rank FID reconstruction passes (maximum difference
> 2.33378e-5), along with 463 artifact identities, batch pixels, paired input
> hashes and both cost decisions. No candidate coefficients or precision were
> revised. This fixed method is retired without seed/scale/window searches or
> larger sampling. This is not a new requirement for 1K gains to exceed 5%,
> nor a statistically established claim that all scales are ineffective.
> See the [frozen experiment protocol](RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md)
> and [current results record](RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md).

> **Primary reading expanded to 46 papers; mechanisms under examination —
> 2026-09-06.** Formal PAG, SEG, Characteristic Guidance, linear CFG/CPC and
> Sliding Window Guidance papers, relevant appendices and official
> implementations have been studied. CPC separates mean forcing from posterior
> covariance contrast; SWG distinguishes input cropping from decoder query
> information deletion. Neither supplies same-class Full/Base error semantics
> or an automatically determined gain. The fixed 80-row teacher paired-error
> follow-up is complete: all 10 time-group mean cross-products C_W are positive
> (59/80 individual rows positive), but the best common scalar in the empirical
> quadratic can lower this equal-time denoising risk by only 0.00204468%.
> Independent reconstruction from all stored heads/targets passes, with
> maximum scalar difference 1.14e-13. This is not FID improvement, and no
> oracle coefficient was deployed. All
> teacher bridge states and prior FP32 gap vectors reproduce bitwise. See the
> [paired-error results](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_RESULTS_20260906_ZH.md).
> Infinite query
> smoothing has an exact reverse-KL attention barycenter interpretation; the
> SEG paper's general curvature guarantee does not carry through its proof.
> A fixed audit of both DDT decoder blocks completed all 160 historical rows
> with native Full/Base bitwise parity. All 5120 weak attention head records
> have identical query rows and retain nonuniform key preference. Relative to
> the original gap, the new response has pooled cosine 0.18254 and 95.3291%
> of its energy orthogonal to each sample's original gap. This establishes a
> distinct structural response, without error-compatibility or quality evidence;
> there was no layer, smoothing-scale or gain search. Independently rebuilt
> direction statistics, identities and observed call counts pass. See the
> [completed decoder results](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md).
> Separately, Characteristic Guidance converts
> to shifted-query clean consensus in RAE coordinates. An independent Gaussian
> derivation gives convergence for a proper power target at w=1.78, but permits
> very slow convergence near the noise endpoint. All 792 existing directional
> Jacobian entries pass one necessary positivity test; this does not establish
> actual-head solvability, exact-score semantics or FID improvement. No new
> sampler is admitted by these results. See the
> [reading index](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md),
> [attention mechanism](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md),
> [fixed decoder audit protocol](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md),
> and [consensus analysis](RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md).
> ERG additionally provides an exact attention-entropy mechanism and valid
> Hopfield energy descent under explicit K/V compatibility. Its positive
> repeated-seed DiT FID evidence is retained, while layer, onset and strength
> selection and unmeasured latency prevent direct transfer as a parameter-free
> RAEv2 method. See the [ERG reading](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md).
> Two further primary studies connect finite-group inference averaging to
> orthogonal score-error decomposition and conditional flow-error bounds.
> The implementation review counts the internal 2/4/8 denoiser queries that
> the published sampler's outer NFE counter omits. The RAE clean projection
> prevents directly inheriting a velocity-Lipschitz improvement over the raw
> official sampler. See the [symmetry reading](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md).
> Fitted CFG derives a coefficient from a specified terminal contraction mode;
> ERK-Guid uses embedded solver differences and has positive low-step ImageNet
> results, but retains tuned strengths and thresholds. Neither mode has yet
> been identified in RAE's same-class Full/Base dynamics. See the
> [solver reading](RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md).
> ICG additionally motivates distinguishing a geometric condition reference
> from the unconditional mixture; random embedding responses require separate
> mean, covariance and curvature analysis. See the
> [ICG/TSG reading](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md).
> Two learned-guidance papers distinguish marginal consistency from the
> stronger per-clean-image self-consistency surrogate. ImageNet-64 FID
> 2.11->1.99 is retained as positive published evidence, with training/selection
> cost limitations; the adversarial follow-up improves alignment rather than
> the strongest reported FID. See the
> [learned consistency reading](RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md).
> Particle Guidance's balanced joint target and Doob conditional-expectation
> potential provide another source of natural time dependence. Marginal
> preservation, changed joint coverage, and improved population FID remain
> distinct goals. See the [PG/EDDY reading](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md).
> Foundational denoising theory gives exact noise-calibrated score and
> posterior-Jacobian identities for same-space Gaussian L2 estimation;
> nonlinear RAE cycles do not automatically satisfy them. See the
> [autoencoder-score reading](RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md).
> Two manifold studies distinguish support recovery from density recovery,
> and show how tangent posterior second moments create legitimate normal
> curvature bias. These are structural design clues, not evidence for another
> hard projection. See [score geometry](RAEV2_GUIDANCE_READING_SCORE_GEOMETRY_20260906_ZH.md)
> and [finite-noise decomposition](RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md).
> A DPS path-reaction study connects local approximation to terminal density
> distortion; an independently checked extension retains the extra PDE
> residual of an arbitrary finite denoiser. Oracle identities, normalized
> reaction, finite-horizon initialization and actual-head semantics remain
> distinct. See the [path-bias reading](RAEV2_GUIDANCE_READING_DPS_REACTION_20260906_ZH.md).
> No new training or sampling arm follows from these readings alone.

> **Encoder-derived raw-token support bound is inactive on cached queries —
> 2026-09-06.** The K7 encoder yields the deterministic ideal bound
> ||a_j||<=64 per raw token. Across all 184320 frozen teacher/rollout
> Full/Base/native-IG token predictions, the largest norm is 46.65022205;
> none violate the bound. This is not a new quality result and supplies no
> active correction on those queries. No radius, gain or sampling run follows.
> See the [CPU support check](RAEV2_RAW_TOKEN_SUPPORT_BOUND_20260906_ZH.md).

> **Two spatial energy balls: frozen paired 1K completed, negative —
> 2026-09-06.** Official100/global100/spatial100 FID is
> 38.1598808139 / 38.1611443514 / 38.3169688602. The candidate worsens official100
> by 0.411658% and the same-reference global control by 0.408333%; this finite
> implementation ends without radius, band-count, gain, window or seed searches.
> Fourteen CPU checks and a 16-image native endpoint/pixel parity passed before
> sampling; an independent audit verified all 100 steps on the full N=1000
> cohort. Measured trajectory time selected K=100 under the frozen rule before
> any FID was computed. Different same-model GPUs limit timing interpretation,
> and historical reference acquisition cost remains unclosed. Neither a
> full-cost success nor the 5% goal is established. See the
> [complete results](RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md).

> **Primary reading expanded to 26 papers; geometry and numerical checks —
> 2026-09-06.** New full-text studies cover CFG++ (including its ICLR flow
> appendix), ADG, PTQD, Q-Diffusion and Covariance Mismatch. The flow CFG++
> formula reduces to a step-dependent coefficient on the existing Euler gap;
> a new manifold effect cannot be claimed from that rewrite alone. A fixed-head
> CPU BF16 audit measures mixing error around 1.06–1.07% of the gap norm,
> primarily from final addition, and does not support catastrophic subtraction
> as the mechanism. This does not evaluate high-precision model-forward error.
> In a separate existing 1K real-latent cache, radius CV is 8.835%; nominal
> dimensionality does not justify a fixed isotropic shell. A Gaussian posterior
> derivation additionally separates real-sample norm from denoiser-mean norm.
> These results have not produced a new sampler, training run or FID gain.
> See the [reading index](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md),
> [ADG analysis](RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md), and
> [mixing audit](RAEV2_GUIDANCE_MIX_NUMERICS_20260906_ZH.md).

> **Fixed spatial energy analysis identifies cancellation — 2026-09-06.**
> Across both historical 5K banks, IG total latent energy is close to real,
> while spatial-DC energy is about 6–7% deficient and its orthogonal complement
> about 5.7–5.8% excessive. The complete channel residuals reproduce across
> banks (DC/AC cosine 0.87447/0.99754); source-identity deduplication preserves
> the result. These are raw second energies, not covariance eigenvalues or a
> quality guarantee. This supports a fixed two-subspace extension of the old
> energy-ball projection, with budgets derived from the Gaussian corruption
> path and no manual guidance schedule. Mathematical review permits one fixed
> 1K experiment. Native checks subsequently passed and the negative quality
> result is reported above. See [energy evidence](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md)
> and [frozen candidate protocol](RAEV2_SPATIAL_ENERGY_BALLS_PROTOCOL_20260906_ZH.md).

> **Primary reading expanded to 21 papers; endpoint response pilot negative —
> 2026-09-06.** Four further full-text studies (APG, CFG-Zero*, C2FG, Guidance
> Matters) distinguish projection coordinates, score bounds, coefficient design
> and evaluation effects. A frozen eight-image full-suffix adjoint intervention
> completed with no gain search or image selection: the FP32 finite response was
> 23.5503% of the linear prediction, while native BF16/uint8 response reversed
> sign at -15.1659%. This implementation does not enter distillation/deployment.
> Its 3992 backbone forwards and 792 input VJPs were a mechanism audit, with no
> FID evaluation, training or fair-cost performance claim. See
> [`RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md`](RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md)
> and the [reading index](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md).
> A separate CPU cache check at t=1 estimates native IG minus zero-velocity
> risk as -1.18912893 per coordinate, opposing the checkpoint-specific
> zero-initialization hypothesis; no zero-step/window search follows. A precise
> class-mean-risk derivation also shows why positive endpoint reward response
> alone does not establish a quality improvement. See
> [`RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md`](RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md)
> and [`RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md`](RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md).

> **Paired 5K scale audit complete; frozen candidate retired — 2026-09-06.**
> Official100/potential100/official105 FID is 6.9748978477 / 6.9746474293 /
> 6.9604170334. The candidate improves official100 by only 0.00359028% and
> worsens official105 by 0.20444746%; official105 uses 0.34737% more measured
> trajectory time than the candidate. All three paired arms completed, including
> a verified continuation that reused the first two arms without resampling.
> This finite candidate's quality line ends without another seed, 50K or larger
> training. This is not an equivalence claim across all sample sizes. A primary-paper reading and
> fixed historical features show IG gains of 8.4659% / 5.0949% at 5K while all
> ten balanced 1K folds fall below 5%. The previous mandatory 1K ≥5% gate was
> too strong; correcting it remains justified despite this negative result.
> Preparation remains 660.5193496611901 seconds plus an unclosed nonnegative
> remainder, so complete total-cost matching and the goal remain unverified. See
> [`RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md`](RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md)
> and [`RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md`](RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md).

> **Omitted-potential witnesses complete; broaden primary-paper reading —
> 2026-09-06.** All 100 times × 1K validation images have been evaluated for two
> fixed gradient tests outside the frozen potential's rank128 input rowspace.
> The radial witness is about 4.85 descriptive SEMs from zero, but the two-test
> same-bank variational plug-in is only 0.4002% of the old achieved proxy gain.
> This is not an upper bound on the complete omitted error, a causal explanation
> of the failed FID screen, or grounds to launch a width/time-weight sweep.
> All 100K old residual MSE values match bitwise; independent CPU aggregation
> agrees within 3.55e-15. No new training or FID follows from this diagnostic.
> At the user's request, reading now extends beyond fixed-point/PFR to strong/
> weak error compatibility, actual guided distributions, flow couplings and
> learned guidance. Eleven primary papers have been read with their core
> derivations, experiments and specified appendices. Sobolev Descent additionally
> motivates actual-state Jacobian-Gram feedback with explicit moving-target
> moment balance, under stated ideal conditions; no new candidate is trained.
> A concrete issue is that correcting X-G can also remove
> the existing guidance bias; better bridge MSE alone is not the quality goal.
> See [`RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md`](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md),
> [`RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md`](RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md)
> and the complete reading index
> [`RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md`](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md).

> **Extremal curvature and distribution structure — 2026-09-06.** A completed
> FP32 JVP/VJP Krylov audit detects positive Full symmetric score-proxy Rayleigh
> directions at all 40 fixed states, while all paired random directions are
> negative. The earlier three-direction audit did not inspect this subspace.
> Neither positivity nor the substantial nonsymmetric response identifies a
> density saddle or an error to suppress. Separately, CPU analysis of both
> complete historical 5K banks shows within-class variance expansion and
> between-class contraction together under IG; the changes are not simple
> homotheties and do not by themselves explain image quality. Four-cell
> predicted-clean features also show a large change in the current head's image
> feature response across Full/IG histories, with no new causal suffix or FID
> result. Full protocols, numerical checks and costs are archived in
> [`RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md`](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md),
> [`RAEV2_CLASS_MOMENT_AUDIT_20260906_ZH.md`](RAEV2_CLASS_MOMENT_AUDIT_20260906_ZH.md)
> and [`RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md`](RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md).

> **Observable-error finite solver: paired 1K screen failed — 2026-09-06.** A
> density-weighted continuity correction derives a potential-gradient guidance
> field as the unique minimum-energy correction, without treating finite F/B
> heads as exact scores. A known Gaussian test verifies the weighted projection,
> distinguishes it from ordinary Jacobian symmetrization, and preserves the
> continuous/discrete boundary. This is residual flow matching in a gradient
> field space. A fixed 608,000-parameter scalar solver has passed nine CPU tests
> and a real RAEv2 GPU derivative/parity pilot. Its one fixed 2,048-update fit
> on 5K real images is complete. Disjoint 1K fit-heldout validation across all
> 100 times reduces weighted bridge coupling MSE by 1.0968%, while retaining
> a nonzero weak residual and negative mean gains at the first 17 query times.
> The complete 100-step B8 benchmark adds 4.783% sampling time. Paired 1K FID
> is 37.5627037 for official100 and 37.5316644 for the fixed potential: only
> about 0.083% relative reduction. The original operational 1K gate failed,
> and no later stages ran in that initial screen. That gate was subsequently
> corrected: the frozen candidate and official100/105 completed a paired 5K
> scale audit, reported above, and the candidate is now retired on that evidence.
> Full preparation-cost matching and independent confirmation did not run.
> The goal remains unmet. See
> [`RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md`](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md).

> **Decoded class moments — 2026-09-06.** CPU analysis of both complete
> historical 5K banks separates Inception within-class and noise-corrected
> between-class traces. Under IG, decoded within-class trace decreases and
> between-class trace increases; both move toward original-image values in
> all six bank/split summaries. Raw-latent changes have the opposite signs.
> The positive decoded pooled-trace increment comes from the between-class
> term, but within-class contraction also repairs Full's excess variance.
> Reconstruction references give the same direction; the reversal alone does
> not identify decoder nonlinearity, a causal guidance direction, or FID gains.
> No new model calls or sampling were performed. See
> [`RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md`](RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md).

> **Mechanism admission clarified — 2026-09-06.** A learnable controller and a
> terminal distribution loss do not alone meet the mechanism-to-design goal.
> The proposed 2049-parameter affine F/B token gate has not derived its structure
> or useful control direction from a concrete error mechanism. Its module passes
> ten CPU tests, but the eight-image full-rollout GPU gradient audit has not run;
> that audit, training and FID sampling are now on hold. Auxiliary training is
> allowed in principle, with disclosed costs; the issue is the missing derivation.
> See [`RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md`](RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md).

> **Decoder finite-displacement audit — 2026-09-06.** FP32 decoder JVPs on two
> historical 1K endpoint subsets confirm a large nonlinear remainder, about
> 1.24–1.25 times the Full-to-IG pixel-change norm on average, opposing the linear
> component. This does not identify that remainder as an error to remove.
> A follow-up CPU check over both complete historical 5K banks does not reproduce
> the heldout block-mean improvement across seeds; all four class-aggregated
> projection intervals cross zero. Expansion of this specific block-mean JVP
> attribution is stopped. These are retrospective diagnostics, not new guidance,
> independent confirmation or FID results. See
> [`RAEV2_DECODER_LINEARIZATION_AUDIT_20260906_ZH.md`](RAEV2_DECODER_LINEARIZATION_AUDIT_20260906_ZH.md).

> **Depth/readout and density-decontamination checks — 2026-09-06.** The genuine
> depth-by-readout crossing completed with bitwise native parity. Crossed heads
> produce interactions roughly 13–31 times the original gap at time-mean level;
> the large opposing decomposition terms do not justify a shared-readout weak
> model. A separate FBG/SuperDiff-inspired linear-decontamination calculation
> passes exact-mixture CPU identities and refinement checks, but its RAEv2
> prerequisite audit fails: two independent trace estimates agree on 82
> negative posterior-covariance margins across all eight trajectories.
> This does not isolate model inconsistency from ratio-estimation error, and
> neither test is an FID result. Neither candidate proceeds to 1K, and no
> gain normalization, density temperature, bias, or clipping rescue is planned.
> Full formulas, numerical qualifications and costs are in
> [`RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md`](RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md)
> and [`RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md`](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md).

> **Normal-noise and curvature measurements — 2026-09-06.** Both frozen-state
> diagnostics completed without a sampling intervention or FID evaluation.
> Across 32 classes and all 100 steps, the normal component accounts for only
> about `8e-6` of the rollout full/base-gap energy. The separate FP32 derivative
> audit contains 480 rows from eight classes, ten times and two state domains;
> none of its three measured directions has positive score-curvature proxy.
> These measurements do not support launching simple normal projection,
> reflection averaging, or positive-curvature suppression as a quality method.
> They do not establish a negative-definite full Hessian or a failed FID result.
> The 5% target remains unmet. See
> [`RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md`](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md).

> **RAEv2 guidance objective revised — 2026-09-06.** The user has reopened
> guidance research with an explicit restriction: do not bind the method to
> NeurIPS-2025 fixed points or PFR, and do not seek gains through extensive
> manual tuning or hand-designed extrapolation schedules/windows. A candidate
> needs a provable, testable mechanism linked to model error or distribution
> quality; consistency/orthogonality alone is insufficient. The improvement
> must be reached by a guidance intervention inside the denoising trajectory.
> Post-generation rejection/ranking or generating extra complete images to
> choose from is outside the intended method scope, as clarified by the user.
> A generic reweighting KL identity does not explain the RAEv2 guidance mechanism.
> The initial target
> is roughly 5% paired FID-1K reduction under the frozen official RAEv2
> protocol, followed by independent confirmation. The newly started manually
> configured semantic/routing screens were stopped without evaluation after
> this clarification. See
> [`RAEV2_GUIDANCE_RESTART_20260906_ZH.md`](RAEV2_GUIDANCE_RESTART_20260906_ZH.md)
> for the current constraints and the explicitly superseded initial plan.

> **First constrained RAEv2 screens — 2026-09-06.** The diagonal finite-sample
> calibration failed its heldout mechanism audit. Two radial designs without
> tuned guidance strengths passed their mathematical checks but failed FID-1K:
> official `38.3977875`, global proximal `38.4083633`, actual-cohort energy
> ball `38.4086042`. Noise, labels, official model/decoder and evaluator were
> matched. The 5% goal remains unmet; no coefficient/window rescue is planned.
> Posterior acceptance was a scope error: its four GPU jobs were stopped at the
> user's correction, after 832 proposals and before a complete paired cohort or
> any FID. Its conditional KL certificate is retained only as an archived
> mathematical result; the method is retired and will not be rescued or evaluated.

> **RAEv2 guidance theory archive closed — 2026-09-05.** The SiT PFR/OU
> certificate remains a strong within-model result, but it did not transfer to
> RAEv2 at formal 5K.  The follow-up RAEv2 mechanisms are now fully archived:
> semigroup value, future-flow pullback, radial/tangential decomposition, and
> relative flow-map iteration all failed their paired 1K release gates.  The
> strongest reproducible lightweight RAEv2 result in this phase is still the
> piecewise internal-guidance schedule; its two 1K banks improve only about
> 0.5%, not the requested 5%.  See
> [`RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md`](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)
> and its compact machine-readable bundle under
> [`data/raev2_guidance_exploration_20260905/`](data/raev2_guidance_exploration_20260905/).

> **Active PFR mainline — 2026-09-03.** The current inference-time guidance
> result is Projected Future Reference (PFR), not the paused DiT bad/good line
> below.  The new theory treats PFR as counterfactual weak-reference
> residualization with a boundary/schedule-jump/cross-Jacobian terminal-risk
> decomposition, rather than
> as a numerical-integration or more-accurate-posterior method.  Derivation,
> exact counterexample, terminal ADM audit, non-overlapping FID-5K confirmation,
> and claim limits are in
> [`PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md).
> The previous mechanism ledger remains
> [`PFR_MECHANISM_AUDIT_20260903_ZH.md`](PFR_MECHANISM_AUDIT_20260903_ZH.md).
>
> A repository-wide RNG audit also found that historical adjacent run seeds
> `0/1` shared 99.84% of the FID-5K sample bank under the legacy
> `seed + batch_index` scheme.  Their within-bank paired method comparisons
> remain valid, but they are not independent replications.  A disjoint
> `seed=1000003` FID-5K bank confirms ordinary IG `40.7983` versus PFR
> `37.6459`; future sampling uses a versioned namespaced RNG schema.

> **Pause notice — 2026-08-28.** The DiT model-relative bad/good line is
> paused by user request. No sampling, model download, metric search, or
> intervention experiment is active. Statements such as “current”, “now”, or
> “in progress” below are dated execution-history snapshots unless the work is
> explicitly resumed. The authoritative pause boundary, evidence tiers, data
> retention policy, and Git inventory are in
> [`DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md`](DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md);
> the corrected method-level outcomes are in
> [`DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`](DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md).

## Paused DiT research snapshot

Latest internal-signal decision on 2026-08-28: the v2.2 cross-scale e-process
`E` is retired as both a generic bad-image alarm and a suffix-repair trigger.
Its operational likelihood-ratio martingale remains mathematically valid, but
the frozen quality alternative did not survive external tests. In the matched
suffix pilot, only 3/8 joint `E+B` paths versus 6/8 `B`-only paths had a stable
visible repair opportunity. The protocol-consistent strict outcome was 10/23
opportunity successes for joint and 17/48 for B-only, but these pooled rates
come from different paths and are not an E comparison. Across the eight
matched path pairs, joint-minus-B-only differences were four negative, zero
positive, and four tied. The immutable corrigendum result is
`f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358`.

The first conditional-branch proposal has also been falsified in its frozen
form. At horizon 10, choosing the four-scout medoid repaired 5/18 opportunities
(`27.8%`), below the exact same-four-scout uniform-policy value of `37.5%`;
higher attempt-0 outlier ratio predicted opportunity with AUC `0.286`.
Therefore medoid/high-O is `STOP_CURRENT_FORM`, and low-O AUC `0.714` is only
the algebraic sign reversal of the same failed statistic.

A reveal-after-freeze audit found one reverse, post-hoc clue: the horizon-10
maximum-nonconformity scout repaired 10/18 (`55.6%`), `+18.1pp` over uniform.
The effect was much weaker at horizon 5 (`+6.9pp`) and horizon 20 (`+1.4pp`),
concentrated in class 795, and accompanied by slightly worse small-sample
safety readouts. It is therefore not a result to promote, only a narrow
early-horizon escape hypothesis.

That exact hypothesis is now frozen in the V1.2 new-suffix prospective lock
`cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9`.
It uses only the common-prefix-normalized, multi-scale `pred_xstart` geometry
of four symmetric baseline-P scouts at step149/h10, seals the argmax choice
before opening endpoints, and compares it with the exact uniform policy over
the same four scouts. The 128-prefix design contains 96 class-795 confirmation
prefixes and 16 each from classes 207/602 as gross-harm sentinels. It tests a
branch selector, not a bad-prefix detector. External blind severity, FID, and
representations remain evaluation only and cannot enter the selector.

Execution is now complete: all `128/128` prefixes and `640/640` endpoints
finished without deletion or failed jobs. Before opening any prospective PNG,
the frozen internal extractor sealed all 128 h10 choices in product
`1c2ee25f6ca7b07c6259a0d9abeb176df9e7b26a0413d197417d24a62d24c72e`;
an exact-validation rerun returned the same identity. The downstream builder,
qualification gate, and analyzer were separately frozen in blind-evaluation
lock V1.1
`42529439647c1647800ff7bdbed587c6e991d816ef7c1909d4ca0fcdc12db7db`.
The public blind pack contains 640 anonymous absolute images, 512 anonymous
attempt0-vs-fresh pairs, and seven old frozen qualification anchors. Two
initial reviewers passed 6/7; an initial third scored 5/7 and was excluded
before formal review; a fresh replacement passed 6/7. Three qualified reviews
are now in progress. No efficacy result is available until all three complete
and the frozen reliability, effect, index, and safety gates are evaluated.

The complete method ledger, including formulas, positive clues, failures, and
claim boundaries, is in
[`DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`](DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md).
The detailed E/B repairability audit remains in
[`DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md`](DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md).

Latest evidence on 2026-08-28: the disjoint third pool is complete at 1,800
trajectories, but its preregistered label-only Stage A found only six
clear-bad endpoints, four with blur/soft fusion, versus required event counts
of 30 and 15. The formal run therefore ended as
`EVENT_GATE_FAILED_NO_SCORE_ACCESS`; Stage B never opened row-level labels or
either frozen internal score. This is an event-limited no-test result, not a
formal pass or failure of B/C. Label reliability is also inadequate: the three
independent model reviewers have binary clear-bad Fleiss kappa 0.147, and a
single adjudicator downgraded 23 of 29 raw-majority bad cases. A future pool
must qualify reviewers against frozen anchors and use two independent
adjudicators before any candidate score is opened.

After that failed gate, a separately frozen, aggregate-only sensitivity audit
read exactly the two pre-existing internal candidates under five already
available blind-label definitions. It did not search features, change
directions or thresholds, compute confirmatory p-values, or authorize an
intervention. Mean decoded-pred-xstart local blur severity B has AUC
0.668--0.848 across definitions and fixed-threshold TPR/FPR ranges
0.250--0.593 / 0.045--0.073. Its direction is comparatively robust, but it is
only an optional hand-crafted blur/fusion subtype detector for a future
event-rich pool, not the current theory or intervention mainline. The latent-channel-3
low-structure-jump candidate C has AUC 0.428--0.728 and TPR 0--0.483, so it is
label-sensitive and remains a falsification/control candidate rather than an
intervention trigger.

A hard method/evaluation boundary now applies. The method may use only
preterminal sampler-internal state: latent trajectories, predicted-clean
states, score/variance outputs, realized innovations, and predeclared
cross-scale counterfactuals. Human labels and AUC evaluate a fixed detector;
Inception, DINO, and FID are terminal external judges. None may enter online
features, thresholds, class ranking, rollback decisions, or intervention
directions. The earlier endpoint representation audit is retained only as
retrospective label/batch context and cannot rescue B/C. No intervention is
authorized. The current formulas, evidence boundaries, third-pool result, and
next-pool plan are in
[`BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md`](BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md).

The current mainline is **model-relative bad/good trajectory-metric
discovery**. The goal is to find a computable, pre-terminal quantity that
separates conspicuous outputs below the frozen model's ordinary quality from
ordinary outputs, and then validate it on new blind trajectories. The search
keeps predictable, online-causal, and retrospective measurements separate and
covers predicted-clean dynamics, reverse-kernel uncertainty, realized
innovation, cross-scale disagreement, spatial/edge structure, and sparse
combinations. The protocol, notation, current post-hoc clue, controls, and
falsification gates are in
[`BAD_GOOD_TRAJECTORY_METRICS_ZH.md`](BAD_GOOD_TRAJECTORY_METRICS_ZH.md).

The earlier cross-scale likelihood-ratio theorem remains mathematically valid,
but its frozen higher-noise quality alternative failed the prospective
cross-prefix screen and is retired. It is now one candidate family inside the
broader diagnostic inventory, not the assumed quality mechanism. No guidance,
rejection, or rollback intervention is authorized by the present evidence.
The complete historical theory audit remains in
[`CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md`](CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md).

Execution status through 2026-08-28:

- The exact CFG-Rejection/EDM2 replication is complete at the paper's
  50-class × 200-seed = 10,000-image scale. Its released early-path statistic
  is at most a weak semantic/prototypicality signal; full-resolution review
  finds clear structural failures in both score tails.
- The OpenAI ADM ImageNet-64 stochastic baseline is reproduced with 249
  Gaussian reverse transitions plus its deterministic final transition. A
  fixed class/seed smoke is bitwise identical to the upstream
  `p_sample_loop` under the same CUDA random stream.
- The primary alternative is now uniquely defined in normalized VP heat
  coordinates, with an additive heat-variance shift, same-normalized-state
  score pullback, the actual baseline learned variance, and a predictable KL
  cap. Its schema-v2 observe-only smoke is complete for six ADM paths: the
  instrumented and pure-P float tensors match bitwise on a real audit path,
  all six PNGs match the frozen baseline, and no intervention occurred. The
  locked per-checkpoint `kappa=0.2` reference produced 0/6 crossings; this tiny
  result is not a detector failure. Because roughly 30 active capped steps can
  still accumulate a large information budget and collapse likelihood-ratio
  power, the next exploratory comparison uses predeclared total-K budgets 0.5
  and 1.0 while preserving the old run unchanged. The implementation is still
  observe-only for every formal claim. Separately, one explicitly post-hoc
  mechanics run used the single shift 1, alpha 0.25 and a fixed one-checkpoint
  rollback on six already viewed paths. It completed 6/6 exact state restores
  and fresh suffix draws while preserving all original-P pixels, but the
  outputs mix improvements and regressions and are not evidence of efficacy.
- The visual audit was upgraded to a false-negative-oriented v2 protocol with
  mandatory regional checks, native/nearest/smooth views, hard structural
  flags, delayed re-review of apparent clean/mild images, and separate
  semantic, absolute-defect, possible-defect, and clean outcomes. Under the
  corrected full-resolution review, none of the six ADM smoke images is
  eligible for a clean label; class recognizability is never used as a
  structural pass. This does **not** mean all six are below ADM's normal
  quality. The six ADM images and eight FKC CFG images are absolute
  defect-present but have `relative_bad=not_evaluated`. The documented v3
  primary endpoint instead requires a material, localized failure plus blind
  losses against five frozen, same-baseline, same-class typical anchors.
- The rollback theorem now distinguishes a future-dependent last-exit anchor
  from a stopping time. Conditional Ville does not apply to the already failed
  suffix. It applies to a newly and independently sampled baseline-P suffix
  from a saved prefix with `E_j <= rho_rb / alpha`; repeated fresh failures are
  then conditionally bounded by `rho_rb`, giving at most
  `1 / (1 - rho_rb)` new suffix draws in expectation under the stated kernel,
  state, independence, and termination conditions.
- The FKC EDM2 image path is source-audited: XS conditional/unconditional
  0.045, CFG 1.4, 64-step Heun with churn 40, and eight-particle resampling.
  Its released class flag is ineffective and its CPU resampling RNG is not
  seeded by the CLI seed, so a strict reproduction must record an additional
  resampling seed. A fail-closed local wrapper reproduces both released CFG
  and FKC smoke outputs byte-for-byte and validates source, checkpoint,
  manifest, and PNG hashes on re-entry. This nonlinear Heun transition is a
  prior-art baseline, not the first exact-Gaussian-LR test bed.
- The official DiT-XL/2 ImageNet-256 demo checkpoint is pinned and reproduced
  with 250 ancestral DDPM steps, CFG 4 and the MSE VAE. The strict wrapper's
  seed-0 grid is byte-identical to a separate direct execution of upstream
  `sample.py` (SHA-256
  `909d19ece2a0ed77b1318eb9e79d967980341322db72ccea6cf55473ccbb3a55`).
- A post-hoc DiT class-207/seed-2 suffix screen now contains exact replay plus
  four fresh baseline-P suffixes at internal timesteps 225, 180, 120, and 60.
  Its corrected v2 review separates endpoint quality from prefix preservation
  and distinguishes a merely identifiable tail from a naturally attached,
  tapered, feathered tail. The clearest late matched failure is a looped hind
  leg, while the best matched branch has the most natural tail. Decoded
  `pred_xstart` histories and hash-bound offline metrics show no common
  one-sided jump signal across three clear failures: one branch is burst-like,
  whereas others commit early to wrong topology. Both ImageNet classifiers
  still predict golden retriever for all 20 endpoints.
- Exact multi-scale shifted-DiT replay is now complete for all 16 fresh
  branches at `delta_nu={0.25,1,4}`, with global plus 4x4 spatial components,
  shared baseline covariance, and reconstructable Gaussian innovations. The
  original one-sided high-noise LR fails: the t120/t180 failures require the
  opposite sign, and clear-good branches can have larger running maxima. A
  fixed path-level sign mixture is still an exact e-process. At
  `delta_nu=0.25`, its terminal value ranks each of the three clear failures
  first within its four-branch checkpoint, but a clear-good t225 branch
  overlaps; the corresponding anytime change-point maxima are only
  `0.099/0.050/0.162` log-e, so catching all three would require
  `alpha >= 0.951`. This is a weak post-hoc ranking clue, not a usable Ville
  trigger.
- The strongest local t60 component was also spatially re-audited. Its
  nominal `tile_12` box is mostly lower-left snow, not the tail or malformed
  hind leg; tail pixels are mainly in tiles 4/5 and the leg is nearer tile 13.
  The signal can only be treated as a frozen black-box candidate, possibly
  due to VAE receptive-field spillover or chance. Tail quality is now graded
  separately by root attachment, continuous taper, and feathered hair flow:
  attempt 004 is the most natural, while merely tail-like broad/blunt/filament
  shapes are not called fully normal. All of these remain discovery results
  from one viewed prefix. New baseline-P suffixes must be blindly labeled
  before revealing frozen scores; failure to enrich clear structural errors
  retires the candidate rather than prompting more tile selection.
- A prospective within-prefix pool is now frozen and sampled for that narrow
  falsification. The protocol binds the exact observer/trace/baseline/x60 and
  schedule hashes, 32 branch-local RNG streams, `delta_nu=0.25`, black-box
  `+theta/tile_12`, `K=0.5`, and a `log(5)` alarm. Four 8-branch GPU shards all
  completed under unchanged baseline-P sampling and passed full state,
  innovation, CFG, LR, image, and closed-bundle reconstruction. No endpoint
  image or evidence value was inspected during generation. An initial attempt
  reached the publication validator but installed no bundle because NumPy's
  contiguous-copy helper changed the scalar `per_step_K_cap` from shape `()`
  to `(1,)`; runner `edb18e9...` was replaced by `104b099...` only to preserve
  zero-dimensional NPZ arrays, with no sampler/evidence/protocol change. The
  four failed staging directories were automatically removed. Evidence-free
  dual review had very low raw agreement on the hind-limb endpoint (9/32 exact,
  Cohen's kappa about 0.056), so a third reviewer conservatively adjudicated all
  native images before the complete annotation was self-hashed and locked. The
  final labels contain 2 clear hind-limb failures, 18 not-clear failures, and 12
  uncertain images; overall structure has 3 clear bad cases. Tail naturalness
  remains separate at 9 natural, 12 odd, 8 malformed, and 3 unscorable. The
  one-time unseal then found **0/32 alarms** for the frozen
  `+theta/tile_12/log(5)` candidate, including 0/2 clear failures and 0/3 overall
  clear bad cases. Thus `TPR=FPR=0`; the run is event-limited by its preregistered
  `<3` rule, while the specific tile-12 quality interpretation is retired for
  having no usable absolute crossing. The frozen 34-path signed spatial
  mixture is only a descriptive lead: its running-maximum AUC is 0.944 on the
  2 versus 18 non-uncertain labels, whereas terminal AUC is 0.611. This supports
  a fresh-prefix test of transient mixed evidence, not a cutoff search or a
  rollback claim on this pool. These data remain conditional on one shared x60.
- The carried-forward 34-way signed spatial mixture has now received its
  prospective cross-prefix test on 64 new class-207 baseline-P trajectories.
  Two independent reviews plus evidence-blind adjudication locked 4 clear
  overall bad, 59 not-clear bad, and 1 uncertain endpoint before the sole
  aggregate unseal. The fixed `E_mix>=5` event occurred **0/64** times:
  TP=`0/4`, FP=`0/59`, so TPR=FPR=0 and one-sided Fisher `p=1`. The frozen
  outcome is `frozen_threshold_failed_to_pass` (also event-limited at 4 clear
  bad); it cannot be rescued by threshold, sign, tile, or label changes.
  Crucially, the within-prefix running-max AUC 0.944 did not replicate:
  cross-prefix AUC is 0.441 and terminal AUC is 0.169. The only notable frozen
  secondary is peak-to-terminal drawdown AUC 0.805, but maximum positive jump
  is only 0.610 and there are 4 positives across 11 related features. This is
  a hypothesis-generating reversal/termination clue, not a validated
  `high-noise-like` detector or an intervention trigger. Raw cross-scale K is
  also lower for bad paths (running-max AUC 0.131), opposing the original
  large-scale-discrepancy story. The exact theorem is retained; the fixed
  empirical quality alternative is **NO-GO**.
- Tail identity remains explicitly separate from naturalness in the new pool.
  Only 7/64 tails were reliably scorable: 1 natural, 5 odd, 1 malformed; 57
  were hidden, cropped, or identity-ambiguous. Among scorable tails, distal
  termination and taper/volume are the dominant visible problems; fluffy hair
  flow alone never upgrades an odd attachment/volume/tip to fully natural.
- The failed design also lacked a matched-alternative power gate. With
  per-component total KL `K<=0.5`, 34 uniform components, and threshold 5, a
  single dominant component pays roughly `log(34)` dilution and would need
  log-LR on the order of `log(170)=5.14`, versus matched-Q mean 0.5 and standard
  deviation 1. Future exact-evidence candidates must demonstrate Q-side power
  before any visually labeled P pool is generated. A switching-state/HMM
  mixture can preserve exact martingale calibration while allowing inactive,
  sign-changing, or location-changing alternatives, but it remains untested
  and cannot authorize current rollback.

Everything below is retained as failed or completed research history and is
not an active branch.

## Archived engineering mainline

This former run had moved to the official RAEv2 DINOv3-L-K7 full training state.
It used a staged continuation gate: first evaluate a 2,000-step Flow-only
control, then run the preregistered 5,000-step Flow/LPL comparison only if the
control remains healthy. Checkpoints are saved every 1,000 steps and all
comparisons reuse labels and initial sampling noise. The protocol is recorded
in `docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md`.

The remainder of this document records earlier RAE geometry, transport,
spectral, prior-decoder, and LPL evidence. These experiments remain useful
audit history but are not the current training mainline.

## Scope

The archived work contains distinct geometry and spectral lines.  The geometry
line asked whether frozen RAE tokenizers contain a usable group response; its
adapter attempts did not improve generation.  The spectral line asked why a
fixed inverse-variance direction loss improves held-out teacher metrics but
worsens rollout.  These lines must not be combined into a claim that
RAE-DINOv2 has a global group representation.

## Latest controlled generative result: prior-decoder mismatch

The Imagenette-64 `16/64/256d x 5 seeds` study confirms a stable trade-off:
empirical latent decoding improves with capacity, while a same-budget latent
prior produces modeling gaps of `2.79/9.92/22.42` FID. Ordinary flow loss,
SWD, effective rank, and condition-space C2ST do not predict the decoded gap.
An equal-angle intervention rejects the stronger claim that prior errors align
with unusually sensitive local decoder directions. Cross-fitted class-mass
reweighting recovers only `15.9%` of the 256d gap; the exact remaining FID
decomposition is dominated by covariance/diversity mismatch.

Post-hoc spectra show strong diminishing intrinsic variance: raw effective
rank changes only `9.94 -> 16.31 -> 20.14`, and decoder-condition effective
rank changes `8.36 -> 16.17 -> 17.91`. This supports low-dimensional added
variance, not yet low-dimensional added information. A within-model PCA
truncation/residual decoding intervention is still required for the stronger
claim. Full evidence and boundaries are in
`docs/IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md`.

The next registered direction is decoder-aware prior training. Literature and
the current covariance-dominated decoded gap support exposing the frozen
decoder's response to the prior objective before jointly updating model
parameters. The first comparison is flow-only versus latent, condition,
pairwise decoder-feature, and batch decoder-response distribution losses. Full
rationale and stop/go criteria are in
`docs/PRIOR_DECODER_ALIGNMENT_LITERATURE_AND_PLAN_ZH.md`.

That direction has now failed its mandatory no-training gate. A frozen-decoder
response atlas over `16/64/256d x 5 seeds`, repeated at 256 and 1024 samples,
finds no adjacent layers whose conditional-response Fréchet predicts modeling
gap under both leave-seed-out and leave-dimension-out evaluation. At 1024
samples every primary LODO correlation is negative; up2 is `-0.907`. The
decoder does use latent conditions, as shuffling worsens paired velocity MSE by
`6.87x/3.49x` at high/mid noise in all 15 runs. Formula, trace, hash, power and
projection-seed audits pass. This supports a pathwise joint-mismatch
interpretation, not the proposed fixed-time marginal response loss. Moment
repair, A3-A5 training, decoder adapters and joint training are not authorized.
See `docs/IMAGENETTE_DECODER_RESPONSE_ATLAS_RESULTS_ZH.md`.

## Current active result: latent trust spectrum

The latest five-seed SPC study closes the current subspace-path curriculum as a
method.  SPC worsens paired 5k-sample FID by `+6.30` on average (`0/5` seeds
improve both FID and KID).  Its fitted rank-16 basis captures `94.65%` of the
top-PCA energy optimum, so the intervention suppressed high-variance,
high-SNR directions rather than a cleanly separated detail subspace.

A variance-normalized generalized eigen diagnostic now separates cross-layer
predictability from final-latent variance.  On held-out ImageNet validation,
the whitened rank-16 basis reaches `R2=0.931` while overlapping top PCA by only
`0.198`; its train-half stability is `0.926`.  Across five frozen stage-2
training seeds, 128 held-out latents, six noise times, and 24 rank-16 blocks,
directional response changes smoothly from variance-dominated at low noise to
jointly variance- and predictability-dependent at high noise.  At `t=0.95`,
the two standardized coefficients are `0.542` and `0.558`, and together explain
`94.7%` of block-level log-gain variance.

The inference-only teacher-path rollout gate has now passed on the latent side.
Across 24 blocks, one-step gain predicts endpoint leverage with mean Spearman
`0.961` at `t=0.85` and `0.990` at `t=0.95`, versus `0.665` at `t=0.30`.
Three variance-matched high-/low-predictability pairs have endpoint gain ratios
of `2.83x`, `4.01x`, and `4.87x` at `t=0.95`, with all five seeds agreeing and
the same ordering at two perturbation amplitudes. Removing the high-
predictability clean component also causes more clean-endpoint error at high
noise, supporting a useful-signal-anchor interpretation rather than a harmful
shortcut.

This remains a mechanism result, not a generation improvement.  A small frozen-
decoder spot check is the only remaining gate before one single-seed reverse
trust-curriculum pilot.  Full evidence and stop/go criteria are in
`docs/RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md`.

Keep the following concepts separate:

1. Direct spatial equivariance: `E(gx) ~= P_g E(x)`.
2. Per-transform linear alignability: a separately fitted channel map aligns
   `P_g E(x)` with `E(gx)`.
3. Group consistency: maps fitted only for generators predict composed actions
   on held-out images, for example `C_90^2` predicts `rot180`.

Only the third level is evidence for an approximate group representation.
Decoder output from `D(P_g E(x))` is an OOD-latent robustness test, not an
encoder-side group-structure test.

## What the current evidence says

The largest frozen layerwise study uses 5,196 ImageNet test images and is
stored outside Git at
`$EQVAE_DATA_ROOT/artifacts/layerwise_imagenet/imagenet_test_n5196_3rae_d4`.
For RAE-DINOv2, sample-centered direct `rot90` error falls from `1.366` at the
patch input to `0.852` at `final_raw`; for `flip_h`, it falls from `1.263` to
`0.434`. This is meaningful deep-layer geometric alignment, but the residual
error is far from a direct equivariance identity. RAE-MAE does not share the
same `rot90` trend in this study: `1.186` at the patch input and `1.217` at
`final_raw`.

The appropriate current conclusion is therefore: DINOv2 becomes more
geometrically aligned with depth, especially for flips, but this alone does
not establish a clean global `C4` or `D4` representation. Any claim about
channel maps must still pass held-out generator-power and D4-relation tests.

A stricter 5,196-image correspondence audit now rejects the simple
homogenization explanation but also rejects a monotonic-depth story.  For
`rot90`, exact token correspondence rises from `1.77%` at patch input to
`67.97%` at hidden layer 9, then falls to `43.00%` at final raw while direct
error continues to improve.  For `flip_h`, it peaks at `92.22%` at layer 9 and
ends at `83.77%`.  Spatial effective rank falls from `226.0` to `38.7` tokens.
Thus middle layers form genuine correspondence, while late semantic/low-rank
compression improves average alignment but reduces spatial uniqueness.  See
`docs/RAE_LAYERWISE_CORRESPONDENCE_RESULTS_ZH.md`.

## Adapter attempts and outcome

The encoder-side adapter experiment uses an invertible additive-coupling map
and trains on ImageNet train images with held-out validation/test splits. The
latest full-image variant used `flip_h`, `flip_v`, and `rot180` with the RAE
encoder and decoder frozen. It did not establish a generation improvement.

The decoder-side inverse-adapter pilot is a useful negative control. It
optimized only the inverse adapter through a frozen RAE decoder on 8,000
ImageNet training images. On a fixed-noise 2,048-image validation check,
noisy reconstruction L1 decreased from `0.17418` to `0.16324`, while noisy
latent relative error increased from `0.35395` to `0.36675`. In matched 5,000
sample ADM-FID evaluation, the exact-inverse baseline scored `19.1944` and the
reconstruction-trained inverse adapter scored `19.7137`. Thus improving this
image reconstruction objective did not improve generation.

| Run | Samples | ADM-FID | Interpretation |
|---|---:|---:|---|
| official DINOv2 DiT-S epoch 14 | 50,000 | 12.8641 | official reference checkpoint |
| encoder-adapter DiT-S fine-tune, step 3,750 | 50,000 | 13.7027 | no improvement over that reference |
| encoder-adapter DiT-S exact inverse | 5,000 | 19.1944 | comparison baseline for inverse-adapter pilot |
| frozen-decoder inverse-adapter pilot | 5,000 | 19.7137 | worse generation despite better noisy reconstruction |

The 5,000- and 50,000-sample values must not be compared as a single ranking;
they use different sample counts and serve different controlled comparisons.

### Transport-compatibility audit

The active geometry-related question is no longer whether equivariance is
intrinsic to the RAE latent.  The invertible adapter is now treated as a
controlled coordinate intervention.  A phase-0 audit established that the old
adapted stage-2 runs used a standard Gaussian source and a straight path to
`f(z)`, rather than the transformed source `f(eps)` or the pushforward path
`f((1-t)z+t eps)`.  Those runs therefore confound source-prior mismatch and
conditional-path mismatch.

The real checkpoint passes cycle, decoder-identity, JVP, identity-path and
orthogonal-path numerical checks.  However, the old 20k original/adapted runs
used different world sizes, and the training code derives the pre-init rank
seed from world size.  Their model initialization and random streams were not
paired.  See `docs/LATENT_TRANSPORT_PHASE0_AUDIT_ZH.md` for the evidence and
`docs/LATENT_TRANSPORT_RESEARCH_PROTOCOL_ZH.md` for the staged stop/go rules.

The phase-2 no-training audit is now complete on 2,048 disjoint ImageNet
validation images. It cleanly separates source-prior mismatch from nonlinear
path curvature: anisotropic linear maps increase prior SW1 monotonically while
their chord defect remains numerical zero; the scaled real adapter increases
bridge defect from `0.091` to `0.293` and chord/pushforward velocity gap from
`0.175` to `0.903`. All implementation controls passed. At the same time,
projected VIV is nearly unchanged and the Gaussian-straight local ambiguity
proxy decreases as adapter strength grows. These proxies are therefore not
accepted as transport-quality predictors. See
`docs/LATENT_TRANSPORT_PHASE2_RESULTS_ZH.md`; only a paired four-path toy is
authorized next.

That four-path toy is now complete and stops the transport-recovery method
line. The experiment is valid: all numerical/solver controls pass, Base covers
all eight modes in 5/5 seeds, and Gaussian-straight is at least 10% worse than
Base in 4/5 primary-strength seeds. Strict Pushforward nevertheless recovers
half the gap in 0/5 seeds and averages `2.60x` Base sliced-W1. A post-hoc
no-training conjugacy audit pushes the trained Base field through the exact
Jacobian and recovers its same-noise endpoint to at most `7.56e-6` mean relative
error at 400 Heun steps. Thus the path implementation is correct; the ordinary
finite MLP class and Euclidean velocity loss are not closed under nonlinear
coordinate changes. Per the preregistered gate, no small-image, CIFAR, or RAE
stage-2 experiment is authorized. Full evidence and theory are in
`docs/LATENT_TRANSPORT_PHASE3_RESULTS_ZH.md`.

## Current repository entry points

- `notebooks/latent_playground.ipynb`: reconstruction and latent editing
  interface. Use it for decoder closure and visual inspection.
- `notebooks/dinov2_token_diagnostic.ipynb`: encoder-side token correspondence,
  orbit alignment, Procrustes, and group-consistency diagnostics.
- `notebooks/rae_layerwise_playground.ipynb`: simple per-image and batched
  layerwise visualizations for DINOv2, MAE, and SigLIP2.
- `notebooks/mnist_spectral_rollout_toy.ipynb`: low-cost paired MNIST
  teacher-path/rollout diagnostic with the simple `T(...)` and `V(result)`
  interface.
- `experiments/rae_layerwise_imagenet_study.py`: reproducible larger-scale
  layerwise study.

## Spectral flow-matching direction

The geometry-adapter line above has not produced a generation gain. A separate
mechanism audit found a stronger real-RAE asymmetry in stage-2 flow matching:

- radial DCT residual energy spans about `23.5x` in the executed audit and is
  much more anisotropic than a random orthogonal basis;
- low-frequency bands have higher teacher predictability;
- the frozen decoder is relatively more sensitive to higher-frequency,
  lower-residual bands;
- direction-only `gamma=0.5` flattens output-head gradient-variance slopes, but
  does not improve per-band GSNR and reduces total projected GSNR.

This supports an objective/capacity-allocation hypothesis, not a pure numerical
preconditioning or training-acceleration claim. The mini-DiT toy result is also
slightly negative, so only one strict real-RAE tiny screen is authorized:
`DiTDH-S step-5000 -> step-10000`, three paired seeds, `gamma=0` versus `0.5`,
fixed fp32 numerics, fixed train-only weights, and fixed-noise 5k KID/FID proxy.
The pass rule is preregistered in
`$HOME/data/eqvae/experiments/rae_spectral_tiny/protocol.json`.

The screen is now complete and failed its preregistered practical threshold.
Across three paired seeds, fixed held-out velocity diagnostics at 5k branch
updates improved raw MSE by `1.09%`, the decoder-sensitivity proxy by `1.60%`,
high-frequency MSE by `3.81%`, and the highest-frequency band by `4.17%`, while
the lowest-frequency band worsened by `1.26%`. This confirms that the fixed
weighting changes direction allocation on unseen images. However, exact 5k
generation improved KID by only `0.65%` to `1.89%` (`0/3` seeds reached the
preregistered `5%` threshold), and the 5k FID proxy worsened on `2/3` seeds.
The method line is therefore stopped without tuning gamma or adding seeds.

### Teacher-forced to rollout gap

The failure is now localized more precisely with a no-training audit of all
six endpoint EMA branches.  Each branch uses the same 64 cached ImageNet
validation latents, fixed noise and labels, official shifted 50-step Euler
times, fp32, and the exact frozen ViT-XL decoder.  Decoder/Inception metrics use
12 fixed samples per branch.  The aggregate tables live outside Git at
`$HOME/data/eqvae/experiments/rae_spectral_tiny/teacher_rollout_gap_*.csv`.
The 64 audit indices have zero overlap with the 2,048 ImageNet-train indices
used for the fixed spectral statistics; they are contained in the disjoint
512-image validation cache.

The most important result is counterintuitive.  On known linear interpolation
states, `gamma=0.5` genuinely improves the supervised vector field: for the
middle times it lowers latent error, raises per-band prediction/target cosine,
moves prediction energy toward target energy, and moves the regression slope
toward one on all three seeds.  This is not a train-set-only effect or a trivial
zero-amplitude predictor.  Yet its self-generated ODE states are farther from
the corresponding validation interpolation marginal at every measured time.
The final state has a larger energy deficit in bands 1--7; for band 7 the mean
log-energy ratio changes from about `-0.30` to `-0.49`.

There are two additional sign reversals:

- at `t ~= 0.69`, teacher-forced latent MSE improves by about `3.8%` and
  Inception cosine distance by `8.9%`, while decoded pixel MSE worsens by
  `3.2%`;
- early in rollout, the treatment has a smoother same-noise vector-field
  secant, but after `t ~= 0.69` its secant sensitivity becomes larger and ends
  about `8.8%` above baseline.

The supported mechanism is therefore a teacher-forcing/transport gap:
per-sample velocity accuracy does not control multi-step marginal covariance,
volume change, or perceptual distribution quality.  Pixel reconstruction is
also not a monotone proxy for semantic quality.  The next quality candidate
must directly constrain a short rollout's per-band covariance or log energy,
possibly with a one- or two-step differentiable rollout, rather than tuning the
fixed DCT `gamma`.

A hard vector-field time-switch probe tests the initially plausible causal
story that the high-noise band-0 sacrifice is the main trigger.  It does not
support that simple story.  On the same 64 noises and validation distribution,
using the partial vector field only at high noise (`t >= 0.85`), only in the
middle (`0.30 <= t < 0.85`), or only in the final two low-noise evaluations
worsens endpoint summary SWD by `5.49%`, `11.74%`, and `2.52%`, respectively.
Every result has the same sign on all three training seeds.  The middle window
has the largest cumulative effect, while the final two evaluations have the
largest damage per evaluation.  This SWD is deliberately a low-dimensional
probe over projected token means and log-band energies, not a full latent
Wasserstein distance or a replacement for FID.

The frequency direction is also counterintuitive: every partial-only window
makes endpoint band 0 slightly closer to the validation energy while making
bands 1--7 more under-dispersed.  Thus the deficit is not the direct transport
of one underfit coarse coefficient.  It is consistent with cross-frequency
coupling and excessive contraction in the learned vector-field Jacobian.  A
time-gated splice of the existing checkpoints is not a quality candidate.

There is a useful theoretical boundary.  The experiment's fixed positive
frequency matrix `W(t)` is independent of the sample.  With an unrestricted
function class, weighted and unweighted squared losses have the same
population minimizer, `E[u | z_t,t]`.  The observed changes therefore come
from finite capacity, shared parameters, and optimization-path reallocation;
the weighted loss is not a pure preconditioner that preserves the finite-model
solution.

### Low-cost MNIST mechanism gate

A bounded MNIST experiment now reproduces the qualitative teacher/rollout gap
without a large tokenizer or decoder.  It uses normalized pixels as an exactly
invertible latent, the same radial DCT mean-one weighting, paired small
convolutional velocity fields, 8,192 official training images, 1,024 disjoint
official test images, 1,000 updates, and three seeds.  All weighting statistics
and the independent MNIST feature classifier use training data only.  Decoded
feature and pixel metrics clamp generated pixels to the valid `[0, 1]` range;
the raw latent metric remains unclipped.

For the width-24 model, `gamma=0.5` improves the combined held-out teacher MSE
at `t in {0.1, 0.3, 0.5}` by `3.62%` on average, but worsens it at
`t in {0.7, 0.9}` by `1.50%`; averaged over all five measured times the net
change is only `-0.25%`.  Despite that small aggregate teacher improvement,
all three seeds worsen every rollout distribution metric.  Mean
weighted/baseline ratios are `1.093` for raw latent SWD, `1.109` for decoded
pixel SWD, `1.054` for feature SWD, and `1.163` for the MNIST feature FID.

A preliminary width check adds an important limit to the finite-capacity
story.  Increasing width makes the teacher-path reallocation much smaller,
but does not monotonically remove the raw-latent rollout penalty and can expose
seed-dependent rollout instability.  Finite capacity therefore helps explain
why the weighted objective changes teacher errors, but capacity alone is not a
sufficient explanation of the accumulated transport error.  The later
matched-basis and exact-skip experiments identify data-geometry alignment and
an unprotected high-energy subspace as the missing factors.

This is a cheap mechanism reproduction, not a replacement for ImageNet FID and
not a reason to tune `gamma` further.  Its practical use is to screen rollout-
aware losses, time windows, and Jacobian/covariance diagnostics before any new
RAE run.

### Mechanism resolution

The follow-up causal probes now resolve the teacher/rollout gap more sharply;
the full Chinese evidence report is in
[`TEACHER_ROLLOUT_MECHANISM_ZH.md`](TEACHER_ROLLOUT_MECHANISM_ZH.md).

The final supported mechanism has two phases.  At high noise, partial's
pointwise teacher MSE improvements do not preserve the band marginal drift
`2 E[<z_b,v_b>]`: its teacher-state drift RMSE is `2.77x` baseline at
`t ~= 0.95` and `3.24x` at `t ~= 0.85`.  In the middle interval, partial drift
is better on teacher states (`0.434x` at `t ~= 0.54`, `0.752x` at
`t ~= 0.315`) but worse when both fields are evaluated on the same baseline
rollout states (`1.121x` and `1.197x`).  This is direct evidence for an
on-path objective/transport mismatch followed by off-path generalization
failure.

Time-and-band splicing is causal evidence rather than a correlation.  On RAE,
the nonzero bands explain about `66%` of the high-window summary-SWD damage and
`81%` of the middle-window damage; band 0 explains only about `34%` and `16%`.
Thus the MNIST coarse-band mechanism is real but not a complete RAE account.
Euler discretization, radial-band energy alone, monotone global divergence, and
a single coarse band are all rejected as sufficient explanations.  No more
fixed-gamma tuning is justified.  A future candidate must constrain marginal
drift on one- or two-step self-generated states while preserving high-noise
coarse-direction MSE.

The bounded follow-ups are now:

1. directly match train-only per-band covariance or log energy after one or
   two differentiable rollout steps, concentrating diagnostics in the middle
   window and final two high-leverage steps;
2. test train-only endpoint spectral variance calibration before any retraining;
3. if revisiting the FxLMS analogy, use a parameter-space optimizer
   preconditioner designed to preserve the original MSE stationary points,
   rather than another output-weighted loss.

All three require KID/FID validation because amplifying decoder-sensitive high
bands can also amplify artifacts.

### Predictive small-image mechanism resolution

The preregistered five-seed MNIST/FashionMNIST follow-up is complete.  Its full
Chinese report is
[`TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md`](TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md),
with sequential predictions in
[`SMALL_TRANSPORT_GAP_PREREG_ZH.md`](SMALL_TRANSPORT_GAP_PREREG_ZH.md).

DCT, train-only PCA, and random orthogonal losses use the same time-dependent
weight spectrum and bit-identical per-seed baseline checkpoints.  On
FashionMNIST, DCT/PCA improve low/middle teacher MSE but worsen feature FID in
all five seeds, with mean ratios `1.461/1.787`; random orientation is neutral
at `1.003`.  MNIST reproduces the ordering in four of five seeds.  High-window
group-0 field splices explain about `90%-101%` of DCT/PCA FID damage on both
datasets, whereas the same random-basis group is neutral.  Keeping baseline
group 0 while using weighted nonzero groups therefore removes almost all
high-window damage.

The mechanism is time-dependent covariance-aligned risk reallocation.  At high
noise, inverse-residual-variance weights downweight the high-energy DCT/PCA
subspace; a shared raw-velocity network sacrifices that subspace while reducing
many low-energy errors.  Removing the exact linear skip in the analytic mixture
toy makes high-variance direction error and endpoint coordinate W1 worse in all
five seeds at both tested widths, while the skip-protected residual model does
not reverse.  Off-path amplification remains present in RAE, but teacher
restart proves it is not required for the small-image failure.

The subsequent preregistered training-side and gradient interventions sharpen
this into a four-factor mechanism: risk-budget shift, weak or conflicting
coarse/detail gradients, high endpoint leverage in the neglected subspace, and
no protected transport path.  On FashionMNIST, DCT/PCA shared-trunk gradient
cosines are `+0.165/-0.318`, versus `+0.947` for the matched-spectrum random
basis; their weighted coarse-descent ratios are `0.213/0.161`, versus `0.992`.
MNIST reproduces the separation at `0.238/0.224/0.968`.  Thus PCA can exhibit
true gradient conflict, while DCT is better described as weak-coupling neglect.

A direction-specific additive coarse guardrail improves Fashion DCT/PCA
feature-FID ratios from `1.461/1.787` to `1.053/0.923`, while a matched-total-
loss-scale control remains at `1.405/1.709`.  This is predictive causal support
for the risk direction, not a finished method: the guardrail retains only about
`9%/15%` of the original detail-MSE gain.  Parameter-matched split and
asymmetric paths protect coarse outputs but either reduce per-task capacity or
underfit the coarse branch.  The mechanism is now well localized, but no
training-side intervention has passed the preregistered absolute-quality and
detail-retention Pareto criteria.

### Frozen RAE mechanism bridge

The small-image mechanism has now passed two preregistered, no-training tests
on the three existing tiny RAE step-10000 EMA pairs.  P20 computes actual
autograd gradients only for the final DiTDH-S transformer block and output
linear on held-out cached ImageNet latents.  DCT and fixed random orthogonal
bases use the same eight weight eigenvalues.  At `t=0.85/0.95`, baseline DCT
has a band-0/nonzero gradient cosine of `0.098` and a band-0 descent ratio of
`0.244`; the matched-spectrum random basis gives `0.836` and `0.952`.
Partial checkpoints and the output head reproduce the same separation.  All
24 seed/time/checkpoint/parameter-group DCT conditions have descent ratio below
`0.5`.

P21 then tests the prediction at the rollout endpoint.  In high-window
individual-band field splices, band 0 is the most damaging band in all three
training seeds.  For stable bands 0--4, `1 - descent_ratio` and endpoint
summary-SWD damage have Spearman `1.0` in every seed.  The eight individual
effects add to high-all within `-1.03%` to `+2.37%`; band 0 contributes `35.8%`
and seven smaller nonzero effects contribute `64.2%`.  This resolves the old
apparent contradiction: nonzero bands dominate in aggregate because they
accumulate, not because one high-frequency direction has more leverage than
the coarse band.

The supported tiny-RAE mechanism is therefore two-stage.  High-noise
covariance-aligned risk allocation combines with weak cross-band gradient
coupling and endpoint leverage; the resulting state shift is then amplified
by the already measured middle-window off-path sign reversal.  This is a
frozen latent-proxy mechanism result, not a 50k ImageNet FID result and not a
successful new training method.

### Mechanism-to-quality gate

Three further preregistered small-image tests ask whether the resolved
mechanism can improve absolute generation quality.  A zero-initialized
width-12 residual adapter freezes the complete raw baseline field and can only
emit groups 1--7.  Same-state group-0 error is therefore exactly unchanged.
On FashionMNIST DCT, this reduces the raw weighted feature-FID ratio from
`1.461` to `1.009`, removing about `98%` of the added damage; PCA falls from
`1.787` to `1.208`, removing about `74%`.  However, DCT retains only `42%` of
the detail-MSE gain and PCA remains materially worse than baseline.  The
absolute-quality Pareto gate fails.

The initially proposed self-generated-state target is specifically rejected.
Off-path paired MSE alone gives DCT FID ratio `1.131`, almost reproducing the
combined rollout/drift result `1.147`; normalized drift alone gives `1.034`,
close to teacher-only `1.009`.  The original pair velocity is not a valid
conditional target after the state leaves the interpolation marginal.

Finally, a no-training trust-region study excludes all previously viewed test
images, selects residual scale on 1,024 new validation images, and reports on
another 1,024 images with new noise.  Three of five seeds select scale zero,
two select scale one, and none select an intermediate scale.  Final-test mean
FID ratio is only `0.9978`, below the preregistered `2%` improvement threshold.
Thus the mechanism currently supports harm prevention, not a stable positive
quality method.  MNIST replication and tiny-RAE adapter training are not
authorized by the gate.  The full report is
[`MECHANISM_TO_QUALITY_STUDY_ZH.md`](MECHANISM_TO_QUALITY_STUDY_ZH.md).

### Sampling-time probe

A three-seed numerical probe compares reduced Euler grids with the same-model
50-step endpoint.  Contrary to the initial intuition, moving time points toward
the low-noise region is much worse than preserving the official shifted time
warp.  Although state updates are largest late, velocity change per unit time
is largest near `t > 0.97`, so small high-noise updates are not automatically
safe to skip.

Subsampling the official 50-point grid to 25 points gives about `2.02x`
measured speedup.  The directly deployable, officially recomputed
`num_steps=25` grid has mean endpoint latent relative RMS `0.0503` and
Inception cosine distance `0.0296`; the exact 50-grid subsample is only
marginally better at `0.0490` and `0.0285`.  Sixteen points increase these
proxies to about `0.112` and `0.074`.

The fixed 5k check is complete under the same two-process, equal-label,
fp32/no-TF32 sampling contract as the 50-step reference.  For seed 3407
baseline, reducing from 50 to 25 steps changes FID from `132.6809` to
`135.9916` (`+2.50%`, worse), KID from `0.121259` to `0.126041` (`+3.94%`,
worse), and IS from `8.4163` to `8.0732` (`-4.08%`).  An earlier four-process
run with a different noise stream gave the same direction and similar
magnitude.  The 25-step grid is therefore a real twofold quality--speed trade,
not a lossless acceleration.  One checkpoint and one sampling seed are enough
to reject the lossless claim, but not to estimate a precise Pareto curve.

### Transport-risk atlas and path-dependent reversal

A leakage-audited baseline-only atlas was evaluated on the completed
FashionMNIST and MNIST isospectral studies. It perfectly ranks the six
dataset/basis means, but its within-basis seed Spearman is `-0.027`; aggregate
Spearman, ROC-AUC, and fixed-threshold sign accuracy are `0.568`, `0.733`, and
`0.733`, so every prospective gate fails.

The failure is not endpoint sampling noise. A 4,096-image, five-rollout-seed
checkpoint resample confirms a stable reversal: MNIST seed 3 DCT/PCA ratios are
`1.753/2.261`, while seed 4 gives `0.672/0.789`. Exact hash-matched training
replays show that both structured bases look favorable during the first 100
steps; the final sign is set by later basin drift. A step-100 band-0 signal
found exploratorily on MNIST has Spearman `0.855`, but the same frozen test on
FashionMNIST gives only `0.067`. Final train-only rollout moments correlate
with FID damage (`0.802` across 20 structured-basis conditions), but only after
the full training path and with weak sign accuracy.

Therefore static gradient geometry, a final-checkpoint local directional
derivative, and a 10%-budget point probe are all rejected as general endpoint
predictors. The full report is
[`TRANSPORT_RISK_ATLAS_RESULTS_ZH.md`](TRANSPORT_RISK_ATLAS_RESULTS_ZH.md).

### Spectral theory audit and fixed-multiset order gap

A two-round literature audit and a new set of controlled toys now separate the
parts of the spectral result that existing theory explains from the remaining
gap.  The full Chinese report is
[`SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md`](SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md).

Three old ambiguities are resolved.  First, an exact linear least-squares toy
shows that isospectral output metrics can produce effective Hessian condition
numbers `10000` versus `2164.8`; the spectrum of `W` alone is insufficient
because training is controlled by `J^T W J`.  Second, DCT sign scrambling
preserves every coefficient power and eight-band energy to fp32 precision but
reduces MNIST/FashionMNIST classifier accuracy to about `0.16`; marginal band
power cannot certify semantics.  Third, a nested seed factorial localizes the
MNIST sign reversal to stochastic-interpolant time draws first, Gaussian bridge
noise second, and not ordinary minibatch order or initialization.

The most important new result fixes the entire seed-3 time multiset: all
`128000` float32 values, data, initialization, minibatches, Gaussian noise,
optimizer, and evaluation are identical, and only the 1000 training-step order
is permuted.  Across 12 permutations, baseline feature-FID ranges from `66.34`
to `219.25`, while the weighted/baseline ratio ranges from `0.736` to `2.112`;
four permutations help and eight hurt.  Source-stream hashes are identical on
all four GPUs, and tests verify exact sorted-multiset equality.

A 12-time-seed IID sweep also corrects the two-seed narrative.  On MNIST,
baseline and weighted mean FID are `104.57/104.61`; weighting mainly reduces
cross-seed variance in that particular setup.  FashionMNIST does not reproduce
this: means are `96.24/139.62`, with weighting harmful in `11/12` seeds and
increasing variance.  Per-batch stratified time sampling reduces some baseline
variance but does not stabilize the paired effect.  Therefore the current
method has no stable average benefit, and generic Monte Carlo variance
reduction is not a sufficient explanation.

Existing NTK, anisotropic-SGD-noise, random-reshuffling, phase, and flow-error
theories explain why these effects are possible.  They do not predict the sign
or magnitude of endpoint semantic quality for a held-out time ordering.  The
remaining research gap is a pathwise theory linking the time-conditioned
non-commuting gradient/Hessian cocycle under `W(t)` to rollout endpoint
quality.  The next justified low-cost diagnostic is a held-out-prediction test
of Hessian commutator and endpoint-adjoint statistics, not another gamma or RAE
training sweep.

## Research decision

### Latent trust and decoder alignment

The failed SPC line produced a reproducible mechanism result rather than a new
training method.  Across five small stage-2 seeds, matched-variance latent
directions cross over from variance-dominated response at low noise to
cross-layer-predictability-dominated response at high noise.  Fifty-step
teacher-path rollouts preserve this crossover.

Frozen decoding reveals a two-stage leverage handoff.  At low noise the more
predictable directions contract more in the ODE but are amplified more by the
decoder; at high noise the ODE leverage itself dominates.  All three matched
pairs and all five seeds preserve larger decoded L1/LPIPS effects for the more
predictable direction across the tested times.

A direct 24-block decoder atlas strengthens the population-level mechanism:
held-out encoder cross-layer predictability alone explains `0.819--0.860` of
the variance in decoder hidden sensitivity, versus `0.156--0.182` for latent
direction variance.  This relation holds within each absolute, fractional,
and PCA basis family.  It is not a simple decoder-input scaling effect: the
exact linear embed gain spans only `1.47x`, while the first recorded decoder
hidden response spans about `11.4x`.

The method gate nevertheless fails.  A static predictability metric reaches
only `0.065` median per-time Spearman with full decoder LPL and `0.021` median
gradient cosine.  Even an oracle static metric fitted from the decoder atlas
reaches only `0.036/0.029`.  Dynamic decoder prefixes improve as depth grows,
but prefix 3 already uses most of the decoder and reaches only `0.728/0.493`,
with high-noise gradient cosine `0.229`.

Therefore no static predictability-weighted loss, SPC schedule, or truncated
decoder training is authorized.  The supported finding is the gap between a
stable population-level semantic axis hierarchy and state-dependent,
sample/token-specific perceptual correction.  The Chinese report is
[`RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md`](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md).

The spectral failure mechanism is sufficiently localized for the current
scope; more fixed-`gamma`, basis, time-splice, residual-scale, seed, or 50k-FID
diagnosis is not the next useful spend.  The protected-residual gate prevented
harm but did not beat the absolute baseline, and paired off-path MSE was
actively harmful.  No paired tiny-RAE screen is authorized.

The final low-cost candidate, train-only rollout checkpoint selection, has now
been prospectively rejected on unseen seeds 5--9.  In MNIST, selected DCT/PCA
remain `1.238x/1.346x` worse than the fairly selected baseline, with zero of
five seed wins.  FashionMNIST gives `1.249x/1.422x`, with one and zero seed
wins.  The proxy itself improves by about 46% across all conditions, but its
alignment with FID is weak (`0.265` overall Spearman and `0.029` on
FashionMNIST), and every selected path has a larger class-entropy gap.

This closes the current spectral-preconditioning method line: do not retune
gamma, bases, band moments, or checkpoint schedules, and do not move it to RAE.
The supported contribution is the negative mechanism result that isospectral
basis orientation and path-dependent semantic transport, rather than spectrum
alone, govern endpoint behavior.  The prospective report is
[`ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md`](ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md).

The group-structure line remains separate.  If it is resumed, only held-out
generator-power and D4-relation tests can justify a group claim; decoder
reconstruction or independently fitted per-transform maps cannot.  Do not
continue decoder inverse-adapter reconstruction training as the primary method.

The speed line is also closed for now: 25 steps is available when a roughly
`2x` speedup is worth a measured 5k quality loss; 16-step and late-dense
schedules are not justified.
