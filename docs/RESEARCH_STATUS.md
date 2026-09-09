# Research Status

> **Current archive checkpoint (2026-09-09): all JiT follow-up arms complete.**
> See [whole-workspace archive](WORKSPACE_ARCHIVE_20260909_ZH.md) for current
> results, portable metadata and raw-data locations. JiT nine alternative PFR
> arms all lose to selected IG55.480326; best projected rho=.05 gives56.855438.
> No training/sampling processes were active at status inspection. Lifting's
> best exploratory1K55.214220 has ~3.59x batch GPU cost and no independent5K.
> The two initial brainstorm ideas are withdrawn; no mature new method yet.
> Existing drafts are archived as historical files, not resumed paper writing.
> **Entries below are chronological research history, not current queue state.**

> **Brainstorm audit: do not relaunch old future-gap or norm-clipping as new ideas.**
> User requested theory/idea discussion only; current nine PFR arms continue.
> Earlier suggested future-gap-change overlaps documented SiT decomposition:
> ordinary64.85216, gap-change66.86920, RMS-matched66.62607; combined weak
> evolution63.27067. Query differs from some canonical PFR settings, so this
> is a direct warning against novelty/generalization, not every variant ruled out.
> Common time/OU-response controls do NOT imply arbitrary common change is
> beneficial; RAE common response was already found without quality benefit.
> APG already contains norm rescaling/projection; simple clipping is a control,
> not an established novel method. Existing endpoint protocol also explicitly
> warns that depth readouts are not consecutive iterates permitting Aitken.

> **Active follow-up: JiT alternative PFR query/strength/horizon controls.**
> User requests alternatives after canonical PFR FID110.23. Fixed selected
> IG early.3/late0, depth4, old noise/labels; no baseline reruns.
> Nine arms: projected h1/32 rho .05/.1/.25/.5; time-only rho .1/.25/1;
> projected/time-only h1/128 rho1. Four GPUs per arm, sequential1K.
> Preflight reproduced first4 canonical PFR and selected IG pixels exactly;
> rho0 equals IG. No time/sign/conversion mismatch identified in source review.
> On four fixed IG paths, projected correction RMS / IG velocity RMS averages
> .157 at t0, .114 at .1, .093 at .4; time-only comparable. No claim that
> correction-size diagnostic explains quality failure. Queue launched at
> jit_pfr_variants_20260909; script run_jit_pfr_variants.py.

> **JiT overnight study complete: depth4 IG helps, fixed PFR fails badly.**
> 86 IG configurations including reused Full, one PFR, six lifting settings;
> total93 result records,92 newly sampled paired1K candidates. Best IG early.30/
> late0 FID55.48033 versus Full63.67457 and officialCFG38.77898.
> Canonical projected PFR h1/32 at selected IG:110.23098, clear negative.
> Lifting alpha.30 m1=55.22358, m2=55.21422; best improvement only.48%,
> batch GPU time962.02/1477.87 versus IG411.85 (~2.34x/3.59x).
> Neither method passed predeclared1% improvement trigger; no independent5K
> was launched. All are exploratory1K, not evidence of reliable lifting gain.
> Queue phasecomplete, GPUs idle. Fixed PFR failure also in pixel JiT weakens
> a universal explanation based solely on RAEv2 encoder-layer averaging;
> does not prove the cause or rule out all PFR settings/heads.

> **JiT training50K complete; corrected overnight fine sweep running.**
> User clarified: finish lifting on JiT, NOT add depth8 sampling. Depth4 only.
> New run_jit_fine_study.py preserves trained heads and runs IG early/late joint
> grid (0:.15:1.5 x 0:.1:.6), conditional boundary extensions and local .05
> refinement, then selected-config PFR, then lifting multipliers1/.8/1.2 x m1/m2.
> If a method beats selected IG by>=1% exploratory FID, freeze settings and
> run independent paired5K seed202609932; threshold is not statistical significance.
> All candidates sequential, four GPUs per candidate, baseline reused. Actual
> status sampling first nonzero IG e.15/l0, trainingcomplete confirmed.
> New root jit_fine_sweep_20260909; older jit_ordered controller terminated,
> adopted training was never interrupted. Latest instructions restore JiT lifting
> only; smallSiT/RAE lifting stays shelved.
> Theory correction: old strong/weak time-defect controls already show Strong
> variants improve too. No evidence that weak-only use is theoretically necessary.

> **Latest user direction: shelve lifting; await JiT IG/PFR results.**
> All new lifting, including pending JiT lifting and RAE tail lifting, paused.
> Replaced only JiT orchestration controller, preserved active training PID1719837;
> new controller adopts it, then tunes depth4 IG and compares PFR, no lifting.
> Verified training step44300/50000 before controller replacement.
> Do not interpret PFR as training weak to catch strong or removing guidance.
> Canonical PFR queries weak at a projected future point; original weak anchor
> remains in G=W(p)+(1+gamma)*(S(p)-W(q)). Quality benefit is empirical.

> **Historical-config SiT lifting complete; JiT resumed and training.**
> Same historical bank IG64.85130/PFR61.85921. Lifting FID1K (m1/m2):
> alpha .6/.7=65.28696/65.27038; .7/.8=66.08276/66.13323;
> .8/.9=66.91876/67.35254. All six negative versus historical IG/PFR.
> No consistent gain from second half-strength write; greater early alpha
> worsens quality in this grid. No additional SiT runs queued.
> JiT controller automatically resumed checkpoint24K; verified step38100/50000,
> four GPUs active. Latest heldout step35K d4=.094728/d8=.094788,
> frozen Full=.067599. Continue training then depth4 IG search -> PFR -> lifting.

> **Authorized automatic continuation after SiT: JiT ordered method study.**
> Controller run_jit_ordered_study.py is waiting for the six-arm SiT study.
> Resume JiT saved24K->50K, retain original two-head training but sample only
> depth4. Tune early IG at late0, then late IG, then revisit early; freeze
> selected observed configuration, compare PFR next, lifting m1/m2 last.
> Full/official CFG banks reused. New single-depth sampler preserves paired
> JiT input bank, verifies shared Full/prefix and baseline4 pixel parity before
> quality sampling. GPU checks deferred until training completes; Python
> compilation passed. No current claim of new JiT quality or parity checks.
> Protocol JIT_ORDERED_METHOD_STUDY_20260909_ZH.md. Queue live status:
> /home/zhoushunyu/data/eqvae/experiments/jit_ordered_20260909/status.json.
> Latest authorization supersedes earlier 'do not auto-resume JiT'.

> **User correction: validate lifting at historical strong IG/PFR configuration.**
> Located historical depth4 schedule gamma .6/.7/0 at data-time .25/.5,
> Dopri5 rtol1e-3 atol1e-6, seed0 per-batch legacy paired bank B8.
> Exact historical IG64.851298/PFR61.859207 reused. Earlier fixed-.35
> bank and Euler sweep did NOT establish improvement over historical PFR.
> Historical-bank lifting anchor launched on four GPUs, .6/.7/0 strengths;
> global input hashes and Strong/Weak interface parity verified.
> Main Strong Dopri5 restarts at finite-write boundaries; auxiliary Heun
> preserves lifting's existing fine grid. Solver restart difference disclosed.
> User then requested richer early strengths and two iteration counts:
> .6/.7, .7/.8, .8/.9 each m1/m2, late0; m2 uses alpha/2 per write.
> Five new arms queued after anchor, no duplicate anchor/IG/PFR sampling.
> Roots small_sit_best_config_lifting_20260909 and small_sit_best_lifting_sweep_20260909.
> JiT remains paused, XL deferred. No paper.

> **Two-stage small SiT IG: all eight paired 1K arms complete.**
> FID cutoff .2 / .5: gamma .35=69.09210/72.32272;
> .50=70.48061/68.65158; .65=75.66677/66.89813;
> .78=81.11084/66.45396. Existing full IG71.16979, fixed PFR68.35781 reused.
> Best observed .78 cut.5 improves FID by1.90385 versus existing fixed PFR,
> at100 Full/image and78.10 batch GPU seconds. Exploratory selection on same1K,
> not independent validation or comparison against equivalently tuned PFR.
> At matched gamma.35 cut.2 lifting68.73780 versus ordinary69.09210, a small
> .35430 gain at219.56 versus78.09 batch GPU seconds. Strong evidence here
> for interaction of guidance strength and window, not a lifting breakthrough.
> GPUs idle; JiT remains paused at24000. No additional experiments launched.

> **Active: small SiT ordinary two-stage IG controls and early-strength sweep.**
> User requested matching IG tail controls, then noted early gamma need not .35.
> First gamma .35 at noise cutoffs .2/.5, then gamma .5/.65/.78 each at both
> cutoffs. All 1K paired with existing bank; only new methods sampled,
> four GPUs cooperate per arm, arms sequential. No PFR/full-IG reruns.
> Samplers/queues: sample/run_small_sit_two_stage_ig[_strength]_*.
> Strength queue waits for .35 queue completion. JiT paused, XL deferred.

> **Small SiT lifting four-arm experiment complete.**
> Paired FID1K: ordinary reused71.16979; PFR reused68.35781;
> lifting a035 constant71.07050, a078 constant93.00245,
> a035 noise cutoff .2=68.73780, cutoff .5=71.13327.
> All four 1000-image banks and evaluations complete; queue exited.
> Cutoff .2 improves over continuous lifting by2.33271 FID and reduces
> measured batch GPU seconds249.26->219.56, but still trails existing PFR
> and costs more. No matched native-IG cutoff control yet, so the gain
> cannot be attributed specifically to lifting. No 5K confirmation.
> Alpha .78 degrades here; RAEv2 coefficient plateau does not transfer.
> GPUs idle at verification; JiT remains paused at saved step24000.

> **User hypothesis to revisit after JiT: RAEv2 multi-layer latent averaging.**
> Source check: dinov3mls k7 averages seven encoder-layer patch features,
> then adds the final selected layer's spatial mean at each patch.
> This defines the encoded training representation; stage1.RAE.decode only
> denormalizes and reshapes generated latents before decoder, with no new
> multi-layer averaging at generation time. Possible representation effects
> remain hypotheses, not demonstrated cancellation of PFR updates.
> User wants JiT evidence before a conclusion; small SiT remains current priority.

> **Current execution priority: small SiT lifting; JiT paused by user.**
> JiT controller/training terminated after preserving last.pt at step24000;
> optimizer and all four RNG states verified. Do not auto-resume JiT.
> New small-SiT four-card queue launched: a035_constant, a078_constant,
> a035_cut02, a035_cut05, sequential candidates. Existing IG and PFR
> quality banks are reused; user reiterated PFR must NOT be rerun.
> Only eight old IG images reproduced for pixel parity, passed. Constant-field
> sign/alpha-zero/grid/cutoff checks passed; new first lifting batch saved.
> Live status: /home/zhoushunyu/data/eqvae/experiments/small_sit_lifting_20260909/status.json.
> RAEv2 tail controls remain pending; XL deferred. No manuscript.

> **Latest correction: lifting transfer starts on small SiT; XL deferred.**
> Use ImageNet-100 SiT-S/2 v800K + depth4 v readout, existing paired
> seed202609417/B8 ordinary and PFR baselines. Small-model data time runs
> 0 to 1: noise cutoffs .2/.5 mean native t>=.8/.5. No XL launch.
> JiT verified live at step21500/50000 (~43%); four GPUs active.
> At step20K heldout EMA readout errors d4=.11316/d8=.11634 versus
> frozen Full=.06760. Prediction errors do not establish guidance quality.
> JiT baseline FID1K: Full Euler100=63.67457, CFG Heun50=38.77898.
> IG/PFR/OU quality evaluation remains pending readout completion.

> **Additional authorized follow-up: transfer RAEv2 lifting to official SiT-XL.**
> After JiT and the queued RAEv2 tail controls: constant alpha .35/.78,
> plus alpha .35 with noise-time cutoff .2/.5. Alpha .35 matches existing
> SiT IG1.35 to first order. Reuse paired ordinary/PFR baselines; four GPUs
> per new candidate. Pending, not launched. See SIT_CAPACITY_LIFTING_TRANSFER_20260909_ZH.md.

> **Pending after JiT: user authorized low-noise guidance-off controls.**
> RAEv2 lifting alpha .65/.78, each with t<=.2 or t<=.5 switched to
> Strong-only Euler. Reuse completed constant samples and native baseline.
> Four GPUs cooperate per candidate, candidates sequential. Not launched.
> Wait for JiT training AND method sampling/evaluation; readouts_ready alone
> is not completion. See IG_CAPACITY_LIFTING_LOW_NOISE_CONTROLS_20260909_ZH.md.

> **Latest active direction: user requests true JiT reproduction of SiT IG/PFR/OU.**
> Found official JiT outside worktree; recovered complete B/16 weights from
> interrupted ZIP with CRC verification. Frozen-backbone depth4/8 clean
> readout training and first-ever JiT paired baselines launched in queue.
> Preparation session88772; baseline workerPIDs1540813/1540814/1540815/1540816.
> FP32/BF16 full-head parity passed; source131320320/readouts3545088 params. See
> JIT_INTERNAL_GUIDANCE_TRANSFER_20260908_ZH.md. No JiT result yet.
> Lifting five settings all complete: .5/.65/.78/.85/.9 =
> 38.737965/38.452305/38.468476/38.456582/38.497212 vs38.264239.
> No improvement; old228 a078 images reused unchanged. No extra sweep.
> FK remains paused500 samples. Manuscript stays paused.

> **Latest: user requests append0.78 after .5/.65/.85/.9 constant grid.**
> Reuse old a078228 saved images with exact frozen sampler/protocol.
> Dedicated append controller waits for current queue completion, then four
> GPUs finish only a078 and evaluate. No fade/1.25/FK restart.
> Current .5/.65/.85 results38.737965/38.452305/38.456582 all above
> reused native38.264239; .9 still sampling at last verified state.

> **Latest user coefficient grid: .5/.65/.85/.9, constant only.**
> Old lifting queue terminated; a078 retained228 images. No1.25/fade run.
> New separate sampler/protocol/root preserve old frozen source hashes.
> Four constant settings run sequentially, four GPUs cooperate on each1K.
> FK remains paused at500 images. No baseline reruns; no manuscript.
> See IG_CAPACITY_LIFTING_CONSTANT_20260908_ZH.md.

> **Latest user steering: pause FK; test finite capacity lifting.**
> FK saved500 images; four workers and watcher stopped to release GPUs.
> Exact-request atomic batches retained. Do not restart FK automatically.
> Implemented partial L=(PhiW)^-1 PhiS then Strong Euler advancement,
> no additional native IG. Four predeclared alpha/schedule settings, run
> sequentially with four GPUs per setting. Baseline reused, no manuscript.
> See IG_CAPACITY_LIFTING_20260908_ZH.md. Smoke passed,11.232s/B4.
> Four-setting queue launched; process status/log in
> experiments/results/terminal_defect_20260908/capacity_lifting/.

> **Latest user-authorized experiment: direct stochastic FK path correction.**
> Read new attachment a11a1076 in full. Same ideal math as existing semigroup
> derivation; replace failed fitted value approximation with direct antithetic
> path log-moment input gradients. Protocol IG_FK_PATH_20260908_ZH.md.
> Smoke passed: 4 images saved, 98.031 s, peak9.216 GB. Four-GPU 1K launched;
> sessions87390/33267/4613/58259, ranks0/1/2/3, baseline reused.
> Watcher68956 automatically merges/evaluates after all four summaries.
> Low-noise two-particle weights concentrate; log this limitation, no gain tuning.
> No distillation; no manuscript. Older requirement to invent a content/quality
> decomposition is superseded by the user-provided terminal-preference method.

> **Method reset after explicit user correction — latest:**
> No distillation or newly guessed algebra/attention variant launched.
> Re-read AG/FSG and inspected fixed first8 matched images: large temporal
> interventions often change framing/content in these examples, not merely
> local quality. This is qualitative scope only, not causal FID evidence.
> See RESEARCH_METHOD_RESET_AFTER_USER_CORRECTION_20260908_ZH.md.
> Need a concrete quality-error/content distinction before a new method;
> no claim of a mature idea yet. Goal remains active, manuscript paused.

> **User rejected response distillation — latest steering:**
> Stop SiT PFR response distillation. Only scripts were prepared; no
> training or student sampling was launched. Efficiency preservation on
> SiT does not solve the unresolved RAEv2 method-quality problem.
> Do not launch the fit/merge/sampling plan automatically. Prioritize a
> substantive effective method; no manuscript and no baseline reruns.

> **Three-time PFR completed, negative — 2026-09-08:**
> FID47.583198 vs reused native38.264239 (24.354% worse); IS68.113455.
> All four workers and evaluator52649 exited0; no active job.
>100Full+178prefix/image,1173.550 batch GPU seconds. No expansion,
> differencing-order, norm, h or strength rescue. Stop this series of
> temporal-difference algebra variants. See
> RAEV2_PFR_TEMPORAL_CURVATURE_RESULTS_20260908_ZH.md. Goal unmet; no paper.

> **Three-time temporal PFR candidate launched — 2026-09-08:**
> R=W_t-2W_(t-h)+W_(t-2h), h=min(1/32,(t-.5)/2), t>.5,
> nativeIG+1.78R. Standard finite-difference annihilation, no novelty or
> quality guarantee; distinct from older two-time OU degree2 projection.
> Four-card single-method1K,100Full+178prefix/image; baseline reused.
> Sessions97749/95136/13415/27410 launched; check actual handles.
> Frozen RAEV2_PFR_TEMPORAL_CURVATURE_20260908_ZH.md. No paper.

> **Teacher-reference PFR1K completed — 2026-09-08:**
> FID38.199289 vs reused native38.264239, only0.1697% exploratory gain.
> Four workers and evaluator17242 exited0; no jobs remain. No matched
> same-dose raw comparison, so teacher-target attribution is unproven.
>100Full+99prefix+198small-head calls/image,1016.511 batch GPU seconds.
> No5K/parameter search under frozen rule. See
> RAEV2_PFR_TEACHER_REFERENCE_RESULTS_20260908_ZH.md. Goal unmet; no paper.

> **PFR teacher-reference quality run — 2026-09-08:**
> Reuse existing ten-noise affine teacher reference; no new training.
> Native Full/Base IG unchanged; only PFR difference uses teacher head.
> Historical mild rho.05,h1/32,native interval; four-card paired1K only,
>100Full+99prefix and198small-head calls/image. No baseline regeneration.
> This differs from canonical rho1 in both dose and reference, so no
> single-factor attribution against that bank. Sampling sessions
>39555/26001/24872/7039 launched; inspect real handles for current state.
> RAEV2_PFR_TEACHER_REFERENCE_20260908_ZH.md is frozen. No paper.

> **RAE differential PFR completed, negative — 2026-09-08:**
> New four-card1K FID58.661318 vs native38.264239 and prior raw52.942740.
> 189Full/image;1500.084 batch GPU seconds. Four workers and evaluator96266
> exited0; no active job in this experiment. No5K or strength/window rescue.
> Direct strong temporal-response subtraction did not repair transfer.
> See RAEV2_PFR_DIFFERENTIAL_RESULTS_20260908_ZH.md. Goal unmet, paper paused.

> **Independent PFR route resumed — 2026-09-08:**
> Pause AG/FSG carrier variations after the negative history-separated1K.
> New RAE differential PFR samples only the new method on all four GPUs:
> nativeIG+1.78*((W-Wfuture)-(S-Sfuture)), same-state temporal queries,
> h1/32,rho1,t>.5,189Full/image. This operator has negative SiT history,
> so no generic novelty claim. RAE-specific quality is now being tested.
> Sessions56614/23822/89934/60499 launched; inspect handles for liveness.
> Frozen RAEV2_PFR_DIFFERENTIAL_20260908_ZH.md. Reuse existing baselines,
> no paper, no FSG/PFR hybrid. Earlier route states are historical.

> **History-separated IG completed — 2026-09-08:**
> Four-card1K FID38.619525 vs reused native38.264239 (0.9285% worse),
> 198 Full calls/image,1574.962 batch GPU seconds. All four workers and
> evaluator exited0; no jobs from this method remain. No5K or shadow/reset
> tuning. See IG_HISTORY_SEPARATED_RESULTS_20260908_ZH.md. The earlier
> running note below is historical. Goal remains unmet; no manuscript.

> **Latest user constraints and method run — 2026-09-08:**
> No manuscript drafting: focus on method and idea. Only sample the NEW method,
> with all four GPUs sharding the same sample set; reuse existing baseline data
> and metrics. Retain ongoing guidance. Method quality experiments take priority
> over mechanism-only admission gates in older notes below.
> Global/coarse/pulled-back query anchors completed paired 1K at FID
> 38.145579 / 38.146866 / 38.250534, native IG38.264239. None establishes a
> substantive contribution; no more small anchor variants.
> New history-separated method is running: current Full prediction plus .78
> times the Full-Base contrast evaluated at a Full-only shadow state, both paths
> starting from matched noise. Unlike direct cross-state strong-minus-weak,
> the contrast is computed at one state. Main trajectory retains IG throughout
> the original interval. 198 Full calls/image, four-card1K, no baseline rerun.
> Sessions61615/37528/41199/27182 were launched; do not infer current liveness
> from this note. Inspect actual handles and result files. Frozen protocol:
> IG_HISTORY_SEPARATED_20260908_ZH.md. Goal remains unmet; PFR separate/paused.

> **User clarified novelty and two forward routes — 2026-09-08:**
> Writing class information into noise is known/FSG-related, not our novelty.
> Routes are a new CFG guidance method or transferring this taste to AG/IG.
> Prioritize AG/IG research while completing the already-started CFG handoff
>1K as a reference, not a contribution. Its pilot/native/Full/BF16 smoke
> gates passed; driver93827 is running quality at entry time.
> Do not mechanically equate strong/weak future agreement with quality;
> the repository already has negative AG-FSG and anchored-inverse evidence.
> Investigate whether useful guided-quality correction can be written into
> state and retained by the strong sampler after withdrawing weak-model
> consultation. Weak-only continuation is a possible diagnostic, not a
> redefinition of the objective as distillation/speed alone. Simply stopping
> guidance early is also not a novelty claim. PFR remains separate/paused.

> **Condition handoff demonstrated in a64-seed RAE pilot — 2026-09-08:**
> Two initial t1 true-CFG/null Euler calibrations (H.125,K2,gamma2), then
>100 entirely null-labelled steps: target top1 48/64 versus52/64 always
> conditional and0/64 raw null or null-only roundtrip. Wrong-class write
> yields49/64 donor-class and0/64 original-class top1. Independent ConvNeXt
> evaluator, no classifier used during sampling. Same-written-state
> conditional/null endpoint MSE .265926 vs uncalibrated1.395674 (ratio.190536).
> This supports the user's information-relocation intuition; prior local
> MSE/cache gates did not answer it. Not full FSG reproduction or new method.
> One-write internal memory is much weaker:9/64 target; exchanged memory
>5/64 donor,0/64 original. Zero-memory8 endpoints/pixels bitwise null parity.
> Drivers14632/9264/72611 and audits17753/71683/85374 all exited0; no live jobs.
>520 decoded outputs across controls,941.584 seconds sampling+save (~.26155
> GPU hours), loading/evaluation extra. See FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md.
> No FID or quality/novelty claim yet. PFR and paper writing remain paused.

> **User correction adopted: test condition handoff, not only local risk — 2026-09-08:**
> FSG relocates external condition into high-noise state; lack of new
> external information is not an objection to useful relocation. Prior
> token/MSE results do not establish or refute condition handoff.
> Running64-seed end-to-end withdrawal experiment: pure initial noise,
> conditional/null continuations, two-round FSG-style latent calibration
> followed by null or class, null-only roundtrip and wrong-class write.
> Separately testing a layer14 class-contrast memory written once at t1,
> then only null-labelled calls; zero-memory continuation parity required.
> Sessions14632/9264 live when this entry written. Protocol:
> FSG_CONDITION_HANDOFF_PROTOCOL_20260908_ZH.md. No new-method claim.

> **Carrier usefulness checked against real clean targets — 2026-09-08:**
> Same32 existing validation-bank images/noise and4 times as preceding
> foresight probe. Class-only midpoint14 swaps recover .654/.651/.675/.720
> of observed null-to-conditional MSE reduction; image-only also helps.
> Full conditional remains best on average at every time. These are risk
> ratios on fixed32 images, not information fractions or quality gains.
> Driver90562/CPU audit24256 exited0;256 split parity checks and128 saved
> vector Grams passed; input and native residual reproduce prior probe.
>44.933 seconds compute+save, loading/bank verification/audit extra.
> No training/FID or live jobs. See FSG_CARRIER_REAL_RISK_20260908_ZH.md.
> Stop expanding basic token-carrier diagnostics: useful interface exists,
> but no validated update rule or novel algorithm yet. Paper/PFR paused.

> **Foresight class-token write screened on real noisy data — 2026-09-08:**
> Fixed32 existing validation-bank images,4 times1/.9/.75/.55,H=.125,
> midpoint14. Advance z with true conditional Full, replace current class
> contrast by future contrast while preserving current image/time tokens.
> Native split and zero-change parity all128 bitwise. Real-clean MSE rises
> 4.8921%/.02404%/.00748%/.00303%; no positive evidence for this candidate.
> Driver8027 and CPU residual-vector audit68077 exited0;31.603 seconds
> compute+save, loading/bank verification/audit extra. No FID or training.
> Stop this candidate without horizon/gain/layer rescue; not a proof of
> endpoint deterioration. See FSG_CLASS_TOKEN_FORESIGHT_20260908_ZH.md.
> Generic attention-space fixed-point/extrapolation already has direct
> precedent (GAG2603.02531, primary §§2.3–3.2 read). Goal incomplete.

> **One-step carrier persistence tested — 2026-09-08:**
> Same8 cached RAE trajectories, fixed14th layer, steps1/47/73/87,
> preceding-step donor only. Conditional-minus-null class-token memory
> reproduces fresh class-token effects well (cos .99845–.99970), but
> that effect omits much of the full conditional response. Full-gap relative
> squared error .325/.721/.477/.432 versus cached velocity-gap
> .00200/.02285/.00387/.00732. Different-target proxy scores must not be
> compared as superiority. No trained memory or quality run warranted yet.
> Driver46154 and CPU audit15068 exited0;128 split-native outputs bitwise,
> saved32 vector Grams audited.22.619 seconds compute+save, loading extra.
> Primary RIN abstract and MetaState introduction establish prior persistent
> latent memory; generic recurrence is not novelty. See
> FSG_CLASS_TOKEN_PERSISTENCE_20260908_ZH.md. No live jobs; goal incomplete.

> **Independent carrier diagnostics completed — 2026-09-08:**
> RAE midpoint14,8 cached trajectories ×4 times, FP32 noTF32, true
> class/null Full only. Class-token swaps carry donor-directed effects;
> raw tokens from other trajectory states produce large shifts. Donor
> conditional-minus-null token transport reduces shifts but fails to recover
> recipient effects at noninitial times (cos .167/.233/.228).
> All192 native split outputs bitwise; both GPU runs exited0 (69805/28587),
> CPU vector audits8131/19836 passed, shared controls reproduce exactly.
> Compute+save59.405 seconds, loading/hash/audit extra. No live jobs or
> quality claim. See FSG_CLASS_TOKEN_CARRIER_RESULTS_20260908_ZH.md.
> Same-trajectory persistence remains untested. PFR and paper paused.

> **Current route decision — 2026-09-08, supersedes earlier route statuses below:**
> Stage-pause PFR extensions and independently reopen the NeurIPS25 FSG
> information-carrier question, as permitted by the user's two-route instruction.
> This is our evidence-based prioritization, not a user instruction to reject PFR.
> Preserve official SiT raw PFR independent5K positive evidence and all RAE
> negative evidence. No PFR/FSG hybrid, no paper writing, no new GPU job in
> this route-decision step. Reread FSG §§3.1–3.2/Algorithm1 and inspected RAE
> sequence/mask implementation. Internal class-token state is a candidate
> to diagnose, not a validated method or novelty claim. See
> [route decision and first diagnostic](FSG_INDEPENDENT_CARRIER_ROUTE_20260908_ZH.md).

Earlier entries below retain the decisions applicable when those experiments ran.

> **OU finite-scale interpretation corrected — 2026-09-08:**
> Exact Gaussian population paths can have collinear finite-h degree1
> defects with long/short amplitude ratios1.541/1.266 (V64, data times
> .02/.05), despite zero neural error. Independent relative-score formula
> and existing OU implementation agree<1e-11 in10 fixed cases;48729 exited0,
> .008729 CPU seconds. Corrected §16 of PFR_OU_PROBABILITY_WAVELET_THEORY:
> finite ratios alone do not identify neural semigroup inconsistency.
> No new quality claim or sampler change. See OU_FINITE_SCALE_RATIO_CORRECTION_20260908_ZH.md.

> **Official SiT-XL raw-versus-OU PFR comparison completed — 2026-09-08:**
> Fill a positive-evidence gap: small SiT v/x showed OU extra gain, but
> official XL had only raw PFR independent5K. Fixed two1K arms with
> identical125 Full+50 prefix/image, inherited high-noise OU direction
> projection and raw RMS, no new scale/window. Drivers30991/86395 exited0;
> independent audit5359 passed, all1000 raw pixels reproduce prior bank.
> Raw FID40.565427, OU40.554934: only.010492 lower, mean improvement
> .107045 nearly offset by covariance worsening.096553. No reliable
> additional gain established; no OU5K/parameter rescue. CPU analytic
> Gaussian interface58724 passed (defect<=1.78e-15), not quality evidence.
> Sampling.684984 GPU hours plus loading/checks/eval. Same query budget
> uses padding for raw, so this is not an efficiency gain over fastest raw.
> See OFFICIAL_SIT_PFR_OU_PROTOCOL_20260908_ZH.md. No live jobs from this
> comparison; continue PFR only, fixed point and paper paused.

> **PFR adjacent temporal-reference targets screened — 2026-09-08:**
> Limited primary reading of TAG2510.11057v1 and DiFA2607.17972v1;
> no full reproduction claim. Exact same-bridge Gaussian example shows
> oracle time-posterior confidence increases while an already correct
> marginal is distorted: eta.05 variance1.25→1.223280, KL0→.000125315.
> Analytic negative variance derivative, positive global Jacobian, finite
> difference and quadrature checks passed98937 (.028834 CPU seconds).
> No GPU/quality jobs. Do not adopt time-classification confidence or ideal
> independent-history variance as a sufficient PFR error certificate; no
> auxiliary classifier/history-reference training launched. See
> PFR_TEMPORAL_REFERENCE_READING_20260908_ZH.md. PFR only, core unresolved.

> **PFR working-point intervention completed, negative — 2026-09-08:**
> RAE Full versus Full+1.78*(W_t-W_r), keeping canonical h1/32, t>.5,
> rho1 and absolute PFR dose; only ordinary IG removed. Fixed1K native
> BF16 protocol, same seed413/batch4 as existing IG/raw banks. Drivers
> 45987/75354 exited0; independent audit78761 passed. Full FID38.874134,
> Full+PFR54.137366 (mean.269684→5.283816, covariance38.604474→48.853568).
> Native IG is therefore not necessary for degradation in this fixed
> protocol; no inference about all strengths or historical mild5K.
> 100 Full versus100 Full+89 prefix/image; .490316 GPU hours sampling,
> loading/check/evaluation extra. No live jobs, no5K expansion/scale scan. See
> PFR_WORKING_POINT_PROTOCOL_20260908_ZH.md. PFR route only; fixed point
> and paper paused.

> **PFR program time-route attribution checked — 2026-09-08:**
> Shapley finite switches of feature/readout time (plus RAE clean-to-velocity
> divisor) on unchanged32 native trajectories/model. Native corners bitwise,
> original raw energies/endpoints reproduced; sessions75360/95271 and CPU
> audit16857 exited0. RAE direct-readout signed raw-dot shares only
> .003713/-.000934/.005960/.000252, energy ratios .0045/.0101/.0258/.1064;
> feature route dominates raw-direction attribution in both backbones.
> This lowers priority of a direct-readout-bypass explanation, without
> establishing small terminal effects or a quality causal refutation.
> .024146 GPU hours compute, loading/audit extra. No quality intervention;
> see PFR_TIME_ROUTES_PROTOCOL_20260908_ZH.md. Continue PFR alone, no live
> jobs from this probe; fixed point and paper remain paused.

> **Latest user clarification: keep PFR and fixed point separate.**
> Current route is PFR only: finish the direct temporal-component comparison
> to investigate SiT success versus RAE failure. Suspend internal-feature
> information relocation speculation and the inverse-noise branch. Do not
> combine these with fixed point; consider that route independently only
> after an evidence-based decision to leave PFR. Paper remains paused.

> **PFR direct temporal condition-component pilots completed — 2026-09-08:**
> Keep native IG; split weak raw time response into unconditional and
> conditional-interaction components, preserving each component actual size.
> Two fixed1K arms/model on official SiT and RAE; no OU projection or norm
> restoration. Equal within-pair cost100 Full+150/267 prefix per image.
> All four drivers exited0; independent audit42541 passed. SiT conditional
> interaction/unconditional FID42.166757/40.990556 versus native42.212951
> and raw40.565427. RAE41.385438/49.223855 versus native38.264239:
> neither component repairs RAE; both increase mean FID substantially.
> Sampling1.484196 GPU hours, loading/checks/eval extra. Endpoint mean
> nonadditivity audited29229 using existing features; no additional sampling.
> See PFR_CONDITION_COMPONENT_1K_RESULTS_20260908_ZH.md. No live jobs from
> this experiment, no5K or novelty claim; continue PFR only, paper paused.

> **Full inverse roundtrip checked, branch paused — 2026-09-08:**
> Four shards exited0; independent CPU audit44691 recomputed800 saved
> inverse states and8 endpoints.7/8 endpoints recover closely, but all
> recovered noises have relative MSE .00150–.00195; label0 endpoint
> relative MSE .007859 and middle-step residual issues.639.719 seconds
> total compute (.17770 GPU hours),28000 B1 Full calls. No inverse-data
> bank or auxiliary flow training. See RAEV2_INVERSE_ROUNDTRIP_PROTOCOL_20260908_ZH.md.

> **Discrete inverse prerequisite for a target-anchored noise prior — 2026-09-08:**
> Read the directly adjacent ICLR2026 Inverse Noise Correction construction
> (2510.02692v3 §4/5.3/D.8/G); inverse-prior learning itself is not new.
> Known-preimage RAE native Euler tests at0/47/89/99 reproduce cached
> steps bitwise. After32 Picard updates, first3 recover near FP precision;
> last-step median residual ratio3.05e-7 but preimage error ratio.001139.
> Residual alone cannot certify inverse-noise accuracy. Session70464 exited0,
> 1216 B1 Full calls,28.477 seconds compute; source/count/stat checks passed.
> See RAEV2_DISCRETE_INVERSE_PROTOCOL_20260908_ZH.md. No inverse-data bank,
> training or quality claim. Full-trajectory roundtrip remains unchecked.

> **PFR endpoint displacement random-half check — 2026-09-08:**
> Existing5K paired feature shifts reproduce across fixed random halves:
> mean-direction cosine .9424 SiT/.8721 RAE; RAE held-out positive
> projection fractions .5928/.6032, top1% absolute projection shares
> .0713/.0745. Feature hashes and available paired sampling records
> checked, session86679 exited0; no GPU sampling or unseen-class claim.
> See PFR_MEAN_SHIFT_CROSSFIT_20260908_ZH.md. A separate pushforward
> argument rules out marginal-preserving inter-sample noise coupling as
> a repair of population mean bias; it can change finite-bank statistics.
> Conditional noise assignment is different and lacks a target-derived rule.
> Core idea remains unresolved; paper paused.

> **Existing 5K endpoint moment comparison — 2026-09-08:** RAE mild PFR
> raises mean FID .076221→.261877 while covariance FID6.958326→6.962336: 
> 97.886% of its .189666 deterioration is the mean term. Official SiT PFR
> improves mean1.281425→.613261 and covariance8.874477→8.678615.
> Mean displacement points toward target for SiT (cos+.725), away for RAE
> (cos−.257). All six existing feature/pixel banks and FP64 FIDs checked;
> no new sampling. Empirical balanced-class mean-bias correction bounds
> preserve both signs; these are not confidence intervals. Sessions9255/52189
> exited0. See PFR_ENDPOINT_MOMENTS_20260908_ZH.md. No fitted mean controller,
> no novelty/quality claim; target-relative displacement remains the next problem.

> **Shared temporal response survives removal of state direction — 2026-09-08:**
> Fixed 32-trajectory/four-time follow-up preserves all prior raw statistics
> and exact native endpoints. Transverse response cosines remain
> SiT .994/.917/.805/.795 and RAE .926/.682/.711/.657; RAE retains
> .997/.973/.915/.967 of weak response energy. Explicit z/t terms do not
> explain the consensus or its cross-model gap at these probes.
> Sessions22829/59486 and independent audit62102 exited0, .02365 GPU
> hours compute, loading extra. No component deletion or quality expansion.
> See PFR_HEAD_KINEMATIC_RESPONSE_PROTOCOL_20260908_ZH.md. Paper paused;
> target-relative utility of shared nonradial response remains unresolved.

> **Official SiT/RAE Full–Base time-response probe completed — 2026-09-08:**
> Both models have substantial shared raw temporal response: samplewise Full
> projection explains SiT .989/.853/.677/.671 and RAE .862/.484/.548/.449
> at four fixed native times near1/.9/.75/.55. RAE consensus is weaker, but
> shared response alone does not distinguish quality success from failure.
> Explicit common z/t terms also preclude interpreting consensus as truth.
> Sessions28945/26709 exited0; unchanged endpoint parity, source hashes,
> counts and 256 FP64 energy identities independently checked. .02359 GPU
> hours compute, loading extra. No quality expansion or paper writing.
> See PFR_HEAD_TIME_RESPONSE_PROTOCOL_20260908_ZH.md. Core idea unresolved.

> **Direct normalized future weak-query 1K completed, negative — 2026-09-08:**
> FID53.458707 versus native38.264239 and canonical raw52.942740.
> Independent feature-Gram FID53.458726; paired inputs, captured sources,
> native8/method8 pixels, prefix parity and actual calls passed audit.
> Sampling996.688 seconds (.27686 GPU hours), 100 Full+89 prefix/image;
> loading, smoke and evaluation extra. No5K or parameter rescue for this arm.
> Norm/time mismatch is measurable but direct query scale matching does not
> repair transfer. See RAEV2_NORM_QUERY_1K_RESULTS_20260908_ZH.md.
> No active sampling job from this branch. Core method unresolved; paper paused.

> **RAE norm/noise-time information checked — 2026-09-08:** Real cached
> clean radii plus exact Gaussian-corruption norm simulation distinguish t from
> t-1/32 very accurately at high noise; poor near .6. Eight existing native IG
> trajectories match current norm scale early, while unchanged states evaluated
> at future times lie ~12–24 norm standard deviations above the future mean.
> CPU sessions24098/85150 exited0 with cache hashes checked (~21.41s total).
> This is a query-support clue, not proof the model uses norms or a causal quality
> explanation. See RAEV2_NORM_TIME_INFORMATION_20260908_ZH.md. No new GPU jobs.
> Current release code also confirms both backbones' heads use velocity MSE;
> filenames/default loss-type strings do not establish perceptual/Huber weak loss.
> Core method remains unresolved; paper paused.

> **RAE posterior weak-reference 1K completed, negative — 2026-09-08:**
> FID posterior42.801481 versus ordinary10038.264239 and ordinary15038.458631.
> New sampling .65235 GPU hours; candidate is 3.51% faster than150 steps but
> worse quality. Drivers68261/57675 and independent audit22721 exited0;
> native/smoke pixel parity, actual query prefix parity, inputs/sources/calls
> and FP64 feature-Gram FID passed (max2.53e-5). See
> RAEV2_POSTERIOR_REFERENCE_1K_RESULTS_20260908_ZH.md. No5K or parameter rescue
> search. Gaussian cancellation and stable residuals did not give useful quality.
> No active jobs from this branch; core method unresolved, paper paused.

> **RAE posterior weak-reference fixed 1K exploration running — 2026-09-08:**
> Repeated-query probe 66102/17602 completed with unchanged endpoint parity;
> RAE single-pair SNR .72/.63/1.52/8.26 at four fixed times, not quality evidence.
> A separate fixed pilot now compares 100 Full+178 prefix per image against
> ordinary150, retaining the whole t>.5 window, h1/32, IG1.78 and seed413.
> Query seed434 is independent. Drivers 68261/GPU0 and 57675/GPU1 both passed
> native8 pixel parity and method8 checks; actual +/- future prefix checks pass.
> Quality 1K is running, estimated total ~.7 GPU hours sampling, loading extra.
> See RAEV2_POSTERIOR_REFERENCE_1K_PROTOCOL_20260908_ZH.md. Independent audit
> analyze_raev2_posterior_reference.py prepared, not yet run. Paper remains paused.

> **Posterior-coupled weak-query candidate: CPU operator check — 2026-09-08:**
> A finite ancestral posterior-mean query with antithetic noise cancels exactly
> for affine Gaussian denoisers (2304 checks, max1.78e-15). For an exact mixture
> denoiser, the approximate transition still induces bias .01121 at h1/32 and
> stochastic variability. Session62422 exited 0, .155 seconds CPU. This builds
> on existing CDM reverse-martingale theory; no novelty or quality claim and no
> escape from the existing target-identifiability counterexample. See
> PFR_POSTERIOR_COUPLING_CANDIDATE_20260908_ZH.md. Next assess estimator noise
> on real weak heads before considering a quality trial. Paper paused.

> **PFR/IG local direction diagnostic completed — 2026-09-08:** 32 unmodified
> trajectories per model show per-sample/per-time IG rescaling explains only
> 2.766% of SiT revision energy and 1.086%/0.472% of RAE energy (dt/dt²).
> Constant or time-only rescaling explains much less. Sessions 39553/99713
> passed prefix parity and FP64 projection checks, CPU counts/source/residual
> audit passed. Total compute .02740 GPU hours, loading extra. See
> PFR_IG_DIRECTION_PROBE_20260908_ZH.md. This concerns local field energy,
> not causal quality attribution or a new controller. No automatic quality
> expansion; core method and RAE breakthrough remain unresolved. Paper paused.

> **Official SiT fixed PFR 5K confirmation completed — 2026-09-08:**
> Independent new-noise FID ordinary115 10.155902 versus PFR100 9.291876
> (-0.864026, 8.51%). PFR sampling is 1.56% slower; both total 2.85814 GPU hours,
> loading/smoke/evaluation extra. Drivers 24082/19607 exited 0; audit 87253
> passed pixels, inputs, sources, calls and independent FP64 FID (~1e-12 error).
> Preregistered local paired noise approximation 47703 gives SE .203745,
> not a calibrated CI. See OFFICIAL_SIT_PFR_5K_RESULTS_20260908_ZH.md.
> This is existing-method transfer evidence. RAE improvement and core novelty
> remain unresolved; next examine effective-guidance alternative. Paper paused.

> **Official SiT PFR 1K verified; fresh 5K confirmation running — 2026-09-08:**
> FID ordinary10042.212951, PFR40.565427, ordinary11542.023907; PFR improves
> 3.47% over the115-step control at essentially identical measured sampling time.
> Independent audit88107 passed pixels, inputs, checkpoint/VAE/source/calls and
> FP64 FID reconstruction (max2.82e-5). Total1K sampling .81607 GPU hours.
> Fixed seed2026094295K now runs PFR24082/GPU0 and ordinary11519607/GPU1,
> with only the sampler seed changed, all method settings retained. Expected
> total sampling2.82 GPU hours, load/smoke/eval extra. See OFFICIAL_SIT_PFR_5K_PROTOCOL_20260908_ZH.md.
> This remains existing-method transfer evidence, not a core idea; paper paused.

> **Official ImageNet-1K SiT fixed PFR quality test running — 2026-09-08:**
> Interface40374 passed native Full latent and depth8 prefix parity. Three fixed
> 1K arms now run smoke/quality/FID: ordinary100 (74246), PFR100 (12449),
> ordinary115 (45452), GPUs0/1/2. Same seed202609428, labels0..999, scale1.35,
> FP32 model/VAE, FP64 native Euler, no TF32. PFR uses50 extra depth8 prefixes;
> 115-step ordinary has slightly greater block compute, actual time to be measured.
> No quality result yet. See OFFICIAL_SIT_PFR_1K_PROTOCOL_20260908_ZH.md.
> Existing-method transfer control, not a new-method claim; paper remains paused.

> **Official ImageNet-1K SiT PFR interface check running — 2026-09-08:** Existing
> SiT-XL/2+IG 800-epoch joint-head weights offer a control for class-count and
> head-training confounds in the small-SiT/RAE comparison. Session40374 checks
> native Full Euler latent parity, depth8 prefix parity and8-image PFR paths.
> No decoder/FID or new-method claim. See OFFICIAL_SIT_PFR_TRANSFER_INTERFACE_20260908_ZH.md.
> Paper remains paused; core research goal remains unmet.

> **Temporal parity timing correction completed — 2026-09-08:** Checked SiT
> grid indices 5/10/20 yield Base even/raw ratios .29741/.11021/.01868 versus
> RAE .34065/.11508/.03061 at nearby native times. Earlier SiT threshold queries
> had slipped to 6/11/21. The proposed large temporal-even difference weakens
> further; close this diagnostic without a quality intervention. Session21948
> exited successfully; input hashes, source, actual times and648 pair calls checked.
> The ineffective scalar-dtype intermediate run is retained and documented.
> No active jobs from this branch, paper paused, core method still unresolved.

> **Temporal parity trajectory check completed — 2026-09-08:** On 96 FP32 IG
> prefixes/model, early Base even/raw energy ratio is RAE .34065 versus SiT
> .24825, much less separated than Gaussian probes (.86399/.24065). Actual
> first probe times differ (.05085/.06); no causal model-only interpretation.
> This weakens the oversized temporal-even explanation; no quality intervention
> or automatic 1K/5K expansion. Sessions 11742/61885 exited successfully,
> source/input hashes and call counts checked. See PFR_TEMPORAL_PARITY_PROBE_20260908_ZH.md.
> No jobs remain from this test; no core method breakthrough, paper paused.

> **PFR cached-adjoint check completed — 2026-09-08:** All eight cached FP32
> native trajectories reproduced bitwise; baseline endpoint prototype values
> matched within 1e-12. Mean PFR first variation +5.697072 versus fixed central
> differences -0.595739 (2^-12) and +2.776445 (2^-13). Finite-response validation
> is unstable; do not use this derivative for a new controller or quality claim.
> 4800 B1 Full + 3560 prefix, 40 endpoint evaluations, 140.65 seconds compute;
> loading/verification extra. Sessions 25516/32413 exited successfully. No jobs
> remain from this diagnostic. See RAEV2_PFR_CACHED_ADJOINT_RESULTS_20260908_ZH.md.
> The core method remains unresolved; paper writing remains paused.

> **CFG-FSG query consistency completed — 2026-09-08:** Fixed paired 1K FID
> ordinary110 40.551168, original asynchronous 54.343048, consistent query
> 44.283012. Aligning query state and time reduces degradation but does not
> beat ordinary CFG at the same 220 Full branches/sample. New sampling cost
> .48685 GPU hours, loading/smoke/evaluation extra. Driver 60150 and independent
> audit 65403 exited successfully; pixel parity, inputs, actual calls, captured
> source differences and FP64 FID reconstruction passed (max discrepancy 1.93e-5).
> No 5K expansion or parameter search. See RAEV2_CFG_FSG_CONSISTENT_RESULTS_20260908_ZH.md.
> This job is complete; core method breakthrough remains absent, paper paused.

> **CFG-FSG query consistency control running — 2026-09-08:** An8-trajectory
> FP32 probe compares q=z-hG versus q=z-HG against16-step forward queries.
> Later-event correction cosines improve .349/.333 to .967/.995; initial
> approximation remains poor. This is not quality evidence. A fixed matched-state
> query candidate now runs native8/method8/1K with unchanged actual update h=.025,
> H=.125 and220 Full branches/sample. No strength/window search or automatic5K.
> See RAEV2_CFG_FSG_CONSISTENT_PROTOCOL_20260908_ZH.md. Paper remains paused.

> **True-CFG RAE FSG completed — 2026-09-08:** Paired fixed CFG2 1K ordinary110
> FID40.551168 versus asynchronous54.343048, both220 Full branches/sample.
> Actual null1000 reference does not rescue this clock-transfer configuration.
> Drivers and independent FP64 Gram audit passed (max FID discrepancy1.93e-5);
> native Full-only pixel parity, smoke/quality prefix pixels, hashes and calls
> verified. Sampling .967516 GPU hours, loading/smoke/eval extra. No5K expansion
> or parameter search. This is not a full reproduction or refutation of original
> FSG DDIM/CFG++. See RAEV2_TRUE_CFG_FSG_RESULTS_20260908_ZH.md.
> Both jobs complete; research breakthrough remains absent, paper stays paused.

> **True-CFG RAE FSG comparison running — 2026-09-08:** Fixed CFG2 from Full
> conditional/null1000 branches, ordinary Euler110 versus Euler100 + existing
> 2/2/1 asynchronous calibrations; both 220 Full branches/sample. This separates
> actual class conditioning from the earlier synthetic Full/Base reference.
> Both CFG1/native Full-only 8-image pixel parity checks passed. Method smoke
> checks and paired 1K are running. No quality result yet, no training or paper
> writing. See RAEV2_TRUE_CFG_FSG_PROTOCOL_20260908_ZH.md.

> **Direct reverse-flow control completed — 2026-09-08:** Same fixed 4-step IG
> targets, reverse Base Euler 4/16 then forward 4/16/32. Initial 16-reverse/32-forward
> error ratio is .568474 to frozen target but 1.502270 to refined IG target;
> midpath refined ratios .002295/.001217. Direct inversion does not remove the
> initial coarse-target/refined-flow mismatch. 1368 batch4 prefix + 492 batch4
> Full calls, 37.78s compute, hashes/counts verified. No quality run or method
> breakthrough. See FSG_ANCHORED_REVERSE_FLOW_20260908_ZH.md.

> **Multistep anchored inverse completed — 2026-09-08:** Fixed 4-step IG target,
> 4-step Base inverse with Anderson. At step0, inverse residual ratio .057384;
> jointly refining Base and IG to 8/16 steps leaves error ratios .320941/.551204.
> Midpath step73/89 16-step ratios .001654/.001116. Initial mapping remains
> discretization-sensitive. 1128 batch4 prefix + 348 batch4 Full calls, 31.16s
> compute. Inputs/checkpoint and record counts verified. No FID, training, or
> quality/efficiency breakthrough. See FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md.

> **Inverse solver versus transport discretization — 2026-09-08:** Fixed Anderson
> history4/ridge1e-4 reduces the initial RAE Euler inverse residual ratio from
> Picard's 23.348795 to .00006525 at k8. But 8-substep Base transport from this
> solution retains .867424 of the unmodified same-solver target error. Accurate
> inversion of a coarse map is not accurate latent encoding for refined transport.
> Original Picard reproduced within 5.43e-8 on identical inputs/checkpoint.
> 420 batch4 prefix calls plus native path queries, 15.68s compute. No quality
> claim or FID launch. See FSG_ANCHORED_INVERSE_SOLVER_20260908_ZH.md.

> **Anchored inverse probe completed — 2026-09-08:** On 8 FP32 RAE native
> trajectories, a fixed H=.125 single-Euler target was inverted through Base
> Picard iterations. Mean squared residual ratios at k8: step0 23.348795,
> step73 .005029, step89 .00002093. Early naive iteration is unstable; this is
> not an audit of FSG's original multi-step operator or a quality result.
> 54 batch4 prefix calls plus native trajectory queries, 8.95s compute, no training.
> See FSG_ANCHORED_INVERSE_PROBE_20260908_ZH.md. No quality run launched.

> **SiT clock compensation quality completed — 2026-09-08:** Paired fixed 1K
> native/smooth/compensated FID = 71.169791/69.803808/71.941799. Independent
> FP64 feature-Gram audit passed (max difference 3.17e-5), as did native pixel,
> prefix, paired input/model and actual-call checks. Both candidates use
> 100 Full + 50 prefix/sample; total sampling .050664 GPU hours. Compensation
> worsens FID, sFID and IS versus smooth. Do not promote exact-clock cancellation
> as a quality fix; no coefficient/window search or 5K expansion for this candidate.
> See SIT_CLOCK_COMPENSATOR_RESULTS_20260908_ZH.md. No jobs remain from this test;
> paper stays paused and the core research goal remains unmet.

> **Clock compensation check — 2026-09-08:** For a smooth endpoint-fixed clock,
> raw temporal secants omit the velocity-times-clock-derivative term. CPU exact
> Gaussian integration confirms first-order endpoint bias becomes second-order
> after adding that term; exact clock transport preserves the endpoint. This is
> a basic ODE control, not a new method. Real PFR shifts W while evolving G, so
> a relative-field perturbation remains and quality improvement is unproven.
> See PFR_CLOCK_COMPENSATOR_20260908_ZH.md. No GPU jobs launched.

> **Exact-field amplitude audit — 2026-09-08:** A CPU Gaussian linear-flow
> calculation shows time-only PFR changes the endpoint distribution even with
> identical exact Bayes strong/weak fields and continuous integration. At
> gamma=.35, h=1/32, target variance 4 becomes 3.816272 (native exact).
> Independent direct ODE integration agrees with log-scale quadrature to 2.12e-11
> across both fixed horizons. This rules out a universal model-error interpretation
> of raw revision magnitude, not OU projection or the empirical SiT improvement.
> It does not establish the cause of RAE failure. See
> PFR_EXACT_GAUSSIAN_AMPLITUDE_20260908_ZH.md. No new GPU job or paper writing.

> **OU condition-direction quality completed — 2026-09-08:** Fixed paired RAEv2
> 1K FID: native 38.264239, conditional 53.184813, unconditional 48.630232,
> conditional-minus-unconditional 44.000009. All drivers and independent FP64
> feature-Gram audit passed (maximum FID discrepancy 2.44e-5). No candidate beats
> native. Each candidate executes 319 Full + 89 prefix calls per sample; sampling
> totals 2.306 GPU hours, excluding loading/evaluation. No 5K expansion or bank
> strength/window search. See RAEV2_OU_CONDITION_DIRECTION_RESULTS_20260908_ZH.md.
> Direction isolation does not validate the retained raw correction magnitude.
> Core method breakthrough remains absent; paper writing stays paused.

> **Latest user constraint — 2026-09-08:** “不要写论文，先打磨方法和idea到极致”。
> Paper writing is paused. Do not advance the existing working draft; focus on
> method development, causal controls, rigorous comparisons and transfer.
> The broad research objective remains unmet. Positive new 5K clock results
> are on SiT, not RAEv2. All five new RAEv2 1K arms are complete and independently
> audited: native100/native110/short/asynchronous/time-only FID =
> 38.264239/38.392635/38.939462/48.972310/49.931931. Direct transfer failed;
> these are not 5K. No new RAEv2 5K confirmation is justified for these arms.
> SiT common-field IG async 1K = 64.771331 versus original 64.615465: distinct
> calibration field identity is not established as necessary. CFG differs.
> A 32-sample RAE native-path decomposition locates strong initial clock effects,
> but the small native first step inflates relative ratios; latent changes are
> only about 1–2%. Neither this probe nor the FID results establish the cause.
> See FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md and
> RAEV2_CLOCK_DECOMPOSITION_PROTOCOL_20260908_ZH.md. Paper draft stays paused.
> Follow-up: matched SiT native-path decomposition completed (32 samples);
> relative-to-latent initial changes are .687% SiT versus 1.319% RAE, so
> relative-to-native-step ratios alone do not establish numerical explosion.
> Fixed RAE early-only and late-only event interventions completed under
> RAEV2_CLOCK_EVENT_INTERVENTION_20260908_ZH.md, with discarded-event calls
> retained for compute matching and exact 8-image none/all parity controls.
> Their 1K FID is 48.659778 and 38.144204 respectively, versus native 38.264239
> and all-events 48.972310; independently audited. Initial calibration accounts
> for the principal observed degradation. Late-only's .1200 reduction is not
> independently validated improvement. Fixed initial-noise shell probes
> completed: two initial async iterations move the normalized squared radius
> by 1.815 standard-Gaussian SD in RAE versus .088 in SiT (32 samples each).
> This is a radial-distribution clue, not yet a cause of quality degradation.
> No normalization repair or new training has been introduced.
> Next causal probe completed: initial radial-only and angular-only
> interventions, each fixed 1K, protocol RAEV2_INITIAL_RADIUS_DIRECTION_PROTOCOL_20260908_ZH.md.
> Native and full-initial 8-image pixel parity passed; radial direction and
> angular radius constraint errors are below 2.6e-8 on smoke samples.
> Independent 1K audit: radial-only FID 38.260656, angular-only 48.703479,
> versus native 38.264239 and full-initial 48.659778. The radial-cause hypothesis
> is contradicted: preserving the original radius does not remove degradation.
> Do not promote shell normalization or launch its 5K based on the prior probe.
> Angular change retains the failure, but its substantive field mechanism remains
> unidentified. No new training or paper writing; the research goal remains unmet.
> Return to original PFR: historical RAE 5K used time-only rho=.05; the existing
> pathwise RAE branch uses full-velocity Euler displacement, not SiT's forward-ray
> projection. A fixed rho=1, h=1/32, first-half time-only/projected pair is now
> completed on native RAE IG. Native pixel and actual future-query prefix/full
> parity passed. This fills a comparison gap, not a new-idea claim; old negative
> results remain valid for their actual settings. Protocol:
> RAEV2_CANONICAL_PFR_QUERY_PROTOCOL_20260908_ZH.md.
> Fixed RAE rho=1 time-only/projected 1K FID = 52.942740/52.069028 versus
> native 38.264239. Projection improves the time-only arm but does not resolve
> transfer failure. No 5K expansion or h/rho rescue sweep for these arms.
> During that run, existing same-representation SiT-v/x controls completed:
> fixed gamma=.35 Euler100, PFR 1K FID v 71.169791→68.357811 and x
> 71.667794→68.449145. All four inputs paired and independently audited.
> This contradicts clean output alone as a PFR barrier; not a new 5K claim.
> See SIT_PFR_OUTPUT_PARAMETERIZATION_PROTOCOL_20260908_ZH.md.
> Reference-target intervention: two frozen-backbone Base readout copies were
> fitted for exactly 2048 AdamW steps to real clean data or frozen Full outputs.
> Neither improved its own held-out objective; total fit/validation .07085 GPUh.
> An exact affine-only control then reached full rank and ~3.6e-15 equation
> residual, improving training but worsening held-out data risk:
> original .477276, data-fit .484877, teacher-fit .479392 (.01406 GPUh).
> This distinguishes finite-bank generalization failure from unconverged SGD;
> it does not establish a PFR mechanism or imply MSE is a quality criterion.
> The fixed ten-noise-draw-per-image control completed without validation
> selection under RAEV2_REFERENCE_MULTINOISE_PROTOCOL_20260908_ZH.md.
> It retains only 5000 unique training images. Independent CPU Cholesky solve,
> training-risk reconstruction and source/heldout preservation audits passed.
> Heldout data risk: original .477276, data .480509, teacher .478104; more noise
> draws reduce but do not eliminate refit degradation. Cost .12345 GPUh.
> Teacher-minus-data error alignment remains small and positive (.000090),
> identified by risk polarization; this is not a normalized explanatory fraction,
> quality improvement, or justification to repackage prior failed X-F adapters.
> No further draw-count/optimizer sweep, new-head FID, or paper writing started.
> Reference-difference norm audit now resolves the small positive alignment:
> ten-draw training norm²=error inner product=.0023744 (CPU verified), but
> heldout norm²=.002412 and inner product=.000090. Fixed unit subtraction
> raises Full risk by .002231; one-draw raises it by .005600. All stored-head
> validation values reproduce, and direct risk/identity/polarization agree.
> Thus these fits did not recover a generalizable unit error projection;
> the positive inner product is insufficient evidence for a method candidate.
> Cost .00206 GPUh; no new training, scaling search or quality sampling.
> See RAEV2_REFERENCE_DIFFERENCE_RISK_PROTOCOL_20260908_ZH.md. The research
> goal and RAEv2 transfer problem remain unresolved; paper writing stays paused.
> Exact 1D Gaussian-mixture transport audit now gives a concrete counterexample
> to gap minimization as a distribution-quality target: U^-1 C once gives the
> exact conditional distribution, but repeated application has no finite fixed
> point and moves into the tail. On all three fixed separations, endpoint gap
> falls while W2 grows after the first transport. Sampled local derivative²<1
> does not establish a global contraction. This violates FSG's global theorem
> assumptions, rather than refuting its conditional theorem or neural results.
> See FSG_EXACT_GAUSSIAN_TRANSPORT_20260908_ZH.md. It is a method-design
> constraint, not a new sampler or an explanation established on RAEv2.
> Follow-up fixed-budget oracle experiment rejects scalar 1/K rescaling as
> sufficient to factor that finite transport: m=2 W2² remains .687 at K256,
> despite total budget=1. Frozen original displacement is exact but trivial
> and offers no new computational capability. Correct factorization needs
> the evolving map/inverse, not repeated evaluation of T-id at the new state.
> PathGuide (arXiv 2608.29107v1, abstract through §4.2 read) already studies
> on-policy weak transport alignment for scalar CFG; broad path-matching
> language is not a novelty claim. See FSG_TRANSPORT_BUDGET_20260908_ZH.md.
> Real RAE causal control launched: same guided field on both calibration legs
> (R=G), original native IG sampler unchanged; fixed paired 1K, h=.025/H=.125,
> all original events and 110 Full calls/sample. This tests whether distinct
> calibration fields are necessary for degradation, not whether Full/Base
> disappears from the guided field. Driver gates quality on native/original
> 8-image pixel parity. Protocol:
> RAEV2_COMMON_FIELD_CALIBRATION_PROTOCOL_20260908_ZH.md.
> Common-field 1K now completed and independently audited: FID50.002931 versus
> native38.264239 and original two-field48.972310. Independent Gram error1.70e-5,
> source/input/call/pixel checks passed. Cost .24485 GPUh sampling. Distinct
> calibration fields are not necessary for this degradation; the common guided
> roundtrip is itself harmful here. G still includes native IG. Future-only
> remains running, so no conclusion yet on its standalone effect or interaction.
> Complementary future-IG-only arm launched before observing the common-field
> result: retain only 2h(G'-S') from the exact same-query decomposition of
> A-z into common roundtrip plus future contrast. Same paired 1K and calls,
> GPU1, independent native/original pixel gates; protocol
> RAEV2_FUTURE_IG_CALIBRATION_PROTOCOL_20260908_ZH.md. Together with existing
> none/all and running common, this tests components and their trajectory
> interaction. FIDs are not additive; no new-method or 5K claim is implied.
> Both arms are now complete: future-IG-only FID41.881665 (independent error
> 2.80e-5), versus none38.264239/common50.002931/all48.972310. Inputs paired,
> source/model/call/pixel checks passed; combined new sampling .48745 GPUh.
> Neither isolated component beats native. Deleting the common roundtrip
> does not turn the remaining future contrast into a successful RAE method.
> No 5K expansion or strength/event rescue sweep for these arms. All associated
> jobs finished; the research goal remains unmet and paper writing paused.
> Best historical SiT OU-polar 5K assets re-audited: CUDA noise banks fully
> reconstruct seed5/6 manifest hashes; 5000 unique each and zero cross-bank
> identical samples. Labels reproduce. Independent retained-feature FIDs
> 36.19015462/35.75879212 agree with reported 36.19015567/35.75879012.
> Historical image NPZ files are absent, so this is not pixel-to-feature or
> full sampling reproduction. See PFR_BEST_5K_ASSET_REAUDIT_20260908_ZH.md.
> Raw PFR's SiT-x success does not yet establish output-parameterization
> independence of the additional OU direction-selection gain.
> Fixed SiT-v/x OU direction control launched: constant gamma=.35/Euler100,
> paired seed202609417/B8 1K, time-only versus strong OU direction/raw norm.
> Both execute 125 Full and 50 prefix calls/sample, including discarded OU
> queries in the time-only control. Native and old projected-PFR pixel parity,
> future-prefix parity all passed on both models; quality runs are active.
> This is a common-protocol mechanism comparison, not reproduction of the
> historical best Heun/piecewise-gamma result. Protocol:
> SIT_OU_OUTPUT_CONTROL_PROTOCOL_20260908_ZH.md. No training or paper writing.
> All four OU controls now complete and independently audited: v time-only
> 68.447133→OU65.779885, x68.022327→65.440884. Same input hashes and actual
> 125 Full+50 prefix/sample, FID reconstruction max3.05e-5. Sampling .12322 GPUh.
> sFID worsens in both (211.346→213.172,209.602→211.510); not an all-metric win.
> Clean output alone is not a universal barrier to the extra OU FID gain in
> this setting. This is one 1K bank, not a new 5K confirmation or causal proof
> about RAE's joint weak head. All associated jobs finished; paper stays paused.
> Gaussian rotation stencil probe now tests whether RAE OU is mainly affine:
> exact affine-annihilation identity verified. At data times .05/.1/.2, residual
> over pair-difference energy is .105/.084/.097 SiT-v, versus .805/.886/.842
> RAE BF16 and .800/.881/.827 RAE FP32 on identical RAE inputs. Precision
> does not explain the pattern; simple affine-covariance dominance is unsupported
> by this probe. These are Gaussian-coordinate queries, not paired native paths,
> and the ratio is not a nonlinear energy fraction. No sampler or training added.
> See OU_AFFINE_RESPONSE_PROBE_20260908_ZH.md; substantive goal remains unmet.
> Independent SiT-v/x OU confirmation launched: four fixed 5K arms on new
> seed202609423, unchanged sampler source and common gamma=.35/Euler100
> protocol, one arm per GPU. Native/prefix/first-batch parity gates precede
> or validate quality runs. Both FID and sFID will be independently reconstructed;
> the prior 1K sFID regression must remain visible. Protocol:
> SIT_OU_OUTPUT_5K_PROTOCOL_20260908_ZH.md. No new training or paper writing.
> Four independent 5K arms completed: v FID41.510229→39.842740, sFID
> 71.710647→71.191108; x FID40.556181→39.579555, sFID70.099348→69.243289.
> Independent FP64 errors max3.60e-6 FID/1.71e-6 sFID, paired input/model/calls
> and pixel gates passed. Sampling .59975 GPUh; all jobs finished.
> Protocol erratum: actual reused code draws uniform random labels, not exactly
> 50/class (5K counts30–68; prior1K counts4–20). All four arms share identical
> labels and cover100 classes. Frozen original protocol copies retained; do not
> describe these as exactly class-balanced confirmations. No RAE or new-method
> breakthrough is implied; the overall objective remains unmet, paper paused.
> Time×condition OU geometry probe completed (96 Gaussian-coordinate inputs,
> three times/model): raw versus posterior-difference cosine is .169/.180/.146
> SiT and .197/.076/.030 RAE, much smaller than raw versus full conditional
> certificate. Removing unconditional response substantially changes direction;
> it cannot be treated as harmless noise removal. These are not quality results
> or evidence that low cosine forbids benefit. No new sampler, strength choice,
> or training started. See OU_CONDITION_DIFFERENCE_PROBE_20260908_ZH.md.
> A subsequent fixed quality test is now launched to avoid treating geometry
> as a quality gate: RAE paired1K directions D_c/D_u/(D_c-D_u), same raw revision
> norm, nativeIG1.78, Euler100, rho1, first-half PFR and t>.75 OU selection.
> All arms execute identical extra conditional/null queries; ordinary8 and
> prefix parity gate sampling. Compare against native as well as each other.
> Protocol RAEV2_OU_CONDITION_DIRECTION_PROTOCOL_20260908_ZH.md, estimated
> 2.2 GPUh sampling total. No training, strength sweep, or paper writing.

> **Research reopened by explicit user request — 2026-09-08.**
> The new goal is a substantive ICLR-level paper, not another archive closeout.
> A finite temporal-defect pilot did not support the proposed cancellation
> explanation. A subsequent frozen-head subspace-control pilot has a positive
> three-seed toy result: learned PCA split gates reach mean ambient SWD 0.074337
> versus a matched scalar gate 0.098277, with lower normal residual; random
> subspaces and simple projection do not reproduce the gain. This is only a
> linear rank-2 D512/H128 toy with a hard bottleneck, not an established novel
> method or an ImageNet breakthrough. The paper goal remains incomplete.
> See [the full pilot report](SUBSPACE_GATE_PILOT_RESULTS_20260908_ZH.md).
> Follow-up: independent 8K banks confirm the narrow result; H512 width
> controls also retain gains, but base-model convergence remains unproven.
> A real 450K SiT rank32 time-only split-gate audit shows no meaningful extra
> held-out local-risk gain. No real-image quality breakthrough is established.
> Further controls retain the toy gain after 30K training. A curved-support
> three-seed experiment also improves full-dimensional SWD, while worsening one
> geometry residual. Coordinate and
> ball oracles show that PCA geometry is not necessary for oracle headroom;
> a matched-parameter learned coordinate gate still trails PCA on three seeds.
> See [maturity and curvature results](SUBSPACE_GATE_MATURITY_CURVATURE_RESULTS_20260908_ZH.md).
> Real-image state-dependent gates now have paired ADM-FID-5K evidence:
> native 71.265543, scalar 71.223144, PCA 71.132263, pure epsilon 70.512364.
> PCA fails to beat the stronger single head; the small native-relative gain
> is not a breakthrough. Held-out local error deterioration is concentrated
> near time endpoints. The fixed native-DDO weighting control also failed:
> scalar FID71.034772, PCA71.223001. Subspace gates are no longer the primary
> paper direction. Following explicit user steering, research turns to PFR
> transfer and FSG's condition-bearing latent intuition; the first controlled
> test erases condition input after a fixed PFR/IG prefix.
> See [the new protocol](PFR_CONDITION_RETENTION_PROTOCOL_20260908_ZH.md).
> The first paired condition-erasure pilot is complete on both models.
> A matched-ImageNet100, 200-sample replication retains the large middle-time
> contrast: ordinary IG conditional/unconditional top1 is 85%/84.5% on RAEv2
> and 49.5%/5% on SiT. Paired interactions do not establish that PFR specifically
> reduces dependence on future conditioning. All three cuts are now complete,
> with exact within-model noise and cross-model label parity. The late interaction
> is also inconclusive; conditional retention is not supported as PFR's direct
> quality mechanism in this intervention.
> A CPU audit of released FSG code finds nominal versus executed DDIM clock
> differences in the tested environment; this is not a reproduction or refutation
> of the authors' image results. See [the audit](FSG_RELEASED_CLOCK_AUDIT_20260908_ZH.md).
> A fixed CFG/IG Euler clock quality screen is complete: long state/long query,
> short state/short query, short state/long query, plus closed40/closed50 controls.
> Both families pass 8-image original-long pixel parity and input/call checks.
> At paired 1K, CFG closed50/short/asynchronous FID is 61.463529/55.530113/53.403568;
> IG is 69.147854/66.055254/64.615465. IG long severely degrades to 122.303940.
> All ten arms share exact noise/label hashes and frozen sampling sources.
> Independent seed202609412 5K confirmation is complete and independently
> reconstructed from features: CFG closed50/short/asynchronous is
> 34.627031/28.960328/28.205314; IG is 41.533513/39.783258/38.395560.
> All six banks pass input/source/call checks; sampling cost totals 0.514 GPU h.
> Fixed 1K controls find time-only nearly matches asynchronous in both families;
> forward state movement is not established as necessary. IG reference gamma
> also changes at future query boundaries; the fixed-reference-gamma control
> passes exact short8 pixel parity and gives 1K FID64.482336 versus 64.615465.
> This coefficient switch is not supported as the sole cause of the gain.
> All clock quality/control and same-bank reference runs are complete.
> Ordinary DOPRI5 CFG/IG FID is 33.372389/41.278966; original PFR DOPRI5
> reaches 37.786383, better than asynchronous IG 38.395560 at higher cost.
> Budget-derived PFR Euler42 (588 Transformer blocks versus 600) gives
> 39.591289, suggesting a low-budget advantage for asynchronous calibration.
> All four reference FIDs independently reconstruct within 3e-6 and paired
> inputs/checkpoint/head settings pass. This is not a full Pareto curve or
> a RAEv2 transfer success. See [matched reference protocol](FSG_PFR_MATCHED_BANK_PROTOCOL_20260908_ZH.md).
> Direct RAEv2 clock transfer is now running under a frozen five-arm 1K protocol:
> native100/native110/short/asynchronous/time-only, h=.025 and H=.125.
> Native100 matches the original sampler pixel-for-pixel on eight images;
> native110/short/asynchronous smoke input and call checks pass.
> Candidates use 110 shared full/base forwards, matching native110.
> No RAEv2 clock FID result exists yet. See [transfer protocol](RAEV2_FSG_CLOCK_TRANSFER_PROTOCOL_20260908_ZH.md).
> No training or parameter search.
> This is a quality signal, not established novelty or RAEv2 transfer success.
> See [the frozen quality protocol](FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md).
> See [real SiT protocol and results](SIT_SUBSPACE_GATE_PROTOCOL_20260908_ZH.md).

> **Latest three-round guidance study complete and closed — 2026-09-07.**
> The user-imposed final three idea rounds are finished. The3% paired-5K FID
> target remains unmet: round1 failed its mechanism gate; round2 reached
> FID6.925725594 (+0.345952%, 1.483697x inference cost); round3 reached
> FID6.970196781 (−0.293942%, 1.991345x), versus official6.949768478.
> All paired inputs, shard/merged pixels, frozen sources and independent FID
> checks pass. The recent comparable best remains spatial covariance6.906138435
> (+0.627791%) on the repeatedly explored discovery bank. All final sampling
> processes exited. No fourth idea round, parameter search or confirmation is
> queued. Theory, positive/negative results, code and data manifests are archived;
> archival completion is not quality success. See the
> [final report](RAEV2_FINAL_THREE_ROUNDS_CLOSEOUT_20260907_ZH.md) and
> [three-round record](RAEV2_FINAL_THREE_ROUNDS_20260907_ZH.md).
> Earlier limits and “active” entries below are historical snapshots.

> **Fixed mild-negative 5K follow-up complete and closed — 2026-09-07.**
> All four candidates, controls and cost accounting are complete. Against each
> family's official100, pooled5K gains are global proximal −0.107106%, legacy
> energy −0.194903%, native global energy +0.230464%, reflection +0.195719%.
> Reflection gains +0.227356% against the cost-covered official201, but loses
> 0.128630% on the new4K alone. Native global gains +0.185095% on new4K.
> None approaches 5%; total research-cost success is unproved. The user returned
> to guidance and requested only completion and Git: all adjacent directions
> are stopped, with historical notes retained. No additional quality search.
> See [supplement closeout](RAEV2_MILD_NEGATIVE_5K_CLOSEOUT_20260907_ZH.md).

> **Limited 5K follow-up authorized — 2026-09-07.** The user explicitly
> reopened only supplementation of mildly negative 1K candidates to 5K.
> Preserve all original methods/calibrations and original 1K images; add four
> fixed independent 1K cohorts, report new4K separately, and retain actual
> inference/preparation cost limits. Selected: global proximal, legacy global
> energy ball, native global energy ball and affine reflection. No new method,
> gain/window or training search. This newer instruction supersedes the prior
> closeout's no-resume boundary for this bounded follow-up only. See
> [frozen supplement protocol](RAEV2_MILD_NEGATIVE_5K_EXTENSION_PROTOCOL_20260907_ZH.md).


> **Closed at the user's final round 8/8 limit — 2026-09-07.** All fixed quality experiments ended in round6 below the3% target; best1K gain2.031820%, best5K0.564060%. Round7 verified all6220 inventory files directly from the quality-closeout Git commit. Final state and archive review are now closed; no new fitting, sampling, parameter changes or automatic research continuation. The objective is unmet, and archival completion is not success. Further research requires a new explicit user instruction. See [final report](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md) and [completed eight-round ledger](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md).

> **Quality closeout, final round 6/8 — 2026-09-07.** The final unchanged semantic_add5K is complete: FID7.250174077, 4.322527% worse than official, inference1.994621×; all pixels,625 paired input batches, original parameters, frozen sources and independent FID pass audit. All17 candidates have1K and9 have5K; none meets3%. Best1K gain is2.031820%, best5K0.564060%. Quality research is now closed; remaining rounds are only final evidence/archive/Git review. No Heun cost run or new candidate follows this failure. See [final report](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md). Archival completion is not goal success.

> **Final round 5/8 — 2026-09-07.** The directional candidate is fully audited at 5K FID6.910567613, a 0.564060% improvement over official, with inference1.013986×; its negative1K and positive5K are both retained, and neither meets3%. The unchanged old semantic_add .15 is now the last additional quality candidate, completing its own5K after original8 pixel parity; no more methods or coefficient changes follow. All Git-owned research, including historical root-level results, is being indexed in the [complete workspace archive](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md), with [reference/asset review](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md). A new exact scalar counterexample shows that even lower Gaussian KL can have higher FID; see [theory conclusions](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md). This does not identify the empirical cause of RAEv2 failures. The3% goal remains unmet; the eight-round cap is unchanged.

> **Final round 4/8 — 2026-09-07.** The fixed spherical conditional-variance5K is complete and independently audited: FID6.957439858, 0.110383% worse than official and 0.179169% worse than the old global variance; inference1.036520×. Both scales fail3%. A final covariance extension retains the same fitted head and adds exactly one global ratio along the native Full/Base difference. Its closed-form Gaussian-NLL optimum is the training mean, kappa27921.8994, with no optimizer or parameter sweep. All72K recomputed native features and MSE match the original cache bitwise; held-out per-coordinate NLL change−.0532327 passes. Total covariance trace rises10.651%, not27922×. Three analytic checks and original/unit-ratio8 full-pixel parity pass; fixed1K is independently audited at FID38.610371195 (0.321142% worse, inference1.013030×); same-parameter5K is running despite the negative1K. See [single-scalar protocol](RAEV2_DIRECTIONAL_VARIANCE_PROTOCOL_20260907_ZH.md) and [spherical5K audit](../experiments/results/raev2_guidance_20260907/conditional_variance_confirm5k_audit.json). The3% goal remains unmet and the eight-round cap stands.

> **Final round 3/8 — 2026-09-07.** Both unchanged legacy5K arms are complete and independently audited: probability-calibrated ratio FID6.938002015 (+0.169307%, inference1.015546×), global guided variance6.944996553 (+0.068663%, inference1.000462×), versus official6.949768478. One audit compatibility failure was a metadata-only `source_law` annotation; all actual parameters and original samples remain unchanged, with the failure/recovery chain retained. The predeclared conditional variance candidate now has complete native64K/8K features and one convex fit: validation per-coordinate NLL change−.005166167, class two-SE upper−.004877652. It also improves over a fresh train-only time-MSE diagnostic (−.005186223), so the signal is not merely old time-calibration bias. Original8 and zero-head8 full-pixel checks pass. Fixed1K is independently audited at FID38.547114655 (0.156783% worse, inference1.036949×); the same head's fixed5K is running despite the slightly negative1K. This is not a quality success. See [conditional variance protocol and results](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md), [legacy5K audit](../experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json), and [eight-round ledger](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md). The 3% goal remains unmet.

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
