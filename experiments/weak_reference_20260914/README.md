# Weak-reference guidance experiments

Chinese research report: [full analysis and results](../../docs/WEAK_REFERENCE_GUIDANCE_RESEARCH_20260914_ZH.md).

This is a frozen-model research pilot. It contains same-time antithetic log-score extrapolation, a two-scale score contrast, a Gaussian-preserving rotation reference, a diagonal moment-matched convolution reference, and fixed-kernel DSM calibration. Rotation and moment matching are tested on SiT with CFG; the two-scale image experiment uses SiT without CFG. The real calibration fits scalar weights for fixed kernels; it does not train a general weakener network.

Raw artifacts: `/home/zhoushunyu/data/eqvae/experiments/weak_reference_20260914/`. Audited tables, checks, fixed-ID contact sheets, bootstrap replicates, and the chart-data workbook: `docs/data/weak_reference_20260914/`.

All image arms use 1,000 samples. SiT uses batch 16, and the cross-model phase uses batch 32. The following phase registry specifies the complete study (34 arms, 34,000 images):

| Phase | SiT sampler suffix | Configuration file | Seed | Arms |
|---|---|---|---:|---:|
| sit_screen_1k | sampler | configs.json | 2026091407 | 9 |
| angular_screen_1k | angular | angular_configs.json | 2026091407 | 4 |
| moment_screen_1k | moment | moment_configs.json | 2026091407 | 3 |
| strong_confirm_1k | sampler | strong_confirm_configs.json | 2026091408 | 3 |
| strong_sg_1k | sampler | strong_sg_configs.json | 2026091408 | 2 |
| strong_band_1k | calibrated | strong_band_configs.json | 2026091408 | 2 |
| strong_band_confirm_1k | calibrated | strong_band_confirm_configs.json | 2026091409 | 5 |
| cross_screen_1k | cross_core | cross_core.configs() for jit,raev2 | 2026091407 | 6 |

Use the existing `myenv` Python and assets recorded in each phase's `request.json`. For a SiT phase, substitute the registry values into:

```bash
PYTHON=/home/zhoushunyu/miniconda3/envs/myenv/bin/python
"$PYTHON" -m experiments.weak_reference_20260914.run prepare --phase PHASE --config experiments/weak_reference_20260914/CONFIG.json --sampler experiments.weak_reference_20260914.SAMPLER --samples 1000 --seed SEED --batch 16
"$PYTHON" -m experiments.weak_reference_20260914.run controller --phase PHASE --gpus 0
"$PYTHON" -m experiments.weak_reference_20260914.report --phase PHASE
```

Use `cross_run prepare/controller` and `report --models jit,raev2` for the cross-model phase, as shown in the research report. On a fresh run, execute `check --model`, `angular_check --model`, `moment_check --model`, `band_check`, and `cross_check --model MODEL` on available GPUs. The two calibration scripts and the two toy scripts are separate from image generation. `moment.py` uses the moment statistics whose source and hash are archived in `moment_stats.json`; they are estimated from cached training VAE posterior moments, not exact model moments.

Once all registry phases have completed and been audited, run `bootstrap`, `bootstrap --band`, `build_data`, then `write_report`. The last command uses `research_verdict.json` for the reviewed interpretation and derives every image-results table from the audit bundle. The workbook includes all plot grid values, every image arm, and calibration/bootstrap statistics.

Requests hash sampling code, model assets, and paired inputs. Use a new phase name for changed configurations. FID recomputation uses the same cached features and does not constitute an independent extractor. The report discusses the small-sample limitations and distinguishes tested prototypes from further proposals.
