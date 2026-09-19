# Copy-based guidance calibration: CPU Gaussian-mixture toy

This runnable toy tests whether an independent image-to-noise inverse lets real-image preservation choose an existing guidance strength. It is a feasibility result, not a new sampler or a win over tuned CFG/AG.

The true conditional law is an equally weighted Gaussian mixture with means ±2 and standard deviation .55. The strong and weak canonical FM models share deliberately aligned underfitting: their means move inward to ±1.7 / ±1.4 and their component standard deviations increase to .70 / .85. The null model has an additional central class. These are controlled model assumptions, not observed neural-model errors.

`run.py` fixes the true inverse and selects one scalar CFG or AG coefficient using 256 independent real anchors. `nonoracle.py` replaces the analytic inverse with KDE CDFs fitted to three independent 1024-sample banks using a fixed bandwidth rule. Neither calibration uses generation W2. Both save every copy from round 0 through round 5. `analyze.py` exports the final PNG and PDF.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.identity_copy_toy_20260913.run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.identity_copy_toy_20260913.nonoracle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.identity_copy_toy_20260913.analyze
```

[Chinese report, figure, complete JSON and NPZ records](../../docs/research/identity_copy_toy_20260913/README.md).

The exact risk identity uses one-dimensional monotone transport and a correct common inverse. Higher-dimensional coupling, biased references, and five-round ranking have explicit counterexamples in the report. CPU only; no GPU or neural sampler is used.
