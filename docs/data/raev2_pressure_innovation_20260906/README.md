# RAEv2 final Round 2: pressure and symmetric innovation

Completed CPU-only mechanism audit; no new training, GPU sampling or image FID. The current variants were not admitted to training.

Eight original small outputs are copied byte-for-byte. `per_time.csv` covers every fixed time; `channel_moments.npz` retains all 100 × 1024 channel differences. `independent_review.json` records an independent source-cache recomputation, including 50-digit Decimal accumulation.

`request.json` and the independent review identify upstream source files by path and SHA-256. Original 10K teacher rows (3.74 MB) and the full moment cache (6.56 MB) remain in the external experiment directory and are not copied here. Thus the package supports inspection of the reported aggregates and provenance, while rerunning the source-cache review requires those originals. Original residual and W/Y tensors were not saved by the producer; their absence is not repaired by this export.

Population moment identities, this finite cohort, and the analytic Gaussian counterexample have separate scopes; none is a generated-image quality result. The Gaussian number is a one-dimensional exact example with an identity decoder.

See [Chinese result record](../../RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md).
