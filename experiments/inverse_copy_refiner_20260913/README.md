# Real-anchor inverse-copy refiner

`model.py` defines `Refiner(num_classes=100, width=32, context_dim=64)`, a 592,100-parameter residual UNet. `model(z, labels)` returns the corrected latent directly; it does **not** return the residual. The last convolution is zero-initialized, making the initial model exactly identity. Neither corruption time nor direction is an input.

`train.py` reads a frozen NPZ containing:

- `clean`: float32 `[N,4,32,32]`;
- `corrupted`: float32 `[N,M,4,32,32]`, with arbitrary `M>=1`;
- `labels`: int64 `[N]`, `split`: int8 `[N]` (`0=fit`, `1=heldout`);
- `source_ids` (or `source_id`): unique integer/string `[N]`;
- optional `taus`: `[M]`, repeated values allowed, and `directions`: `[M]` in `{-1,+1}`.

Production requires 400 fit and 100 heldout sources and rejects source-ID overlap. Current M=2 is `.75,.5`; the symmetric M=4 input can use times `.75,.5,.75,.5` with directions `+1,+1,-1,-1` without changing this code.

Every step samples 16 distinct fit sources, uses every supplied corruption (32 repair inputs for M=2 or 64 for M=4), and evaluates identity on the same 16 clean sources once. The objective is exactly `repair_MSE + 2 * identity_MSE`. There is no augmentation, idempotence objective, schedule, clipping, or autocast.

Defaults: 1500 steps, AdamW learning rate `2e-4`, weight decay `1e-4`, seed `2026091361`; heldout evaluation every 100 steps and at the final step. The initial identity model at step 0 participates in best-checkpoint selection. Output directories must not already exist.

Production execution is owned by the root task. Example:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.inverse_copy_refiner_20260913.train \
  --device cuda \
  --pairs /home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/pairs.npz \
  --output /home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/training
```

Outputs: `run.json`, `zero_p_baseline.json`, step-level `rawmetrics.jsonl`, `best.pt`, `last.pt`, and final `summary.json`. They record per-corruption repair loss, identity loss, weighted objective, source IDs, input/source SHA256 hashes, model parameter count, and model configuration.

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model = Refiner(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model"])
model.eval()
refined = model(latents, labels)
```

CPU-only smoke:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.inverse_copy_refiner_20260913.smoke
```

It uses a clearly synthetic shape fixture, performs three CPU optimization steps, checks checkpoint reload and nonzero gradients, rejects source leakage and output overwrite, and checks the M=4 metadata/evaluation interface. It does not run production training or establish any image-quality benefit.

Scientific limits: a lower heldout reconstruction loss is not an FID improvement; identity and repair targets can conflict on overlapping supports. If a refiner learns to exactly undo `C=G_weak G_strong^-1`, its action on the matched weak generator tends toward the strong reference. Applying it to another CFG/APG distribution therefore requires separate evaluation. Five-round behavior is also evaluated separately by the root task.
