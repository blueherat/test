# RAE Geometry Research Status

## Scope

The working research question is whether frozen RAE tokenizers contain a
usable geometric response that can improve latent-space generation. This is
not yet a claim that RAE-DINOv2 has a global group representation.

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

## Current repository entry points

- `notebooks/latent_playground.ipynb`: reconstruction and latent editing
  interface. Use it for decoder closure and visual inspection.
- `notebooks/dinov2_token_diagnostic.ipynb`: encoder-side token correspondence,
  orbit alignment, Procrustes, and group-consistency diagnostics.
- `notebooks/rae_layerwise_playground.ipynb`: simple per-image and batched
  layerwise visualizations for DINOv2, MAE, and SigLIP2.
- `experiments/rae_layerwise_imagenet_study.py`: reproducible larger-scale
  layerwise study.

## Recommended next experiment

Do not continue decoder inverse-adapter reconstruction training as the primary
method. First finish the held-out generator-only group test: fit `C_90` (and
optionally `C_flip_h`) on training images, predict `rot180`, `rot270`, and D4
relations on test images, and compare functional errors with independently
fitted maps. If this fails, the method direction should be a light,
position-aware adapter or an intermediate-layer tokenizer, rather than forcing
global group action in the final RAE latent.
