# Spectral preconditioning toy data

This directory contains the six raw generation CSV files and six raw teacher
CSV files used by the three-seed summary in `summary.csv`.

The linear runs use `curvature=0.0`; the curved runs use `curvature=0.5`.
Every run uses `D=64`, output rank 16, 5,000 training steps, 4,096 generated
samples, and matched initialization/random streams within a seed. Teacher MSE
in `summary.csv` is averaged over five evaluation times and three seeds.

Source hashes before copying:

The archived copies normalize line endings from CRLF to LF; numeric/textual
CSV content is unchanged, so their Git hashes intentionally differ from these
source-file hashes.

| file | SHA-256 |
|---|---|
| `linear_seed20260821.csv` | `e35c60298fb56d44e73d22c7b99b6d0cb4114719481cd0f78021872f5707fb0f` |
| `linear_seed20260822.csv` | `8edc7420e0daaaa701018f754f3f92d66ba152082e8b5952ad43dab01974ca33` |
| `linear_seed20260823.csv` | `e4099eba88a8d3c1defae20b8fe48268dcae38422525773258addfb27797cd6c` |
| `curved_seed20260821.csv` | `6597fbcef2ad5ef3cbde5282841fadf5fe3bec0d329013a739451d0a1c20d9e7` |
| `curved_seed20260822.csv` | `d0b2fa1f3cdaf69f54a56fc73095d5affd78187aae0f322dfa55a5a18b12c4c6` |
| `curved_seed20260823.csv` | `73559d6022c0b39836e2fa11c6c0139e08e27c5a4b5b672378c410b4f248aa73` |
| `linear_teacher_seed20260821.csv` | `840521c039bf0668132120ded7d227fdec0ec9c3c2d9ce4717b4725ef3b6eb4e` |
| `linear_teacher_seed20260822.csv` | `73d6217bc4b9d25c034a77fd101610af033c27c252627002e7e7ceaad17956b7` |
| `linear_teacher_seed20260823.csv` | `ffc8879b5e01ec316bd1a51ffc17295ba91bcafacc87a510fd079ccfb556fdbe` |
| `curved_teacher_seed20260821.csv` | `92fe0186051c8f81481fe8b2b1095b6d8b8e70d957af4b896e0714b03cdfd4e4` |
| `curved_teacher_seed20260822.csv` | `b60aa22de62930e67344f615b068980bed7789ca06a4e1be33f3e6e2236cc410` |
| `curved_teacher_seed20260823.csv` | `cd9154a1ac2a651df39e5acbb8d8f860a1463c6fd6aad28d22cd19b34e14b7ae` |

The original manifests, plots, and training histories remain outside Git under
`$EQVAE_DATA_ROOT/experiments/`.
