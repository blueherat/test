#!/usr/bin/env bash
set -euo pipefail

# Public baseline mirrors and the exact CFG-Rejection/EDM2 checkpoints.
# All large files stay outside Git under EQVAE_DATA_ROOT.

DATA_ROOT="${EQVAE_DATA_ROOT:-/home/zhoushunyu/data/eqvae}"
BASELINE_ROOT="$DATA_ROOT/baselines"
EDM2_ROOT="$BASELINE_ROOT/edm2"
CHECKPOINT_ROOT="$EDM2_ROOT/checkpoints"
ADM_ROOT="$BASELINE_ROOT/guided-diffusion"
ADM_CHECKPOINT_ROOT="$ADM_ROOT/checkpoints"
FKC_ROOT="$BASELINE_ROOT/fkc-diffusion"
FKC_CHECKPOINT_ROOT="$FKC_ROOT/applications/images/edm2/checkpoints"
DIT_ROOT="$BASELINE_ROOT/DiT"
DIT_CHECKPOINT_ROOT="$DIT_ROOT/pretrained_models"

mkdir -p "$BASELINE_ROOT"

download_if_incomplete() {
  local destination="$1"
  local expected_bytes="$2"
  local url="$3"
  if [[ -f "$destination" ]] && [[ "$(stat -c '%s' "$destination")" == "$expected_bytes" ]]; then
    echo "already complete: $destination"
    return
  fi
  mkdir -p "$(dirname "$destination")"
  curl -fL --continue-at - --output "$destination" "$url"
}

if [[ ! -d "$BASELINE_ROOT/CFG-Rejection/.git" ]]; then
  git clone --depth 1 https://github.com/WSX20003/CFG-Rejection.git "$BASELINE_ROOT/CFG-Rejection"
fi

if [[ ! -d "$EDM2_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/NVlabs/edm2.git "$EDM2_ROOT"
fi

if [[ ! -d "$BASELINE_ROOT/fkc-diffusion/.git" ]]; then
  git clone --depth 1 https://github.com/martaskrt/fkc-diffusion.git "$BASELINE_ROOT/fkc-diffusion"
fi

if [[ ! -d "$BASELINE_ROOT/Self-Guidance/.git" ]]; then
  git clone --depth 1 https://github.com/maple-research-lab/Self-Guidance.git "$BASELINE_ROOT/Self-Guidance"
fi

if [[ ! -d "$BASELINE_ROOT/Radon-Nikodym-Estimator/.git" ]]; then
  git clone --depth 1 https://github.com/jiajunhe98/Radon-Nikodym-Estimator.git \
    "$BASELINE_ROOT/Radon-Nikodym-Estimator"
fi

if [[ ! -d "$BASELINE_ROOT/DiT/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/DiT.git "$BASELINE_ROOT/DiT"
fi

if [[ ! -d "$ADM_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/openai/guided-diffusion.git "$ADM_ROOT"
fi

download_if_incomplete \
  "$CHECKPOINT_ROOT/edm2-img512-s-2147483-0.025.pkl" \
  560565890 \
  https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.025.pkl

download_if_incomplete \
  "$CHECKPOINT_ROOT/edm2-img512-xs-uncond-2147483-0.025.pkl" \
  248541796 \
  https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.025.pkl

# Exact EDM2-XS conditional/unconditional pair used by the released FKC image
# script (`edm2-img512-xs-guid-fid`, CFG 1.4, 64 steps).
download_if_incomplete \
  "$FKC_CHECKPOINT_ROOT/edm2-img512-xs-2147483-0.045.pkl" \
  249566482 \
  https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.045.pkl

download_if_incomplete \
  "$FKC_CHECKPOINT_ROOT/edm2-img512-xs-uncond-2147483-0.045.pkl" \
  248541796 \
  https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.045.pkl

download_if_incomplete \
  "$ADM_CHECKPOINT_ROOT/64x64_diffusion.pt" \
  1183736577 \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt

download_if_incomplete \
  "$ADM_CHECKPOINT_ROOT/64x64_classifier.pt" \
  261889658 \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt

download_if_incomplete \
  "$DIT_CHECKPOINT_ROOT/DiT-XL-2-256x256.pt" \
  2700611775 \
  https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt

python - "$CHECKPOINT_ROOT" "$ADM_CHECKPOINT_ROOT" "$FKC_CHECKPOINT_ROOT" "$DIT_CHECKPOINT_ROOT" <<'PY'
import hashlib
from pathlib import Path
import sys

edm_root = Path(sys.argv[1])
adm_root = Path(sys.argv[2])
fkc_root = Path(sys.argv[3])
dit_root = Path(sys.argv[4])
expected = {
    edm_root / "edm2-img512-s-2147483-0.025.pkl": (
        560_565_890,
        "aa40d3ee46f3db358df37bc4387ce90020df593e4839c1ee056203d448357239",
    ),
    edm_root / "edm2-img512-xs-uncond-2147483-0.025.pkl": (
        248_541_796,
        "42d0ca453ba0dc1efc6e8b0c6c21063b423ffa12630b8a786eb51eabef9fb442",
    ),
    adm_root / "64x64_diffusion.pt": (
        1_183_736_577,
        "a18558f9a2499615a3ff9759ad12299690ad36ee3378c395adbb94855e2b634f",
    ),
    adm_root / "64x64_classifier.pt": (
        261_889_658,
        "d5c4c240e4f0d36460f58520c2803a11490db7e5540e42d2ad75f0cf75bb3586",
    ),
    fkc_root / "edm2-img512-xs-2147483-0.045.pkl": (
        249_566_482,
        "27a6c6eaf697b68a74f9c7b72e82f91c2e898d22f629b4546c053865cfe3da68",
    ),
    fkc_root / "edm2-img512-xs-uncond-2147483-0.045.pkl": (
        248_541_796,
        "2ea8fffdf0e32d68da3b4050e77c3f9defb1a50a2c9a4a845eb8f927355dea08",
    ),
    dit_root / "DiT-XL-2-256x256.pt": (
        2_700_611_775,
        "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521",
    ),
}
for path, (size, digest) in expected.items():
    actual = path.stat().st_size
    if actual != size:
        raise SystemExit(f"bad size for {path}: {actual} != {size}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            hasher.update(chunk)
    actual_digest = hasher.hexdigest()
    if actual_digest != digest:
        raise SystemExit(f"bad SHA-256 for {path}: {actual_digest} != {digest}")
    print(f"verified {path} ({actual} bytes, sha256={actual_digest})")
PY
