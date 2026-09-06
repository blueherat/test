"""Encode disjoint ranges of a preallocated official FP32->FP16 real bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_raev2_calibration_cache import (
    DEFAULT_CONFIG, DeterministicImageNetPacked, install_raev2_decoder_config_compat,
    instantiate_from_config, load_config)
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    root = DATA/'real_ratio_bank64k'
    folder = root/args.split
    metadata = np.load(folder/'metadata.npz')
    rows, labels = metadata['rows'], metadata['labels']
    selection = json.loads((root/'selection.json').read_text())
    os.environ['DINOV3_CKPT_DIR'] = str(DATA.parent.parent/'models/RAEv2/encoders/dinov3')
    os.environ['DINOV3_REPO_DIR'] = str(DATA.parent.parent/'models/RAEv2/dinov3_repo')
    install_raev2_decoder_config_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dataset = DeterministicImageNetPacked(Path(selection['packed_path']), split='train', image_size=256,
                                         horizontal_flip=False, index_map_path=None)
    rae = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_1)
    del rae.decoder
    rae = rae.float().cuda().eval().requires_grad_(False)
    assert rae.do_normalization and rae.eps == 1e-5 and rae.noise_tau == 0
    for tensor in [*rae.parameters(), *rae.buffers(), rae.latent_mean, rae.latent_var]:
        if tensor is not None and tensor.is_floating_point():
            assert tensor.dtype == torch.float32
    old_root = DATA.parent/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/train'
    old_meta = np.load(old_root/'metadata.npz')
    old_latents = np.load(old_root/'latents.npy', mmap_mode='r')
    anchor = []
    for index in range(8):
        image, label, row = dataset[int(old_meta['rows'][index])]
        assert label == old_meta['labels'][index] and row == old_meta['rows'][index]
        anchor.append(image)
    encoded = rae.encode(torch.stack(anchor).cuda()).cpu().half().numpy()
    assert np.array_equal(encoded, old_latents[:8]), 'original FP32 encoder/B8 stored-latent parity failed'
    latents = np.load(folder/'latents.npy', mmap_mode='r+')
    image_hashes = np.load(folder/'image_hashes.npy', mmap_mode='r+')
    digest = hashlib.sha256()
    began, count = time.perf_counter(), 0
    for batch in range(args.shard, len(rows)//8, 4):
        begin, end = batch*8, batch*8+8
        images = []
        for index in range(begin, end):
            image, label, row = dataset[int(rows[index])]
            assert label == labels[index] and row == rows[index]
            assert image.dtype == torch.float32 and tuple(image.shape) == (3, 256, 256)
            image_hashes[index] = hashlib.sha256(image.numpy().tobytes()).hexdigest().encode()
            images.append(image)
        fresh = rae.encode(torch.stack(images).cuda()).cpu()
        assert fresh.dtype == torch.float32 and torch.isfinite(fresh).all()
        half = fresh.half().numpy()
        assert np.isfinite(half).all()
        # Each worker owns complete, disjoint global B8 ranges. The parent
        # created the NPY header once before any worker opened the mapping.
        latents[begin:end] = half
        digest.update(half.tobytes())
        count += 8
        if count % 256 == 0:
            progress = {'count': count, 'seconds': time.perf_counter()-began}
            (folder/f'shard{args.shard}_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps(progress), flush=True)
    latents.flush(); image_hashes.flush()
    dataset.close()
    summary = {'complete': True, 'count': count, 'shard': args.shard,
               'seconds': time.perf_counter()-began, 'original_encoder_anchor_bitwise_equal': True,
               'owned_latent_slices_sha256': digest.hexdigest(), 'source_sha256': sha(Path(__file__).resolve()),
               'sampling_or_fid': False, 'precision': 'FP32 encoder, TF32 off, official normalization once, FP16 storage'}
    (folder/f'shard{args.shard}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
