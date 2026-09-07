#!/usr/bin/env python3
"""CPU diagnostic of fixed 16-pixel decoder phases on historical images only.

No model, FID, filter fitting, image modification, or new extension-image reads.
This measures a possible structural signature, not a quality objective.
"""
from __future__ import annotations

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
from experiments.raev2_training_core import DeterministicImageNetPacked
from experiments.extract_raev2_decoder_audit_real_blocks import selected_protocol, verify_source_order

DATA = Path('/home/zhoushunyu/data/eqvae/experiments')
OUTPUT = DATA / 'raev2_guidance_restart_20260906/decoder_patch_phase_audit_v1'
SEEDS = [20260801, 20260802]


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def phase_statistics(image):
    image = np.asarray(image, dtype=np.float64)
    assert image.shape == (256, 256, 3) and np.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    result = []
    for axis in (1, 0):  # horizontal, vertical; phase is the left/upper pixel
        x = image if axis == 1 else image.transpose(1, 0, 2)
        # Every phase occurs exactly 14 times; omit outer 16 pixels.
        first = x[16:240, 17:241] - x[16:240, 16:240]
        second = x[16:240, 17:241] - 2*x[16:240, 16:240] + x[16:240, 15:239]
        for difference in (first, second):
            energy = np.square(difference).mean(axis=(0, 2)).reshape(14, 16).mean(axis=0)
            result.append(energy)
    return np.asarray(result)


def summarize(rows):
    # rows: image, [dx,dxx,dy,dyy], phase.
    total = rows.mean(axis=0)
    curve = total / np.maximum(total.mean(axis=1, keepdims=True), np.finfo(float).tiny)
    normalized = rows / np.maximum(rows.mean(axis=2, keepdims=True), np.finfo(float).tiny)
    result = {}
    for i, name in enumerate(('dx', 'dxx', 'dy', 'dyy')):
        boundary = [15] if i % 2 == 0 else [0, 15]
        interior = [j for j in range(16) if j not in boundary]
        contrast = normalized[:, i, boundary].mean(axis=1) - normalized[:, i, interior].mean(axis=1)
        result[name] = {'phase_mean_energy': total[i].tolist(), 'normalized_phase_curve': curve[i].tolist(),
                        'pooled_boundary_minus_interior': float(curve[i, boundary].mean()-curve[i, interior].mean()),
                        'image_mean_boundary_minus_interior': float(contrast.mean()),
                        'image_median_boundary_minus_interior': float(np.median(contrast)),
                        'phase_rms_deviation': float(np.sqrt(np.mean((curve[i]-1)**2)))}
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'refusing overwrite: {OUTPUT}')
    torch.set_num_threads(1)
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True)
    request = {'protocol': 'raev2_historical_decoder_patch_phase_v1', 'seeds': SEEDS,
               'script': artifact(__file__), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'selection': 'Fixed first global sample_id per class in each existing 5K bank; all 1000 classes.',
               'phases': 16, 'interior': '[16:240) with 14 full periods; first and second pixel differences.',
               'first_difference_boundary_phase': [15], 'second_difference_boundary_phases': [0, 15],
               'hypothesis': 'Generated decoded images may show an excess signature tied to the known decoder patch lattice.',
               'scope': 'Descriptive diagnostic only; no FID, fitting, thresholds, filters, or inferred quality gain.',
               'limitations': ['Original decoded banks store clamped FP16 pixels, not original FP32 or official uint8.',
                              'Real source images are uint8/255; JPEG and resizing can themselves induce phase structure.',
                              'A phase signature does not identify the decoder as its cause or imply FID improvement after removing it.']}
    write_json(OUTPUT/'request.json', request)
    banks = []
    for seed in SEEDS:
        bank = DATA / f'raev2_ig_scale_response/n5000_seed{seed}_scales7_v1'
        manifest_path = bank/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete' and manifest['seed'] == seed
        assert manifest['world_size'] == 4 and manifest['samples'] == 5000
        arrays = selected_protocol(bank, manifest)
        files = [manifest_path, bank/'sample_protocol.npz']
        decoded = {}
        for name, prefix in [('reconstruction', 'real'), ('generated_ig1p78', 'scale_s1p780000')]:
            decoded[name] = []
            for rank in range(4):
                path = bank/'decoded'/f'{prefix}_rank{rank:02d}.npy'
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                assert value.shape == (1250, 256, 256, 3) and value.dtype == np.float16
                decoded[name].append(value)
                files.append(path)
        dataset = DeterministicImageNetPacked(Path(manifest['packed_data_path']), split='train', image_size=256, horizontal_flip=False)
        source_order = verify_source_order(manifest, dataset)
        statistics = {name: np.empty((1000, 4, 16), dtype=np.float64)
                      for name in ('source_real', 'reconstruction', 'generated_ig1p78')}
        pixel_hashes = {name: hashlib.sha256() for name in statistics}
        try:
            for i, (sample_id, label, source_row) in enumerate(zip(arrays['sample_ids'], arrays['labels'], arrays['source_rows'], strict=True)):
                image, observed_label, observed_row = dataset[int(source_row)]
                assert observed_label == label and observed_row == source_row
                original = image.numpy().transpose(1, 2, 0)
                # Original runner uses strided rank IDs, verified against its source.
                rank, local_index = int(sample_id % 4), int(sample_id // 4)
                images = {'source_real': original, **{name: values[rank][local_index] for name, values in decoded.items()}}
                for name, pixels in images.items():
                    pixel_hashes[name].update(np.ascontiguousarray(pixels).tobytes())
                    statistics[name][i] = phase_statistics(pixels)
                if (i+1) % 250 == 0:
                    print(json.dumps({'seed': seed, 'images_per_branch': i+1}), flush=True)
        finally:
            dataset.close()
        destination = OUTPUT/f'seed{seed}_phase_statistics.npz'
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **arrays, **statistics)
        banks.append({'seed': seed, 'samples_per_branch': 1000, 'source_order': source_order,
                      'source_files': [artifact(path) for path in files], 'array_output': artifact(destination),
                      'selected_pixel_hashes': {name: digest.hexdigest() for name, digest in pixel_hashes.items()},
                      'statistics': {name: summarize(value) for name, value in statistics.items()}})
    result = {'complete': True, 'banks': banks, 'cpu_wall_seconds': time.perf_counter()-started,
              'model_or_gpu_calls': 0, 'fid_calls': 0, 'new_5k_extension_images_read': 0,
              'quality_improvement_claim': False}
    write_json(OUTPUT/'summary.json', result)
    print(json.dumps({'complete': True, 'output': str(OUTPUT), 'cpu_wall_seconds': result['cpu_wall_seconds']}))


if __name__ == '__main__':
    main()
