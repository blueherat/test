#!/usr/bin/env python3
"""CPU-only within/between-class traces from the original paired 2x5K banks.

No new features, covariance matrices, model calls, fitted scale, or sampling.
All moment arithmetic is FP64 on stored FP16 raw normalized latent values.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_fullbank_block_means import protocol_arrays
from experiments.extract_raev2_decoder_audit_real_blocks import (
    DeterministicImageNetPacked, atomic_json, sha256, verify_source_order,
)

DATA = Path('/home/zhoushunyu/data/eqvae/experiments')
NAMES = ('real', 'full', 'ig')
FILES = ('real', 'scale_s1p000000', 'scale_s1p780000')
PAIRS = ((0, 1, 'full_minus_real'), (0, 2, 'ig_minus_real'), (1, 2, 'ig_minus_full'))
M, DIMENSION = 5, 1024*16*16
PROTOCOL = 'raev2_raw_latent_within_between_class_moments_v1'


def squared(value):
    flat = value.reshape(-1)
    return float(np.dot(flat, flat))


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else None


def summarize(mask, class_stats, mean_sums):
    count = int(mask.sum())
    samples = count*M
    means = mean_sums/count
    norms = np.array([squared(value) for value in means])
    within = class_stats['within'][mask].mean(0)
    mean_square_sum = class_stats['mean_squared_norm'][mask].sum(0)
    scatter = (mean_square_sum-count*norms)/(count-1)
    corrected = scatter-within/M
    total = (class_stats['raw_squared_sum'][mask].sum(0)-samples*norms)/(samples-1)
    between_factor = M*(count-1)/(samples-1)
    reconstructed = within+between_factor*corrected
    if not np.allclose(total, reconstructed, rtol=1e-11, atol=1e-8):
        raise AssertionError('pooled trace differs from exact within/between ANOVA identity')
    branches = {}
    for index, name in enumerate(NAMES):
        branches[name] = {
            'pooled_covariance_trace_ddof1': float(total[index]),
            'unbiased_within_class_covariance_trace': float(within[index]),
            'class_centroid_scatter_trace_observed': float(scatter[index]),
            'class_centroid_scatter_trace_finite_m_corrected': float(corrected[index]),
            'uniform_class_between_covariance_trace_corrected': float((count-1)/count*corrected[index]),
            'mean_squared_within_class_pair_distance': float(2*within[index]),
            'mean_squared_class_centroid_pair_distance_observed': float(2*scatter[index]),
            'mean_squared_class_centroid_pair_distance_corrected': float(2*corrected[index]),
            'global_mean_norm': float(np.sqrt(norms[index])),
            'anova_absolute_error': float(abs(total[index]-reconstructed[index])),
        }
    ratios = {}
    for index, name in ((1, 'full_to_real'), (2, 'ig_to_real')):
        ratios[name] = {'pooled_trace': divide(total[index], total[0]),
                        'within_trace': divide(within[index], within[0]),
                        'centroid_scatter_observed': divide(scatter[index], scatter[0]),
                        'centroid_scatter_corrected': divide(corrected[index], corrected[0])}
    delta_within = float(class_stats['delta_within'][mask].mean())
    cross_within = float(class_stats['full_ig_within_cross'][mask].mean())
    if not np.isclose(delta_within, within[1]+within[2]-2*cross_within, rtol=1e-11, atol=1e-8):
        raise AssertionError('paired within covariance identity failed')
    distances = {}
    for pair_index, (first, second, name) in enumerate(PAIRS):
        observed = float(class_stats['centroid_pair_squared_distance'][mask, pair_index].mean())
        mean_difference_squared = squared(means[second]-means[first])
        noise = delta_within if name == 'ig_minus_full' else float(within[first]+within[second])
        corrected_distance = observed-noise/M
        distances[name] = {
            'global_mean_difference_norm': float(np.sqrt(mean_difference_squared)),
            'global_mean_difference_rms_per_coordinate': float(np.sqrt(mean_difference_squared/DIMENSION)),
            'global_mean_difference_squared_finite_m_corrected': float(mean_difference_squared-noise/samples),
            'mean_class_centroid_squared_distance_observed': observed,
            'rms_class_centroid_distance_observed': float(np.sqrt(observed)),
            'finite_m_noise_subtraction': noise/M,
            'mean_class_centroid_squared_distance_corrected': corrected_distance,
            'rms_class_centroid_distance_corrected_if_nonnegative': float(np.sqrt(corrected_distance)) if corrected_distance >= 0 else None,
            'correction': 'paired Full/IG delta within trace / m' if name == 'ig_minus_full' else '(within_generated + within_real) / m; independent real images and generated noise conditional on class',
        }
    mean_cross = float(np.dot(means[1], means[2]))
    cross_centroid = (class_stats['full_ig_mean_cross'][mask].sum()-count*mean_cross)/(count-1)
    cross_centroid_corrected = cross_centroid-cross_within/M
    delta_mean_sq = class_stats['centroid_pair_squared_distance'][mask, 2].sum()
    delta_scatter = (delta_mean_sq-count*squared(means[2]-means[1]))/(count-1)
    delta_scatter_corrected = delta_scatter-delta_within/M
    changes = {
        'pooled_trace_ig_minus_full': float(total[2]-total[1]),
        'within_trace_ig_minus_full': float(within[2]-within[1]),
        'between_corrected_contribution_ig_minus_full': float(between_factor*(corrected[2]-corrected[1])),
        'between_contribution_multiplier': between_factor,
        'within_fraction_of_total_change': divide(within[2]-within[1], total[2]-total[1]),
        'centroid_corrected_fraction_of_total_change': divide(between_factor*(corrected[2]-corrected[1]), total[2]-total[1]),
    }
    paired_geometry = {
        'within_full_ig_covariance_cross_trace': cross_within,
        'within_delta_covariance_trace': delta_within,
        'within_pair_difference_alignment': divide(cross_within, np.sqrt(within[1]*within[2])),
        'within_pair_distance_rms_ratio_ig_full': float(np.sqrt(within[2]/within[1])),
        'mean_squared_change_of_within_class_pair_differences': 2*delta_within,
        'class_centroid_pair_difference_alignment_observed': divide(cross_centroid, np.sqrt(scatter[1]*scatter[2])),
        'class_centroid_pair_distance_rms_ratio_ig_full_observed': float(np.sqrt(scatter[2]/scatter[1])),
        'class_centroid_cross_scatter_corrected': float(cross_centroid_corrected),
        'class_centroid_pair_alignment_noise_corrected_estimate': divide(cross_centroid_corrected, np.sqrt(corrected[1]*corrected[2])) if corrected[1] > 0 and corrected[2] > 0 else None,
        'class_centroid_pair_distance_rms_ratio_ig_full_corrected': float(np.sqrt(corrected[2]/corrected[1])) if corrected[1] > 0 and corrected[2] >= 0 else None,
        'mean_squared_change_of_class_centroid_pair_differences_observed': float(2*delta_scatter),
        'mean_squared_change_of_class_centroid_pair_differences_corrected': float(2*delta_scatter_corrected),
        'finite_sample_boundary': 'Corrected scatter/distance/alignment estimates are not clipped; they may be negative or outside cosine bounds because of estimation noise.',
    }
    return {'classes': count, 'images_per_class': M, 'samples': samples,
            'branches': branches, 'ratios_to_real': ratios, 'variance_change_decomposition': changes,
            'centroid_distances': distances, 'paired_full_ig_geometry': paired_geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-root', type=Path, default=DATA/'raev2_ig_scale_response')
    parser.add_argument('--output-dir', type=Path, default=DATA/'raev2_guidance_restart_20260906/raw_latent_class_moments_v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started, cpu_started = time.perf_counter(), time.process_time()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    source_paths = (Path(__file__), ROOT/'experiments/audit_raev2_fullbank_block_means.py',
                    ROOT/'experiments/extract_raev2_decoder_audit_real_blocks.py',
                    ROOT/'experiments/raev2_training_core.py', ROOT/'experiments/run_raev2_scale_response_study.py')
    request = {'protocol': PROTOCOL, 'seeds': [20260801, 20260802], 'dimension': DIMENSION,
        'branches': dict(zip(NAMES, FILES)), 'images_per_class': M,
        'splits': 'original 800 train / 200 heldout classes, five images per class; all-classes summary also retained',
        'real': 'saved raw normalized encoder-clean latent, not reconstruction or pixels',
        'arithmetic': 'stored FP16 promoted to FP64; classwise two-pass centering; no covariance matrix',
        'pooled_trace': 'T=sum_i ||x_i-global_mean||^2/(m*C-1)',
        'within_trace': 'W=mean_c[sum_i ||x_ci-class_mean_c||^2/(m-1)]',
        'centroid_scatter': 'B=sum_c ||class_mean_c-global_mean||^2/(C-1); corrected B*=B-W/m, never clipped',
        'anova': 'T=W+m*(C-1)/(m*C-1)*B*; reports contributions to IG-minus-Full trace',
        'finite_sample_assumption': 'Within each fixed class the five real images and five generated noises estimate class-conditional moments. Real/generated draws are independent; Full/IG are paired by common noise. Class means themselves may differ arbitrarily across labels.',
        'pair_geometry': 'Pairwise squared distances and cross products are recovered exactly from class moments. No fitted scale, whitening, clustering, or feature selection.',
        'limits': ['Classes are not identified semantic modes; traces do not establish isotropic or Gaussian heat flow.',
                   'Exact positive scalar scaling about each unchanged Full class center requires unchanged centroids and collinear within-class pair differences.',
                   'Exact contraction of class centroids about a common global center requires collinear centroid-pair differences and reduced centroid-pair scale; preserving class shapes additionally requires unchanged within-class pair differences.',
                   'These are retrospective finite-bank necessary-condition diagnostics, not quality/FID or population guarantees.'],
        'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        'device': 'cpu', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_calls': 0, 'fitted_parameters': 0, 'input_latent_bytes_expected': 2*3*5000*DIMENSION*2,
        'banks': []}
    atomic_json(out/'request.json', request)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    outputs = []
    try:
        for seed in request['seeds']:
            bank_started = time.perf_counter()
            bank = (args.bank_root/f'n5000_seed{seed}_scales7_v1').resolve()
            manifest = json.loads((bank/'manifest.json').read_text())
            if (manifest['status'] != 'complete' or manifest['seed'] != seed or manifest['samples'] != 5000
                    or manifest['world_size'] != 4 or manifest['precision'] != 'bf16'
                    or manifest['state_key'] != 'ema' or manifest['sampler_steps'] != 100
                    or manifest['ig_interval'] != [.1, 1.] or not manifest['same_noise_and_labels_across_scales']):
                raise ValueError('unexpected source sampling protocol')
            identities = protocol_arrays(bank, manifest)
            dataset = DeterministicImageNetPacked(Path(manifest['packed_data_path']), split='train',
                                                 image_size=256, horizontal_flip=False)
            try:
                source_identity = verify_source_order(manifest, dataset)
                for label, row in zip(identities['labels'], identities['source_rows'], strict=True):
                    if not 0 <= row < len(dataset):
                        raise ValueError('invalid source row')
                    file_index = bisect.bisect_right(dataset._row_offsets, int(row))-1
                    local_row = int(row)-dataset._row_offsets[file_index]
                    if int(dataset._labels[file_index][local_row]) != int(label):
                        raise ValueError('original source label does not match sample protocol')
            finally:
                dataset.close()
            maps, files = {}, []
            for name, filename in zip(NAMES, FILES):
                for rank in range(4):
                    path = bank/'latents'/f'{filename}_rank{rank:02d}.npy'
                    value = np.load(path, mmap_mode='r', allow_pickle=False)
                    if value.shape != (1250, 1024, 16, 16) or value.dtype != np.float16:
                        raise ValueError(f'unexpected source array {path}')
                    stat = path.stat()
                    files.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
                    maps[name, rank] = value
            request['banks'].append({'seed': seed, 'bank': str(bank), 'source_identity': source_identity,
                'source_labels_checked': 5000, 'manifest_sha256': sha256(bank/'manifest.json'),
                'protocol_sha256': sha256(bank/'sample_protocol.npz'), 'files': files})
            atomic_json(out/'request.json', request)
            class_stats = {name: np.empty((1000, 3), dtype=np.float64) for name in
                           ('within', 'mean_squared_norm', 'raw_squared_sum', 'centroid_pair_squared_distance')}
            class_stats.update({name: np.empty(1000, dtype=np.float64) for name in
                                ('delta_within', 'full_ig_within_cross', 'full_ig_mean_cross')})
            class_stats['labels'] = np.arange(1000, dtype=np.int64)
            class_stats['sample_ids'] = np.empty((1000, M), dtype=np.int64)
            class_stats['source_rows'] = np.empty((1000, M), dtype=np.int64)
            class_stats['test_mask'] = np.empty(1000, dtype=bool)
            sums = np.zeros((2, 3, DIMENSION), dtype=np.float64)
            digests = {name: hashlib.sha256() for name in NAMES}
            for label in range(1000):
                slots = np.where(identities['labels'] == label)[0]
                ids = identities['sample_ids'][slots]
                heldout = bool(identities['test_mask'][slots[0]])
                if len(slots) != M or not np.all(identities['test_mask'][slots] == heldout):
                    raise ValueError('class cohort/split mismatch')
                class_stats['sample_ids'][label] = ids
                class_stats['source_rows'][label] = identities['source_rows'][slots]
                class_stats['test_mask'][label] = heldout
                means, centered = [], []
                for branch, name in enumerate(NAMES):
                    block = np.empty((M, DIMENSION), dtype=np.float64)
                    for offset, sample_id in enumerate(ids):
                        stored = np.asarray(maps[name, int(sample_id)%4][int(sample_id)//4])
                        digests[name].update(stored.tobytes())
                        block[offset] = stored.reshape(-1)
                    if not np.isfinite(block).all():
                        raise FloatingPointError('nonfinite source latent')
                    mean = block.mean(0)
                    deviations = block-mean
                    class_stats['within'][label, branch] = squared(deviations)/(M-1)
                    class_stats['raw_squared_sum'][label, branch] = squared(block)
                    class_stats['mean_squared_norm'][label, branch] = squared(mean)
                    sums[int(heldout), branch] += mean
                    means.append(mean)
                    centered.append(deviations)
                for pair_index, (first, second, _) in enumerate(PAIRS):
                    class_stats['centroid_pair_squared_distance'][label, pair_index] = squared(means[second]-means[first])
                class_stats['delta_within'][label] = squared(centered[2]-centered[1])/(M-1)
                class_stats['full_ig_within_cross'][label] = float(np.einsum('nd,nd->', centered[1], centered[2]))/(M-1)
                class_stats['full_ig_mean_cross'][label] = float(np.dot(means[1], means[2]))
                if (label+1) % 50 == 0:
                    progress = {'status': 'extracting', 'seed': seed, 'classes_complete': label+1,
                                'classes_total': 1000, 'elapsed_seconds': time.perf_counter()-started, 'model_calls': 0}
                    atomic_json(out/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            for record in files:
                stat = Path(record['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
                    raise ValueError('source cache changed during extraction')
            mask = class_stats['test_mask']
            if mask.sum() != 200 or any(not np.isfinite(value).all() for value in class_stats.values()):
                raise ValueError('invalid class moments or split')
            analyses = {'train': summarize(~mask, class_stats, sums[0]),
                        'heldout': summarize(mask, class_stats, sums[1]),
                        'all': summarize(np.ones(1000, dtype=bool), class_stats, sums.sum(0))}
            path = out/f'seed{seed}_class_statistics.npz'
            with path.open('wb') as handle:
                np.savez_compressed(handle, **class_stats)
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(class_stats) or any(not np.array_equal(saved[key], value) for key, value in class_stats.items()):
                    raise AssertionError('class statistics serialization mismatch')
            result = {'seed': seed, 'splits': analyses, 'class_statistics': {'path': str(path), 'sha256': sha256(path)},
                'latent_content_sha256': {name: digest.hexdigest() for name, digest in digests.items()},
                'content_hash_order': 'ascending class, then ascending original global sample ID within class; original FP16 CHW row bytes',
                'source_label_checks': 5000, 'latent_rows_read': 15000, 'finite_and_serialization_checks_passed': True,
                'wall_seconds': time.perf_counter()-bank_started}
            atomic_json(out/f'seed{seed}_analysis.json', result)
            outputs.append(result)
            atomic_json(out/'partial_summary.json', {'status': 'running', 'outputs': outputs})
            del maps, sums, class_stats, centered, means, block
        summary = {'protocol': PROTOCOL, 'complete': True, 'outputs': outputs,
            'request_sha256': sha256(out/'request.json'), 'runner_sha256': sha256(Path(__file__)),
            'total_wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu_started,
            'max_resident_memory_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'input_latent_bytes_read': request['input_latent_bytes_expected'], 'model_calls': 0,
            'covariance_matrices_built': 0, 'fitted_scales': 0, 'fid': False,
            'boundary': 'Conditional moment diagnostic only; five images per class, retrospective banks, no semantic mode identity or heat-flow/quality guarantee.'}
        atomic_json(out/'summary.json', summary)
        atomic_json(out/'progress.json', {'status': 'complete', 'complete': True,
                    'total_wall_seconds': summary['total_wall_seconds'], 'model_calls': 0})
        print(json.dumps({'complete': True, 'summary': str(out/'summary.json'),
                          'total_wall_seconds': summary['total_wall_seconds']}), flush=True)
    except BaseException as exc:
        atomic_json(out/'progress.json', {'status': 'failed', 'error': repr(exc),
                    'completed_seeds': [item['seed'] for item in outputs], 'elapsed_seconds': time.perf_counter()-started})
        raise


if __name__ == '__main__':
    main()
