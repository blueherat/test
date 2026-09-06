#!/usr/bin/env python3
"""Read-only CPU class-moment audit of existing paired RAEv2 Inception caches.

No model, fitted transformation, candidate scale, sampling, or new FID.
The reconstruction branch distinguishes D(E(x)) from original source images.
"""
from __future__ import annotations

import time

PROCESS_START, CPU_START = time.perf_counter(), time.process_time()

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments')
NAMES = ('source', 'reconstruction', 'full', 'ig')
FILES = ('source', 'real', 'scale_s1p000000', 'scale_s1p780000')
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SEEDS = (20260801, 20260802)
M, DIMENSION = 5, 2048
PROTOCOL = 'raev2_decoded_inception_class_moments_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': sha256(path)}


def save_json(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def norm2(value, axis=None):
    return np.sum(value * value, axis=axis, dtype=np.float64)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()),
            'descriptive_class_sem': float(values.std(ddof=1) / np.sqrt(len(values))),
            'negative_classes': int((values < 0).sum()), 'classes': len(values)}


def ratio(a, b):
    return float(a / b) if b != 0 else None


def class_statistics(arrays, labels):
    """Arrays have original global-ID order. No class ordering is inferred."""
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.arange(1000)) or not np.all(counts == M):
        raise ValueError('expected exactly five observations of each fixed class')
    order = np.argsort(labels, kind='stable')
    means, within, centered = [], [], []
    for value in arrays:
        block = value[order].reshape(1000, M, DIMENSION)
        mean = block.mean(axis=1)
        residual = block - mean[:, None, :]
        means.append(mean)
        within.append(norm2(residual, axis=(1, 2)) / (M - 1))
        centered.append(residual)
    result = {'class_means': np.stack(means, axis=1),
              'within': np.stack(within, axis=1),
              'sample_ids_by_class': order.reshape(1000, M)}
    for first, second in PAIRS:
        key = f'{NAMES[second]}_minus_{NAMES[first]}'
        result[key + '_within_delta'] = norm2(centered[second] - centered[first], axis=(1, 2)) / (M - 1)
        result[key + '_within_cross'] = np.sum(centered[first] * centered[second], axis=(1, 2)) / (M - 1)
        np.testing.assert_allclose(result[key + '_within_delta'],
            within[first] + within[second] - 2 * result[key + '_within_cross'], rtol=1e-12, atol=1e-10)
    return result


def summarize(stats, mask, arrays, labels):
    count = int(mask.sum())
    samples = M * count
    means = stats['class_means'][mask]
    global_means = means.mean(axis=0)
    mean_residuals = means - global_means
    within = stats['within'][mask].mean(axis=0)
    between = norm2(mean_residuals, axis=(0, 2)) / (count - 1)
    corrected = between - within / M
    factor = M * (count - 1) / (samples - 1)
    # Independent direct pooled calculation checks the complete class identity.
    image_mask = mask[labels]
    pooled = np.array([value[image_mask].var(axis=0, ddof=1).sum() for value in arrays])
    np.testing.assert_allclose(pooled, within + factor * corrected, rtol=1e-12, atol=1e-10)
    branches = {name: {'pooled_trace': float(pooled[j]), 'within_trace': float(within[j]),
        'centroid_scatter_observed': float(between[j]), 'centroid_scatter_corrected': float(corrected[j]),
        'uniform_class_between_trace_corrected': float((count - 1) / count * corrected[j]),
        'anova_absolute_error': float(abs(pooled[j] - within[j] - factor * corrected[j]))}
        for j, name in enumerate(NAMES)}
    ratios = {reference: {name: {'pooled_trace': ratio(pooled[j], pooled[r]),
        'within_trace': ratio(within[j], within[r]),
        'centroid_scatter_corrected': ratio(corrected[j], corrected[r])}
        for j, name in enumerate(NAMES) if j != r}
        for r, reference in ((0, 'source'), (1, 'reconstruction'))}
    distances = {}
    for first, second in PAIRS:
        key = f'{NAMES[second]}_minus_{NAMES[first]}'
        observed = norm2(means[:, second] - means[:, first], axis=1)
        paired = (first, second) in ((0, 1), (2, 3))
        noise_by_class = (stats[key + '_within_delta'][mask] if paired else
                         stats['within'][mask, first] + stats['within'][mask, second])
        corrected_distance = observed - noise_by_class / M
        global_distance = float(norm2(global_means[second] - global_means[first]))
        distances[key] = {'centroid_squared_distance_observed': describe(observed),
            'centroid_squared_distance_corrected': describe(corrected_distance),
            'finite_m_noise_subtraction_mean': float(noise_by_class.mean() / M),
            'global_mean_squared_distance': global_distance,
            'global_mean_squared_distance_corrected': global_distance - float(noise_by_class.mean() / samples),
            'noise_correction': 'paired within-difference trace / m' if paired else 'sum of independent branch within traces / m'}
    w_cross = float(stats['ig_minus_full_within_cross'][mask].mean())
    b_cross = float(np.sum(mean_residuals[:, 2] * mean_residuals[:, 3]) / (count - 1))
    b_cross_corrected = b_cross - w_cross / M
    changes = {'pooled_ig_minus_full': float(pooled[3] - pooled[2]),
        'within_ig_minus_full': float(within[3] - within[2]),
        'between_corrected_contribution_ig_minus_full': float(factor * (corrected[3] - corrected[2])),
        'between_multiplier': factor,
        'within_change_by_class': describe(stats['within'][mask, 3] - stats['within'][mask, 2]),
        'within_full_ig_pair_alignment': ratio(w_cross, np.sqrt(within[2] * within[3])),
        'centroid_full_ig_alignment_corrected': ratio(b_cross_corrected, np.sqrt(corrected[2] * corrected[3]))
            if corrected[2] > 0 and corrected[3] > 0 else None}
    for reference in ('source', 'reconstruction'):
        r = NAMES.index(reference)
        f = norm2(means[:, 2] - means[:, r], axis=1) - (stats['within'][mask, 2] + stats['within'][mask, r]) / M
        g = norm2(means[:, 3] - means[:, r], axis=1) - (stats['within'][mask, 3] + stats['within'][mask, r]) / M
        changes['centroid_risk_ig_minus_full_relative_to_' + reference] = describe(g - f)
        for quantity, values in (('pooled', pooled), ('within', within), ('between_corrected', corrected)):
            changes[quantity + '_absolute_error_relative_to_' + reference] = {
                'full': float(abs(values[2] - values[r])), 'ig': float(abs(values[3] - values[r]))}
        delta = means[:, 3] - means[:, 2]
        cross_noise = stats['ig_minus_full_within_cross'][mask] - stats['within'][mask, 2]
        cross = np.sum((means[:, 2] - means[:, r]) * delta, axis=1) - cross_noise / M
        delta_risk = norm2(delta, axis=1) - stats['ig_minus_full_within_delta'][mask] / M
        np.testing.assert_allclose(g - f, 2 * cross + delta_risk, rtol=1e-11, atol=1e-10)
        changes['centroid_risk_decomposition_relative_to_' + reference] = {
            'twice_error_shift_cross_corrected': describe(2 * cross),
            'shift_norm_squared_corrected': describe(delta_risk)}
    reference_effect = -2 * np.sum((means[:, 3] - means[:, 2]) * (means[:, 0] - means[:, 1]), axis=1)
    risk_source = changes['centroid_risk_ig_minus_full_relative_to_source']['mean']
    risk_recon = changes['centroid_risk_ig_minus_full_relative_to_reconstruction']['mean']
    np.testing.assert_allclose(risk_source - risk_recon, reference_effect.mean(), rtol=1e-11, atol=1e-10)
    changes['centroid_risk_reference_effect_source_minus_reconstruction'] = describe(reference_effect)
    return {'classes': count, 'samples': samples, 'branches': branches,
            'ratios': ratios, 'distances': distances, 'ig_minus_full': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA / 'raev2_guidance_restart_20260906/decoded_class_moments_v1')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    request = {'protocol': PROTOCOL, 'seeds': SEEDS, 'branches': dict(zip(NAMES, FILES)),
        'dimension': DIMENSION, 'images_per_class': M, 'fit_or_hyperparameter_search': False,
        'question': 'Does IG decoded pooled trace calibration arise from increased between-class structure rather than within-class recovery? Compare the full paired change to previously audited raw-latent results.',
        'reference': 'source=original RGB crop Inception; reconstruction=Inception of D(E(source)); generated Full/IG have common noise, not source-image noise',
        'arithmetic': 'all original float32 cache entries promoted to float64; no whitening, clipping corrected statistics, covariance regularization or fitted transformation',
        'splits': 'all 1000 fixed classes plus original train800/heldout200, five images/class; retrospective reuse',
        'identity': 'global_id = rank + 4*local_row; original manifest and sample_protocol are authoritative',
        'anova': 'T=W+[m(C-1)/(mC-1)]*(B-W/m); exact sample identity, no PSD claim for corrected B',
        'estimation': 'W/m removes independent class-mean sampling noise; paired Full/IG and source/reconstruction differences retain their cross covariance. Fixed-label superpopulation sampling assumption, not a finite-ImageNet-population risk certificate.',
        'sem': 'descriptive heterogeneity across fixed classes; no exact conditional sampling CI or multiple-testing claim',
        'limits': ['traces do not determine covariance orientation, FID, density, semantic modes, causal suffix effects, or a usable guidance direction',
                   'decoder effect includes clamp, uint8 conversion and Inception; raw encoder latent is a different geometry',
                   'historical decode uses float() before clamp/multiply255; do not treat these banks as current native-BF16 sampling baselines'],
        'source_records': [record(Path(__file__)), record(ROOT/'experiments/run_raev2_scale_response_study.py')],
        'device': 'cpu', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_calls': 0, 'new_fid': False, 'banks': []}
    raw_audit = DATA/'raev2_guidance_restart_20260906/raw_latent_class_moments_v1'
    raw_request = json.loads((raw_audit/'request.json').read_text())
    request['prior_raw_identity_audit'] = record(raw_audit/'request.json')
    request['source_identity_scope'] = 'Recheck hashes of original manifest/protocol against completed raw audit; reuse its 10000 source-label checks rather than reread packed RGB. Compare IDs, labels, split and source rows against its saved class tables.'
    save_json(out/'request.json', request)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    outputs = []
    for seed in SEEDS:
        bank = DATA / 'raev2_ig_scale_response' / f'n5000_seed{seed}_scales7_v1'
        manifest = json.loads((bank/'manifest.json').read_text())
        if not (manifest['status'] == 'complete' and manifest['seed'] == seed and manifest['samples'] == 5000
                and manifest['world_size'] == 4 and manifest['sampler_steps'] == 100
                and manifest['ig_interval'] == [.1, 1.] and manifest['precision'] == 'bf16'
                and manifest['state_key'] == 'ema' and manifest['same_noise_and_labels_across_scales']):
            raise ValueError('unexpected historical sampling protocol')
        with np.load(bank/'sample_protocol.npz', allow_pickle=False) as data:
            protocol = {key: data[key] for key in data.files}
        ids, labels, heldout = (protocol[key] for key in ('sample_ids', 'labels', 'test_mask'))
        if not np.array_equal(ids, np.arange(5000)) or not np.array_equal(labels, ids % 1000):
            raise ValueError('original identity mismatch')
        expected = np.isin(labels, np.random.default_rng(seed + 17).permutation(1000)[:200])
        if not np.array_equal(heldout, expected):
            raise ValueError('original heldout split mismatch')
        info = {'seed': seed, 'manifest': record(bank/'manifest.json'),
                'sample_protocol': record(bank/'sample_protocol.npz'), 'features': []}
        previous = next(item for item in raw_request['banks'] if item['seed'] == seed)
        if (info['manifest']['sha256'] != previous['manifest_sha256'] or
                info['sample_protocol']['sha256'] != previous['protocol_sha256']):
            raise ValueError('original identities differ from completed raw audit')
        arrays = []
        for filename in FILES:
            values = np.empty((5000, DIMENSION), dtype=np.float64)
            for rank in range(4):
                path = bank/'inception'/f'{filename}_rank{rank:02d}.npy'
                rec = record(path)
                stored = np.load(path, allow_pickle=False)
                if stored.shape != (1250, DIMENSION) or stored.dtype != np.float32 or not np.isfinite(stored).all():
                    raise ValueError(f'invalid feature cache {path}')
                values[rank::4] = stored
                if path.stat().st_size != rec['bytes'] or path.stat().st_mtime_ns != rec['mtime_ns']:
                    raise ValueError('input mutated during read')
                info['features'].append(rec)
            arrays.append(values)
        request['banks'].append(info)
        save_json(out/'request.json', request)
        stats = class_statistics(arrays, labels)
        stats['labels'] = np.arange(1000)
        stats['test_mask'] = heldout[:1000]
        stats['source_rows_by_class'] = protocol['real_source_rows'][stats['sample_ids_by_class']]
        previous_path = raw_audit/f'seed{seed}_class_statistics.npz'
        info['prior_raw_class_statistics'] = record(previous_path)
        with np.load(previous_path, allow_pickle=False) as previous_stats:
            for key, old_key in (('labels', 'labels'), ('test_mask', 'test_mask'),
                    ('sample_ids_by_class', 'sample_ids'), ('source_rows_by_class', 'source_rows')):
                if not np.array_equal(stats[key], previous_stats[old_key]):
                    raise ValueError(f'raw/decoded cohort mismatch: {key}')
        save_json(out/'request.json', request)
        splits = {name: summarize(stats, mask, arrays, labels) for name, mask in (
            ('train', ~stats['test_mask']), ('heldout', stats['test_mask']), ('all', np.ones(1000, dtype=bool)))}
        path = out/f'seed{seed}_class_statistics.npz'
        np.savez_compressed(path, **stats)
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(stats) or any(not np.array_equal(saved[key], val) for key, val in stats.items()):
                raise AssertionError('serialized sufficient statistics differ')
        result = {'seed': seed, 'splits': splits, 'class_statistics': record(path)}
        outputs.append(result)
        save_json(out/f'seed{seed}_analysis.json', result)
        print(json.dumps({'seed': seed, 'all': splits['all']['ig_minus_full']}), flush=True)
    summary = {'protocol': PROTOCOL, 'status': 'complete', 'outputs': outputs,
        'request': record(out/'request.json'), 'runner': record(out/'runner_source.py'),
        'cost': {'cpu_only': True, 'model_calls': 0, 'new_images': 0, 'new_fid': False,
                 'feature_payload_bytes': 2 * 4 * 5000 * DIMENSION * 4,
                 'wall_seconds_from_script_start': time.perf_counter() - PROCESS_START,
                 'cpu_seconds_from_script_start': time.process_time() - CPU_START,
                 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'timing_excludes': 'interpreter startup before time import and final summary serialization/exit'},
        'goal_achieved': False}
    save_json(out/'summary.json', summary)
    print(json.dumps({'summary': str(out/'summary.json'), 'cost': summary['cost']}), flush=True)


if __name__ == '__main__':
    main()
