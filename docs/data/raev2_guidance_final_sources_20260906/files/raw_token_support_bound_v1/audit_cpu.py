#!/usr/bin/env python3
"""Frozen all-cache raw-token support diagnostic. CPU only, no model imports."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
RESTART = OUT.parent
ROOT = Path('/home/zhoushunyu/eqvae')
HIST = RESTART/'normal_noise_audit_seed202609071'
CURRENT = RESTART/'query_mean_error_compatibility_v1'
STEPS = [0, 47, 67, 77, 84, 89, 92, 95, 97, 99]
SHAPE = (1024, 16, 16)
HEADS = ['F', 'B', 'IG_native_reconstructed']
STAT_SHA = '40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77'
CURRENT_REQ_SHA = '1f18f4374530e9c235d3c4a62bfe9349b729d20467d36194077797f229524ead'
CURRENT_SUMMARY_SHA = 'b3e49faf465b5b0a66a3ff2670505063e2ccf936219b822e47dce7d3fe754273'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f'source SHA changed: {path}')
    return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}


def put(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    start, cpu = time.perf_counter(), time.process_time()
    if (OUT/'request.json').exists():
        raise FileExistsError('request already frozen')
    sources = {}
    def add(name, path, expected=None):
        sources[name] = record(path, expected)
    add('audit_source', __file__)
    add('current_request', CURRENT/'request.json', CURRENT_REQ_SHA)
    add('current_summary', CURRENT/'summary.json', CURRENT_SUMMARY_SHA)
    req, summary = read(CURRENT/'request.json'), read(CURRENT/'summary.json')
    assert summary['complete'] is True and summary['rows'] == 80
    for name, key in [('historical_request', 'state_source_request'), ('historical_summary', 'state_source_summary')]:
        add(name, req[key]['path'], req[key]['sha256'])
    hist, hist_summary = read(HIST/'request.json'), read(HIST/'summary.json')
    assert hist['tf32'] is True and req['sample_ids'] == list(range(8)) and req['labels'] == list(range(8))
    assert hist['source_rows'][:8] == req['source_rows']
    snapshots = hist_summary['snapshots']
    assert [s['step_index'] for s in snapshots] == STEPS
    current_snapshots = {s['step_index']: s for s in req['snapshots']}
    for s in snapshots:
        old = current_snapshots[s['step_index']]
        assert s['t'] == hist['time_grid'][s['step_index']] == old['t']
        assert Path(s['path']).resolve() == Path(old['path']).resolve() and s['sha256'] == old['sha256']
        add(f'historical_snapshot_{s["step_index"]:03d}', s['path'], s['sha256'])
    outputs = {Path(row['path']).name: row for row in summary['outputs']}
    for name in ('F.npy', 'B.npy', 'per_image_risk.csv', 'paired_clean_identity.json'):
        source = outputs[name]
        add('current_'+name, source['path'], source['sha256'])
    add('normalization_stats', hist['normalization_stats_path'], STAT_SHA)
    assert hist['normalization_stats_sha256'] == STAT_SHA
    add('historical_producer', HIST/'runner_source.py', hist['source_sha256']['experiments/audit_raev2_guidance_normal_noise.py'])
    for relative in ('external/RAEv2/src/encoders/vision_encoder.py', 'external/RAEv2/src/stage1/rae.py'):
        add(relative, ROOT/relative, hist['source_sha256'][relative])
    identities = []
    for snapshot in snapshots:
        for domain in ('teacher', 'rollout'):
            for sample in range(8):
                identities.append({'row': len(identities), 'cache': 'historical', 'domain': domain,
                    'step_index': snapshot['step_index'], 't': snapshot['t'], 'sample_id': sample,
                    'label': sample, 'source_row': req['source_rows'][sample],
                    'source_file': str(Path(snapshot['path']).resolve()), 'source_array_row': sample})
    with (CURRENT/'per_image_risk.csv').open(newline='') as stream:
        metadata = list(csv.DictReader(stream))
    assert len(metadata) == 80
    for i, row in enumerate(metadata):
        step, sample = STEPS[i//8], i%8
        assert int(row['output_row']) == i and row['domain'] == 'teacher'
        assert int(row['sample_id']) == int(row['label']) == sample and int(row['step_index']) == step
        assert float(row['t']) == current_snapshots[step]['t'] and int(row['source_row']) == req['source_rows'][sample]
        identities.append({'row': len(identities), 'cache': 'current', 'domain': 'teacher',
            'step_index': step, 't': float(row['t']), 'sample_id': sample, 'label': sample,
            'source_row': int(row['source_row']), 'source_file': str(CURRENT.resolve()), 'source_array_row': i})
    assert len(identities) == 240
    request = {'protocol': 'raev2_raw_token_support_bound_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_before_any_tensor_math': True, 'sources': sources, 'rows': identities,
        'cache_scope': {'historical': 'all10 snapshots, teacher+rollout, all8 IDs; old BF16 heads promoted toFP32, TF32on, FP32 historical IG trajectory',
                        'current': 'all80 teacher heads, nativeBF16->FP32, TF32off; same8IDs/10times, not independent confirmation or current rollout'},
        'heads': HEADS, 'latent_shape': list(SHAPE), 'raw_token_radius': 64, 'raw_token_norm_squared_bound': 4*SHAPE[0],
        'raw_coordinate': 'FP64 sigma=sqrt(FP64(var)+1e-5), a=sigma*FP64(saved_or_reconstructed_head)+FP64(mean); reduce squarednorm over1024channels pertoken',
        'native_IG': 'Only after verifying every saved F/B scalar exactly equals its BF16->FP32 roundtrip, use eager CPU BF16 B+1.78*(F-B), thenFP32. Outside existing[.1,1], IG=F. These are arithmetic reconstructions from cached heads, not newly observed CUDA guided outputs.',
        'prerequisites': 'All tensors finite FP32 and exact shape/ID/time match. Check all cached F/B for losslessBF16 before any norm calculation. If any fail, save prerequisite/failure and do not relabel a rounded surrogate native.',
        'statistics': 'Save fullFP64 norm_squared[240,3,16,16] plus240row identities; perstate/head min/mean/max norm andnorm², count/fraction norm²>4096, positive norm andnorm² excess max/mean/sum. Pooled percache/domain/head, historicalall/head, andall240/head. All tokens/rows retained, no CI or independence claim.',
        'threshold': '4096 exact theoretical squaredbound, strictgreater countsoutside; no fittedradius/tolerance or numericalmargin',
        'grouping_note': 't1 teacher/rollout duplicate; sameimages reused times andcurrentprecision. Pooled240 is descriptive only.',
        'output_scope': 'No clip, projection, model, newnoise, GPU, sampling, decoder orFID; no source/model modifications.',
        'geometry_boundary': 'Support applies to idealraw clean encoder andexact posterior means, not noisy states. Normalizedcoordinates are ellipsoidal; rawradialclip does not imply normalizedEuclideanprojection.',
        'model_checkpoint_identity_inherited_not_rehashed_or_loaded': req['checkpoint'],
        'prepare_wall_seconds': time.perf_counter()-start, 'prepare_cpu_seconds': time.process_time()-cpu,
        'source_files_hashed_bytes': sum(x['bytes'] for x in sources.values())}
    put(OUT/'request.json', request)
    print(json.dumps({'prepared': True, 'request_sha256': sha(OUT/'request.json'), 'sources': len(sources),
                      'bytes_hashed': request['source_files_hashed_bytes'], 'wall_seconds': request['prepare_wall_seconds']}), flush=True)


def run():
    start, cpu = time.perf_counter(), time.process_time()
    if (OUT/'summary.json').exists() or (OUT/'failure.json').exists():
        raise FileExistsError('refusing to overwrite completed or failed math')
    request = read(OUT/'request.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('this diagnostic requires CUDA_VISIBLE_DEVICES empty')
    for rec in request['sources'].values():
        if sha(rec['path']) != rec['sha256']:
            raise ValueError('source changed after freeze: '+rec['path'])
    identity_wall = time.perf_counter()-start
    import numpy as np
    import torch
    torch.set_num_threads(4)
    batches, checks = [], []
    source = request['sources']
    # Full BF16-lattice gate precedes ALL norm calculations.
    for step in STEPS:
        p = source[f'historical_snapshot_{step:03d}']['path']
        snap = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        assert snap['step_index'] == step and snap['sample_ids'].tolist() == list(range(8)) and snap['labels'].tolist() == list(range(8))
        for domain in ('teacher', 'rollout'):
            meta = request['rows'][len(batches)*8:(len(batches)+1)*8]
            assert all(m['cache'] == 'historical' and m['domain'] == domain and m['t'] == snap['t'] for m in meta)
            batches.append((meta, snap[domain]['full'], snap[domain]['base']))
    arrays = [np.load(source['current_'+name+'.npy']['path'], mmap_mode='r', allow_pickle=False) for name in ('F', 'B')]
    assert all(a.shape == (80, *SHAPE) and a.dtype == np.float32 for a in arrays)
    for k in range(10):
        meta = request['rows'][160+k*8:168+k*8]
        batches.append((meta, *(torch.from_numpy(np.array(a[k*8:k*8+8], copy=True)) for a in arrays)))
    for meta, full, base in batches:
        for head, value in [('F', full), ('B', base)]:
            assert value.dtype == torch.float32 and value.shape == (8, *SHAPE) and bool(torch.isfinite(value).all())
            restored = value.bfloat16().float()
            per_image = (restored != value).flatten(1).sum(1).tolist()
            differences = (restored-value).abs().flatten(1).amax(1).tolist()
            checks.extend({**m, 'head': head, 'not_exact_bf16_coordinates': int(n),
                           'BF16_roundtrip_max_abs_difference': float(d)} for m, n, d in zip(meta, per_image, differences))
    lattice = {'complete': True, 'head_rows': len(checks),
               'checked_coordinates': len(checks)*int(np.prod(SHAPE)),
               'all_exact': all(row['not_exact_bf16_coordinates'] == 0 for row in checks), 'rows': checks}
    put(OUT/'head_lattice_check.json', lattice)
    if not lattice['all_exact']:
        raise ValueError('saved heads are not exactlyBF16-promoted; nativeIG unavailable, no new norm results issued')
    stats = torch.load(source['normalization_stats']['path'], map_location='cpu', weights_only=False)
    mean, variance = (stats[key].double().numpy() for key in ('mean', 'var'))
    assert mean.shape == variance.shape == SHAPE and np.isfinite(mean).all() and np.isfinite(variance).all() and (variance >= 0).all()
    sigma = np.sqrt(variance+1e-5)
    token_norm2 = np.empty((240, 3, 16, 16), dtype=np.float64)
    for meta, full, base in batches:
        t = meta[0]['t']
        f16, b16 = full.bfloat16(), base.bfloat16()
        guided = (b16+1.78*(f16-b16)).float() if .1 <= t <= 1 else full
        for head_index, value in enumerate((full, base, guided)):
            raw = value.double().numpy()*sigma[None]+mean[None]
            norm2 = np.einsum('bchw,bchw->bhw', raw, raw, optimize=False)
            assert norm2.shape == (8, 16, 16) and np.isfinite(norm2).all() and (norm2 >= 0).all()
            for m, norms in zip(meta, norm2):
                token_norm2[m['row'], head_index] = norms
    assert np.isfinite(token_norm2).all()
    np.save(OUT/'token_norm_squared.npy', token_norm2, allow_pickle=False)
    put(OUT/'row_identity.json', request['rows'])
    def statistics(values):
        flat = values.reshape(-1)
        norm = np.sqrt(flat)
        excess = np.maximum(norm-64., 0.)
        excess2 = np.maximum(flat-4096., 0.)
        return {'tokens': len(flat), 'norm_min': float(norm.min()), 'norm_mean': float(norm.mean()),
                'norm_max': float(norm.max()), 'norm_squared_mean': float(flat.mean()), 'norm_squared_max': float(flat.max()),
                'outside_tokens': int((flat > 4096.).sum()), 'outside_fraction': float((flat > 4096.).mean()),
                'norm_excess_max': float(excess.max()), 'norm_excess_sum': float(excess.sum()), 'norm_excess_mean': float(excess.mean()),
                'norm_squared_excess_max': float(excess2.max()), 'norm_squared_excess_sum': float(excess2.sum()),
                'norm_squared_excess_mean': float(excess2.mean())}
    rows = []
    for m in request['rows']:
        for h, head in enumerate(HEADS):
            rows.append({**m, 'head': head, **statistics(token_norm2[m['row'], h])})
    with (OUT/'per_state_head.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {'historical_teacher': [m['row'] for m in request['rows'] if m['cache'] == 'historical' and m['domain'] == 'teacher'],
              'historical_rollout': [m['row'] for m in request['rows'] if m['cache'] == 'historical' and m['domain'] == 'rollout'],
              'current_teacher': list(range(160, 240)), 'historical_all': list(range(160)), 'all_fixed_rows': list(range(240))}
    pooled = {name: {head: {'state_rows': len(index), **statistics(token_norm2[index, h])}
                       for h, head in enumerate(HEADS)} for name, index in groups.items()}
    per_step = {name: {str(step): {head: statistics(token_norm2[[i for i in index if request['rows'][i]['step_index'] == step], h])
                         for h, head in enumerate(HEADS)} for step in STEPS}
                  for name, index in list(groups.items())[:3]}
    put(OUT/'per_step_pooled.json', per_step)
    # Independent small-output reconciliation: aggregate token counts/maxima via per-state rows.
    for group, index in groups.items():
        for head in HEADS:
            subset = [row for row in rows if row['row'] in index and row['head'] == head]
            actual = pooled[group][head]
            assert sum(r['outside_tokens'] for r in subset) == actual['outside_tokens']
            assert max(r['norm_max'] for r in subset) == actual['norm_max']
            assert sum(r['tokens'] for r in subset) == actual['tokens']
    outputs = [record(OUT/name) for name in ('token_norm_squared.npy', 'row_identity.json', 'per_state_head.csv',
                                            'head_lattice_check.json', 'per_step_pooled.json')]
    summary = {'complete': True, 'protocol': request['protocol'], 'request': record(OUT/'request.json'),
        'state_rows': 240, 'head_rows': 720, 'token_values': token_norm2.size,
        'all_heads_exact_BF16_promotions': lattice['all_exact'], 'head_coordinate_checks': lattice['checked_coordinates'],
        'all_fixed_tokens_inside_bound': bool((token_norm2 <= 4096.).all()),
        'pooled': pooled, 'per_step_pooled': record(OUT/'per_step_pooled.json'), 'outputs': outputs,
        'source_files_rehashed_bytes': sum(r['bytes'] for r in source.values()),
        'identity_recheck_wall_seconds': identity_wall, 'run_wall_seconds_before_summary': time.perf_counter()-start,
        'run_cpu_seconds_before_summary': time.process_time()-cpu, 'model_calls': 0, 'GPU_calls': 0,
        'new_noise_clip_sampling_decoder_FID_calls': 0,
        'no_performance_arm_authorized_by_this_result': True,
        'scope': 'All fixed cached predictions retained. NativeIG is CPU arithmetic reconstruction from losslessBF16 heads. No fullcurrentnative rollout cache, no independent classes/banks, no supportclaim for noisy states or normalizedEuclideanclip.'}
    put(OUT/'summary.json', summary)
    print(json.dumps({'complete': True, 'pooled': pooled, 'run_wall': summary['run_wall_seconds_before_summary'],
                      'run_cpu': summary['run_cpu_seconds_before_summary'], 'summary_sha256': sha(OUT/'summary.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        began = time.perf_counter()
        try:
            run()
        except BaseException as error:
            put(OUT/'failure.json', {'complete': False, 'error': f'{type(error).__name__}: {error}',
                'wall_seconds': time.perf_counter()-began, 'no_native_surrogate_or_fallback_performed': True})
            raise
