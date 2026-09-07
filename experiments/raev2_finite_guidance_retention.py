#!/usr/bin/env python3
"""Finite native IG impulse, followed by Full-only or native IG continuation.

No derivative approximation, fitted guidance, decoder, image selection or FID.
Prepare freezes a metadata-only cohort and verifies each selected cache payload.
Run uses native B8 BF16 heads/mixing and FP32 Euler; every positive suffix time
is measured. A bounded detached coordinator can run four rank workers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SOURCE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/actual_ratio_bank64k/train')
CONFIG = ROOT/'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml'
CHECKPOINT = Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt')
EXPECTED_CHECKPOINT = '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
EXPECTED_CONFIG = '3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342'
PLAN = {
    'protocol': 'raev2_finite_guidance_retention_v1',
    'question': 'Does one actual native IG write change subsequent Full-only readouts in the frozen message direction?',
    'query_indices': [4, 16, 32, 48, 64, 80, 92, 96],
    'batches_per_query': 2, 'batch': 8, 'samples': 128, 'ranks': 4,
    'selection': 'Smallest two SHA256 scores per query over original B8 batch IDs; metadata only.',
    'selection_namespace': 'raev2-finite-retention:202609074',
    'arms': ['none_then_full', 'one_ig_then_full', 'all_ig'],
    'write': 'One unchanged native IG Euler step; its difference from the Full Euler step is the actual successor-state intervention.',
    'reader': 'Full alone at all remaining original query times, with class unchanged.',
    'message': 'Frozen native IG clean minus source Full clean, in FP32.',
    'primary_readout': 'Every Full readout and terminal latent: finite difference projected on the frozen clean message, divided by message squared norm.',
    'secondary': 'One-write terminal contrast projected on full remaining-IG terminal contrast; a response statistic, not information fraction or quality.',
    'independence': 'Previously used exploratory actual-state bank; two B8 clusters per time, not an independent quality confirmation.',
    'rules': 'No time selection after reading values; no derivative, scale search, optimization, images or FID. No automatic quality-run promotion.',
    'precision': 'Native BF16 heads and IG mixing, FP32 states/Euler, TF32 on; original B8 shape.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def put(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sources = [SCRIPT, CONFIG, ROOT/'experiments/cache_raev2_actual_ratio_states.py',
               ROOT/'experiments/raev2_guidance_quadrature.py',
               ROOT/'experiments/sample_raev2_pfr_retiming.py',
               ROOT/'external/RAEv2/src/stage2/models/DDT.py',
               ROOT/'external/RAEv2/src/utils/guidance_utils.py']
    plan = dict(PLAN, created_at=datetime.now(timezone.utc).isoformat(),
                source_sha256={str(p): sha(p) for p in sources})
    put(out/'plan.json', plan)
    candidates, provenance, grids = [], [], []
    for rank in range(4):
        path = SOURCE/f'shard{rank}'
        request, summary = read(path/'request.json'), read(path/'summary.json')
        assert summary['complete'] and sha(path/'request.json') == summary['request_sha256']
        assert sha(path/'metadata.npz') == summary['files']['metadata.npz']
        assert request['checkpoint_sha256'] == EXPECTED_CHECKPOINT
        assert request['config_sha256'] == EXPECTED_CONFIG
        assert request['state_key'] == 'ema' and request['batch'] == 8 and request['samples'] == 64000
        grids.append(request['query_grid'])
        records = np.load(path/'metadata.npz')['records']
        assert records.shape == (16000, 4) and len(summary['batch_records']) == 2000
        for local, record in enumerate(summary['batch_records']):
            b, k = record['batch'], record['query_index']
            rows = records[local*8:(local+1)*8]
            assert b == rank+4*local
            assert np.array_equal(rows[:, 0], np.arange(8*b, 8*b+8))
            assert np.array_equal(rows[:, 1], rows[:, 0] % 1000)
            assert np.all(rows[:, 2] == k) and np.all(rows[:, 3] == request['query_grid'][k])
            if k in PLAN['query_indices']:
                score = hashlib.sha256(f"{PLAN['selection_namespace']}:q:{k}:batch:{b}".encode()).hexdigest()
                candidates.append(dict(record, source_rank=rank, offset=local*8, score=score,
                                       ids=rows[:, 0].astype(int).tolist(), labels=rows[:, 1].astype(int).tolist()))
        provenance.append({'directory': str(path), 'request_sha256': sha(path/'request.json'),
                           'summary_sha256': sha(path/'summary.json'),
                           'metadata_sha256': sha(path/'metadata.npz'),
                           'declared_large_file_sha256': summary['files'],
                           'verification_scope': 'Metadata/request hashes and selected B8 raw array hashes; whole 16GB arrays are not rehashed.'})
    assert all(g == grids[0] for g in grids)
    cohort = []
    for k in PLAN['query_indices']:
        selected = sorted((c for c in candidates if c['query_index'] == k), key=lambda c: c['score'])[:2]
        assert len(selected) == 2
        for c in selected:
            cohort.append(dict(c, ordinal=len(cohort)))
    # This record is written before reading any selected model-state values.
    put(out/'cohort.json', {'selected': cohort, 'time_grid': grids[0], 'source_provenance': provenance})
    assert sha(CONFIG) == EXPECTED_CONFIG and sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    files = {}
    for c in cohort:
        source = SOURCE/f"shard{c['source_rank']}"
        offset = c['offset']
        state = np.array(np.load(source/'states.npy', mmap_mode='r')[offset:offset+8])
        noise = np.array(np.load(source/'noise.npy', mmap_mode='r')[offset:offset+8])
        assert state.shape == noise.shape == (8, 1024, 16, 16)
        assert state.dtype == noise.dtype == np.float32
        assert array_sha(state) == c['state_sha256'] and array_sha(noise) == c['noise_sha256']
        target = out/f"input{c['ordinal']:02d}.npz"
        np.savez(target, state=state, noise=noise, ids=np.array(c['ids']), labels=np.array(c['labels']))
        files[target.name] = sha(target)
    put(out/'prepared.json', {'complete': True, 'plan_sha256': sha(out/'plan.json'),
        'cohort_sha256': sha(out/'cohort.json'), 'input_sha256': files,
        'checkpoint_sha256': EXPECTED_CHECKPOINT, 'config_sha256': EXPECTED_CONFIG,
        'selected_payload_hashes_verified': True, 'seconds': time.perf_counter()-start})
    print(json.dumps({'prepared': True, 'cohort_batches': len(cohort), 'seconds': time.perf_counter()-start}), flush=True)


def run(out, rank):
    import torch
    for p in (ROOT, ROOT/'external/RAEv2/src'):
        sys.path.insert(0, str(p))
    from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid
    from experiments.raev2_guidance_quadrature import native_clean
    from utils.model_utils import instantiate_from_config
    from utils.guidance_utils import forward_with_internalguidance

    prepared, plan, cohort_data = read(out/'prepared.json'), read(out/'plan.json'), read(out/'cohort.json')
    assert prepared['complete'] and prepared['plan_sha256'] == sha(out/'plan.json')
    assert prepared['cohort_sha256'] == sha(out/'cohort.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, f'source changed: {path}'
    target = out/f'rank{rank}'
    target.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(CONFIG)
    model = instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'], strict=True)
    del checkpoint
    grid = cohort_data['time_grid']
    assert shifted_time_grid(100, 8., torch.device('cuda')).cpu().tolist() == grid
    put(target/'request.json', {'rank': rank, 'plan_sha256': prepared['plan_sha256'],
                              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                              'no_grad': True, 'tf32': True, 'batch': 8})

    def euler(x, clean, t, s):
        return x-(t-s)*((x-clean)/max(t, .05))

    def evaluate(x, t, labels):
        times = torch.full((8,), t, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f, b = model(x, times, context=labels, attn_mask=None)
            g = native_clean(f, b, t)
        return f.float(), b.float(), g

    def response(difference, message, prefix):
        a, b = difference.double().flatten(1), message.double().flatten(1)
        aa, bb, ab = a.square().sum(1), b.square().sum(1), (a*b).sum(1)
        safe = bb.clamp_min(1e-300)
        return {prefix+'_projection': (ab/safe).cpu().numpy(),
                prefix+'_rms': (aa/a.shape[1]).sqrt().cpu().numpy(),
                prefix+'_relative_norm': (aa/safe).sqrt().cpu().numpy(),
                prefix+'_cosine': (ab/(aa*bb).sqrt().clamp_min(1e-300)).cpu().numpy()}

    selected = cohort_data['selected'][rank::4]
    count, forward_calls, checks = 0, 0, []
    began = time.perf_counter()
    with torch.no_grad():
        for local, c in enumerate(selected):
            input_path = out/f"input{c['ordinal']:02d}.npz"
            assert sha(input_path) == prepared['input_sha256'][input_path.name]
            inp = np.load(input_path)
            x = torch.from_numpy(inp['state']).cuda()
            labels = torch.from_numpy(inp['labels']).cuda()
            k, t, s = c['query_index'], grid[c['query_index']], grid[c['query_index']+1]
            batch_dir = target/f"batch{c['ordinal']:02d}"
            batch_dir.mkdir()
            torch.cuda.synchronize()
            start = time.perf_counter()
            f, b, g = evaluate(x, t, labels)
            forward_calls += 1
            if local == 0:
                times = torch.full((8,), t, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    official = forward_with_internalguidance(model, torch.cat([x, x]), torch.cat([times, times]),
                        ig_scale=1.78, ig_interval=(.1, 1.), context=torch.cat([labels, labels]), attn_mask=None)[:8]
                forward_calls += 1
                assert torch.equal(official.float(), g), 'native guidance parity'
                # Rebuild one selected original prefix per worker, not a new input draw.
                replay = torch.from_numpy(inp['noise']).cuda()
                for j in range(k):
                    _, _, gj = evaluate(replay, grid[j], labels)
                    replay = euler(replay, gj, grid[j], grid[j+1])
                    forward_calls += 1
                assert torch.equal(replay, x), 'cached native prefix replay'
                checks.append({'ordinal': c['ordinal'], 'native_ig_parity': True, 'cached_prefix_bitwise': True})
            message = g-f
            none, once = euler(x, f, t, s), euler(x, g, t, s)
            all_ig = once.clone()
            impulse = once-none
            formula = ((t-s)/t)*message
            roundoff = (impulse-formula).abs().amax().item()
            # Equal heads remove guidance, including the official BF16 arithmetic.
            null_guided = native_clean(f.to(torch.bfloat16), f.to(torch.bfloat16), t)
            assert torch.equal(euler(x, null_guided, t, s), none)
            assert torch.isfinite(message).all() and message.double().flatten(1).square().sum(1).gt(0).all()
            snapshots = {'ids': inp['ids'], 'labels': inp['labels'], 'source_full': f.cpu().numpy(),
                         'source_base': b.cpu().numpy(), 'source_guided': g.cpu().numpy(),
                         'successor_none': none.cpu().numpy(), 'successor_once': once.cpu().numpy()}
            rows, first_read = [], True
            for j in range(k+1, 100):
                now, following = grid[j:j+2]
                fn, _, _ = evaluate(none, now, labels)
                fo, _, _ = evaluate(once, now, labels)
                # The first shared state all_ig == once provides a native B8
                # determinism control; count the repeated call explicitly.
                if first_read:
                    fa, ba, ga = evaluate(all_ig, now, labels)
                    assert torch.equal(fa, fo), 'same-state native Full determinism'
                    snapshots['first_read_none'], snapshots['first_read_once'] = fn.cpu().numpy(), fo.cpu().numpy()
                    first_read = False
                else:
                    fa, ba, ga = evaluate(all_ig, now, labels)
                forward_calls += 3
                state_difference, read_difference = once-none, fo-fn
                passive = (now/s)*impulse
                values = response(state_difference, message, 'state')
                values.update(response(read_difference, message, 'read'))
                values.update(response(state_difference-passive, message, 'feedback'))
                quantized_changed = (once.to(torch.bfloat16) != none.to(torch.bfloat16)).float().flatten(1).mean(1).cpu().numpy()
                next_none, next_once = euler(none, fn, now, following), euler(once, fo, now, following)
                finite_prediction = (following/now)*state_difference+((now-following)/now)*read_difference
                recurrence_error = ((next_once-next_none)-finite_prediction).double().flatten(1).square().mean(1).sqrt().cpu().numpy()
                for i, sample_id in enumerate(c['ids']):
                    row = {'ordinal': c['ordinal'], 'source_batch': c['batch'], 'id': sample_id,
                           'label': c['labels'][i], 'write_query': k, 'write_time': t, 'successor_time': s,
                           'read_query': j, 'read_time': now, 'quantized_input_changed_fraction': float(quantized_changed[i]),
                           'finite_recurrence_roundoff_rms': float(recurrence_error[i])}
                    row.update({key: float(value[i]) for key, value in values.items()})
                    rows.append(row)
                none, once = next_none, next_once
                all_ig = euler(all_ig, ga, now, following)
            assert all(torch.isfinite(v).all() for v in (none, once, all_ig))
            snapshots.update(endpoint_none=none.cpu().numpy(), endpoint_once=once.cpu().numpy(), endpoint_all=all_ig.cpu().numpy())
            np.savez(batch_dir/'snapshots.npz', **snapshots)
            with (batch_dir/'readouts.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            terminal = response(once-none, message, 'terminal')
            terminal.update(response(all_ig-none, message, 'all_guidance_terminal'))
            terminal.update(response(once-none, all_ig-none, 'one_relative_to_all'))
            terminal_records = []
            for i, sample_id in enumerate(c['ids']):
                terminal_records.append(dict(id=sample_id, label=c['labels'][i],
                    **{key: float(value[i]) for key, value in terminal.items()}))
            torch.cuda.synchronize()
            put(batch_dir/'summary.json', {'complete': True, 'ordinal': c['ordinal'], 'source_batch': c['batch'],
                'write_query': k, 'samples': 8, 'readout_rows': len(rows), 'terminal': terminal_records,
                'actual_write_rms': impulse.double().flatten(1).square().mean(1).sqrt().cpu().tolist(),
                'frozen_message_rms': message.double().flatten(1).square().mean(1).sqrt().cpu().tolist(),
                'write_rearrangement_max_abs': roundoff, 'seconds': time.perf_counter()-start,
                'files': {name: sha(batch_dir/name) for name in ('snapshots.npz', 'readouts.csv')}})
            count += 8
            put(target/'progress.json', {'samples': count, 'target': 32, 'forward_batch_calls': forward_calls,
                                       'seconds': time.perf_counter()-began})
            print(json.dumps({'rank': rank, 'ordinal': c['ordinal'], 'samples': count,
                              'seconds': time.perf_counter()-began}), flush=True)
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, f'source changed during run: {path}'
    put(target/'complete.json', {'complete': True, 'samples': count, 'forward_batch_calls': forward_calls,
        'sample_model_calls': forward_calls*8, 'input_gradients': 0, 'checks': checks,
        'seconds': time.perf_counter()-began, 'max_memory_allocated': torch.cuda.max_memory_allocated()})


def analyze(out):
    cohort = read(out/'cohort.json')['selected']
    summaries, all_rows, artifacts = [], [], []
    for c in cohort:
        batch_dir = out/f"rank{c['ordinal'] % 4}"/f"batch{c['ordinal']:02d}"
        summary = read(batch_dir/'summary.json')
        assert summary['complete'] and summary['ordinal'] == c['ordinal']
        for name, expected in summary['files'].items():
            assert sha(batch_dir/name) == expected
        summaries.append(summary)
        artifacts.append({'directory': str(batch_dir), 'summary_sha256': sha(batch_dir/'summary.json'), **summary['files']})
        with (batch_dir/'readouts.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == (99-c['query_index'])*8
        all_rows.extend(rows)
    groups = []
    for k in PLAN['query_indices']:
        batches = [s for s in summaries if s['write_query'] == k]
        terminal = [r for b in batches for r in b['terminal']]
        first = [r for r in all_rows if int(r['write_query']) == k and int(r['read_query']) == k+1]
        assert len(terminal) == len(first) == 16
        group = {'write_query': k, 'samples': 16, 'batch_clusters': 2}
        for key in ('read_projection', 'read_cosine', 'quantized_input_changed_fraction'):
            arr = np.array([float(r[key]) for r in first])
            group['first_'+key] = {'mean': float(arr.mean()), 'min': float(arr.min()), 'max': float(arr.max())}
        for key in ('terminal_projection', 'terminal_cosine', 'terminal_relative_norm',
                    'one_relative_to_all_projection', 'one_relative_to_all_cosine'):
            arr = np.array([r[key] for r in terminal])
            group[key] = {'mean': float(arr.mean()), 'min': float(arr.min()), 'max': float(arr.max()),
                          'positive_count': int((arr > 0).sum())}
        group['write_rms_mean'] = float(np.mean([x for b in batches for x in b['actual_write_rms']]))
        group['message_rms_mean'] = float(np.mean([x for b in batches for x in b['frozen_message_rms']]))
        groups.append(group)
    ranks = [read(out/f'rank{rank}'/'complete.json') for rank in range(4)]
    assert all(r['complete'] and r['samples'] == 32 for r in ranks)
    result = {'complete': True, 'samples': 128, 'readout_records': len(all_rows), 'groups': groups,
        'goal_complete': False, 'quality_evaluated': False, 'input_gradients': 0,
        'interpretation': 'Finite paired model response to one unchanged native IG impulse. No FID, semantic correctness, mutual information or small-perturbation theorem is inferred.',
        'max_finite_recurrence_roundoff_rms': max(float(r['finite_recurrence_roundoff_rms']) for r in all_rows),
        'sample_model_calls': sum(r['sample_model_calls'] for r in ranks),
        'worker_seconds': [r['seconds'] for r in ranks], 'rank_checks': [r['checks'] for r in ranks],
        'artifacts': artifacts, 'prepared_sha256': sha(out/'prepared.json')}
    put(out/'analysis.json', result)
    print(json.dumps({'complete': True, 'groups': groups, 'sample_model_calls': result['sample_model_calls']}), flush=True)


def coordinate(out):
    assert read(out/'prepared.json')['complete']
    began, jobs, handles = time.perf_counter(), [], []
    for rank in range(4):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
        handle = (out/f'worker{rank}.log').open('wb')
        proc = subprocess.Popen([sys.executable, str(SCRIPT), '--stage', 'run', '--output', str(out), '--rank', str(rank)],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append(proc)
        handles.append(handle)
    put(out/'live_workers.json', {'controller_pid': os.getpid(), 'workers': [
        {'rank': rank, 'pid': p.pid, 'starttime': Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
        for rank, p in enumerate(jobs)]})
    try:
        while any(p.poll() is None for p in jobs):
            failed = [p for p in jobs if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'worker failure: {[(p.pid, p.returncode) for p in failed]}')
            time.sleep(5)
        assert all(p.returncode == 0 for p in jobs)
        analyze(out)
        put(out/'completion.json', {'complete': True, 'exit_codes': [p.returncode for p in jobs],
                                   'seconds': time.perf_counter()-began, 'analysis_sha256': sha(out/'analysis.json')})
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'run', 'analyze', 'coordinate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.rank < 4:
        parser.error('rank must be 0..3')
    {'prepare': lambda: prepare(args.output.resolve()), 'run': lambda: run(args.output.resolve(), args.rank),
     'analyze': lambda: analyze(args.output.resolve()), 'coordinate': lambda: coordinate(args.output.resolve())}[args.stage]()


if __name__ == '__main__':
    main()
