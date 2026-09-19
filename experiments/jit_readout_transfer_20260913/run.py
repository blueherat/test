import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import common as c
from . import sample as s
from experiments.guidance_pasted_20260912.audit import fid_from_features


def collect(arm):
    root = c.ROOT / c.STAGE / arm
    summary_path = root / 'summary.json'
    if summary_path.exists():
        old = c.read(summary_path)
        assert old['complete'] and c.sha(root / 'samples.npz') == old['samples_sha256']
        return old
    assert all(c.read(root / f'rank{rank}/complete.json')['complete'] for rank in range(3))
    noise = np.load(c.ROOT / c.STAGE / 'inputs/noise.npy', mmap_mode='r')
    labels = np.load(c.ROOT / c.STAGE / 'inputs/labels.npy')
    request_hash = c.sha(c.ROOT / c.STAGE / 'request.json')
    pixels = np.empty((c.N, 256, 256, 3), dtype=np.uint8)
    coverage, records, seconds = [], [], 0.
    full = 198 if arm == 'cfg_reference' else 100
    heads = 0 if arm in ('strong', 'cfg_reference') else 50
    for p in sorted(root.glob('rank*/batch*.npz'), key=lambda p: int(p.stem[5:])):
        meta = c.read(p.with_suffix('.json'))
        assert meta['request_sha256'] == request_hash and c.sha(p) == meta['sha256']
        with np.load(p) as data:
            start, count = int(data['start']), len(data['labels'])
            coverage.extend(range(start, start + count))
            np.testing.assert_array_equal(data['labels'], labels[start:start + count])
            assert data['arr_0'].dtype == np.uint8 and data['arr_0'].shape == (count, 256, 256, 3)
            assert str(data['noise_sha256']) == c.array_sha(noise[start:start + count])
            assert str(data['request_sha256']) == request_hash and np.isfinite(data['latents']).all()
            assert int(data['full_calls']) == full and int(data['head_calls']) == heads and int(data['prefix_calls']) == 0
            np.testing.assert_array_equal(data['block_calls'], np.full(12, full))
            np.testing.assert_array_equal(data['arr_0'], np.round(np.clip((data['latents'] + 1).transpose(0, 2, 3, 1) / 2 * 255, 0, 255)).astype(np.uint8))
            pixels[start:start + count] = data['arr_0']
            seconds += float(data['seconds'])
        records.append(dict(file=str(p), sha256=meta['sha256']))
    assert coverage == list(range(c.N))
    np.savez(root / 'samples.npz', arr_0=pixels)
    result = dict(complete=True, arm=arm, primary_samples=c.N, generated_paths=c.N, records=records,
        seconds=seconds, full_calls_per_output=full, prefix_calls_at_inference=0, head_calls_per_output=heads,
        samples_sha256=c.sha(root / 'samples.npz'), raw_images_labels_states_and_counts_verified=True)
    c.atomic(summary_path, result)
    return result


def evaluate():
    command = [sys.executable, 'experiments/evaluate_raev2_official_samples.py']
    for arm in c.ARMS:
        command += ['--branch', arm + '=' + str(c.ROOT / c.STAGE / arm / 'samples.npz')]
    command += ['--output', str(c.ROOT / c.STAGE / 'metrics.csv'), '--batch-size', '32',
                '--device', 'cuda', '--fid-reference', str(s.REFERENCE),
                '--feature-cache-dir', str(c.ROOT / c.STAGE / 'features')]
    return command


def audit(summary, metric):
    assert metric['sample_sha256'] == summary['samples_sha256']
    assert metric['branch'] == summary['arm']
    paths = list((c.ROOT / c.STAGE / 'features').glob(summary['arm'] + '-' + summary['samples_sha256'][:16] + '*.features.pt'))
    assert len(paths) == 1, paths
    features = torch.load(paths[0], map_location='cpu', weights_only=True).numpy()
    assert len(features) == c.N and np.isfinite(features).all()
    with np.load(s.REFERENCE) as ref:
        mu, cov = (ref['mu'], ref['sigma']) if 'mu' in ref else (ref['ref_mu'], ref['ref_sigma'])
    fid = fid_from_features(features, mu, cov)
    error = abs(fid - metric['fid'])
    assert error < .002, (summary['arm'], error)
    record = dict(passed=True, samples=c.N, full_calls_per_output=summary['full_calls_per_output'],
        prefix_calls=0, samples_sha256=summary['samples_sha256'], features_sha256=c.sha(paths[0]),
        reference_sha256=c.sha(s.REFERENCE), fid_reported=metric['fid'], fid_fp64_same_features=fid,
        absolute_error=error, independent_feature_extraction=False,
        raw_images_labels_states_and_counts_verified=True)
    c.atomic(c.ROOT / c.STAGE / summary['arm'] / 'audit.json', record)
    return record


def main():
    c.ROOT.mkdir(parents=True, exist_ok=True)
    lock = (c.ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (c.ROOT / 'status.json').exists() and c.read(c.ROOT / 'status.json')['phase'] == 'complete':
        c.verify(c.ROOT / c.STAGE / 'request.json')
        return
    assert c.read(c.ROOT / 'implementation_check.json')['passed']
    request = s.prepare()
    c.verify(request)
    pid = os.getpid()
    workers, logs = [], []
    c.atomic(c.ROOT / 'status.json', dict(pid=pid, phase='sampling', samples_per_arm=c.N, arms=list(c.ARMS)))
    def launch(command, gpu, logname):
        stream = (c.ROOT / logname).open('a')
        logs.append(stream)
        process = subprocess.Popen(command, cwd=c.WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'), stdout=stream, stderr=subprocess.STDOUT)
        workers.append(process)
        return process
    def wait_all():
        while True:
            codes = [p.poll() for p in workers]
            c.atomic(c.ROOT / 'workers.json', dict(parent_pid=pid,
                children=[dict(pid=p.pid, returncode=code) for p, code in zip(workers, codes)]))
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('Worker failed: ' + str(codes))
            if all(code == 0 for code in codes):
                return
            time.sleep(5)
    try:
        for rank, gpu in enumerate((1, 2, 3)):
            launch([sys.executable, '-m', 'experiments.jit_readout_transfer_20260913.sample',
                '--rank', str(rank), '--world', '3', '--parent', str(pid)], gpu, f'sample{rank}.log')
        wait_all()
        workers.clear()
        summaries = [collect(arm) for arm in c.ARMS]
        c.atomic(c.ROOT / 'status.json', dict(pid=pid, phase='evaluating_and_benchmarking', images=c.N * len(c.ARMS)))
        launch(evaluate(), 1, 'evaluation.log')
        launch([sys.executable, '-m', 'experiments.jit_readout_transfer_20260913.sample', '--benchmark'], 2, 'benchmark.log')
        wait_all()
        c.verify(request)
        metrics = {row['branch']: row for row in c.read(c.ROOT / c.STAGE / 'metrics.json')}
        assert set(metrics) == set(c.ARMS)
        audits, results = [], []
        for summary in summaries:
            row = metrics[summary['arm']]
            audits.append(audit(summary, row))
            results.append(dict(**summary, fid=row['fid'], inception_score=row['inception_score'], metrics=row))
        c.atomic(c.ROOT / c.STAGE / 'results.json', results)
        by = {r['arm']: r for r in results}
        passed = by['mlp']['fid'] <= min(by[a]['fid'] for a in ('native_base', 'adg')) - 1 and by['mlp']['inception_score'] >= .9 * by['native_base']['inception_score']
        decision = dict(passes_transfer_gate=passed, worth_independent_5k=passed,
            mlp_minus_native=by['mlp']['fid'] - by['native_base']['fid'],
            mlp_minus_adg=by['mlp']['fid'] - by['adg']['fid'],
            mlp_minus_matched=by['mlp']['fid'] - by['native_fresh']['fid'],
            mlp_minus_official_cfg_reference=by['mlp']['fid'] - by['cfg_reference']['fid'],
            readout_bundle_gate=by['mlp']['fid'] <= by['native_fresh']['fid'] - 1,
            statistical_significance_established=False, core_novelty_established=False, goal_complete=False)
        c.atomic(c.ROOT / 'decision.json', decision)
        c.atomic(c.ROOT / c.STAGE / 'audit.json', dict(passed=True, arms=audits,
            request_sha256=c.sha(request), max_fid_error=max(a['absolute_error'] for a in audits)))
        c.atomic(c.ROOT / 'status.json', dict(pid=pid, phase='complete', arms=len(c.ARMS), images=c.N * len(c.ARMS)))
        print('JiT transfer complete', decision, flush=True)
    except BaseException as error:
        c.atomic(c.ROOT / 'status.json', dict(pid=pid, phase='failed', error=repr(error)))
        raise
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            p.wait()
        for stream in logs:
            stream.close()


if __name__ == '__main__':
    main()
