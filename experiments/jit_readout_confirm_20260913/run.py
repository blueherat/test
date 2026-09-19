import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import core as m
from experiments.jit_readout_transfer_20260913 import common as c
from experiments.report_readout_validation_20260912 import fid_covariance


def collect(arm):
    root = m.ROOT / m.STAGE / arm
    path = root / 'summary.json'
    if path.exists():
        row = c.read(path)
        assert row['complete'] and c.sha(root / 'samples.npz') == row['samples_sha256']
        return row
    assert all(c.read(root / f'rank{rank}/complete.json')['complete'] for rank in range(3))
    noise = np.load(m.ROOT / m.STAGE / 'inputs/noise.npy', mmap_mode='r')
    labels = np.load(m.ROOT / m.STAGE / 'inputs/labels.npy')
    request_hash = c.sha(m.ROOT / m.STAGE / 'request.json')
    full, heads = (98, 0) if arm == 'cfg_heun25' else (100, 50)
    pixels = np.empty((m.N, 256, 256, 3), dtype=np.uint8)
    records, coverage, seconds = [], [], 0.
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
    assert coverage == list(range(m.N))
    np.savez(root / 'samples.npz', arr_0=pixels)
    row = dict(complete=True, arm=arm, primary_samples=m.N, generated_paths=m.N, records=records,
        seconds=seconds, full_calls_per_output=full, prefix_calls_at_inference=0, head_calls_per_output=heads,
        samples_sha256=c.sha(root / 'samples.npz'), raw_images_labels_states_and_counts_verified=True)
    c.atomic(path, row)
    return row


def evaluate_command():
    command = [sys.executable, 'experiments/evaluate_raev2_official_samples.py']
    for arm in m.ARMS:
        command += ['--branch', arm + '=' + str(m.ROOT / m.STAGE / arm / 'samples.npz')]
    return command + ['--output', str(m.ROOT / m.STAGE / 'metrics.csv'), '--batch-size', '32',
        '--device', 'cuda', '--fid-reference', str(m.base.REFERENCE),
        '--feature-cache-dir', str(m.ROOT / m.STAGE / 'features')]


def audit(summary, metric):
    assert summary['samples_sha256'] == metric['sample_sha256']
    paths = list((m.ROOT / m.STAGE / 'features').glob(summary['arm'] + '-' + summary['samples_sha256'][:16] + '*.features.pt'))
    assert len(paths) == 1, paths
    features = torch.load(paths[0], map_location='cpu', weights_only=True).numpy()
    assert len(features) == m.N and np.isfinite(features).all()
    with np.load(m.base.REFERENCE) as ref:
        mu, cov = (ref['mu'], ref['sigma']) if 'mu' in ref else (ref['ref_mu'], ref['ref_sigma'])
    fid = fid_covariance(features, mu, cov)
    error = abs(fid - metric['fid'])
    assert error < .002, (summary['arm'], error)
    row = dict(passed=True, arm=summary['arm'], samples=m.N, samples_sha256=summary['samples_sha256'],
        features_sha256=c.sha(paths[0]), reference_sha256=c.sha(m.base.REFERENCE),
        fid_reported=metric['fid'], fid_fp64_same_features=fid, absolute_error=error,
        independent_feature_extraction=False, method='symmetric_fp64_covariance',
        raw_images_labels_states_and_counts_verified=True)
    c.atomic(m.ROOT / m.STAGE / summary['arm'] / 'audit.json', row)
    return row


def main():
    m.ROOT.mkdir(parents=True, exist_ok=True)
    lock = (m.ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = m.prepare()
    c.verify(request)
    if (m.ROOT / 'status.json').exists() and c.read(m.ROOT / 'status.json')['phase'] == 'complete':
        return
    pid, workers, logs = os.getpid(), [], []
    c.atomic(m.ROOT / 'status.json', dict(pid=pid, phase='sampling', samples_per_arm=m.N, arms=list(m.ARMS)))
    def launch(command, gpu, logfile):
        stream = (m.ROOT / logfile).open('a')
        logs.append(stream)
        p = subprocess.Popen(command, cwd=c.WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'), stdout=stream, stderr=subprocess.STDOUT)
        workers.append(p)
    def wait_all():
        while True:
            codes = [p.poll() for p in workers]
            c.atomic(m.ROOT / 'workers.json', dict(parent_pid=pid,
                children=[dict(pid=p.pid, returncode=code) for p, code in zip(workers, codes)]))
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('Worker failed: ' + str(codes))
            if all(code == 0 for code in codes):
                return
            time.sleep(5)
    try:
        for rank, gpu in enumerate((1, 2, 3)):
            launch([sys.executable, '-m', 'experiments.jit_readout_confirm_20260913.core',
                '--rank', str(rank), '--world', '3', '--parent', str(pid)], gpu, f'sample{rank}.log')
        wait_all()
        workers.clear()
        summaries = [collect(a) for a in m.ARMS]
        c.atomic(m.ROOT / 'status.json', dict(pid=pid, phase='evaluating_and_benchmarking', images=m.N * len(m.ARMS)))
        launch(evaluate_command(), 1, 'evaluation.log')
        launch([sys.executable, '-m', 'experiments.jit_readout_confirm_20260913.core', '--benchmark'], 2, 'benchmark.log')
        wait_all()
        c.verify(request)
        metrics = {r['branch']: r for r in c.read(m.ROOT / m.STAGE / 'metrics.json')}
        assert set(metrics) == set(m.ARMS)
        results, audits = [], []
        for summary in summaries:
            metric = metrics[summary['arm']]
            audits.append(audit(summary, metric))
            results.append(dict(**summary, fid=metric['fid'], inception_score=metric['inception_score'], metrics=metric))
        c.atomic(m.ROOT / m.STAGE / 'results.json', results)
        by = {r['arm']: r for r in results}
        decision = dict(passes_ig_5k_gate=by['mlp']['fid'] <= min(by[a]['fid'] for a in ('native_base', 'adg')) - 1
                and by['mlp']['inception_score'] >= .9 * by['native_base']['inception_score'],
            mlp_minus_native=by['mlp']['fid'] - by['native_base']['fid'],
            mlp_minus_adg=by['mlp']['fid'] - by['adg']['fid'],
            mlp_minus_cfg25=by['mlp']['fid'] - by['cfg_heun25']['fid'],
            mlp_fid_better_than_cfg25=by['mlp']['fid'] < by['cfg_heun25']['fid'],
            statistical_significance_established=False, core_novelty_established=False, goal_complete=False)
        c.atomic(m.ROOT / 'decision.json', decision)
        c.atomic(m.ROOT / m.STAGE / 'audit.json', dict(passed=True, arms=audits,
            request_sha256=c.sha(request), max_fid_error=max(a['absolute_error'] for a in audits),
            controller_sha256=c.sha(Path(__file__)), fid_recalculation_source_sha256=c.sha(c.WORK / 'experiments/report_readout_validation_20260912.py')))
        c.atomic(m.ROOT / 'status.json', dict(pid=pid, phase='complete', arms=len(m.ARMS), images=m.N * len(m.ARMS)))
        print('JiT independent 5K complete', decision, flush=True)
    except BaseException as error:
        c.atomic(m.ROOT / 'status.json', dict(pid=pid, phase='failed', error=repr(error)))
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
