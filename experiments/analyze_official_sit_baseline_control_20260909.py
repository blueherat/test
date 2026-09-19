"""Independent sample identity and rank-space FID verification for fixed IG controls."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909')
OUT = ROOT / 'docs/data/guidance_goal_20260909'
REF = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    request = json.loads((DATA / 'request.json').read_text())
    assert json.loads((DATA / 'status.json').read_text())['phase'] == 'complete'
    for path, digest in request['sources'].items():
        assert sha(Path(path)) == digest, path
    assert sha(REF) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    with np.load(REF) as ref:
        mu, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
    assert json.loads((DATA / 'parity_passed.json').read_text())['images'] == 16
    inputs = np.load(DATA / 'inputs.npz')
    assert hashlib.sha256(inputs['noise'].tobytes()).hexdigest() == request['noise_sha256']
    assert hashlib.sha256(inputs['labels'].tobytes()).hexdigest() == request['label_sha256']
    assert sha(DATA / 'inputs.npz') == request['input_file_sha256']
    entries = []
    for arm, ref in request['references'].items():
        s = ref['summary']
        directory = Path(ref['directory'])
        assert s['samples'] == 1000 and s['batch_size'] == 4
        assert s['noise_sha256'] == request['noise_sha256'] and s['label_sha256'] == request['label_sha256']
        assert s['checkpoint_sha256'] == request['checkpoint_sha256'] and s['vae_state_sha256'] == request['vae_state_sha256']
        entries.append((arm, directory, 1.35, 1.0, s['steps'], 50 if arm == 'pfr' else 0, s['seconds']))
    for arm, (scale, high) in request['arms'].items():
        result = json.loads((DATA / arm / 'result.json').read_text())
        shards = [json.loads((DATA / arm / f'rank{r}.json').read_text()) for r in range(4)]
        assert sum(s['samples'] for s in shards) == 1000
        assert sum(s['full_sample_calls'] for s in shards) == 115000
        assert all(s['steps'] == 115 and s['prefix_calls'] == 0 and s['scale'] == scale and s['guidance_high'] == high for s in shards)
        entries.append((arm, DATA / arm, scale, high, 115, 0, result['gpu_inference_seconds_sum']))
    rows, feature_hashes, evaluator_ids = [], {}, set()
    for arm, directory, scale, high, full, prefix, seconds in entries:
        metrics = json.loads((directory / 'fid.json').read_text())[0]
        assert sha(directory / 'samples.npz') == metrics['sample_sha256']
        with np.load(directory / 'samples.npz') as pixels:
            assert pixels['arr_0'].shape == (1000, 256, 256, 3) and pixels['arr_0'].dtype == np.uint8
        feature, = (directory / 'features').glob('*.features.pt')
        x = torch.load(feature, map_location='cpu', weights_only=True).numpy().astype(float)
        assert x.shape == (1000, 2048) and np.isfinite(x).all()
        mean = x.mean(0)
        centered = x - mean
        # The nonzero spectrum equals that of C_ref^(1/2) C_gen C_ref^(1/2).
        # Use sample-space eigenvalues instead of the evaluator's nonsymmetric sqrtm.
        gram = centered @ cov @ centered.T / 999
        eigenvalues = np.linalg.eigvalsh((gram + gram.T) * .5)
        assert eigenvalues.min() > -1e-7
        mean_term = float(np.square(mean - mu).sum())
        cov_term = float(np.square(centered).sum()/999 + np.trace(cov) - 2*np.sqrt(np.maximum(eigenvalues, 0)).sum())
        independently_computed = mean_term + cov_term
        assert abs(independently_computed - metrics['fid']) < 2e-4
        feature_hashes[arm] = sha(feature)
        evaluator_ids.add((metrics['evaluator_commit'], metrics['fid_reference']))
        rows.append(dict(arm=arm, scale=scale, guidance_high=high, samples=1000,
            fid=metrics['fid'], independent_fid=independently_computed,
            mean_term=mean_term, covariance_term=cov_term, inception_score=metrics['inception_score'],
            full_per_image=full, prefix8_per_image=prefix, inference_gpu_seconds=seconds,
            noise_sha256=request['noise_sha256'], label_sha256=request['label_sha256'],
            pixel_sha256=metrics['sample_sha256']))
    assert evaluator_ids == {('19dfb4c2705333eb8b97e454fb354d47d1fe135b', 'imagenet_256_fid_stats')}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'official_sit_baseline_controls.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min((r for r in rows if r['arm'] != 'pfr'), key=lambda r:r['fid'])
    pfr = next(r for r in rows if r['arm'] == 'pfr')
    audit = dict(passed=True, research_goal_achieved=False, sampling_bank_role='reused_1k_discovery',
        source_sha256=sha(Path(__file__)), request_sha256=sha(DATA / 'request.json'),
        reference_sha256=sha(REF), feature_sha256=feature_hashes,
        maximum_fid_reconstruction_error=max(abs(r['fid']-r['independent_fid']) for r in rows),
        best_ordinary_arm=best['arm'], best_ordinary_fid=best['fid'], pfr_fid=pfr['fid'],
        best_ordinary_minus_pfr=best['fid']-pfr['fid'],
        limitation='Point estimates on one reused discovery bank. No independent confirmation, mechanism equivalence, or new-method claim.')
    (OUT / 'official_sit_baseline_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'rows':rows, 'audit':audit}, indent=2))


if __name__ == '__main__':
    main()
