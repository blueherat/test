"""Run specification and asset fingerprints. Importing does not initialize CUDA."""
import hashlib
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS / 'weak_reference_loss_20260914'
PROTOCOL = WORK / 'docs/WEAK_REFERENCE_LOSS_PROTOCOL_20260914_ZH.md'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
MODEL = 'sit_small'
STEPS, BATCH, SEED = 3000, 32, 2026091461
SCREEN_SEED, CONFIRM_SEED = 2026091462, 2026091463
DATA = EXPS / 'sit_strong_reference_20260912/data'
OLD = EXPS / 'guidance_distribution_20260912/sit_small/input_local_training'
SPECS = {
    'real': dict(dataset='real', smooth_probability=0., weights=None),
    'self': dict(dataset='strong', smooth_probability=0., weights=None),
    'gaussian': dict(dataset='strong', smooth_probability=1., weights=None),
    'mixture': dict(dataset='strong', smooth_probability=.5, weights=None),
    'excess': dict(dataset='strong', smooth_probability=0., weights='excess'),
    'shuffled': dict(dataset='strong', smooth_probability=0., weights='shuffled'),
}
SCREEN_ARMS = ['real', 'self', 'gaussian', 'gaussian_half', 'mixture', 'context', 'native', 'excess', 'shuffled']


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class RequestedStop(RuntimeError):
    pass


def check_stop():
    if (ROOT / 'STOP_AFTER_CURRENT').exists():
        raise RequestedStop('STOP_AFTER_CURRENT requested for this experiment')


def prepare():
    import numpy as np
    from experiments.lifting_scale_sweep_20260909 import asset_paths, source_paths
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = set(Path(__file__).parent.glob('*.py')) | {PROTOCOL}
    sources.update(map(Path, source_paths(MODEL)))
    sources.add(Path('/home/zhoushunyu/data/research_repos/SiT/models.py'))
    for relative in (
        'experiments/guidance_pasted_20260912/common.py',
        'experiments/guidance_distribution_20260912/local_head.py',
        'experiments/imagenet100_sit_internal_v_head.py',
        'experiments/compute_adm_fid.py',
        'experiments/small_sit_carrier_flow_20260909.py',
    ):
        sources.add(WORK / relative)
    assets = list(asset_paths(MODEL)) + [OLD / 'head.pt', OLD / 'summary.json', OLD / 'request.json']
    assets += [DATA / 'real_clean.npy', DATA / 'strong_clean.npy', DATA / 'complete.json']
    assets += [EXPS.parent / 'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz']
    strong = np.load(DATA / 'strong_clean.npy')
    real = np.load(DATA / 'real_clean.npy')
    assert strong.shape == real.shape == (100, 20, 4, 32, 32)
    assert np.isfinite(strong).all() and np.isfinite(real).all()
    tau = .5 * float(np.sqrt(strong.var(axis=(0, 1), dtype=np.float64).mean()))
    summary = read(OLD / 'summary.json')
    assert summary['complete'] and summary['steps'] == 3000
    assert sha(OLD / 'head.pt') == summary['head_sha256']
    receipt = read(DATA / 'complete.json')
    assert receipt['passed'] and receipt['samples_per_dataset'] == 2000
    for source in (DATA / 'real_clean.npy', DATA / 'strong_clean.npy'):
        assert receipt['files'][str(source)] == sha(source)
    request = dict(
        model=MODEL, steps=STEPS, batch=BATCH, seed=SEED, lr=3e-4, weight_decay=1e-4,
        adam_betas=[.9, .999], ema=.995, timestep=[.01, .99], label_dropout=0.,
        head='fixed depth4 Context MLP', normalization='copied frozen buffers from original Context head',
        tau=tau, tau_rule='half pooled per-coordinate centered RMS of frozen generated training bank',
        specs=SPECS, screen_arms=SCREEN_ARMS, screen_samples=1000, screen_seed=SCREEN_SEED,
        confirm_samples=5000, confirm_seed=CONFIRM_SEED, extra_ig=.8,
        ig_schedule='existing 6/7 factor before .25, full factor before .5, zero afterward',
        solver='Heun64', sampling_batch=8, full_calls=128, prefix_calls=0,
        source_classifier=dict(features='mean and std of frozen depth4 tokens at deterministic endpoint X,t=.99,class c',
            classifier='StandardScaler + L2 logistic regression', C=1., max_iter=2000,
            folds=2, fold_rule='within-class endpoint index modulo2, same split in both sources',
            seed=SEED+40, strength=1., prediction='held-out fold only',
            feature_ratio_only=True, degeneracy_threshold=1e-3),
        promotion='candidate beats every specified control by >=1 FID and IS >= .9 native; single best candidate then fresh5K with best control and native',
        sources={str(p.resolve()): sha(p) for p in sorted(sources)},
        assets={str(p): sha(p) for p in assets},
    )
    path = ROOT / 'request.json'
    if path.exists():
        assert read(path) == request, 'Frozen request changed; create an explicit amendment before running'
    else:
        atomic(path, request)
        snapshot = ROOT / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for index, source in enumerate(sorted(sources)):
            (snapshot / f'{index:03d}_{source.name}').write_bytes(source.read_bytes())
    return request


def verify():
    request = read(ROOT / 'request.json')
    for group in ('sources', 'assets'):
        for path, fingerprint in request[group].items():
            assert sha(path) == fingerprint, (group, path, 'fingerprint changed')
    return request
