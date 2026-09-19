"""Frozen experiment specification. Safe to import in the CPU supervisor."""
import hashlib
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS / 'guidance_loss_50k_20260914'
LEGACY = EXPS / 'weak_reference_loss_20260914'
PROTOCOL = WORK / 'docs/GUIDANCE_LOSS_50K_PROTOCOL_20260914_ZH.md'
REPORT = WORK / 'docs/GUIDANCE_LOSS_50K_RESULTS_20260914_ZH.md'
MODULE = 'experiments.guidance_loss_50k_20260914'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
MODEL = 'sit_small'
STEPS, BATCH, SEED = 50000, 32, 2026091461
SCREEN_SEED, CONFIRM_SEED = 2026091462, 2026091463
DATA = EXPS / 'sit_strong_reference_20260912/data'
OLD = EXPS / 'guidance_distribution_20260912/sit_small/input_local_training'
ALPHA = .8

# Every learned vector head, including mixture-mean nuisance heads, gets 50K.
ARMS = {
    'real': dict(loss='fm', source='real', base='strong'),
    'self': dict(loss='fm', source='strong', base='strong'),
    'gaussian': dict(loss='fm', source='strong', base='strong', smooth=1.),
    'mixture': dict(loss='fm', source='strong', base='strong', smooth=.5),
    'excess': dict(loss='fm', source='strong', base='strong', weight='excess'),
    'shuffled': dict(loss='fm', source='strong', base='strong', weight='shuffled'),
    'real_residual': dict(loss='residual', source='real', base='strong'),
    'guided_weak': dict(loss='guided_weak', source='real', base='strong'),
    'contrast_s': dict(loss='contrast', source='strong', base='strong', baseline='strong'),
    'contrast_m': dict(loss='contrast', source='strong', base='strong', baseline='mixture'),
    'contrast_weak': dict(loss='contrast_weak', source='strong', base='strong', baseline='strong'),
    'covariance': dict(loss='covariance', source='strong', base='strong', baseline='mixture'),
    'contrast_null': dict(loss='null', source='strong', base='strong'),
    'weakmix_fm': dict(loss='fm', source='weakmix', base='strong'),
    'weakmix_contrast': dict(loss='contrast', source='weakmix', base='strong', baseline='strong'),
    'ig_residual': dict(loss='residual', source='real', base='ig'),
    'ig_contrast': dict(loss='contrast', source='ig', base='ig', baseline='strong'),
    'cfg_residual': dict(loss='residual', source='real', base='cfg'),
    'cfg_contrast': dict(loss='contrast', source='cfg', base='cfg', baseline='strong'),
}
CONTROLS = ['native', 'context', 'gaussian_half', 'cfg_native']
SCREEN_ARMS = list(ARMS) + CONTROLS
NUISANCE_SOURCES = ['strong', 'weakmix', 'ig', 'cfg']
WEAK_LOSSES = {'fm', 'guided_weak', 'contrast_weak'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class RequestedStop(RuntimeError):
    pass


def check_stop():
    if (ROOT / 'STOP_AFTER_CURRENT').exists():
        raise RequestedStop('STOP_AFTER_CURRENT requested for the 50K queue')


def spec(arm):
    if arm == 'cfg_native':
        return dict(loss='none', source='real', base='cfg')
    if arm == 'native':
        return dict(loss='none', source='real', base='ig')
    if arm == 'context':
        return dict(loss='fm', source='real', base='strong')
    return ARMS[arm.removesuffix('_half')]


def inference_counts(arm):
    value = spec(arm)
    return dict(full=224 if value['base'] == 'cfg' else 128, prefix=0,
                head=0 if arm in ('native', 'cfg_native') else 64,
                native_head=64 if value['base'] == 'ig' else 0)


def candidate_controls():
    common = ['real', 'self', 'context', 'native']
    return {
        'mixture': common + ['gaussian', 'gaussian_half'],
        'excess': common + ['shuffled', 'gaussian', 'gaussian_half', 'mixture'],
        'contrast_s': common + ['real_residual', 'guided_weak', 'contrast_null'],
        'contrast_m': common + ['real_residual', 'guided_weak', 'contrast_null'],
        'contrast_weak': common + ['guided_weak', 'contrast_null'],
        'covariance': common + ['real_residual', 'contrast_null'],
        'weakmix_contrast': common + ['weakmix_fm', 'contrast_s', 'contrast_m'],
        'ig_contrast': common + ['ig_residual'],
        'cfg_contrast': ['cfg_native', 'cfg_residual'],
    }


def prepare():
    import numpy as np
    # A successor request, preserving the previous request and its GPU receipt.
    old_request = read(LEGACY / 'request.json')
    assert old_request['steps'] == 3000
    sources = {Path(p) for p in old_request['sources']}
    sources.update(Path(__file__).parent.glob('*.py'))
    sources.update((WORK / 'experiments/endpoint_contrast_20260914').glob('*.py'))
    sources.add(PROTOCOL)
    assets = {p: sha(p) for p in old_request['assets']}
    for p, digest in old_request['assets'].items():
        assert assets[p] == digest, ('legacy asset changed', p)
    for name in ('request.json', 'gpu_preflight.json'):
        p = LEGACY / name
        assets[str(p)] = sha(p)
    generated = np.load(DATA / 'strong_clean.npy', mmap_mode='r')
    tau = .5 * float(np.sqrt(generated.var(axis=(0, 1), dtype=np.float64).mean()))
    request = dict(model=MODEL, steps=STEPS, nuisance_steps=STEPS, batch=BATCH, seed=SEED,
        lr=3e-4, weight_decay=1e-4, adam_betas=[.9,.999], ema=.995, timestep=[.01,.99],
        tau=tau, arms=ARMS, historical_controls=CONTROLS, alpha=ALPHA,
        screen_samples=1000, screen_seed=SCREEN_SEED, confirm_samples=5000, confirm_seed=CONFIRM_SEED,
        checkpoint_every=500, retained_steps=[3000,10000,20000,30000,40000,50000],
        validation_every=5000, validation_pairs=256,
        clean_split='per class and endpoint component: indices0..17 train;18..19 validation',
        nuisance=dict(folds=2, split='clean endpoint index modulo two; held-out predictions only',
            steps=STEPS, eta='pooled depth4 feature MLP', mean='same Context vector head',
            sources=NUISANCE_SOURCES, eta_label_one='real', loss='BCE and native FM, separate optimizers'),
        inference={arm:inference_counts(arm) for arm in SCREEN_ARMS},
        candidate_controls=candidate_controls(),
        promotion='>=1 FID better than every specified control; IS>=.9 corresponding native; best per 128/224-full-call family',
        source_classifier=dict(old_request['source_classifier'],training_endpoints_per_class=18,
            scaler_excludes_clean_validation=True,normalization='mean of training endpoints within class',
            shuffle='within class and train/validation partition',label_one='generated'),
        source_policy='freeze clean endpoint banks; weakmix=(G+standalone self50K weak endpoints)/2; one native IG and one native CFG endpoint round',
        predecessor_request_sha256=sha(LEGACY / 'request.json'),
        sources={str(p.resolve()):sha(p) for p in sorted(sources)}, assets=assets)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'request.json'
    if path.exists():
        assert read(path) == request, 'Frozen 50K request changed; explicit amendment required'
    else:
        atomic(path, request)
        snap = ROOT / 'source_snapshot'
        snap.mkdir(exist_ok=True)
        for i, source in enumerate(sorted(sources)):
            (snap / f'{i:03d}_{source.name}').write_bytes(source.read_bytes())
    return request


def verify():
    request = read(ROOT / 'request.json')
    for group in ('sources', 'assets'):
        for path, digest in request[group].items():
            assert sha(path) == digest, (group, path, 'fingerprint changed')
    return request
