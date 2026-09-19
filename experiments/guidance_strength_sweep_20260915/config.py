"""Immutable search policy; original model and training requests are retained."""
from pathlib import Path
from experiments.guidance_loss_50k_20260914 import config as old

WORK, EXPS, PYTHON = old.WORK, old.EXPS, old.PYTHON
ROOT = EXPS / 'guidance_strength_sweep_20260915'
MODULE = 'experiments.guidance_strength_sweep_20260915'
PROTOCOL = WORK / 'docs/GUIDANCE_STRENGTH_EVALUATION_20260915_ZH.md'
REPORT = WORK / 'docs/GUIDANCE_STRENGTH_RESULTS_20260915_ZH.md'
BANK = old.ROOT / old.MODEL / 'screen1000/inputs'
METHODS = ('guided_weak', 'self', 'mixture', 'gaussian', 'excess',
           'contrast_s', 'contrast_m', 'contrast_weak', 'covariance',
           'weakmix_contrast', 'ig_contrast', 'cfg_contrast',
           'real', 'shuffled', 'real_residual', 'contrast_null',
           'weakmix_fm', 'ig_residual', 'cfg_residual')
POINT_METHODS = (*METHODS, 'native', 'context', 'cfg_native', 'strong')
STEPS = (16, 8, 4, 1)  # Exact integer arithmetic in units of 0.025.
COARSE = tuple(range(0, 81, 16))
atomic, read, sha, array_sha = old.atomic, old.read, old.sha, old.array_sha
RequestedStop = old.RequestedStop


def check_stop():
    if (ROOT / 'STOP_AFTER_CURRENT').exists() or (old.ROOT / 'STOP_AFTER_CURRENT').exists():
        raise RequestedStop('Strength search stop requested; checkpoint current work')


def value(tick):
    return tick / 40


def key(method, tick):
    assert method in POINT_METHODS and isinstance(tick, int) and 0 <= tick <= 80
    return f'{method}__c{tick:04d}'


def parse(point):
    method, raw = point.rsplit('__c', 1)
    tick = int(raw)
    assert key(method, tick) == point
    return method, tick


def canonical(method, tick):
    if tick:
        return key(method, tick)
    if method in ('native', 'context', 'cfg_native', 'strong') or old.spec(method)['base'] == 'strong':
        return key('strong', 0)
    return key('native', 32) if old.spec(method)['base'] == 'ig' else key('cfg_native', 50)


def anchor(point):
    method, tick = parse(point)
    if method == 'strong':
        return None
    default = 50 if method == 'cfg_native' else 32 if method in ('native', 'context') or old.spec(method)['loss'] in old.WEAK_LOSSES else 40
    if tick == default:
        return method
    if method == 'gaussian' and tick == 16:
        return 'gaussian_half'
    return None


def counts(point):
    method, tick = parse(point)
    if method == 'strong':
        return dict(full=128, prefix=0, head=0, native_head=0)
    if method == 'cfg_native':
        return dict(full=224 if tick else 128, prefix=0, head=0, native_head=0)
    if method == 'native':
        return dict(full=128, prefix=0, head=0, native_head=64 if tick else 0)
    base = old.spec(method)['base']
    return dict(full=224 if base == 'cfg' else 128, prefix=0,
                head=64 if tick else 0, native_head=64 if base == 'ig' else 0)


def prepare():
    old.verify()
    assert not (ROOT / 'request.json').exists(), 'Use verify() for an existing search'
    from experiments.guidance_loss_50k_20260914 import planning
    existing = []
    for method in METHODS:
        if method not in old.ARMS or planning.verify_output(dict(action='train', arm=method)):
            existing.append(method)
    files = list(Path(__file__).parent.glob('*.py')) + [PROTOCOL, WORK / 'experiments/guidance_loss_parallel_20260914.py']
    manifest = read(BANK / 'complete.json')
    assets = {str(old.ROOT / 'request.json'): sha(old.ROOT / 'request.json'),
              str(old.ROOT / 'dispatch_request.json'): sha(old.ROOT / 'dispatch_request.json'),
              str(BANK / 'complete.json'): sha(BANK / 'complete.json'), **manifest['files']}
    request = dict(model='sit_small', methods=list(METHODS), steps=50000,
        samples_per_point=1000, input_seed=old.SCREEN_SEED, initial_range=[0., 2.],
        coefficient_steps=[.4, .2, .1, .025], step_ticks=list(STEPS), coarse_ticks=list(COARSE),
        interval_selection='rank adjacent valid intervals by mean endpoint FID, tie by min endpoint FID then lower coefficient; take top two and fill their entire connected span',
        connected_refinement=True, no_automatic_boundary_expansion=True,
        initially_completed_methods=existing, idea_order=list(METHODS),
        execution='one idea at a time: its necessary 50K training, all four search levels and evaluation, then next idea',
        excluded_standalone_scans=['context', 'native', 'cfg_native'],
        retained_training='pending original training is resumed only when required for the current idea; checkpoints preserved',
        max_gpu_workers=4, max_cpu_workers=4, quality_models=['sit_small', 'jit'],
        jit_policy='transfer follows SiT selection and uses the same four-stage 1K evaluation policy',
        coefficient_definition=dict(weak='actual IG extra coefficient alpha', native='actual native IG extra coefficient alpha',
            cfg_native='actual CFG extra coefficient beta, conventional scale=1+beta',
            correction='gain lambda on the existing correction; fixed IG/CFG base; default lambda=1'),
        base_ig_alpha=.8, base_cfg_extra=1.25, inference_windows_unchanged=True,
        no_5k=True, final_scores_are_selection_scores=True,
        sources={str(p.resolve()): sha(p) for p in files}, assets=assets)
    atomic(ROOT / 'request.json', request)
    return request


def verify():
    request = read(ROOT / 'request.json')
    for group in ('sources', 'assets'):
        for path, digest in request[group].items():
            assert sha(path) == digest, (group, path)
    return request
