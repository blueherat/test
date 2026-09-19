"""A single follow-up: train guided weak with its deployed interval/operator."""
from pathlib import Path
from experiments import guidance_dynamic_recovery_20260915 as recovery
from experiments.guidance_dynamic_50k_20260915 import config as k

NAME = 'guided_weak_deployed'
MODULE = 'experiments.guided_weak_deployed_20260915'
REQUEST = k.ROOT / 'guided_weak_deployed_request.json'


def verify():
    result = recovery.verify()
    request = k.read(REQUEST)
    assert request['scientific_request_sha256'] == k.sha(k.ROOT / 'request.json')
    assert request['runtime_amendment_sha256'] == k.sha(recovery.AMENDMENT)
    for filename, digest in request['sources'].items():
        assert k.sha(filename) == digest, filename
    return result


def prepare():
    recovery.verify()
    assert not REQUEST.exists()
    paths = list(Path(__file__).parent.glob('*.py'))
    paths.append(k.WORK / 'experiments/guidance_dynamic_recovery_20260915/pipeline_theory.py')
    k.atomic(REQUEST, dict(scientific_request_sha256=k.sha(k.ROOT / 'request.json'),
        runtime_amendment_sha256=k.sha(recovery.AMENDMENT), candidate=NAME,
        position='after guided_weak within each model; all SiT before all JiT',
        objective='E || S + alpha*rho(t)*(S-W) - Y ||^2 over the existing active interval',
        training_time='SiT uniform [0,.5); JiT original logistic-normal conditioned on t<.5',
        profile='SiT rho=6/7 for t<.25, else 1; JiT rho=1; inference remains existing t<.5',
        discrete_boundary='Existing Heun second query retains the left-step profile; objective matches the continuous interval, not those boundary queries exactly.',
        training_anchor=dict(sit_small=.8,jit=.3), training_steps=50000,
        data='same full dynamic real data; new head initialized from the frozen normalization template',
        global_batch=256, world_size=2, architecture_unchanged=True, extra_inference_backbones=0,
        evaluation='unchanged .4/.2/.1/.025 1K scan and best-two extension to 5K',
        limitation='Compared with guided_weak, both the time allocation and SiT early profile in the loss change.',
        sources={str(p.resolve()):k.sha(p) for p in paths}))


def install():
    recovery.install()
    verify()
    k.verify = verify
    k.MODULE = MODULE
    methods = list(k.METHODS)
    if NAME not in methods:
        methods.insert(methods.index('guided_weak')+1,NAME)
    k.METHODS = tuple(methods)
    k.ARMS[NAME] = dict(loss='guided_weak',source='real',base='strong',deployed_objective=True)
    from .training import install_training
    install_training()
