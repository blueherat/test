"""Audited preflight repair; the scientific request and production code stay frozen."""
from pathlib import Path
from experiments.guidance_dynamic_50k_20260915 import config as k

AMENDMENT = k.ROOT / 'repair_20260915' / 'request.json'
MODULE = 'experiments.guidance_dynamic_recovery_20260915'
_original_verify = k.verify


def verify():
    request = _original_verify()
    amendment = k.read(AMENDMENT)
    assert amendment['scientific_request_sha256'] == k.sha(k.ROOT / 'request.json')
    for filename, digest in amendment['sources'].items():
        assert k.sha(filename) == digest, filename
    return request


def install():
    verify()
    k.verify = verify
    k.MODULE = MODULE
    k.WORLD = 2
    from experiments.guidance_dynamic_50k_20260915 import checks as original
    from .checks import gpu
    original.gpu = gpu
    from experiments.guidance_dynamic_50k_20260915 import training
    old_save = training.save_torch
    def save(path, value):
        return old_save(path, dict(value, runtime_amendment_sha256=k.sha(AMENDMENT)))
    training.save_torch = save


def prepare():
    _original_verify()
    assert not AMENDMENT.exists(), 'Existing repair amendments are immutable'
    k.atomic(AMENDMENT, dict(
        scientific_request_sha256=k.sha(k.ROOT / 'request.json'),
        reason='JiT four-rank preflight failed exact next-step comparison; one/two ranks reproduce exactly.',
        change='Separate exact checkpoint/data/local-gradient/optimizer replay from bounded FP32 collective summation.',
        production_training_unchanged=True, sampling_and_selection_unchanged=True,
        require_runtime_world_preflight_before_training=True,
        user_sequence='Complete all SiT ideas, then all JiT ideas; retain each idea 1K scan and top-two 5K.',
        training_world_size=2, global_batch=256, local_batch=128, max_sampling_gpus=4,
        user_resource_override='Use the two currently free GPUs; do not block SiT behind JiT preflight.',
        monitoring_seconds=900,
        sources={str(p.resolve()): k.sha(p) for p in Path(__file__).parent.glob('*.py')}))
