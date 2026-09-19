from pathlib import Path
from experiments.guidance_strength_sweep_20260915 import config as original
from experiments.guidance_strength_sweep_20260915.config import *

MODULE = 'experiments.guidance_strength_four_gpu_20260915'
AMENDMENT = ROOT / 'four_gpu_dispatch_request.json'
SHARDS = 4
SAMPLE_BATCH = 8


def batch_starts(rank):
    assert 0 <= rank < SHARDS
    return tuple(range(0, 1000, SAMPLE_BATCH))[rank::SHARDS]


def prepare():
    original.verify()
    assert not AMENDMENT.exists(), 'An existing dispatch amendment is immutable'
    value = dict(scientific_request_sha256=sha(ROOT / 'request.json'),
        original_training_request_sha256=sha(old.ROOT / 'request.json'),
        max_gpu_workers=4, shards_per_point=SHARDS, samples_per_point=1000,
        sampling_batch=SAMPLE_BATCH, batch_assignment='original batch index modulo 4',
        preserve_input_order=True, reuse_completed_batches=True,
        one_idea_at_a_time=True, evaluator_unchanged=True,
        sources={str(p.resolve()): sha(p) for p in Path(__file__).parent.glob('*.py')})
    atomic(AMENDMENT, value)
    return value


def verify():
    request = original.verify()
    value = read(AMENDMENT)
    assert value['scientific_request_sha256'] == sha(ROOT / 'request.json')
    assert value['original_training_request_sha256'] == sha(old.ROOT / 'request.json')
    for path, digest in value['sources'].items():
        assert sha(path) == digest, path
    return request
