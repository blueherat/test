"""Repair endpoint-bank label dtypes without editing frozen experiment sources."""
from pathlib import Path
from experiments import guided_weak_deployed_20260915 as deployed
from experiments.guidance_dynamic_50k_20260915 import config as k

MODULE = 'experiments.guidance_endpoint_recovery_20260915'
ROOT = k.ROOT / 'endpoint_repair_20260915'
REQUEST = ROOT / 'request.json'
_installed = False


def verify():
    request = deployed.verify()
    amendment = k.read(REQUEST)
    assert amendment['scientific_request_sha256'] == k.sha(k.ROOT / 'request.json')
    assert amendment['deployed_request_sha256'] == k.sha(deployed.REQUEST)
    for filename, digest in amendment['sources'].items():
        assert k.sha(filename) == digest, filename
    return request


def prepare():
    deployed.verify()
    assert not REQUEST.exists(), 'Existing endpoint repair amendments are immutable'
    k.atomic(REQUEST, dict(
        scientific_request_sha256=k.sha(k.ROOT / 'request.json'),
        deployed_request_sha256=k.sha(deployed.REQUEST),
        reason='SiT endpoint labels are int16; embedding requires int32 or int64.',
        change='Cast endpoint conditioning labels to int64 before strong/native/CFG or standalone weak integration.',
        label_values_unchanged=True, endpoint_bank_storage_unchanged=True,
        noise_seed_sampler_and_counts_unchanged=True,
        existing_training_and_quality_results_unchanged=True,
        affected_sources=['strong', 'weak', 'ig', 'cfg'],
        sources={str(p.resolve()): k.sha(p) for p in Path(__file__).parent.glob('*.py')},
    ))


def patch_endpoint_calls(module):
    """Patch only the two integration references used by endpoint generation."""
    original_integrate = module.integrate
    original_weak = module.weak_integrate

    def integrate(adapter, head, method, coefficient, noise, labels):
        return original_integrate(adapter, head, method, coefficient, noise, labels.long())

    def weak_integrate(adapter, head, noise, labels):
        return original_weak(adapter, head, noise, labels.long())

    module.integrate = integrate
    module.weak_integrate = weak_integrate


def install():
    global _installed
    if _installed:
        verify()
        return
    deployed.install()
    verify()
    from experiments.guidance_dynamic_50k_20260915 import endpoints, pipeline
    patch_endpoint_calls(endpoints)
    k.verify = verify
    k.MODULE = MODULE
    original_publish = pipeline.Supervisor.publish

    def publish(self, phase=None):
        original_publish(self, phase)
        state = k.read(k.ROOT / 'status.json')
        state['endpoint_repair_sha256'] = k.sha(REQUEST)
        k.atomic(k.ROOT / 'status.json', state)

    pipeline.Supervisor.publish = publish
    _installed = True
