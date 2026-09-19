"""Exercise real stored labels and compare a full endpoint batch exactly."""
import fcntl
import os
from types import SimpleNamespace

import numpy as np
import torch

from . import ROOT, REQUEST, install, k, patch_endpoint_calls


def cpu():
    received = []
    sentinel = object()
    def regular(*args):
        received.append(('regular', args))
        return sentinel
    def weak(*args):
        received.append(('weak', args))
        return sentinel
    module = SimpleNamespace(integrate=regular, weak_integrate=weak)
    patch_endpoint_calls(module)
    labels = torch.tensor([0, 31, 99], dtype=torch.int16)
    noise = torch.zeros(3, 4, 2, 2)
    for method in ('strong', 'native', 'cfg_native'):
        assert module.integrate(None, None, method, .8, noise, labels) is sentinel
    assert module.weak_integrate(None, None, noise, labels) is sentinel
    for _, args in received:
        assert args[-2] is noise
        assert args[-1].dtype == torch.int64
        assert torch.equal(args[-1], labels.long())
    dtypes = {}
    for model in k.MODELS:
        for split in ('train', 'validation'):
            values = np.load(k.model_root(model) / 'data' / f'{split}_labels.npy')
            assert np.issubdtype(values.dtype, np.integer)
            converted = values.astype(np.int64)
            np.testing.assert_array_equal(values, converted)
            dtypes[f'{model}/{split}'] = str(values.dtype)
    return dict(all_endpoint_routes_cast=True, label_values_preserved=True, source_dtypes=dtypes)


def main():
    from experiments.weak_reference_loss_20260914 import idle
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    assert uuid.startswith('GPU-') and ',' not in uuid
    lease = open('/tmp/eqvae_idle_' + uuid + '.lock', 'a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    row = next(row for row in idle.gpu_snapshot() if row['uuid'] == uuid)
    assert idle.eligible(row), 'GPU is no longer idle'
    install()
    checks = cpu()
    from experiments.guidance_dynamic_50k_20260915 import endpoints, sampling
    from experiments.guidance_dynamic_50k_20260915.models import Adapter
    model = 'sit_small'
    batch = k.settings(model)['endpoint_batch']
    labels = np.load(k.model_root(model) / 'data/train_labels.npy')[:batch]
    assert labels.dtype == np.int16
    rng = np.random.default_rng(np.random.SeedSequence([2026091581, 0, 0, 0, 0]))
    initial = rng.standard_normal((batch, *k.settings(model)['shape']), dtype=np.float32)
    noise = torch.from_numpy(initial).cuda()
    small_labels = torch.from_numpy(labels).cuda()
    adapter = Adapter(model)
    reference, original_counts = sampling.integrate(adapter, None, 'strong', .8, noise, small_labels.long())
    repaired, repaired_counts = endpoints.integrate(adapter, None, 'strong', .8, noise, small_labels)
    assert torch.isfinite(repaired).all()
    assert torch.equal(reference, repaired), (reference - repaired).abs().max().item()
    assert original_counts == repaired_counts
    assert repaired_counts['full'] == 128
    k.atomic(ROOT / 'validation.json', dict(
        passed=True, endpoint_repair_sha256=k.sha(REQUEST),
        model=model, batch=batch, gpu_uuid=uuid,
        full_solver_trajectory_endpoint_exact=True,
        original_int64_vs_repaired_int16_exact=True,
        initial_noise_sha256=k.array_sha(initial),
        endpoint_sha256=k.array_sha(repaired.cpu().numpy()),
        counts=repaired_counts, **checks,
    ))
    print(k.read(ROOT / 'validation.json'), flush=True)


if __name__ == '__main__':
    main()
