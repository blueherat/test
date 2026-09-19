"""Install the validated contextual readout in place of SiT's original IG head."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from . import core as m


def install(rt):
    """Mutate this SiT runtime to use one replacement head; do not retain the old head."""
    if rt.name != 'sit_small' or rt.head.depth != 4 or rt.head.prediction_target != 'velocity':
        raise ValueError('This retained checkpoint is specific to the frozen SiT-S/2 depth4 velocity runtime')
    training = m.training_root(rt.name)
    local.verify_request(training / 'request.json')
    summary = c.read(training / 'summary.json')
    path = training / 'head.pt'
    if not summary['complete'] or c.sha(path) != summary['head_sha256']:
        raise ValueError('Retained head checkpoint failed verification')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['step'] != 3000 or state['request_sha256'] != c.sha(training / 'request.json'):
        raise ValueError('Unexpected head training provenance')
    head = local.make_head(rt).eval().requires_grad_(False)
    head.load_state_dict(state['ema']['context'], strict=True)
    original_parameters = sum(p.numel() for p in rt.head.module.parameters())
    rt.head = replace(rt.head, module=head, checkpoint=str(path), checkpoint_sha256=c.sha(path))
    return dict(parameters=sum(p.numel() for p in head.parameters()),
        replaced_parameters=original_parameters, depth=4, extra_prefix_calls=0,
        checkpoint=str(path), checkpoint_sha256=c.sha(path))


@torch.inference_mode()
def sample(rt, noise, labels):
    """Fixed original IG coefficients with the replacement reference, 64-step Heun."""
    rt.grid = torch.linspace(0, 1, 65, device='cuda')
    with rt.context():
        return c.integrate(rt, noise, labels,
            lambda z, t, left, step, substage: rt.guided(z, t, c.amount(rt, left, 'ig')))


@torch.inference_mode()
def check():
    m.configure()
    m.verify('sit_small')
    rt = c.runtime('sit_small')
    provenance = install(rt)
    noise, _, labels = c.bank('sit_small', m.STAGE)
    output, counts = sample(rt, c.cuda(noise[:8]), c.cuda(labels[:8]))
    pixels = rt.decode(output)
    original = m.ROOT / 'sit_small' / m.STAGE / 'context_base/rank0/batch0000.npz'
    with np.load(original) as data:
        np.testing.assert_array_equal(output.cpu().numpy(), data['latents'])
        np.testing.assert_array_equal(pixels, data['arr_0'])
    assert counts == dict(full=128, prefix=0)
    c.atomic(m.ROOT / 'sit_small/replacement_api_verification.json', dict(passed=True,
        original_batch=str(original), original_batch_sha256=c.sha(original), samples=8,
        latent_and_pixel_arrays_exact=True, counts=counts, provenance=provenance,
        sources=c.source_manifest([Path(__file__), Path(local.__file__)])))
    print('Replacement API reproduced the frozen 5K batch exactly', provenance, flush=True)


if __name__ == '__main__':
    check()
