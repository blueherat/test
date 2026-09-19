"""Check an equivalent implementation that replaces, rather than also evaluates, the native head."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from . import core as m


@torch.inference_mode()
def main():
    m.configure()
    m.verify('sit_small')
    rt = c.runtime('sit_small')
    heads = local.load_heads(rt)
    native = rt.head
    checkpoint = m.training_root('sit_small') / 'head.pt'
    direct = replace(native, module=heads['context'], checkpoint=str(checkpoint), checkpoint_sha256=c.sha(checkpoint))
    assert direct.prediction_target == 'velocity' and direct.depth == 4
    noise, _, labels = c.bank('sit_small', m.STAGE)
    checks = []
    for start in (0, 8, 16):
        x, y = c.cuda(noise[start:start + 8]), c.cuda(labels[start:start + 8])
        rt.head = native
        capture = local.Capture(rt)
        first, counts1 = m.sample(rt, heads, capture, x, y, 'context_base')
        pixels1 = rt.decode(first)
        capture.close()
        rt.head = direct
        second, counts2 = m.sample(rt, heads, None, x, y, 'native_base')
        pixels2 = rt.decode(second)
        assert counts1 == counts2 == dict(full=128, prefix=0)
        error = float((first - second).abs().max())
        assert torch.equal(first, second) and np.array_equal(pixels1, pixels2), error
        checks.append(dict(start=start, samples=8, latent_max_difference=error, pixels_exact=True,
            full_calls=128, prefix_calls=0))
    arms = ('native_base', 'context_capture', 'context_direct', 'native_66', 'adg')
    seconds = {arm: [] for arm in arms}
    x, y = c.cuda(noise[:8]), c.cuda(labels[:8])
    for repeat in range(4):
        for arm in arms[repeat:] + arms[:repeat]:
            rt.head = direct if arm == 'context_direct' else native
            kind = 'context_base' if arm == 'context_capture' else 'native_base' if arm == 'context_direct' else arm
            capture = local.Capture(rt) if arm == 'context_capture' else None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = m.sample(rt, heads, capture, x, y, kind)
            rt.decode(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            assert counts == dict(full=132 if arm == 'native_66' else 128, prefix=0)
            if capture is not None:
                capture.close()
            if repeat:
                seconds[arm].append(elapsed)
    medians = {k: float(np.median(v)) for k, v in seconds.items()}
    params = {name: sum(p.numel() for p in spec.module.parameters()) for name, spec in [('native', native), ('context', direct)]}
    record = dict(complete=True, checks=checks, total_equivalence_samples=24, equal_latents_and_pixels=True,
        single_replacement_head=True, additional_prefix_calls=0, seconds=seconds, medians=medians,
        batch=8, repeats=3, includes_decode=True, parameters=params, net_added_parameters=params['context'] - params['native'],
        context_direct_relative_to_native=medians['context_direct'] / medians['native_base'] - 1,
        context_direct_relative_to_capture=medians['context_direct'] / medians['context_capture'] - 1,
        source_sha256=c.sha(Path(__file__)), frozen_method_sha256=c.sha(Path(m.__file__)),
        checkpoint_sha256=c.sha(checkpoint), optimization_does_not_change_frozen_5k_generation=True,
        cost_note='Both modules are resident for paired benchmarking; a deployed replacement needs only its chosen reference head.')
    c.atomic(m.ROOT / 'sit_small/direct_readout_verification.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int, default=0)
    args = p.parse_args()
    while args.wait_pid and Path('/proc', str(args.wait_pid)).exists():
        time.sleep(5)
    main()
