"""Rotating same-device timings for the retained head and native 66-step control."""
import time
import numpy as np
import torch
from pathlib import Path
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from . import core as m


@torch.inference_mode()
def main():
    m.configure()
    m.verify('sit_small')
    rt = c.runtime('sit_small')
    heads = local.load_heads(rt)
    noise, _, labels = c.bank('sit_small', m.STAGE)
    noise, labels = c.cuda(noise[:rt.batch]), c.cuda(labels[:rt.batch])
    seconds = {arm: [] for arm in m.ARMS}
    costs = {}
    for repeat in range(4):
        arms = m.ARMS[repeat:] + m.ARMS[:repeat]
        for arm in arms:
            capture = local.Capture(rt) if arm == 'context_base' else None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z, counts = m.sample(rt, heads, capture, noise, labels, arm)
            rt.decode(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if capture is not None:
                capture.close()
            assert counts == dict(full=132 if arm == 'native_66' else 128, prefix=0)
            costs[arm] = counts
            if repeat:
                seconds[arm].append(elapsed)
    medians = {k: float(np.median(v)) for k, v in seconds.items()}
    record = dict(complete=True, batch=rt.batch, repeats=3, seconds=seconds, medians=medians,
        costs=costs, extra_parameters=304528, includes_decode=True, additional_prefix_calls=0,
        source_sha256=c.sha(Path(__file__)), method_sha256=c.sha(Path(m.__file__)),
        context_relative_to_native=medians['context_base'] / medians['native_base'] - 1,
        native66_relative_to_context=medians['native_66'] / medians['context_base'] - 1)
    c.atomic(m.ROOT / 'sit_small/inference_benchmark.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
