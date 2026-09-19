"""Same-device full sampling timings, with the native head-only path unhooked."""
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from . import core as m


@torch.inference_mode()
def benchmark(model):
    m.configure()
    m.verify(model)
    rt = c.runtime(model)
    heads = local.load_heads(rt)
    noise, _, labels = c.bank(model, m.STAGE)
    noise, labels = c.cuda(noise[:rt.batch]), c.cuda(labels[:rt.batch])
    arms = ('native_base', 'local_base', 'context_base')
    seconds = {arm: [] for arm in arms}
    costs = {}
    for repeat in range(4):
        order = arms[repeat % 3:] + arms[:repeat % 3]
        for arm in order:
            capture = local.Capture(rt) if not arm.startswith('native') else None
            torch.cuda.synchronize()
            start = time.perf_counter()
            z, counts = m.sample(rt, heads, capture, noise, labels, arm)
            rt.decode(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if capture is not None:
                capture.close()
            assert counts == dict(full=128 if model == 'sit_small' else 100, prefix=0)
            costs[arm] = counts
            if repeat:
                seconds[arm].append(elapsed)
    medians = {arm: float(np.median(times)) for arm, times in seconds.items()}
    parameters = c.read(m.training_root(model) / 'summary.json')['parameters_per_head']
    record = dict(complete=True, model=model, batch=rt.batch, repeats=3, seconds=seconds,
                  medians=medians, costs=costs, extra_parameters_per_candidate=parameters,
                  native_capture_hooks=False, candidates_capture_hooks=True,
                  timing_includes_decode=True, additional_prefix_calls=0,
                  relative_changes={arm: medians[arm] / medians['native_base'] - 1 for arm in arms[1:]},
                  source_sha256=c.sha(Path(__file__)), method_sha256=c.sha(Path(m.__file__)))
    c.atomic(m.ROOT / model / 'inference_benchmark.json', record)
    print(model, record['relative_changes'], flush=True)
    del rt, heads
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    while Path('/proc', str(args.wait_pid)).exists():
        time.sleep(5)
    for model in c.MODELS:
        benchmark(model)
