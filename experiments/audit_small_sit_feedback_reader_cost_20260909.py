"""Separate production guidance arithmetic from optional RMS instrumentation."""
from __future__ import annotations

import time

import numpy as np
import torch

from experiments import small_sit_feedback_reader_20260909 as study
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha


@torch.inference_mode()
def main():
    study.install_infrastructure()
    request, request_hash = study.infrastructure.verify_request()
    assert read(study.ROOT / 'status.json')['phase'] == 'complete'
    rt = study.make_runtime('sit_small')
    noise = np.load(study.infrastructure.BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(study.infrastructure.BANK_ROOT / 'labels.npy', mmap_mode='r')
    z = torch.from_numpy(np.array(noise[:8])).cuda()
    rt.labels = torch.from_numpy(np.array(labels[:8])).cuda()
    t, alpha = z.new_tensor(.25), .6

    def pure(arm):
        strong, weak = rt.pair(z, t)
        tokens = study.make_tokens(rt, z, t, strong, weak)
        sp, wp = rt.readers[arm](tokens, arm)
        s, w = study.unpatchify(sp), study.unpatchify(wp)
        return s + alpha * (s - w)

    exact = {}
    for arm in study.ARMS:
        value = pure(arm)
        instrumented, _ = study.feedback_field(rt, z, t, alpha, arm)
        exact[arm] = torch.equal(value, instrumented)
        assert exact[arm]
    methods = ('native_unobserved', 'native_observed', 'strong_only_pure', 'strong_gap_pure',
               'strong_only_instrumented', 'strong_gap_instrumented')
    rows = []
    for round_id in range(6):
        order = methods[round_id:] + methods[:round_id]
        for method in order:
            rt.capture.close()
            if method != 'native_unobserved':
                rt.capture = study.Capture(rt)
            if method.startswith('native'):
                fn = lambda: rt.guided(z, t, alpha)
            else:
                arm = 'strong_gap' if method.startswith('strong_gap') else 'strong_only'
                fn = (lambda: pure(arm)) if method.endswith('_pure') else (
                    lambda: study.feedback_field(rt, z, t, alpha, arm)[0])
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            rows.append(dict(round=round_id, method=method, seconds_per_batch_query=(time.perf_counter()-start)/50))
    medians = {method: float(np.median([r['seconds_per_batch_query'] for r in rows if r['method']==method]))
               for method in methods}
    result = dict(request_sha256=request_hash, source_sha256=sha(__file__),
        fixed_state='first 8 original noise tensors queried at t=.25, alpha=.6',
        batch=8, repetitions=50, rounds=6, cyclic_order=list(methods),
        pure_outputs_equal_instrumented=exact, gpu=torch.cuda.get_device_name(),
        medians_seconds=medians, ratios_to_native={k:v/medians['native_unobserved'] for k,v in medians.items()},
        rows=rows, replaces_no_generation=False, quality_sampling_changed=False,
        limitation='isolated fixed-state latency; not an independent end-to-end throughput trial')
    atomic(study.ROOT / 'cost_audit.json', result)
    print(result['ratios_to_native'], flush=True)


if __name__ == '__main__':
    main()
