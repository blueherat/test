"""Matched SiT readout training, followed by two guidance windows and paired 5K searches."""
import csv
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
from .evaluate_sit_transformer import ROOT, VARIANTS

lock = threading.Lock()
claimed = set()


class Device:
    def __enter__(self):
        while True:
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                raise RuntimeError('Pipeline stop marker')
            with lock:
                row = next((r for r in gpu_snapshot() if r['index'] in (1, 2, 3)
                            and r['index'] not in claimed and eligible(r)), None)
                if row is not None:
                    self.index = row['index']; claimed.add(self.index)
                    return self.index
            time.sleep(5)

    def __exit__(self, *unused):
        with lock:
            claimed.remove(self.index)


def launch(task, output, arguments):
    if (output/'complete.json').exists():
        assert c.read(output/'exit.json')['exit_code'] == 0
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Unfinished output needs inspection: {output}')
    with Device() as gpu:
        cmd = [c.PYTHON, '-u', '-m', 'classifier_guidance.launch', '--gpus', str(gpu),
               '--output', str(output), '--task', task, '--', *arguments]
        print(json.dumps(dict(event='launch', gpu=gpu, output=str(output), command=cmd)), flush=True)
        subprocess.run(cmd, cwd=c.WORK, check=True)
        assert c.read(output/'exit.json')['exit_code'] == 0
        assert c.read(output/'complete.json')['complete']


def train(variant):
    launch('diffusion-head', ROOT/variant/'training_50k',
           ['--variant', variant, '--steps', '50000', '--resume',
            str(ROOT/variant/'preflight100/checkpoint_000100.pt')])
    assert c.read(ROOT/variant/'training_50k/latest.json')['step'] == 50000
    return variant


def point(variant, tick=0, window='full', mode='guided'):
    if mode == 'baseline':
        output = ROOT/'baseline_5k'
    elif mode == 'weak':
        output = ROOT/variant/'weak_only_5k'
    else:
        output = ROOT/variant/'sweep_5k'/window/f'a{tick:03d}'
    arguments = ['--mode', mode, '--window', window, '--coefficient', str(tick/20)]
    if mode != 'baseline':
        arguments += ['--checkpoint', str(ROOT/variant/'training_50k/checkpoint_050000.pt')]
    launch('sit-adapter-eval', output, arguments)
    result = dict(variant=variant, mode=mode, window=window, coefficient=tick/20, output=str(output))
    if not c.read(output/'complete.json')['valid']:
        return dict(result, valid=False, fid=None, reason=c.read(output/'invalid.json')['reason'])
    metrics = c.read(output/'metrics.json')
    assert metrics['n'] == 5000
    assert math.isfinite(metrics['fid'])
    return dict(result, valid=True, n=5000, fid=metrics['fid'],
                inception_score=metrics['inception_score'], metrics=str(output/'metrics.json'))


def rank(values, tick):
    value = values[tick]
    return (value['fid'] if value['valid'] else math.inf, tick)


def fine_ticks(best):
    # Eight adjacent values, including the coarse best; no coefficient 2.0.
    low = max(0, min(best-3, 39-7))
    return list(range(low, low+8))


def scan(variant, window, baseline):
    arm = ROOT/variant/'sweep_5k'/window
    values = {0: dict(baseline, variant=variant, window=window, reused_shared_baseline=True)}

    def save(complete=False):
        best = min(values, key=lambda t: rank(values, t))
        result = dict(complete=complete, variant=variant, window=window,
                      points=[values[t] for t in sorted(values)], best=values[best],
                      independent_validation=False, updated_utc=c.now())
        c.atomic(arm/'results.json', result)
        return result

    # Same starting range as the previous SiT MLP experiment, in EXTRA a units.
    for tick in (20, 24, 28, 32, 36):
        values[tick] = point(variant, tick, window); save()
    lower = 20
    while lower > 4:
        best = min(values, key=lambda t: rank(values, t))
        if best not in (0, lower):
            break
        lower -= 4
        values[lower] = point(variant, lower, window); save()
    best = min(values, key=lambda t: rank(values, t))
    fine = fine_ticks(best)
    assert len(fine) == 8 and all(b-a == 1 for a, b in zip(fine, fine[1:]))
    c.atomic(arm/'fine_plan.json', dict(coarse_best=best/20, coefficients=[t/20 for t in fine],
             count=8, spacing=.05, n_each=5000, reused=[t/20 for t in fine if t in values],
             formula='S+a*f(t)*(S-W)', no_independent_validation=True))
    for tick in fine:
        if tick not in values:
            values[tick] = point(variant, tick, window); save()
    result = save(complete=True)
    result.update(fine_coefficients=[t/20 for t in fine],
                  best_at_upper_boundary=result['best']['coefficient'] == max(values)/20)
    c.atomic(arm/'results.json', result)
    return result


def main():
    assert c.read(ROOT/'structure_audit/result.json')['passed']
    assert c.read(ROOT/'resume_audit/result.json')['passed']
    for name in ('block2_legacy', 'block1_full', 'block2_weak'):
        assert c.read(ROOT/'sampler_audits'/name/'parity.json')['unaccelerated_endpoint_exact']
    plan = dict(created_utc=c.now(), variants=list(VARIANTS), depth=4,
                source='SiT-S/2 ImageNet100 800K EMA, frozen',
                data='full 126689 real training images; fresh VAE posterior, time and noise',
                steps=50000, global_batch=256, lr=1e-4, betas=[.9, .999], ema=.9999,
                precision='BF16 training; FP32/TF32 deployed sampling',
                train_order=['block2', 'block1', 'shallow', 'linear'],
                initialization='fresh MLP; copied strong final for linear; copied last 1/2 blocks for adapters',
                objective='ordinary velocity FM MSE; no GAN and no guided-target objective',
                coarse_extra_a=[1., 1.2, 1.4, 1.6, 1.8],
                lower_extension='0.2 downward if lower boundary or a=0 wins',
                fine_points=8, fine_spacing=.05, upper_fine_a=1.95,
                windows=dict(legacy='6/7 before .25, 1 before .5, then 0', full='1 throughout'),
                formula='S+a*f(t)*(S-W); paper total w=1+a when f=1',
                samples_per_point=5000, paired_noise_labels=True, no_independent_validation=True,
                additional_evaluations='shared no-guidance 5K; weak-only 5K each; paired 50K validation states',
                gpus='idle GPUs 1,2,3 via leases')
    c.atomic(ROOT/'plan.json', plan)
    c.atomic(ROOT/'status.json', dict(phase='training', updated_utc=c.now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for done in as_completed([pool.submit(train, v) for v in plan['train_order']]):
            print('finished 50K', done.result(), flush=True)
    inputs = [c.read(ROOT/v/'training_50k/first_batches.json') for v in VARIANTS]
    assert all(value == inputs[0] for value in inputs)
    c.atomic(ROOT/'paired_training_inputs.json', dict(passed=True, first_four_resumed_batches_identical=True))
    c.atomic(ROOT/'status.json', dict(phase='baseline_and_head_quality', updated_utc=c.now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        baseline_job = pool.submit(point, 'shallow', mode='baseline')
        prediction_job = pool.submit(launch, 'sit-adapter-eval', ROOT/'prediction_audit', ['--mode', 'prediction'])
        weak_jobs = {v: pool.submit(point, v, mode='weak') for v in VARIANTS}
        baseline = baseline_job.result(); prediction_job.result()
        weak = {v: job.result() for v, job in weak_jobs.items()}
    c.atomic(ROOT/'baseline_and_weak.json', dict(baseline=baseline, weak=weak))
    c.atomic(ROOT/'status.json', dict(phase='5k_sweeps', updated_utc=c.now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(scan, v, w, baseline) for v in VARIANTS for w in ('legacy', 'full')]
        arms = [job.result() for job in as_completed(jobs)]
    arms.sort(key=lambda r: (r['variant'], r['window']))
    c.atomic(ROOT/'results.json', dict(complete=True, baseline=baseline, weak=weak, arms=arms,
                                     prediction_audit=str(ROOT/'prediction_audit/result.json')))
    with (ROOT/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'mode', 'window', 'coefficient', 'valid', 'n', 'fid', 'inception_score', 'output'], extrasaction='ignore')
        writer.writeheader(); writer.writerows([baseline, *weak.values()])
        for arm in arms:
            writer.writerows(arm['points'])
    c.atomic(ROOT/'status.json', dict(phase='complete', updated_utc=c.now()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        c.atomic(ROOT/'status.json', dict(phase='error', error=repr(exc), updated_utc=c.now()))
        raise
