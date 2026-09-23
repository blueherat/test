"""Detached two-arm diffusion training followed by paired 5K coefficient search."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys

from experiments.adversarial_weak_training_20260915 import common as c


def fine_ticks(best, upper=36):
    # Ticks are exactly 0.05; eight points, including the coarse best.
    start = max(0, min(best - 3, upper - 7))
    return list(range(start, start + 8))


def main(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    variants = {'moderate': '1', 'large': '2'}
    plan = dict(created_utc=c.now(), variants=variants, steps=50000, global_batch=256,
                objective='ordinary real-data flow-matching MSE; no GAN or guided-target loss',
                samples_per_point=5000, coarse_coefficients=[1.,1.2,1.4,1.6,1.8],
                lower_boundary='extend downward by 0.2 until bracketed or a=0',
                fine='exactly 8 adjacent coefficients spaced 0.05 around coarse best; reuse overlaps',
                upper_bound=1.8, no_independent_5k=True,
                formula='S+a*f(t)*(S-W), f=6/7 before .25, 1 before .5, 0 afterwards',
                selected_weights='50K EMA, matching old diffusion head protocol')
    if not (root/'plan.json').exists(): c.atomic(root/'plan.json', plan)

    def execute(variant, task, output, extra):
        if (output/'exit.json').exists():
            if c.read(output/'exit.json')['exit_code'] != 0:
                raise RuntimeError(f'Failed stage needs inspection: {output}')
            return
        command = [sys.executable, '-u', '-m', 'classifier_guidance.launch',
                   '--gpus', variants[variant], '--output', str(output), '--task', task, '--', *extra]
        with (root/f'{variant}_{output.name}_launcher.log').open('a') as log:
            subprocess.run(command, cwd=c.WORK, stdout=log, stderr=subprocess.STDOUT, check=True)

    def smoke(variant):
        run = root/variant/'smoke100'
        execute(variant, 'diffusion-head', run, ['--variant', variant, '--steps', '100'])
        latest = c.read(run/'latest.json')
        if not (latest['replicas_identical'] and latest['strong_unchanged']
                and latest['step'] == 100 and all(v > 0 and math.isfinite(v) for v in latest['parameter_deltas'].values())):
            raise RuntimeError(f'Invalid training smoke: {variant}')
        if not c.read(run/'objective_audit.json')['passed']: raise RuntimeError('MSE audit failed')
        return variant

    def work(variant):
        run = root/variant/'training_50k'
        execute(variant, 'diffusion-head', run,
                ['--variant', variant, '--steps', '50000', '--resume',
                 str(root/variant/'smoke100/checkpoint_000100.pt')])
        checkpoint = run/'checkpoint_050000.pt'
        values = {}
        def point(tick, phase):
            output = root/variant/'sweep_5k'/f'a{tick:03d}'
            execute(variant, 'capacity-eval', output,
                    ['--checkpoint', str(checkpoint), '--coefficient', str(tick/20)])
            metrics = c.read(output/'samples/metrics.json')
            values[tick] = dict(coefficient=tick/20, fid=metrics['fid'],
                                inception_score=metrics['inception_score'], phase=phase,
                                metrics=str(output/'samples/metrics.json'))
            c.atomic(root/variant/'results.json', dict(complete=False, points=list(values.values()),
                best=min(values.values(), key=lambda r:(r['fid'],r['coefficient'])), updated_utc=c.now()))
        coarse = list(range(20, 37, 4))
        for tick in coarse: point(tick, 'coarse')
        while min(values, key=lambda t:(values[t]['fid'],t)) == min(values) and min(values) > 0:
            point(max(0, min(values)-4), 'coarse_lower_extension')
        best = min(values, key=lambda t:(values[t]['fid'],t))
        fine = fine_ticks(best)
        c.atomic(root/variant/'fine_plan.json', dict(coarse_best=best/20,
            coefficients=[t/20 for t in fine], reused=[t/20 for t in fine if t in values]))
        for tick in fine:
            if tick not in values: point(tick, 'fine')
        result = dict(complete=True, points=sorted(values.values(),key=lambda r:r['coefficient']),
                      best=min(values.values(),key=lambda r:(r['fid'],r['coefficient'])),
                      fine_coefficients=[t/20 for t in fine], independent_validation=False,
                      search_upper_boundary_best=max(values)==min(values,key=lambda t:(values[t]['fid'],t)),
                      checkpoint=str(checkpoint), updated_utc=c.now())
        c.atomic(root/variant/'results.json', result)
        return variant, result

    try:
        c.atomic(root/'status.json', dict(phase='smoke', updated_utc=c.now()))
        with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(smoke, variants))
        a = c.read(root/'moderate/smoke100/first_batches.json')
        b = c.read(root/'large/smoke100/first_batches.json')
        if a != b: raise RuntimeError('Architecture variants used different initial training inputs')
        c.atomic(root/'paired_data_audit.json', dict(passed=True, first_four_batches_identical=True,
            dynamic_noise= len({x['noise_sha256'] for x in a}) == 4))
        c.atomic(root/'status.json', dict(phase='training_then_5k_search', updated_utc=c.now()))
        with ThreadPoolExecutor(max_workers=2) as pool: results = dict(pool.map(work, variants))
        c.atomic(root/'results.json', dict(complete=True, variants=results, updated_utc=c.now()))
        c.atomic(root/'status.json', dict(phase='complete', updated_utc=c.now()))
    except BaseException as error:
        c.atomic(root/'status.json', dict(phase='failed', error=repr(error), updated_utc=c.now()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
