"""Confirm apparent 5K wins using the original CPU ADM evaluator.

The completed GPU metrics and coefficient selection remain immutable. CPU
confirmation uses exactly the same saved samples and a separate output folder.
This launches neither training nor sampling, and never resumes the old queue.
"""

import argparse
import os
import subprocess
import time
from . import common as c


BASELINE = 36.688847797154665


def main(args):
    run = c.ROOT / args.run
    promotion = run / f'promotion_step{args.step:06d}'
    output = run / f'confirmation_step{args.step:06d}'
    output.mkdir(parents=True, exist_ok=True)

    def publish(phase, **values):
        c.atomic(output / 'status.json', dict(phase=phase, updated_utc=c.now(), **values))

    try:
        while not (promotion / 'result.json').exists():
            if c.stopped():
                publish('paused')
                return
            for path in (run / 'training/status.json', promotion / 'status.json'):
                if path.exists() and c.read(path).get('phase') in ('failed', 'paused', 'needs_research_decision'):
                    raise RuntimeError(f'Upstream did not complete full evaluation: {path}')
            publish('waiting_complete_evaluation')
            time.sleep(10)
        result = c.read(promotion / 'result.json')
        weights = result['weights']
        checks = []
        for row in sorted(result['result']['results5k'], key=lambda item: item['fid']):
            if row['fid'] >= BASELINE:
                continue
            point = run / 'quality' / f'step{args.step:06d}_{weights}' / f"c{row['tick']:04d}"
            stage = point / 'n5000_cpu_confirmation'
            stage.mkdir(parents=True, exist_ok=True)
            summary = c.read(point / 'n5000/summary.json')
            if (stage / 'summary.json').exists():
                assert c.read(stage / 'summary.json') == summary
            else:
                c.atomic(stage / 'summary.json', summary)
            assert summary['samples_sha256'] == row['samples_sha256']
            publish('confirming_cpu_adm', coefficient=row['coefficient'], gpu_fid=row['fid'])
            if not (stage / 'metrics.json').exists():
                command = [c.PYTHON, '-u', '-m', 'experiments.adversarial_weak_training_20260915.score',
                           '--stage', str(stage)]
                with (stage / 'confirmation.log').open('a') as stream:
                    subprocess.run(command, cwd=c.WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
            cpu = c.read(stage / 'metrics.json')
            assert cpu['samples_sha256'] == row['samples_sha256']
            assert cpu['reference_sha256'] == row['reference_sha256']
            assert cpu['evaluator'] == 'unchanged CPU ADM'
            delta = cpu['fid'] - row['fid']
            check = dict(coefficient=row['coefficient'], gpu_fid=row['fid'], cpu_fid=cpu['fid'],
                delta_cpu_minus_gpu=delta, same_samples=True,
                beats_previous_best_cpu=cpu['fid'] < BASELINE,
                cpu_metrics=str(stage / 'metrics.json'))
            checks.append(check)
            c.atomic(output / 'checks.json', checks)
            if abs(delta) > 1e-4:
                raise RuntimeError(f'ADM CPU/CUDA FID discrepancy exceeds validated tolerance: {check}')
        report = dict(complete=True, run=args.run, step=args.step, weights=weights,
            previous_best_5k_fid=BASELINE, checks=checks,
            beats_previous_best_cpu=any(item['beats_previous_best_cpu'] for item in checks),
            quality_scope='Same fixed 5K inputs and reference; no statistical significance or cross-model claim',
            completed_utc=c.now())
        c.atomic(output / 'result.json', report)
        publish('complete', cpu_confirmed_improvement=report['beats_previous_best_cpu'], confirmed_candidates=len(checks))
        print(report, flush=True)
    except Exception as error:
        publish('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, required=True)
    main(parser.parse_args())
