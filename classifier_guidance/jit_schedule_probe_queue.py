"""GPU2-only performance comparisons, numerical gates, then a bounded GAN pilot."""
import json
import argparse
import subprocess
import time

import numpy as np

from experiments.adversarial_weak_training_20260915 import common as c
from .jit_schedule import ROOT, HEAD


def run(name, task, arguments):
    output = ROOT/name
    if (output/'complete.json').exists():
        assert c.read(output/'exit.json')['exit_code']==0
        return output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Unfinished output requires inspection: {output}')
    command = [c.PYTHON, '-u', '-m', 'classifier_guidance.launch', '--gpus', '2',
               '--output', str(output), '--task', task, '--', '--model', 'jit',
               '--head-checkpoint', str(HEAD), *arguments]
    print(json.dumps(dict(event='launch', command=command)), flush=True)
    subprocess.run(command, cwd=c.WORK, check=True)
    assert c.read(output/'complete.json')['complete'] and c.read(output/'exit.json')['exit_code']==0
    return output


def compare(base, candidate):
    expected, actual = (c.read(p/'result.json') for p in (base, candidate))
    dg = []
    for i in range(1,4):
        a = np.load(candidate/f'gradient_{i}.npy').astype(np.float64)
        b = np.load(base/f'gradient_{i}.npy').astype(np.float64)
        dg.append(dict(relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
                       cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))))
    m = np.array([r['metrics'] for r in actual['iterations'][1:]])
    n = np.array([r['metrics'] for r in expected['iterations'][1:]])
    d_delta = float(np.max(np.abs(np.load(candidate/'critic_after.npy')-np.load(base/'critic_after.npy'))))
    passed = bool(np.allclose(m[:,[0,1,2,4,5,6]], n[:,[0,1,2,4,5,6]], rtol=1e-6, atol=1e-7)
                  and d_delta==0 and max(x['relative'] for x in dg)<.01
                  and min(x['cosine'] for x in dg)>.9999)
    result = dict(passed=passed, gradients=dg, critic_update_max_abs=d_delta,
                  base=str(base), candidate=str(candidate))
    c.atomic(candidate/'comparison.json', result)
    assert passed, result
    return result


def main():
    assert c.read(ROOT/'audit_fresh_leaf/result.json')['passed']
    base = ROOT/'benchmarks/baseline_b4'
    # This initial baseline was started interactively on GPU2; finish it first.
    deadline = time.monotonic()+900
    while not (base/'exit.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Baseline did not finish')
        time.sleep(3)
    assert c.read(base/'exit.json')['exit_code']==0
    c.atomic(ROOT/'status.json', dict(phase='paired_benchmarks', updated_utc=c.now()))
    choices = dict(graph_b4=[], precast_b4=['--precast'],
                   feedback_b4=['--precast','--checkpoint-feedback'],
                   backbone_b4=['--precast','--checkpoint-feedback','--checkpoint-backbone'])
    results = {'baseline_b4':c.read(base/'result.json')}
    for name, flags in choices.items():
        output = run(f'benchmarks/{name}', 'jit-schedule-probe', ['--mode','optimized','--batch','4',*flags])
        compare(base, output); results[name] = c.read(output/'result.json')
        c.atomic(ROOT/'benchmark_summary.json', dict(complete=False, configurations=results))
    # Training always materializes BF16 weights. Keep recomputation only if its
    # measured tradeoff is useful; compare all configurations at equal batch.
    speed = min(('precast_b4','feedback_b4','backbone_b4'), key=lambda n:results[n]['median_seconds'])
    flags = choices[speed]
    c.atomic(ROOT/'status.json', dict(phase='batch_throughput', fastest_b4=speed, updated_utc=c.now()))
    batches = {4:results[speed]}
    for batch in (8,16):
        output = run(f'benchmarks/selected_b{batch}', 'jit-schedule-probe',
                     ['--mode','optimized','--batch',str(batch),*flags])
        batches[batch] = c.read(output/'result.json')
    selected = max(batches, key=lambda b:b/batches[b]['median_seconds'])
    report = dict(complete=True, configurations=results, throughput=batches,
                  fastest_b4=speed, selected_batch=selected, selected_flags=flags,
                  batch_changes_are_throughput_experiments_not_paired_gradient_comparisons=True,
                  initial_extra_a=.5, initial_total_w=1.5,
                  pilot=dict(critic_warmup=128, schedule_updates=128, lr_a=.001, lr_d=.0001))
    c.atomic(ROOT/'benchmark_summary.json', report)
    c.atomic(ROOT/'status.json', dict(phase='pilot', batch=selected, updated_utc=c.now()))
    run('pilot', 'jit-schedule', ['--global-batch',str(selected),'--warmup','128','--updates','128',
                                 *[f for f in flags if f!='--precast']])
    c.atomic(ROOT/'status.json', dict(phase='complete', updated_utc=c.now()))


def resume_with_batch(batch):
    previous = ROOT/'pilot'
    latest = c.read(previous/'latest.json')
    assert latest['phase']=='paused' and c.read(previous/'exit.json')['exit_code']==0
    summary = c.read(ROOT/'benchmark_summary.json')
    flags = summary['selected_flags']
    target = c.read(previous/'progress.json')['target_step']
    remaining = target-latest['step']
    assert remaining>0
    c.atomic(ROOT/'status.json', dict(phase='requested_batch_benchmark', batch=batch,
        previous_training_paused=True, previous_step=latest['step'], updated_utc=c.now()))
    output = run(f'benchmarks/selected_b{batch}', 'jit-schedule-probe',
                 ['--mode','optimized','--batch',str(batch),*flags])
    summary['throughput'][str(batch)] = c.read(output/'result.json')
    summary.update(active_batch=batch, active_batch_reason='user requested modest increase',
                   active_training=f'pilot_b{batch}')
    c.atomic(ROOT/'benchmark_summary.json', summary)
    c.atomic(ROOT/'status.json', dict(phase='pilot', batch=batch, active_training=f'pilot_b{batch}',
        resumed_step=latest['step'], target_step=target, updated_utc=c.now()))
    run(f'pilot_b{batch}', 'jit-schedule', ['--global-batch',str(batch),'--warmup','128',
        '--updates',str(remaining),'--resume',latest['checkpoint'],'--allow-batch-change',
        *[f for f in flags if f!='--precast']])
    c.atomic(ROOT/'status.json', dict(phase='complete', batch=batch, active_training=f'pilot_b{batch}', updated_utc=c.now()))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-with-batch',type=int)
    args = parser.parse_args()
    try:
        if args.resume_with_batch is None: main()
        else: resume_with_batch(args.resume_with_batch)
    except Exception as error:
        c.atomic(ROOT/'status.json', dict(phase='error', error=repr(error), updated_utc=c.now()))
        raise
