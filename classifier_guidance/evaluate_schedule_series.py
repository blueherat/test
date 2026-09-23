"""Queue 5K evaluations of saved schedules after the existing final evaluation.

The saved total steps are 3000..27000 at 3000-step intervals. The final point
reuses step30128 (30000 coefficient updates after 128 D warmup steps). Initial
and final evaluations are reused; no near-duplicate step30000 evaluation runs.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c


def checkpoint_info(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['objective'] != 'binary_gan_signed_schedule' or state['model'] != 'sit_small':
        raise ValueError(f'Unexpected checkpoint objective/model: {path}')
    coefficients = state['schedule']['coefficients']
    if coefficients.shape != (64,) or coefficients.dtype != torch.float32 or not torch.isfinite(coefficients).all():
        raise ValueError(f'Invalid raw coefficient vector: {path}')
    if state['args']['steps'] != 64 or state['args']['warmup'] != 128 or state['args']['coefficient'] != .6:
        raise ValueError(f'Unexpected sampling or training protocol: {path}')
    return dict(checkpoint=str(path), checkpoint_sha256=c.sha(path),
        step=state['step'], coefficient_updates=state['step']-state['args']['warmup'],
        coefficients=coefficients.tolist(), frozen=state['frozen'], provenance=state['provenance'])


def prepare(root, gpus):
    output = root/'quality_every3k_5k'
    output.mkdir(parents=True, exist_ok=True)
    training = root/'training_30k_monitored'
    targets = [*range(3000, 30000, 3000), 30128]
    points = []
    reference = None
    for step in targets:
        path = training/f'checkpoint_{step:06d}.pt'
        point = dict(step=step, coefficient_updates=step-128, checkpoint=str(path),
            samples=5000, arm='learned', weights='raw',
            evaluation=str(root/'quality_5k' if step==30128 else output/f'step_{step:06d}'),
            reuse_existing_final=step==30128, checkpoint_present=path.exists())
        if path.exists():
            info = checkpoint_info(path)
            if info['step'] != step: raise ValueError(f'Checkpoint filename/step mismatch: {path}')
            identity = (info['frozen'], info['provenance'])
            if reference is None: reference = identity
            if identity != reference: raise ValueError('Frozen model changed between checkpoints')
            point['checkpoint_sha256'] = info['checkpoint_sha256']
        points.append(point)
    plan = dict(created_utc=c.now(), root=str(root), gpus=gpus, samples_per_point=5000,
        step_convention='Total training step, including 128 initial critic-only updates',
        final_step=30128, final_coefficient_updates=30000,
        initialization_evaluation=str(root/'quality_5k'/'initial'),
        initialization_samples=5000, learned_points=10, total_samples_including_initial=55000,
        additional_samples_after_original_pair=45000, points=points,
        sampling='64-step Heun, batch 8, same fixed 5000 noise vectors and labels, 50 images/class',
        driver_source_sha256=c.sha(__file__))
    c.atomic(output/'plan.json', plan)
    (output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return output, plan


def valid_metric(stage, baseline, info=None):
    metric = c.read(stage/'metrics.json')
    if not metric['complete'] or not metric['valid'] or metric['n'] != 5000:
        raise ValueError(f'Incomplete 5K evaluation: {stage}')
    for key in ('noise_sha256', 'labels_sha256', 'reference_sha256', 'solver', 'batch'):
        if metric[key] != baseline[key]: raise ValueError(f'Unpaired {key}: {stage}')
    if c.sha(metric['samples_path']) != metric['samples_sha256']:
        raise ValueError(f'Samples changed: {stage}')
    if not np.isfinite([metric['fid'], metric['inception_score']]).all():
        raise ValueError(f'Nonfinite evaluation: {stage}')
    if info is not None:
        if metric['arm'] != 'learned' or metric['checkpoint_sha256'] != info['checkpoint_sha256']:
            raise ValueError(f'Evaluated wrong checkpoint: {stage}')
        np.testing.assert_array_equal(np.asarray(metric['coefficients'], dtype=np.float32),
                                      np.asarray(info['coefficients'], dtype=np.float32))
    return metric


def summarize(output, plan, baseline, available):
    rows = [dict(checkpoint_step=0, coefficient_updates=0, fid=baseline['fid'],
        inception_score=baseline['inception_score'], delta_fid_from_initial=0.,
        evaluation=plan['initialization_evaluation'])]
    for point, metric in available:
        rows.append(dict(checkpoint_step=point['step'], coefficient_updates=point['coefficient_updates'],
            fid=metric['fid'], inception_score=metric['inception_score'],
            delta_fid_from_initial=metric['fid']-baseline['fid'],
            evaluation=str(Path(point['evaluation'])/'learned')))
    rows.sort(key=lambda row:row['checkpoint_step'])
    c.atomic(output/'results.json', dict(updated_utc=c.now(), samples_per_point=5000,
        paired_inputs=True, completed_learned_points=len(available), requested_learned_points=10,
        complete=len(available)==10, rows=rows,
        selection_note='Comparison uses one shared 5K input set; no independent confirmation sampling.'))
    path=output/'results.csv'
    with path.with_suffix('.tmp').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    # Unscored checkpoints remain gaps; do not connect across pending results.
    by_step={row['checkpoint_step']:row['fid'] for row in rows}
    steps=[0,*[point['step'] for point in plan['points']]]
    values=[by_step.get(step,float('nan')) for step in steps]
    ax.plot(steps,values,'o-',color='#1764ab')
    ax.axhline(baseline['fid'],color='gray',linestyle='--',label='Initial schedule')
    ax.set(xlabel='Total training step (includes 128 critic warmup steps)',ylabel='FID-5K',
           title=f'Saved schedule comparison: {len(available)}/10 checkpoints scored')
    ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'fid_by_step.png',dpi=160);fig.savefig(output/'fid_by_step.pdf');plt.close(fig)


def main(args):
    root=args.root.resolve()
    output=root/'quality_every3k_5k';output.mkdir(parents=True,exist_ok=True)
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output,plan=prepare(root,args.gpus)
        if args.prepare_only:
            print(json.dumps(plan,indent=2));return

        def status(phase,**values):
            c.atomic(output/'status.json',dict(phase=phase,updated_utc=c.now(),**values))

        try:
            # The original driver already owns training and the initial/final
            # 5K pair. Wait for process exit as well as metric files before taking GPUs.
            while True:
                pipeline=c.read(root/'pipeline_status.json')
                if pipeline['phase']=='failed':raise RuntimeError(f'Upstream pipeline failed: {pipeline}')
                exit_path=root/'quality_5k'/'exit.json'
                if exit_path.exists():
                    if c.read(exit_path)['exit_code'] != 0:raise RuntimeError('Upstream final 5K evaluation failed')
                    if pipeline['phase']=='complete':break
                progress=c.read(root/'training_30k_monitored'/'progress.json')
                status('waiting_for_training_and_initial_final_5k',
                    coefficient_updates=progress['coefficient_updates'],target_coefficient_updates=30000,
                    upstream_phase=pipeline['phase'])
                time.sleep(30)
            output,plan=prepare(root,args.gpus)
            if not all(p['checkpoint_present'] for p in plan['points']):
                raise RuntimeError('A requested completed-training checkpoint is missing')
            initial_stage=Path(plan['initialization_evaluation'])
            baseline=valid_metric(initial_stage,c.read(initial_stage/'metrics.json'))
            expected=np.r_[np.full(32,.6,dtype=np.float32),np.zeros(32,dtype=np.float32)]
            np.testing.assert_array_equal(np.asarray(baseline['coefficients'],dtype=np.float32),expected)
            if baseline['arm']!='initial':raise ValueError('Wrong initialization baseline')
            available=[]
            pending=[]
            identity=None
            for point in plan['points']:
                info=checkpoint_info(Path(point['checkpoint']))
                frozen_identity=(info['frozen'],info['provenance'])
                if identity is None:identity=frozen_identity
                if frozen_identity!=identity:raise ValueError('Frozen model differs between requested points')
                stage=Path(point['evaluation'])/'learned'
                if (stage/'metrics.json').exists():
                    if c.read(stage.parent/'exit.json')['exit_code']!=0:raise RuntimeError(f'Previous evaluation failed: {stage}')
                    available.append((point,valid_metric(stage,baseline,info)))
                else:pending.append((point,info))
            summarize(output,plan,baseline,available)
            for point,info in pending:
                run=Path(point['evaluation'])
                if run.exists() and any(run.iterdir()):
                    raise RuntimeError(f'Partial evaluation requires inspection before retry: {run}')
                # A separate job may start after the previous evaluation; honor
                # existing resource leases instead of competing with it.
                from experiments.weak_reference_loss_20260914.idle import gpu_snapshot,eligible
                while True:
                    devices=gpu_snapshot()
                    selected=[next(r for r in devices if str(r['index'])==key or r['uuid']==key)
                              for key in args.gpus.split(',')]
                    if all(eligible(row) for row in selected):break
                    status('waiting_for_idle_gpus',checkpoint_step=point['step'],completed_points=len(available))
                    time.sleep(30)
                command=[c.PYTHON,'-u','-m','classifier_guidance.launch','--gpus',args.gpus,
                    '--task','schedule-eval','--output',str(run),'--','--checkpoint',point['checkpoint'],
                    '--samples','5000','--arms','learned']
                status('evaluating',checkpoint_step=point['step'],coefficient_updates=point['coefficient_updates'],
                       completed_points=len(available),requested_points=10,command=command)
                subprocess.run(command,cwd=c.WORK,check=True)
                available.append((point,valid_metric(run/'learned',baseline,info)))
                summarize(output,plan,baseline,available)
                print(f"step={point['step']} FID-5K={available[-1][1]['fid']}",flush=True)
            status('complete',completed_points=len(available),requested_points=10,
                   results=str(output/'results.json'))
        except BaseException as error:
            status('failed',error=repr(error));traceback.print_exc();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpus',default='1,2,3')
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())
