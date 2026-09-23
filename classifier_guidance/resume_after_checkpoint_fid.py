"""After all paired 5K results pass validation, resume both complete GAN states."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from experiments.adversarial_weak_training_20260915 import common as c
from .checkpoint_fid_pipeline import ROOT, BASELINES, PAUSE, TRAINS, validate_metric, wait_idle

TARGET = 30128  # 128 initial critic warmup steps plus 30,000 generator updates.


def training_arguments(model, state, checkpoint):
    if state['step'] < 128 or state['step'] >= TARGET:
        raise ValueError('Expected a partially completed 30K GAN run after warmup')
    arguments = ['--model', model, '--resume', str(checkpoint), '--updates', str(TARGET-state['step']),
                 '--save-every', '300']
    if model == 'jit':
        assert state['objective'] == 'jit_block1_gan_signed_schedule'
        config = state['config']
        arguments += ['--head-checkpoint', state['provenance']['weak_checkpoint'],
                      '--coefficient', str(config['initial_extra_a']), '--diagnostic-every', '16']
        for key in ('global_batch', 'microbatch', 'warmup', 'lr_a', 'lr_d', 'r1', 'ema', 'seed'):
            arguments += ['--'+key.replace('_', '-'), str(config[key])]
        for key in ('checkpoint_feedback', 'checkpoint_backbone'):
            if config[key]:
                arguments += ['--'+key.replace('_', '-')]
    else:
        assert model == 'sit_small' and state['objective'] == 'sit_joint_signed_schedule_gap_anchor_v1'
        config = state['args']
        for key in ('head_checkpoint', 'global_batch', 'microbatch', 'warmup', 'coefficient',
                    'lr_w', 'lr_a', 'lr_d', 'norm_weight', 'r1', 'ema', 'seed', 'profile_every'):
            arguments += ['--'+key.replace('_', '-'), str(config[key])]
        arguments += ['--stop-at-epoch', '0']
        if config.get('eager'):
            arguments += ['--eager']
    return arguments


def prepare(root):
    jobs = []
    for model in ('jit', 'sit_small'):
        receipt = c.read(root/PAUSE[model])
        assert receipt['phase'] == 'paused'
        checkpoint = Path(receipt['checkpoint'])
        assert c.sha(checkpoint) == receipt['sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['step'] == receipt['step']
        # Resume the live state, including its EMA and optimizers. A FID winner
        # never substitutes an earlier or EMA-only checkpoint into training.
        for key in ('critic', 'optimizer', 'optimizer_d', 'data_rng', 'ema'):
            assert key in state
        name = 'training_after_fid_20260923_gpu0_gpu2' if model == 'jit' else 'single_after_fid_20260923'
        output = TRAINS[model].parent/name
        if output.exists():
            raise RuntimeError(f'Refusing duplicate continuation: {output}')
        gpus = '0,2' if model == 'jit' else '1'
        command = [sys.executable, '-u', '-m', 'classifier_guidance.launch',
                   '--gpus', gpus, '--allow-desktop-pid', '2766127',
                   '--task', 'jit-schedule' if model == 'jit' else 'sit-joint',
                   '--output', str(output), '--', *training_arguments(model, state, checkpoint)]
        jobs.append(dict(model=model, checkpoint=str(checkpoint), checkpoint_sha256=receipt['sha256'],
            start_step=state['step'], target_step=TARGET, remaining_updates=TARGET-state['step'],
            output=str(output), gpus=gpus, command=command, global_batch=32, microbatch=8))
    return dict(authorized_utc=c.now(), trigger='all 12 paired 5K raw/EMA evaluations completed and validated',
        target_generator_updates=30000, target_global_step=TARGET, jobs=jobs,
        optimizer_and_rng_restoration=True, automatic_restart_on_failure=False,
        initialization='latest raw checkpoint with its own EMA, D, optimizer states and RNGs',
        save_every=300, gpu3_excluded=True)


def validate_all(root):
    result = c.read(root/'results.json')
    complete = c.read(root/'pipeline_complete.json')
    if not result['complete'] or result['completed'] != 12 or not complete['complete'] or complete['errors']:
        raise ValueError('All 12 evaluations must complete successfully before continuation')
    keys = set()
    for row in result['rows']:
        key = (row['model'], row['step'], row['weights'])
        assert key not in keys
        keys.add(key)
        output = Path(row['output'])
        assert c.read(output/'exit.json')['exit_code'] == 0
        assert c.read(output/'complete.json')['complete'] and c.read(output/'parity.json')['passed']
        metric = c.read(output/'metrics.json')
        validate_metric(metric, c.read(BASELINES[row['model']]))
        assert metric['fid'] == row['fid'] and c.sha(metric['checkpoint']) == metric['checkpoint_sha256']
    assert len(keys) == 12
    return result


def start_monitor(job):
    parent = Path(job['output']).parent
    if job['model'] == 'jit':
        command = [sys.executable, '-u', '-m', 'classifier_guidance.monitor_jit_schedule',
                   '--root', str(parent), '--watch', '30']
        shell = shlex.join(command)+' > '+shlex.quote(str(parent/'monitor_after_fid.log'))+' 2>&1'
        subprocess.run(['tmux', 'new-window', '-t', 'checkpoint_fid_0923', '-n', 'jit_curves', shell], check=True)
    else:
        runs = [parent/name for name in ('training_30k', 'dual_resume_check_20260923',
            'single_resume_check_20260923', 'dual_until_deadline_20260923',
            'single_after_deadline_20260923', 'single_after_fid_20260923')]
        command = [sys.executable, '-u', '-m', 'classifier_guidance.monitor_sit_joint',
                   '--runs', *map(str, runs)]
        shell = shlex.join(command)+' > '+shlex.quote(str(parent/'monitor_after_fid.log'))+' 2>&1'
        subprocess.run(['tmux', 'respawn-window', '-k', '-t', 'sit_joint_gan_0922:curves_png', shell], check=True)


def launch(root, job):
    for gpu in job['gpus'].split(','):
        wait_idle(int(gpu))
    if (root/'STOP_RESUME').exists():
        raise RuntimeError('User stop marker cancels continuation')
    output = Path(job['output'])
    if output.exists():
        raise RuntimeError(f'Continuation already exists: {output}')
    assert c.sha(job['checkpoint']) == job['checkpoint_sha256']
    state = dict(job, phase='resuming', updated_utc=c.now())
    c.atomic(root/f"resume_{job['model']}.json", state)
    if job['model'] == 'jit':
        old = c.read(output.parent/'status.json')
        old.update(phase='resuming', active_training=output.name, resumed_step=job['start_step'],
                   target_step=TARGET, remaining_coefficient_updates_at_launch=job['remaining_updates'],
                   tmux='checkpoint_fid_0923:resume_training', started_utc=c.now())
        c.atomic(output.parent/'status.json', old)
    c.atomic(output.parent/'continuation_after_fid.json', state)
    start_monitor(job)
    with (root/f"resume_{job['model']}_launcher.log").open('w') as stream:
        process = subprocess.Popen(job['command'], cwd=c.WORK, stdout=stream, stderr=subprocess.STDOUT)
        state.update(phase='launching', launcher_pid=process.pid, updated_utc=c.now())
        c.atomic(root/f"resume_{job['model']}.json", state)
        confirmed = False
        while process.poll() is None:
            if not confirmed and (output/'progress.json').exists():
                progress = c.read(output/'progress.json')
                if progress['step'] > job['start_step']:
                    restore = c.read(output/'resume.json')
                    assert restore['weights_restored_exactly'] and restore['both_optimizers_restored']
                    assert restore['exact_global_stream']
                    state.update(phase='training', first_resumed_step=progress['step'],
                        restoration=restore, updated_utc=c.now())
                    c.atomic(root/f"resume_{job['model']}.json", state)
                    c.atomic(output.parent/'continuation_after_fid.json', state)
                    if job['model'] == 'jit':
                        status = c.read(output.parent/'status.json')
                        status.update(phase='training', step=progress['step'])
                        c.atomic(output.parent/'status.json', status)
                    confirmed = True
            time.sleep(5.)
        latest = c.read(output/'latest.json') if (output/'latest.json').exists() else {}
        state.update(phase=latest.get('phase', 'failed'), exit_code=process.returncode, updated_utc=c.now())
        c.atomic(root/f"resume_{job['model']}.json", state)
        if process.returncode:
            raise RuntimeError(f"{job['model']} continuation failed with exit {process.returncode}")


def main(root, prepare_only=False):
    lock = (root/'resume_after_fid.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = prepare(root)
    c.atomic(root/'resume_after_evaluation_plan.json', plan)
    if prepare_only:
        return
    while not (root/'pipeline_complete.json').exists():
        if (root/'STOP_RESUME').exists():
            return
        if (root/'errors_live.json').exists():
            raise RuntimeError('Evaluation errors need inspection; automatic resume withheld')
        time.sleep(5.)
    result = validate_all(root)
    best = {model: min((r for r in result['rows'] if r['model'] == model), key=lambda r: r['fid'])
            for model in BASELINES}
    c.atomic(root/'evaluation_finished_before_resume.json', dict(completed_utc=c.now(),
        completed=12, rows=result['rows'], best=best,
        summary_report=str(c.WORK/'docs/classifier_guidance/GAN_CHECKPOINT_FID_20260923_ZH.md')))
    print('All 12 paired 5K FIDs validated. Resuming the two saved training states.', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(launch, root, job) for job in plan['jobs']]):
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.root, args.prepare_only)
