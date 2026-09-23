"""Unattended 5K raw/EMA evaluations after two trainers safely pause."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.adversarial_weak_training_20260915 import common as c

BASE = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance')
ROOT = BASE/'checkpoint_fid_20260923'
BASELINES = {
    'jit': BASE/'jit_ssg_capacity_20260920/blocks1/points/guided_cfg1_w1.50/metrics.json',
    'sit_small': BASE/'sit_transformer_capacity_20260921/block1/sweep_5k/full/a015/metrics.json',
}
TRAINS = {
    'jit': BASE/'jit_block1_gan_schedule_20260922/training_30k_gpu0_gpu2_g32_m8',
    'sit_small': BASE/'sit_joint_gan_20260922/single_after_deadline_20260923',
}
MIDDLE = {
    'jit': TRAINS['jit'],
    'sit_small': BASE/'sit_joint_gan_20260922/dual_until_deadline_20260923',
}
PAUSE = {'jit': 'jit_pause.json', 'sit_small': 'sit_pause.json'}
FIGURE = c.WORK/'docs/classifier_guidance/figures/gan_checkpoint_fid_20260923.png'
REPORT = c.WORK/'docs/classifier_guidance/GAN_CHECKPOINT_FID_20260923_ZH.md'


def points(root, model):
    receipt = c.read(root/PAUSE[model])
    latest = receipt.get('step', receipt['target'])
    latest_path = Path(receipt.get('checkpoint', str(TRAINS[model]/f'checkpoint_{latest:06d}.pt')))
    return [(latest, latest_path), (6000, MIDDLE[model]/'checkpoint_006000.pt'),
            (3000, MIDDLE[model]/'checkpoint_003000.pt')]


def output_for(root, model, step, weights):
    return root/model/f'step{step:06d}_{weights}'


def validate_metric(metric, baseline):
    assert metric['complete'] and metric['valid'] and metric['n'] == 5000
    assert math.isfinite(metric['fid']) and metric['evaluated_weights_unchanged']
    for key in ('noise_sha256', 'labels_sha256', 'reference_sha256', 'solver', 'precision'):
        if metric[key] != baseline[key]:
            raise ValueError(f'Unmatched evaluation protocol: {key}')
    assert metric['records'] == baseline['records']
    source_sha = metric['provenance'].get('weak_sha256', metric['provenance'].get('sha256'))
    assert source_sha == baseline['checkpoint_sha256']
    assert metric['initial_coefficient'] == baseline.get('extra_a', baseline.get('coefficient'))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    statuses = []
    baselines = {m: c.read(p) for m, p in BASELINES.items()}
    for model in BASELINES:
        for step, checkpoint in points(root, model):
            for weights in ('ema', 'raw'):
                output = output_for(root, model, step, weights)
                metric_path = output/'metrics.json'
                row = dict(model=model, step=step, weights=weights, n=5000,
                           checkpoint=str(checkpoint), output=str(output))
                if metric_path.exists() and (output/'complete.json').exists():
                    metric = c.read(metric_path)
                    validate_metric(metric, baselines[model])
                    assert metric['step'] == step and metric['weights'] == weights
                    assert c.read(output/'parity.json')['passed']
                    row.update(fid=metric['fid'], delta=metric['fid']-baselines[model]['fid'],
                               inception_score=metric['inception_score'], metrics=str(metric_path))
                    rows.append(row)
                elif (output/'progress.json').exists():
                    row.update(c.read(output/'progress.json'))
                else:
                    row['phase'] = 'waiting'
                statuses.append(row)
    c.atomic(root/'results.json', dict(complete=len(rows) == 12, completed=len(rows), total=12,
        rows=rows, status=statuses, baselines={m: dict(fid=v['fid'], metrics=str(BASELINES[m]),
            metrics_sha256=c.sha(BASELINES[m])) for m, v in baselines.items()}, updated_utc=c.now()))
    columns = ('model', 'step', 'weights', 'n', 'fid', 'delta', 'inception_score', 'metrics')
    with (root/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = {'raw': '#d55e00', 'ema': '#0072b2'}
    for ax, model, title in zip(axes, BASELINES, ('JiT: learned scales', 'SiT: joint weak head + scales')):
        baseline = baselines[model]['fid']
        ax.axhline(baseline, color='#777777', linestyle='--', label=f'Initial constant: {baseline:.4f}')
        for weights in ('ema', 'raw'):
            selected = sorted([r for r in rows if r['model'] == model and r['weights'] == weights],
                              key=lambda r: r['step'])
            if selected:
                ax.plot([r['step'] for r in selected], [r['fid'] for r in selected], 'o-',
                        color=colors[weights], label=weights.upper())
                for r in selected:
                    ax.annotate(f"{r['fid']:.3f}", (r['step'], r['fid']),
                                xytext=(0, 8 if weights == 'ema' else -16), textcoords='offset points',
                                ha='center', fontsize=9, color=colors[weights])
        if not any(r['model'] == model for r in rows):
            ax.text(.5, .4, '5K evaluations pending', transform=ax.transAxes, ha='center')
        saved_steps = sorted(step for step, _ in points(root, model))
        ax.set_xlim(0, saved_steps[-1]*1.08)
        ax.set_xticks([0, *saved_steps])
        ax.set(title=title, xlabel='Saved global training step', ylabel='FID-5K (lower is better)')
        ax.margins(y=.12)
        ax.grid(alpha=.18); ax.legend(fontsize=9)
    fig.suptitle(f'Paired checkpoint evaluation | completed {len(rows)}/12 | {c.now()[:19]} UTC', fontsize=12)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    temp = FIGURE.with_suffix('.tmp.png'); fig.savefig(temp, dpi=160); plt.close(fig); temp.replace(FIGURE)
    text = ['# 两组 GAN 的 checkpoint FID-5K 评估', '',
        f'实时更新：{c.now()}；完成 {len(rows)}/12 组。', '',
        '![FID 曲线](figures/gan_checkpoint_fid_20260923.png)', '',
        '每组均为 5,000 张，raw 与 EMA 分开测。SiT 的 weak head 与 scale 必须来自同一套 raw/EMA。'
        '每个模型内部复用原固定噪声和标签，不改变系数曲线、NFE、精度、像素量化与 FID 参考集。', '',
        'JiT：ImageNet-1000，每类 5 张，50 intervals / 99 NFE，无 CFG，初始全程 a=0.5。'
        'SiT：ImageNet-100，每类 50 张，64 Heun / 128 NFE，初始全程 a=0.75。'
        '两模型的 FID 参考集不同，绝对值不作跨模型排名。', '',
        '| 模型 | 存档步数 | 权重 | FID-5K | 相对初始 ΔFID | 状态 |',
        '|---|---:|---|---:|---:|---|']
    for model, baseline in baselines.items():
        text.append(f"| {model} | 0 | 初始常数 | {baseline['fid']:.6f} | 0 | 复用已核验基线 |")
        for row in sorted([r for r in statuses if r['model'] == model], key=lambda r: (r['step'], r['weights'])):
            if 'delta' in row:
                text.append(f"| {model} | {row['step']} | {row['weights']} | {row['fid']:.6f} | {row['delta']:+.6f} | 完成 |")
            else:
                phase = row.get('phase', 'waiting')
                text.append(f"| {model} | {row['step']} | {row['weights']} | — | — | {phase} {row.get('samples', 0)}/5000 |")
    text.extend(['', '负 ΔFID 表示相对同模型初始常数基线改善。5K 用于本轮配对筛选，'
                 '未测独立重复样本，不把很小的差值当成已证实的显著差异。', '',
                 f'- [原始结果 JSON]({root}/results.json)', f'- [结果 CSV]({root}/results.csv)'])
    for model, path in BASELINES.items():
        text.append(f'- [{model} 初始基线]({path})')
    if (root/'resume_after_evaluation_plan.json').exists():
        text.extend(['', '## 评估后续训', '',
            '按用户授权，从本轮暂停的最新完整 checkpoint 恢复训练；保留 raw、EMA、判别器、'
            '两个优化器和数据随机状态。目标为累计30K次系数/联合更新（含128步D预热共30128步），'
            '有效batch32、microbatch8，每300步保存。', '',
            '| 模型 | 恢复起点 | 当前步数（本报告刷新时） | GPU | 状态 |',
            '|---|---:|---:|---|---|'])
        for model in BASELINES:
            path = root/f'resume_{model}.json'
            if path.exists():
                state = c.read(path)
                progress_path = Path(state['output'])/'progress.json'
                progress = c.read(progress_path) if progress_path.exists() else {}
                text.append(f"| {model} | {state['start_step']} | {progress.get('step', '—')} | "
                            f"{state['gpus']} | {state['phase']} |")
        text.extend(['', f'- [完整续训参数与起点记录]({root}/resume_after_evaluation_plan.json)',
            '- [JiT 系数实时 PNG](figures/jit_gan_schedule.png)',
            '- [SiT 系数实时 PNG](figures/sit_joint_schedule.png)'])
    temp = REPORT.with_suffix('.tmp.md'); temp.write_text('\n'.join(text)+'\n'); temp.replace(REPORT)


def wait_idle(gpu):
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
    while True:
        row = next(r for r in gpu_snapshot() if r['index'] == gpu)
        # Launch revalidates that the sole allowlisted process really is the desktop service.
        row = dict(row, compute_pids=[p for p in row['compute_pids'] if p != 2766127])
        if eligible(row):
            return
        time.sleep(5.)


def lane(root, model, gpu, weights):
    while c.read(root/PAUSE[model])['phase'] != 'paused':
        if (TRAINS[model]/'exit.json').exists() and c.read(TRAINS[model]/'exit.json')['exit_code'] != 0:
            raise RuntimeError(f'{model} training exited with an error')
        time.sleep(2.)
    receipt = c.read(root/PAUSE[model])
    assert c.sha(receipt['checkpoint']) == receipt['sha256']
    for step, checkpoint in points(root, model):
        for weight in weights:
            output = output_for(root, model, step, weight)
            if (output/'complete.json').exists():
                validate_metric(c.read(output/'metrics.json'), c.read(BASELINES[model]))
                continue
            if output.exists():
                raise RuntimeError(f'Unfinished output needs inspection before retry: {output}')
            if (root/'STOP_AFTER_CURRENT').exists():
                return
            wait_idle(gpu)
            cmd = [sys.executable, '-u', '-m', 'classifier_guidance.launch', '--gpus', str(gpu),
                   '--allow-desktop-pid', '2766127', '--task', 'learned-guidance-eval',
                   '--output', str(output), '--', '--model', model, '--checkpoint', str(checkpoint),
                   '--weights', weight]
            print(json.dumps(dict(phase='launch', gpu=gpu, model=model, step=step, weights=weight)), flush=True)
            subprocess.run(cmd, cwd=c.WORK, check=True)
            validate_metric(c.read(output/'metrics.json'), c.read(BASELINES[model]))


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c.atomic(root/'evaluation_plan.json', dict(created_utc=c.now(), samples_per_point=5000,
        checkpoints='3000, 6000, paused latest', weights=['raw', 'ema'], total_evaluations=12,
        total_new_samples=60000, lanes={'GPU0': 'JiT EMA', 'GPU2': 'JiT raw', 'GPU1': 'SiT raw+EMA'},
        reuse_initial_baselines={m: str(p) for m, p in BASELINES.items()}, automatic_training_resume=False))
    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane, root, 'jit', 0, ('ema',)),
                   pool.submit(lane, root, 'jit', 2, ('raw',)),
                   pool.submit(lane, root, 'sit_small', 1, ('ema', 'raw'))]
        while not all(f.done() for f in futures):
            report(root)
            failed = [repr(f.exception()) for f in futures if f.done() and f.exception() is not None]
            if failed:
                c.atomic(root/'errors_live.json', dict(errors=failed, updated_utc=c.now()))
            time.sleep(20.)
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(repr(exc))
    report(root)
    result = c.read(root/'results.json')
    c.atomic(root/'pipeline_complete.json', dict(complete=result['complete'] and not errors,
        completed=result['completed'], errors=errors, updated_utc=c.now()))
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    report(args.root) if args.report_only else main(args.root)
