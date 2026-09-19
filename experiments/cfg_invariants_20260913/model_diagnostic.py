"""Real paired-data weak Bayes compatibility audit; diagnostic, never guidance loss.

Run with CUDA_VISIBLE_DEVICES=2 python -m experiments.cfg_invariants_20260913.model_diagnostic
All finite differences are of score gaps t/(1-t)*(v_cond-v_null), in FP32.
The zero reference is an expectation under a compatible model joint, not a
per-sample target. Evaluation on q_data mixes model compatibility and data error.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.guidance_pasted_20260912.common import runtime, atomic, sha
from experiments.lifting_scale_sweep_20260909 import SMALL_CKPT

DATA = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae')
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/diagnostic')
REPORT = Path(__file__).resolve().parents[2] / 'docs/research/cfg_invariants_20260913/model_diagnostic_results.md'
TIMES = (.15, .35, .5, .7, .9)
DIM = 4 * 32 * 32


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def make_inputs(args):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['source']['posterior_layout'] == 'channels 0:4 mean, channels 4:8 standard deviation'
    moments = np.load(DATA / 'validation_moments.npy', mmap_mode='r')
    labels_all = np.load(DATA / 'validation_labels.npy')
    source_all = np.load(DATA / 'validation_source_indices.npy')
    assert moments.shape == (5000, 8, 32, 32)
    assert len(labels_all) == len(moments)
    assert args.samples <= len(moments)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(moments), args.samples, replace=False)
    selected = np.array(moments[indices], dtype=np.float32)
    assert np.isfinite(selected).all() and (selected[:, 4:] >= 0).all()
    posterior_noise = rng.standard_normal((args.samples, 4, 32, 32), dtype=np.float32)
    clean = (selected[:, :4] + selected[:, 4:] * posterior_noise) * np.float32(.18215)
    labels = labels_all[indices].astype(np.int64)
    permutation = rng.permutation(args.samples)
    shuffle = labels[permutation]
    noise = rng.standard_normal((len(TIMES), args.samples, 4, 32, 32), dtype=np.float32)
    probes = (2 * rng.integers(0, 2, (len(TIMES), args.samples, args.probes, 4, 32, 32), dtype=np.int8) - 1).astype(np.float32)
    # Fixed across images: random probes varying by image would make a first-order
    # mean vanish trivially, for an arbitrary g. These fixed directions do not.
    fixed = (2 * rng.integers(0, 2, (4, 4, 32, 32), dtype=np.int8) - 1).astype(np.float32) / np.sqrt(DIM)
    np.savez(args.output / 'inputs.npz', indices=indices, source_indices=source_all[indices],
             labels=labels, shuffle_labels=shuffle, shuffle_permutation=permutation,
             clean=clean, posterior_noise=posterior_noise, noise=noise, probes=probes,
             fixed_first_order_directions=fixed, times=np.array(TIMES))
    files = ['manifest.json', 'validation_moments.npy', 'validation_labels.npy', 'validation_source_indices.npy']
    return dict(clean=clean, labels=labels, shuffle_labels=shuffle, noise=noise,
                probes=probes, fixed=fixed), dict(
        real_data=True, split='official validation cache, no generated images',
        population_count=len(moments), unique_classes=int(len(np.unique(labels))),
        shuffle_unchanged_count=int(np.sum(labels == shuffle)),
        posterior_layout=manifest['source']['posterior_layout'], vae_scaling_factor=.18215,
        source_manifest=manifest, source_sha256={str(DATA / f): sha(DATA / f) for f in files})


@torch.inference_mode()
def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('This delegated diagnostic is assigned physical GPU2 only; set CUDA_VISIBLE_DEVICES=2.')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.npz').exists():
        raise RuntimeError('Completed result exists: use --summarize or choose a new output.')
    bank, data_meta = make_inputs(args)
    start_time = time.monotonic()
    log('Loading SiT-S/2 EMA; 128 real validation samples by default, five noise strata.')
    rt = runtime('sit_small')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rt.model.float().eval().requires_grad_(False)
    assert not rt.autocast
    assert all(p.dtype == torch.float32 for p in rt.model.parameters())
    # Decoder and internal head are runtime conveniences, unused in this audit.
    n, k = args.samples, args.probes
    trace = np.zeros((len(TIMES), n, 2, k), np.float64)
    fisher = np.zeros_like(trace)
    first = np.zeros((len(TIMES), n, 2, 4), np.float64)
    score_energy = np.zeros((len(TIMES), n, 2), np.float64)
    fm_mse = np.zeros((len(TIMES), n, 3), np.float64)
    stable_n = min(args.stability_samples, n)
    stable_k = min(2, k)
    stable = np.zeros((len(TIMES), stable_n, 2, stable_k, 2), np.float64)
    eval_samples = 0
    max_repeat_diff = 0.
    fixed = torch.from_numpy(bank['fixed']).cuda().flatten(1).double()
    metadata = dict(samples=n, probes=k, times=TIMES, seed=args.seed, batch=args.batch,
                    fd_coordinate_step=args.step, stability_samples=stable_n,
                    stability_probes=stable_k, stability_multipliers=[.5, 2.],
                    score_gap='t/(1-t)*(v_cond-v_null)',
                    trace_estimator='u.T @ (g(z+h*u)-g(z-h*u))/(2*h) / D; iid Rademacher coordinates',
                    fisher_estimator='(u.T @ g)**2 / D',
                    step_scaling='h=step*max(1,RMS(z)), held fixed at +/- probes',
                    first_order='four fixed unit Rademacher directions dotted with g; fixed over images',
                    fm_target='clean - epsilon; z=t*clean+(1-t)*epsilon',
                    expectation_scope='q_data real joint; combines internal model mismatch and model/data mismatch',
                    dtype='float32 network and perturbations; float64 scalar accumulation',
                    autocast=False, tf32=False, gpu=torch.cuda.get_device_name(0),
                    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    checkpoint=str(SMALL_CKPT), checkpoint_sha256=sha(SMALL_CKPT),
                    source_sha256={str(Path(__file__)): sha(__file__), **rt.sources},
                    runtime_metadata=rt.metadata, data=data_meta,
                    first_order_ci_note='four exploratory readouts; no multiple-testing correction',
                    bootstrap_note='pointwise 95% image bootstrap; each image includes its probes; 2000 resamples')
    atomic(args.output / 'metadata.json', metadata)

    def branches(z, t, label, shuffled):
        nonlocal eval_samples
        rt.labels = torch.cat([label, torch.full_like(label, 100), shuffled])
        with torch.autocast('cuda', enabled=False):
            prediction = rt.field(torch.cat([z, z, z]), t, 'full')
        assert prediction.dtype == torch.float32 and torch.isfinite(prediction).all()
        eval_samples += len(z) * 3
        return prediction.chunk(3)

    def gaps(z, t, label, shuffled):
        cond, null, shuf = branches(z, t, label, shuffled)
        return torch.stack([cond - null, shuf - null], 1) * (t / (1 - t))

    for ti, time_value in enumerate(TIMES):
        stratum_start = time.monotonic()
        t = torch.tensor(time_value, device='cuda', dtype=torch.float32)
        for a in range(0, n, args.batch):
            b = min(n, a + args.batch)
            clean = torch.from_numpy(bank['clean'][a:b]).cuda()
            noise = torch.from_numpy(bank['noise'][ti, a:b]).cuda()
            label = torch.from_numpy(bank['labels'][a:b]).cuda()
            shuffled = torch.from_numpy(bank['shuffle_labels'][a:b]).cuda()
            z = t * clean + (1 - t) * noise
            target = clean - noise
            predictions = branches(z, t, label, shuffled)
            baseline_gap = torch.stack([predictions[0] - predictions[1], predictions[2] - predictions[1]], 1) * (t / (1 - t))
            flat_gap = baseline_gap.flatten(2).double()
            score_energy[ti, a:b] = flat_gap.square().mean(2).cpu().numpy()
            first[ti, a:b] = torch.einsum('bcd,kd->bck', flat_gap, fixed).cpu().numpy()
            for branch, prediction in enumerate(predictions):
                fm_mse[ti, a:b, branch] = (prediction.double() - target.double()).square().flatten(1).mean(1).cpu().numpy()
            if ti == 0 and a == 0:
                repeat = gaps(z, t, label, shuffled)
                max_repeat_diff = float((repeat - baseline_gap).abs().max())
            steps = args.step * z.square().flatten(1).mean(1).sqrt().clamp_min(1)
            for pi in range(k):
                u = torch.from_numpy(bank['probes'][ti, a:b, pi]).cuda()
                delta = steps[:, None, None, None] * u
                plus = gaps(z + delta, t, label, shuffled)
                minus = gaps(z - delta, t, label, shuffled)
                derivative = (plus.double() - minus.double()) / (2 * steps[:, None, None, None, None].double())
                directional = (flat_gap * u.flatten(1)[:, None].double()).sum(2)
                trace[ti, a:b, :, pi] = (derivative.flatten(2) * u.flatten(1)[:, None].double()).sum(2).cpu().numpy() / DIM
                fisher[ti, a:b, :, pi] = directional.square().cpu().numpy() / DIM
                if a < stable_n and pi < stable_k:
                    count = min(b, stable_n) - a
                    for mi, multiplier in enumerate((.5, 2.)):
                        plus2 = gaps(z[:count] + multiplier * delta[:count], t, label[:count], shuffled[:count])
                        minus2 = gaps(z[:count] - multiplier * delta[:count], t, label[:count], shuffled[:count])
                        derivative2 = (plus2.double() - minus2.double()) / (2 * multiplier * steps[:count, None, None, None, None].double())
                        stable[ti, a:a+count, :, pi, mi] = (derivative2.flatten(2) * u[:count].flatten(1)[:, None].double()).sum(2).cpu().numpy() / DIM
            if a % (args.batch * 4) == 0 or b == n:
                torch.cuda.synchronize()
                log(f't={time_value:.2f}: {b}/{n}; elapsed={time.monotonic()-start_time:.1f}s, sample-forwards={eval_samples}')
                atomic(args.output / 'progress.json', dict(time=time_value, completed_in_stratum=b,
                       completed_strata=ti, total_strata=len(TIMES), total_samples=n,
                       elapsed_seconds=time.monotonic()-start_time, sample_forwards=eval_samples))
        np.savez(args.output / f'stratum_{ti}.npz', trace=trace[ti], fisher=fisher[ti],
                 first=first[ti], score_energy=score_energy[ti], fm_mse=fm_mse[ti], stability_trace=stable[ti])
        log(f't={time_value:.2f} done in {time.monotonic()-stratum_start:.1f}s')
    torch.cuda.synchronize()
    np.savez(args.output / 'results.npz', trace=trace, fisher=fisher, first=first,
             score_energy=score_energy, fm_mse=fm_mse, stability_trace=stable, times=np.array(TIMES))
    metadata.update(elapsed_seconds=time.monotonic()-start_time, model_sample_forwards=eval_samples,
                    runtime_field_calls=rt.counts, repeated_baseline_max_absolute_difference=max_repeat_diff,
                    results_sha256=sha(args.output / 'results.npz'), complete=True)
    atomic(args.output / 'metadata.json', metadata)
    summarize(args)


def summarize(args):
    metadata = json.loads((args.output / 'metadata.json').read_text())
    d = np.load(args.output / 'results.npz')
    rng = np.random.default_rng(metadata['seed'] + 91)
    indices = rng.integers(0, metadata['samples'], (2000, metadata['samples']))

    def ci(x):
        x = np.asarray(x, dtype=np.float64)
        values = x[indices].mean(1)
        low, high = np.quantile(values, [.025, .975], axis=0)
        return dict(mean=float(x.mean()), low=float(low), high=float(high))

    def fmt(x):
        return f"{x['mean']:.5g} [{x['low']:.5g}, {x['high']:.5g}]"

    residual = (d['trace'] + d['fisher']).mean(3)
    # The Fisher trace is known exactly once g is available. Keep directional
    # probes as a check, but do not introduce their avoidable variance here.
    trace_estimate = d['trace'].mean(3).copy()
    autograd_path = args.output / 'autograd_traces.npz'
    autograd_metadata = None
    if autograd_path.exists():
        autograd_values = np.load(autograd_path)
        trace_estimate[3] = autograd_values['trace_t070'].mean(2)
        autograd_metadata = json.loads((args.output / 'autograd_metadata.json').read_text())
    exact_residual = trace_estimate + d['score_energy']
    rows = []
    for ti, t in enumerate(TIMES):
        r = dict(t=t, paired=ci(residual[ti, :, 0]), shuffled=ci(residual[ti, :, 1]),
                 main_derivative_estimator='autograd VJP' if ti == 3 and autograd_metadata else 'central finite difference',
                 shuffled_minus_paired=ci(residual[ti, :, 1] - residual[ti, :, 0]),
                 exact_fisher_paired=ci(exact_residual[ti, :, 0]),
                 exact_fisher_shuffled=ci(exact_residual[ti, :, 1]),
                 exact_fisher_shuffled_minus_paired=ci(exact_residual[ti, :, 1]-exact_residual[ti, :, 0]),
                 trace_paired=ci(trace_estimate[ti, :, 0]),
                 fisher_paired=ci(d['fisher'][ti, :, 0].mean(1)),
                 exact_score_energy_paired=ci(d['score_energy'][ti, :, 0]),
                 fm_cond=ci(d['fm_mse'][ti, :, 0]), fm_null=ci(d['fm_mse'][ti, :, 1]),
                 fm_shuffle=ci(d['fm_mse'][ti, :, 2]),
                 fm_null_minus_cond=ci(d['fm_mse'][ti, :, 1]-d['fm_mse'][ti, :, 0]),
                 fm_shuffle_minus_cond=ci(d['fm_mse'][ti, :, 2]-d['fm_mse'][ti, :, 0]),
                 first_order_paired=[ci(d['first'][ti, :, 0, j]) for j in range(4)],
                 first_order_shuffle=[ci(d['first'][ti, :, 1, j]) for j in range(4)])
        baseline = d['trace'][ti, :metadata['stability_samples'], :, :metadata['stability_probes']]
        r['fd_stability'] = []
        for mi, mult in enumerate((.5, 2.)):
            changed = d['stability_trace'][ti, :, :, :, mi]
            difference = changed-baseline
            r['fd_stability'].append(dict(multiplier=mult,
                rms_difference=float(np.sqrt(np.mean(difference**2))),
                rms_baseline=float(np.sqrt(np.mean(baseline**2))),
                relative_rms=float(np.sqrt(np.mean(difference**2))/max(np.sqrt(np.mean(baseline**2)), 1e-30)),
                max_absolute_difference=float(np.max(np.abs(difference)))))
        rows.append(r)
    summary = dict(complete=True, metadata_path=str(args.output / 'metadata.json'), noise_strata=rows,
                   summarization_source_sha256=sha(__file__),
                   interpretation='Weak empirical consistency diagnostics only. No per-sample zero requirement, no quality or guided-loss claim.')
    atomic(args.output / 'summary.json', summary)
    lines = ['# 真实数据 Bayes 二阶相容性弱诊断（2026-09-13）', '',
             f"已完成 SiT-S/2 EMA 的 {metadata['samples']} 张真实 ImageNet100 validation 图、5 个时间层、每图 {metadata['probes']} 个 Rademacher 探针。没有生成图替代真实配对数据。",
             '', '## 协议与解释边界', '',
             '`z_t=t*x+(1-t)*epsilon`，`v_target=x-epsilon`；`g=t/(1-t)*(v_cond-v_null)`。',
             '主读数为 `mean_probe(uᵀ J_g u)/D + ||g||²/D`，D=4096。已知 g 后 Fisher 的 trace 可以精确计算，故主表避免额外的探针方差；原来的 `[uᵀ J_g u + (uᵀg)²]/D` 仍保留作核对。探针坐标独立取 ±1，先按图平均探针，再对图 bootstrap 2000 次，报告逐时间层的 95% CI。',
             '零只是在精确相容模型及其自身联合分布下的总体期望；单图和单探针不需要为零。真实数据联合分布 q_data 与模型联合分布的差异也会进入残差，不能凭该表将责任归于某个头，也不能把残差直接加在放大的 guided gap 上当损失。',
             '标签打乱对照复用同一图、噪声与探针，保留标签边际；它不再是配对联合分布，因此不要求满足零恒等式。CI 是本次图抽样及四探针的探索性读数，未校正多重检验，也未涵盖 checkpoint 或数据域变化。',
             f"所有网络前向与扰动为 FP32，标量累加为 FP64；autocast/TF32 均关闭。t 以 noise→data 方向增加。重复前向最大差为 {metadata['repeated_baseline_max_absolute_difference']:.3g}。",
             '', '## 二阶弱读数', '',
             '| t | 真实配对残差 /D [95% CI] | 标签打乱残差 /D | 打乱−配对 |',
             '|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['t']:.2f} | {fmt(r['exact_fisher_paired'])} | {fmt(r['exact_fisher_shuffled'])} | {fmt(r['exact_fisher_shuffled_minus_paired'])} |")
    lines += ['', '本次 128 图观察：配对残差在 t=0.15 的 CI 含零，在其余四层的逐层 CI 为正；所有层的“打乱−配对”CI 为正，说明正确标签对应的弱读数与打乱联合分布有区别。这是诊断的区分能力，不是生成质量收益。跨 t 的绝对量还受 score 换算因子 t/(1−t) 影响，不能把末段数值大直接解释为模型变差幅度。']
    if autograd_metadata:
        lines += ['', '数值复查后，主表 t=0.7 的 Jacobian 项已改为全部 128 图、4 探针的 autograd VJP；其他层使用经步长检查的中央差分。原始有限差分文件保留。']
    lines += ['', '原始有限差分 + 方向探针 Fisher 版本（附加方差更大）：', '',
              '| t | 真实配对残差 /D [95% CI] | 标签打乱残差 /D | 打乱−配对 |',
              '|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['t']:.2f} | {fmt(r['paired'])} | {fmt(r['shuffled'])} | {fmt(r['shuffled_minus_paired'])} |")
    lines += ['', '残差由可能为负的 trace 项与非负的 Fisher 项相加；其绝对值不是 FID 或图像质量。', '',
              '| t | 配对 trace /D | 配对 Fisher 探针 /D | 精确 ||g||²/D |', '|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['t']:.2f} | {fmt(r['trace_paired'])} | {fmt(r['fisher_paired'])} | {fmt(r['exact_score_energy_paired'])} |")
    lines += ['', '## 原始 FM 预测误差', '', '| t | conditional MSE | null MSE | shuffle MSE | null−conditional | shuffle−conditional |', '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['t']:.2f} | {fmt(r['fm_cond'])} | {fmt(r['fm_null'])} | {fmt(r['fm_shuffle'])} | {fmt(r['fm_null_minus_cond'])} | {fmt(r['fm_shuffle_minus_cond'])} |")
    lines += ['', '## 一阶弱读数', '', '使用四个固定、与数据独立且在各图间共享的单位 Rademacher 方向读出 `rᵀg`；随机方向若逐图重抽，均值会因方向对称性自动接近零，故本实验不这样做。', '', '| t | 方向1 | 方向2 | 方向3 | 方向4 |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['t']:.2f} | " + ' | '.join(fmt(x) for x in r['first_order_paired']) + ' |')
    lines += ['', '## 有限差分可靠性', '',
              f"中央差分坐标步长 h={metadata['fd_coordinate_step']}*max(1,RMS(z))；首 {metadata['stability_samples']} 图、前 {metadata['stability_probes']} 探针额外比较 h/2 与 2h。下表为 trace/D 改变量的 RMS，括号内为相对原 trace RMS。", '',
              '| t | h/2 对 h | 2h 对 h |', '|---|---|---|']
    for r in rows:
        columns = [f"{v['rms_difference']:.3g} ({100*v['relative_rms']:.3g}%)" for v in r['fd_stability']]
        lines.append(f"| {r['t']:.2f} | " + ' | '.join(columns) + ' |')
    refinement_path = args.output / 'fd_refinement.json'
    if refinement_path.exists():
        refinement = json.loads(refinement_path.read_text())
        lines += ['', 't=0.7 的初始差分敏感性触发独立复查：同一首 16 图、前 2 探针、配对/打乱两条件，对比 h=0.00025, 0.0005, 0.001, 0.002，并额外以 FP32 autograd 反向微分计算同一 uᵀJ_gu。步长改变的是每坐标扰动；原始 128 图结果未被覆盖。', '',
                  '| h | 相对 autograd 的 RMS 差 | / autograd RMS | 最大绝对差 |', '|---|---|---|---|']
        for row in refinement['comparisons']:
            lines.append(f"| {row['step']:.5g} | {row['rms_difference']:.5g} | {100*row['relative_rms']:.4g}% | {row['max_absolute_difference']:.5g} |")
        lines += ['', f"该子集 autograd trace/D 均值：配对 {refinement['autograd_mean_paired']:.6g}，打乱 {refinement['autograd_mean_shuffled']:.6g}。h=0.001 的配对均值偏差 {refinement['baseline_mean_bias_paired']:.6g}，打乱均值偏差 {refinement['baseline_mean_bias_shuffled']:.6g}。", '局部有限差分误差与总体采样不确定性必须分开；子集复查本身不能代替全部 128 图的精确散度。']
    if autograd_metadata:
        lines += ['', f"为消除 t=0.7 的差分步长选择，追加全部 128 图×4 探针的 autograd VJP，均值差（FD−autograd）：配对 {autograd_metadata['t070']['mean_bias_paired']:.6g}，打乱 {autograd_metadata['t070']['mean_bias_shuffled']:.6g}；主表已使用 autograd。另在 t=0.9 原首16图×2探针核验，FD 对 autograd 相对 RMS 差 {100*autograd_metadata['t090']['relative_rms']:.3g}%。该数值差不能包装为模型机制。",
                  f"追加 autograd 成本 {autograd_metadata['elapsed_seconds']:.1f} 秒，{autograd_metadata['forward_samples']} 次样本前向，{autograd_metadata['backward_batches']} 次批量 VJP；原始成本与哈希未改写。"]
    lines += ['', '## 可复查文件', '',
              f"- 输出目录：`{args.output}`。`inputs.npz` 保存真实图索引、源索引、真实/打乱标签、latent、posterior noise、5 层噪声及全部探针。",
              '- `results.npz` 保存逐图逐探针 trace、Fisher、一阶方向读数、FM MSE 与有限差分敏感性；`summary.json` 保存完整 CI，含标签打乱的一阶读数。',
              '- `metadata.json` 保存数据 manifest、实际文件哈希、checkpoint 哈希、原始运行代码哈希、dtype 与成本；汇总代码更新后的哈希另存 `summary.json`，不改写历史运行哈希。',
              f"- 共 {metadata['model_sample_forwards']} 次样本前向，耗时 {metadata['elapsed_seconds']:.1f} 秒；仅 physical GPU2。",
              '- 运行：`CUDA_VISIBLE_DEVICES=2 python -m experiments.cfg_invariants_20260913.model_diagnostic`；汇总：同命令增加 `--summarize`。',
              '- 追加复查分别增加 `--refine` 与 `--autograd-full`；不会覆盖原始 results.npz 或原运行代码哈希。', '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines))
    log(f'Complete. Report: {REPORT}')


def refine(args):
    """Only resolve the observed t=.7 FD concern, with the exact same 16 images."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=2 for the assigned GPU.')
    metadata = json.loads((args.output / 'metadata.json').read_text())
    bank = np.load(args.output / 'inputs.npz')
    old = np.load(args.output / 'results.npz')
    rt = runtime('sit_small')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rt.model.float().eval().requires_grad_(False)
    ti = list(TIMES).index(.7)
    n, k = metadata['stability_samples'], metadata['stability_probes']
    value = np.zeros((n, 2, k), np.float64)
    analytic = np.zeros_like(value)
    t = torch.tensor(.7, device='cuda')
    started = time.monotonic()
    for a in range(0, n, 4):
        b = min(n, a + 4)
        clean = torch.from_numpy(bank['clean'][a:b]).cuda()
        noise = torch.from_numpy(bank['noise'][ti, a:b]).cuda()
        z = (t * clean + (1-t) * noise).requires_grad_(True)
        labels = torch.from_numpy(bank['labels'][a:b]).cuda()
        shuffle = torch.from_numpy(bank['shuffle_labels'][a:b]).cuda()
        rt.labels = torch.cat([labels, torch.full_like(labels, 100), shuffle])

        def evaluate(state):
            with torch.autocast('cuda', enabled=False):
                cond, null, shuffled = rt.field(torch.cat([state]*3), t, 'full').chunk(3)
            return torch.stack([cond-null, shuffled-null], 1)*(t/(1-t))

        gap = evaluate(z)
        for pi in range(k):
            u = torch.from_numpy(bank['probes'][ti, a:b, pi]).cuda()
            for ci in range(2):
                # Independent examples in eval mode make the summed scalar VJP
                # recover each example's directional quadratic form separately.
                gradient = torch.autograd.grad((gap[:, ci]*u).sum(), z, retain_graph=True)[0]
                analytic[a:b, ci, pi] = (gradient.double()*u.double()).flatten(1).sum(1).detach().cpu().numpy()/DIM
            with torch.no_grad():
                steps = .00025*z.square().flatten(1).mean(1).sqrt().clamp_min(1)
                delta = steps[:, None, None, None]*u
                derivative = (evaluate(z+delta).double()-evaluate(z-delta).double())/(2*steps[:, None, None, None, None].double())
                value[a:b, :, pi] = (derivative.flatten(2)*u.flatten(1)[:, None].double()).sum(2).cpu().numpy()/DIM
        log(f't=.7 gradient/FD refinement {b}/{n}')
    candidates = [(.00025, value), (.0005, old['stability_trace'][ti, :, :, :, 0]),
                  (.001, old['trace'][ti, :n, :, :k]), (.002, old['stability_trace'][ti, :, :, :, 1])]
    comparisons = []
    for step, arr in candidates:
        diff = arr-analytic
        rms = float(np.sqrt(np.mean(diff**2)))
        comparisons.append(dict(step=step, rms_difference=rms,
            relative_rms=rms/float(np.sqrt(np.mean(analytic**2))),
            max_absolute_difference=float(np.abs(diff).max()),
            mean_bias_paired=float(diff[:, 0].mean()), mean_bias_shuffled=float(diff[:, 1].mean())))
    np.savez(args.output / 'fd_refinement.npz', trace_h00025=value, autograd_trace=analytic)
    baseline_diff = candidates[2][1]-analytic
    atomic(args.output / 'fd_refinement.json', dict(t=.7, samples=n, probes=k,
        source_sha256=sha(__file__), original_source_sha256=metadata['source_sha256'][str(Path(__file__))],
        input_sha256=sha(args.output/'inputs.npz'), comparisons=comparisons,
        autograd_mean_paired=float(analytic[:, 0].mean()), autograd_mean_shuffled=float(analytic[:, 1].mean()),
        baseline_mean_bias_paired=float(baseline_diff[:, 0].mean()),
        baseline_mean_bias_shuffled=float(baseline_diff[:, 1].mean()),
        elapsed_seconds=time.monotonic()-started, dtype='FP32 network/gradient, FP64 scalar accumulation',
        autocast=False, tf32=False, physical_gpu=2))
    summarize(args)


def autograd_full(args):
    """Replace disputed t=.7 derivatives, retaining the original FD results."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=2 for the assigned GPU.')
    original = json.loads((args.output/'metadata.json').read_text())
    bank = np.load(args.output/'inputs.npz')
    fd = np.load(args.output/'results.npz')
    started = time.monotonic()
    rt = runtime('sit_small')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rt.model.float().eval().requires_grad_(False)
    assert not rt.model.training and not rt.autocast
    arrays, comparisons = {}, {}
    forward_samples, backward_batches = 0, 0
    for ti, n, k, name in ((3, original['samples'], original['probes'], 't070'),
                           (4, original['stability_samples'], original['stability_probes'], 't090')):
        arr = np.zeros((n, 2, k), np.float64)
        t = torch.tensor(TIMES[ti], device='cuda', dtype=torch.float32)
        for a in range(0, n, 4):
            b = min(n, a+4)
            clean = torch.from_numpy(bank['clean'][a:b]).cuda()
            noise = torch.from_numpy(bank['noise'][ti, a:b]).cuda()
            with torch.inference_mode(False), torch.enable_grad():
                z = (t*clean+(1-t)*noise).clone().requires_grad_(True)
                assert not z.is_inference()
                labels = torch.from_numpy(bank['labels'][a:b]).cuda()
                shuffled = torch.from_numpy(bank['shuffle_labels'][a:b]).cuda()
                rt.labels = torch.cat([labels, torch.full_like(labels, 100), shuffled])
                with torch.autocast('cuda', enabled=False):
                    c, u0, s = rt.field(torch.cat([z]*3), t, 'full').chunk(3)
                forward_samples += 3*(b-a)
                gap = torch.stack([c-u0, s-u0], 1)*(t/(1-t))
                for pi in range(k):
                    probe = torch.from_numpy(bank['probes'][ti, a:b, pi]).cuda()
                    for ci in range(2):
                        gradient = torch.autograd.grad((gap[:, ci]*probe).sum(), z, retain_graph=True)[0]
                        arr[a:b, ci, pi] = (gradient.double()*probe.double()).flatten(1).sum(1).detach().cpu().numpy()/DIM
                        backward_batches += 1
            if b % 32 == 0 or b == n:
                log(f'autograd t={TIMES[ti]:.2f}: {b}/{n}, {k} probes')
        arrays['trace_'+name] = arr
        diff = fd['trace'][ti, :n, :, :k]-arr
        comparisons[name] = dict(samples=n, probes=k, rms_difference=float(np.sqrt(np.mean(diff**2))),
            relative_rms=float(np.sqrt(np.mean(diff**2))/np.sqrt(np.mean(arr**2))),
            mean_bias_paired=float(diff[:, 0].mean()), mean_bias_shuffled=float(diff[:, 1].mean()))
    torch.cuda.synchronize()
    np.savez(args.output/'autograd_traces.npz', **arrays)
    atomic(args.output/'autograd_metadata.json', dict(**comparisons,
        elapsed_seconds=time.monotonic()-started, source_sha256=sha(__file__),
        original_source_sha256=original['source_sha256'][str(Path(__file__))],
        input_sha256=sha(args.output/'inputs.npz'), checkpoint_sha256=sha(SMALL_CKPT),
        result_sha256=sha(args.output/'autograd_traces.npz'), forward_samples=forward_samples,
        backward_batches=backward_batches, physical_gpu=2, tf32=False, autocast=False,
        dtype='FP32 network/gradient, FP64 accumulation', model_eval=True, inference_mode=False,
        derivative='autograd.grad(sum(g_c*u), z) dot u; no cross-example interactions in SiT eval'))
    summarize(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--probes', type=int, default=4)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--step', type=float, default=.001)
    parser.add_argument('--stability-samples', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026091357)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--autograd-full', action='store_true')
    args = parser.parse_args()
    if args.autograd_full:
        autograd_full(args)
    elif args.refine:
        refine(args)
    elif args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
