"""Frozen 59-arm experiment using the existing audited sampling infrastructure."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import catalog, core, train
from .data import ROOT, WORK, METHODS
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

STAGE = 'measure_screen_1k'
MODULE = 'experiments.sit_measure_guidance_20260912.pipeline'
PROTOCOL = WORK/'docs/SIT_MEASURE_GUIDANCE_PROTOCOL_20260912_ZH.md'
REPORT = WORK/'docs/SIT_MEASURE_GUIDANCE_RESULTS_20260912_ZH.md'
RESEARCH = WORK/'docs/FIVE_MECHANISM_IDEAS_20260912_ZH.md'
_verify_parent = engine.verify_parent


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'), PROTOCOL, RESEARCH,
        Path(engine.__file__), Path(analysis.__file__), Path(core.previous.__file__),
        Path(core.previous.angular.__file__),
        WORK/'experiments/resume_sit_broad_20260912.py',
        WORK/'docs/SIT_BROAD_RESUME_20260912_ZH.md']))


def verify_parent():
    parent = _verify_parent()
    request = train.verify()
    assert read(ROOT/'training_status.json')['phase'] == 'complete'
    extra = {}
    for method in METHODS:
        folder = ROOT/'training'/method
        receipt = read(folder/'complete.json')
        assert receipt['passed'] and receipt['request_sha256'] == sha(ROOT/'training_request.json')
        assert receipt['checkpoint_sha256'] == sha(folder/'model.pt')
        for name in ('complete.json', 'model.pt'):
            extra[str(folder/name)] = sha(folder/name)
    for name in ('training_request.json', 'gaussian_reference.pt', 'gaussian_reference.json',
                 'mechanism_check.json', 'mechanism_results.csv', 'theory_checks.json',
                 'local_partners.npy', 'random_partners.npy'):
        extra[str(ROOT/name)] = sha(ROOT/name)
    assert read(ROOT/'mechanism_check.json')['passed']
    if (ROOT/'external_baselines.json').exists():
        extra[str(ROOT/'external_baselines.json')] = sha(ROOT/'external_baselines.json')
    return dict(parent, assets={**parent['assets'], **extra},
                sources={**parent['sources'], **request['sources']})


def configure():
    engine.ROOT, engine.MODULE, engine.PROTOCOL = ROOT, MODULE, PROTOCOL
    engine.STAGES = {STAGE:(1000, catalog.NOISE_SEED)}
    engine.fusion, engine.additional_sources, engine.verify_parent = core, sources, verify_parent
    analysis.write_progress = write_progress


def write_progress(base, request, rows):
    fields = ['arm', 'idea_id', 'family', 'role', 'source', 'strength', 'theta',
        'fid', 'full_calls_per_image', 'prefix_calls_per_image', 'sum_batch_gpu_seconds', 'complete']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    (base/'results.csv').write_text(stream.getvalue())
    portable = WORK/'docs/data/sit_measure_guidance_20260912'
    portable.mkdir(parents=True, exist_ok=True)
    (portable/'all_results.csv').write_text(stream.getvalue())
    valid = [r for r in rows if r['complete']]
    external = read(ROOT/'external_baselines.json')
    def best(pool):
        return min(pool, key=lambda r:r['fid']) if pool else None
    baselines = {source:best([r for r in external if r['source'] == source and
        r['family'] in ('ig_local', 'cfg_native', 'cfg_apg')]) for source in ('ig', 'cfg')}
    training = [read(ROOT/'training'/m/'complete.json') for m in METHODS]
    seconds = sum(r['training_seconds'] for r in training)
    lines = ['# 五个分布机制候选的实验结果\n',
        f'已提交 {len(rows)}/59 组1K：24组候选、35组对照；数值失败 {sum(not r["complete"] for r in rows)}。\n',
        '与旧control队列相同的配对噪声和标签，不是独立5K确认。强模型不更新；四种训练候选使用完整弱模型，不是单次主干IG。\n',
        f'八个匹配弱模型均已完成1500步训练；累计训练GPU时间 {seconds:.1f} 秒，'
        '不含准备/验证。训练成本单列，以下成本为主采样、全部模型调用、随机更新及最终解码。\n',
        '|假设|完成|最低FID|同源旧最强基线FID|差值|采样/解码成本倍数|参数|',
        '|---|--:|--:|--:|--:|--:|---|']
    for key, idea in catalog.IDEAS.items():
        pool = [r for r in valid if r['idea_id'] == idea['id']]
        row = best(pool)
        planned = sum(c['idea_id'] == idea['id'] for c in request['configs'])
        count = sum(r['idea_id'] == idea['id'] for r in rows)
        if row is None:
            lines.append(f'|{idea["id"]} {idea["title"]}|{count}/{planned}|—|—|—|—|—|')
            continue
        baseline = baselines[row['source']]
        lines.append(f'|{idea["id"]} {idea["title"]}|{count}/{planned}|{row["fid"]:.6f}|'
            f'{baseline["fid"]:.6f}|{row["fid"]-baseline["fid"]:+.6f}|'
            f'{row["sum_batch_gpu_seconds"]/baseline["sum_batch_gpu_seconds"]:.3f}|'
            f'α={row["strength"]}, θ={row["theta"]}|')
    lines += ['\n这里比较旧同源强基线，不能替代相应同预算/同转移核消融；完整曲线见CSV。\n',
        '|对照family|完成|最低FID|Full / prefix|', '|---|--:|--:|---|']
    for family in dict.fromkeys(c['family'] for c in request['configs'] if c['role'] != 'candidate'):
        pool = [r for r in valid if r['family'] == family]
        row = best(pool)
        if row:
            lines.append(f'|{family}|{len(pool)}|{row["fid"]:.6f}|'
                f'{row["full_calls_per_image"]:.0f} / {row["prefix_calls_per_image"]:.0f}|')
    lines += ['\n|训练分布|自身验证MSE，t=.5：训练前→后|原分布验证MSE，t=.5|训练GPU秒|',
        '|---|--:|--:|--:|']
    for row in training:
        lines.append(f'|{row["method"]}|{row["target_mse_before"][1]["mse"]:.6f} → '
            f'{row["target_mse_after"][1]["mse"]:.6f}|{row["native_mse_after"][1]["mse"]:.6f}|'
            f'{row["training_seconds"]:.1f}|')
    if (base/'analysis_audit.json').exists():
        lines.append('\n全部配置覆盖审计通过；每个有完整结果的family最佳点另核验原始产物与FP64 FID/sFID。\n')
    lines += ['\n离线128图机制诊断只用于区分解释，不用于在线选择或调强度。'
        '数学不变量与验证MSE均不证明真实IG机制或生成优势。'
        '完整生成和审计完成后自动补旧优先7组，再恢复原709/201队列。\n',
        '[推导与失败条件](FIVE_MECHANISM_IDEAS_20260912_ZH.md) · '
        '[生成协议](SIT_MEASURE_GUIDANCE_PROTOCOL_20260912_ZH.md) · '
        '[逐组结果](data/sit_measure_guidance_20260912/all_results.csv) · '
        '[机制诊断](data/sit_measure_guidance_20260912/mechanism_results.csv)\n',
        f'请求SHA256：{sha(base/"request.json")}。\n']
    text = '\n'.join(lines)
    (base/'report.md').write_text(text)
    REPORT.write_text(text)


@torch.inference_mode()
def check_rank(rank):
    configure()
    verify_parent()
    rt = engine.parent.operators.make_runtime()
    bank = catalog.OLD/'inputs'
    noise = torch.from_numpy(np.load(bank/'noise.npy')[:8].copy()).cuda()
    labels = torch.from_numpy(np.load(bank/'labels.npy')[:8].copy()).cuda()
    rt.labels = labels
    hooks = engine.parent.hook_counts(rt)
    limits = core.limiting_checks(rt, noise, labels)
    configs = catalog.configurations()
    anchor = next(c for c in configs if c['family'] == 'strong')
    native, _ = core.sample(rt, noise[:2], labels[:2], anchor)
    records = []
    for config in configs[rank::4]:
        z, stats = core.sample(rt, noise[:2], labels[:2], config)
        assert torch.isfinite(z).all()
        assert engine.parent.hook_counts(rt) == hooks
        zero, _ = core.sample(rt, noise[:2], labels[:2], config, zero=True)
        assert torch.equal(zero, native), config['arm']
        records.append(dict(arm=config['arm'], full_calls=stats['full_calls'],
            prefix_calls=stats['prefix_calls'], max_abs=float(z.abs().max()), zero_exact=True))
        print(dict(rank=rank, checked=config['arm'], full_calls=stats['full_calls']), flush=True)
    golden_config = next(c for c in configs if c['arm'] ==
        ('strong_00', 'ig_local_00', 'cfg_native_04', 'cfg_apg_07')[rank])
    z, _ = core.sample(rt, noise, labels, golden_config)
    golden_path = catalog.OLD/golden_config['arm']/'rank0/batch0000.npz'
    with np.load(golden_path) as data:
        np.testing.assert_array_equal(z.cpu().numpy(), data['latents'])
    # Actual conditional flow and inverse, with no local quality criterion.
    state = core.rk4(rt, noise[:2], 0., .5, labels[:2], 32)
    inverse = core.rk4(rt, state, .5, 0., labels[:2], 32)
    error = float((inverse-noise[:2]).norm()/noise[:2].norm())
    assert error < .003, error
    again, _ = core.sample(rt, noise[:2], labels[:2], anchor)
    assert torch.equal(again, native)
    atomic(ROOT/'development'/f'rank{rank}.json', dict(passed=True, rank=rank,
        trajectories=records, limiting_checks=limits, inverse_relative_error=error,
        native_after_interventions_exact=True, original_golden_exact=True,
        original_golden_sha256=sha(golden_path), source_hashes={str(p):sha(p) for p in sources()},
        runtime_sources=rt.sources, no_fid_used=True))


def development_check():
    configure()
    verify_parent()
    assert not (ROOT/'STOP_AFTER_CURRENT').exists()
    folder = ROOT/'development'
    folder.mkdir(exist_ok=True)
    processes, streams = [], []
    try:
        for rank in range(4):
            stream = (folder/f'rank{rank}.log').open('a')
            streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON, '-u', '-m', MODULE,
                '--check-rank', str(rank)], cwd=WORK, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')))
        atomic(folder/'processes.json', dict(controller_pid=os.getpid(),
            worker_pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Preflight failed: {[p.poll() for p in processes]}')
            time.sleep(2)
        ranks = [read(folder/f'rank{r}.json') for r in range(4)]
        coverage = [r['arm'] for rank in ranks for r in rank['trajectories']]
        assert set(coverage) == {c['arm'] for c in catalog.configurations()} and len(coverage) == 59
        expected = {str(p):sha(p) for p in sources()}
        assert all(r['passed'] and r['source_hashes'] == expected for r in ranks)
        atomic(ROOT/'development_check.json', dict(passed=True, trajectories=59,
            source_hashes=expected, ranks=[sha(folder/f'rank{r}.json') for r in range(4)],
            maximum_inverse_error=max(r['inverse_relative_error'] for r in ranks),
            four_original_baselines_exact=True, all_zero_strength_exact=True, no_fid_used=True))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait(timeout=30)
        for s in streams:
            s.close()


def prepare():
    configure()
    if not (ROOT/'external_baselines.json').exists():
        rows = read(catalog.OLD/'results.json')
        rows = [r for r in rows if r['role'] != 'candidate' and r['complete']]
        atomic(ROOT/'external_baselines.json', rows)
    engine.prepare_stage(STAGE, catalog.configurations())
    actual = read(ROOT/STAGE/'request.json')['bank']
    expected = read(catalog.OLD/'request.json')['bank']
    assert actual['noise_sha256'] == expected['noise_sha256']
    assert actual['label_sha256'] == expected['label_sha256']
    prior = ROOT/STAGE/'results.json'
    write_progress(ROOT/STAGE, read(ROOT/STAGE/'request.json'), read(prior) if prior.exists() else [])


def pipeline():
    configure()
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT/'STOP_AFTER_CURRENT').exists():
            return
        prepare()
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase'] != 'complete':
            return
        rows = read(ROOT/STAGE/'results.json')
        selected = [min([r for r in rows if r['family'] == family and r['complete']],
                        key=lambda r:r['fid'])['arm']
                    for family in dict.fromkeys(r['family'] for r in rows if r['complete'])]
        analysis.audit_stage(ROOT/STAGE, selected)
        write_progress(ROOT/STAGE, read(ROOT/STAGE/'request.json'), rows)
        atomic(ROOT/'status.json', dict(phase='complete', total_arms=59,
            candidate_arms=24, controls=35, numerical_failures=sum(not r['complete'] for r in rows),
            all_arms_accounted=True, final_audit_passed=True, no_automatic_5k=True))


if __name__ == '__main__':
    configure()
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--check-rank', type=int)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--pipeline', action='store_true')
    actions.add_argument('--worker', type=int)
    parser.add_argument('--stage')
    parser.add_argument('--run-id')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.check:
        development_check()
    elif args.check_rank is not None:
        check_rank(args.check_rank)
    elif args.prepare:
        prepare()
    elif args.pipeline:
        pipeline()
    else:
        engine.worker(args.stage, args.worker, args.run_id, args.parent_pid)
