"""Fifty hypotheses plus three priorities; one paired 1K sweep, no promotion."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
from experiments import run_sit_control_50ideas_20260910 as catalog
from experiments.sit_control_50_20260910 import core, checks
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, atomic, read, sha

ROOT = EXPS/'sit_control_output_50ideas_20260910'
WAIT_ROOT = EXPS/'sit_fsg_pasted_followup_20260910'
STAGE = 'control_screen_1k'
MODULE = 'experiments.run_sit_control_50ideas_20260910'
PROTOCOL = WORK/'docs/SIT_CONTROL_53_IDEAS_PROTOCOL_20260910_ZH.md'
RESEARCH = WORK/'docs/FSG_CONTROL_50_IDEAS_20260910_ZH.md'
PRIORITIES = WORK/'docs/FSG_CONTROL_THREE_PRIORITY_IDEAS_20260910_ZH.md'
REPORT = WORK/'docs/SIT_CONTROL_53_IDEAS_RESULTS_20260910_ZH.md'
_verify_parent = engine.verify_parent
_factory = engine.parent.operators.make_runtime
_evaluate = engine.infrastructure.evaluate


def sources():
    return list(dict.fromkeys([
        Path(__file__).resolve(), Path(core.__file__).resolve(), Path(checks.__file__).resolve(),
        Path(catalog.__file__).resolve(), Path(core.seven.__file__).resolve(),
        WORK/'experiments/sit_control_50_20260910/write_docs.py',
        Path(core.seven.catalog.__file__).resolve(), Path(core.prior.__file__).resolve(),
        Path(core.prior.catalog.__file__).resolve(), Path(engine.__file__).resolve(),
        Path(analysis.__file__).resolve(), WORK/'experiments/sit_guidance_fusion_20260910.py',
        PROTOCOL, RESEARCH, PRIORITIES,
    ]))


def verify_parent():
    value = _verify_parent()
    return dict(value, assets={**value['assets'], **{str(path): sha(path) for path in core.source_assets()}})


def make_runtime():
    rt = _factory()
    rt.pasted_semantic = core.seven.SemanticReadout(rt)
    with torch.inference_mode():
        rt.pasted_semantic.probabilities(torch.zeros((2, 4, 32, 32), device='cuda'))
    return rt


def evaluate(arm, request, request_hash):
    row = _evaluate(arm, request, request_hash)
    if not row['complete']:
        return row
    config = next(c for c in request['configs'] if c['arm'] == arm)
    base = ROOT/STAGE/arm
    totals = dict(auxiliary_decoder_images=0., auxiliary_classifier_images=0.,
        accepted_calibrations=0., calibration_events=0., calibration_full_calls=0.,
        calibration_loss_before_sum=0., calibration_loss_after_sum=0.,
        calibration_shift_norm_sum=0., response_prediction_error=0.,
        qp_infeasible=0., qp_clipped_progress_shortfall=0., resolution_rejections=0.,
        verified_fine_gain=0., finite_target_attained=0., finite_zero_selected=0.)
    traces = []
    for rank in range(4):
        for rec in read(base/f'rank{rank}/summary.json')['files']:
            with np.load(base/f'rank{rank}'/rec['file']) as batch:
                for key in totals:
                    if key in batch:
                        totals[key] += float(batch[key])
                if 'priority_trace_kind' in batch:
                    traces.append(checks.audit_priority_trace(batch, config))
    if config['key'] in ('verified_semantic_step', 'finite_semantic_control'):
        assert len(traces) == request['samples']//request['batch']
    row.update(auxiliary_counts=totals,
        auxiliary_decoder_images_per_output=totals['auxiliary_decoder_images']/request['samples'],
        auxiliary_classifier_images_per_output=totals['auxiliary_classifier_images']/request['samples'],
        priority_batch_traces_audited=len(traces), posthoc_semantic_evaluation=False)
    atomic(base/'result.json', row)
    return row


def configure():
    engine.ROOT = ROOT
    engine.MODULE = MODULE
    engine.PROTOCOL = PROTOCOL
    engine.STAGES = {STAGE: (1000, catalog.NOISE_SEED)}
    engine.fusion = core
    engine.additional_sources = sources
    engine.verify_parent = verify_parent
    engine.parent.operators.make_runtime = make_runtime
    engine.infrastructure.evaluate = evaluate
    analysis.write_progress = write_progress


def dependency_complete(status):
    return (status.get('phase') == 'complete' and status.get('no_further_sampling_queued') is True
            and 'stage' not in status and status.get('total_arms') == 140)


def write_progress(base, request, rows):
    valid = [row for row in rows if row['complete']]
    def best(predicate):
        pool = [row for row in valid if predicate(row)]
        return min(pool, key=lambda row: row['fid']) if pool else None
    fields = ['arm', 'idea_id', 'family', 'key', 'role', 'solver', 'strength', 'theta',
        'external_semantics', 'fid', 'full_calls_per_image', 'prefix_calls_per_image',
        'auxiliary_decoder_images_per_output', 'auxiliary_classifier_images_per_output',
        'sum_batch_gpu_seconds', 'priority_batch_traces_audited', 'complete']
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    (base/'results.csv').write_text(output.getvalue())
    portable = WORK/'docs/data/sit_control_output_50ideas_20260910'
    portable.mkdir(parents=True, exist_ok=True)
    (portable/'all_results.csv').write_text(output.getvalue())
    lines = ['# 控制输出 53 个候选的 1K FID 筛选\n',
        f'已完成 {len(rows)}/{len(request["configs"])} 组，数值失败 {sum(not r["complete"] for r in rows)}。53个方向各12组，另有73组对照。\n',
        '每个配置使用同一套新噪声，每类10张。全部参数按本轮1K筛选，不自动追加5K。最佳值是网格筛选结果，不能据此声称显著性或独立确认。\n']
    for family, title in [('strong', '强条件模型'), ('cfg_native', '原生CFG'), ('cfg_apg', 'APG')]:
        row = best(lambda row: row['family'] == family)
        if row:
            lines.append(f'{title}：FID {row["fid"]:.6f}，{row["arm"]}。\n')
    lines += ['|方向|完成/12|最低FID|Δ匹配对照|对照|a / 参数|采样成本×对照|Full/prefix|内部解码/分类张数|',
              '|---|--:|--:|--:|---|---|--:|---|---|']
    for method in catalog.IDEAS:
        done = sum(row['idea_id'] == method['id'] for row in rows)
        row = best(lambda row: row['idea_id'] == method['id'])
        if row is None:
            lines.append(f'|{method["id"]} {method["title"]}|{done}/12|—|—|—|—|—|—|—|')
            continue
        baseline = best(lambda other: other['role'] != 'candidate' and other['source'] == 'cfg'
            and other['solver'] == row['solver'] and other['external_semantics'] == row['external_semantics'])
        delta = f'{row["fid"]-baseline["fid"]:+.6f}' if baseline else '—'
        ratio = f'{row["sum_batch_gpu_seconds"]/baseline["sum_batch_gpu_seconds"]:.3f}' if baseline else '—'
        lines.append(f'|{method["id"]} {method["title"]}|{done}/12|{row["fid"]:.6f}|{delta}|'
            f'{baseline["arm"] if baseline else "待完成"}|{row["strength"]}/{row["theta"]}|{ratio}|'
            f'{row["full_calls_per_image"]:.0f}/{row["prefix_calls_per_image"]:.0f}|'
            f'{row["auxiliary_decoder_images_per_output"]:.0f}/{row["auxiliary_classifier_images_per_output"]:.0f}|')
    lines += ['\nΔ为候选减对照，负值更好。匹配对照同时限定ODE/SDE和是否使用额外ConvNeXt语义读出；各自取本轮已完成网格最低值。计算预算没有强制相等，另列实际采样与解码成本。\n',
        'FID为本轮选优依据。继承的ADM提取器同时保存sFID/IS，不运行额外后验分类器或扩展诊断队列。优化用分类读出只属于方法内部，全部额外解码与分类计算计入采样时间。\n',
        '第52项逐批核对粗细未来验收条件及真实16步读出进展；第53项逐批穷尽核对九个候选损失与最小范数选择。两项性质均不等价于最终FID改善。\n',
        '[50项推理目录](FSG_CONTROL_50_IDEAS_20260910_ZH.md) · [三项重点推导](FSG_CONTROL_THREE_PRIORITY_IDEAS_20260910_ZH.md) · [冻结协议](SIT_CONTROL_53_IDEAS_PROTOCOL_20260910_ZH.md) · [逐组CSV](data/sit_control_output_50ideas_20260910/all_results.csv)。\n',
        f'原始记录：{base}；请求SHA256：{sha(base/"request.json")}。\n']
    if (base/'analysis_audit.json').exists():
        audit = read(base/'analysis_audit.json')
        lines.append(f'全部覆盖和元数据审计通过；{len(audit["raw_and_metric_audits"])}个family代表另通过逐批哈希、成本与缓存特征FP64 FID复算。\n')
    text = '\n'.join(lines)
    (base/'report.md').write_text(text)
    REPORT.write_text(text)


@torch.inference_mode()
def check_rank(rank):
    configure()
    verify_parent()
    rt = make_runtime()
    noise = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'noise.npy')[:8].copy()).cuda()
    labels = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'labels.npy')[:8].copy()).cuda()
    rows = catalog.configurations()
    limits = core.limiting_checks(rt, noise, labels)
    hooks = engine.parent.hook_counts(rt)
    original, _ = core.sample(rt, noise, labels, rows[0])
    state = noise.clone()
    for k in range(8):
        state = core.old.strong_heun(rt, state, k/64, 1/64)
    probes = []
    configs = [c for c in rows if c['key'] in core.SPECIAL]
    for index, config in enumerate(configs):
        if index % 4 != rank:
            continue
        rt.labels = labels[:2]
        context = core.event_context(noise[:2], 8)
        value, record = core.calibrate(rt, state[:2], .125, config, context, config['strength'])
        assert torch.isfinite(value).all() and all(np.isfinite(v) for v in record.values()), config['arm']
        assert record['after'] <= record['before']+1e-6
        assert engine.parent.hook_counts(rt) == hooks
        probes.append(config['arm'])
        if len(probes) % 10 == 0:
            print(dict(rank=rank, grid_probes=len(probes)), flush=True)
    selected = [[c for c in rows if c['idea_id'] == method['id']][-1] for method in catalog.IDEAS]
    selected += [[c for c in rows if c['family'] == family][-1] for family in
        ('full_root16_control', 'fsg_operator_control', 'semantic_direct', 'sde_semantic_direct',
         'cfg_smc_control', 'cfg_sde_native')]
    native_sde = next(c for c in rows if c['key'] == 'native_sde')
    sde_zero, _ = core.sample(rt, noise, labels, native_sde, zero=True)
    trajectories = []
    for index, config in enumerate(selected):
        if index % 4 != rank:
            continue
        latent, stats = core.sample(rt, noise, labels, config)
        assert torch.isfinite(latent).all() and np.isfinite(stats['diagnostics']).all()
        trace = checks.audit_priority_trace(stats, config)
        zero, _ = core.sample(rt, noise, labels, config, zero=True)
        assert torch.equal(zero, sde_zero if config['solver'] == 'sde_tail64' else original), config['arm']
        if config['key'] in core.SPECIAL:
            native = next(c for c in rows if c['key'] == ('native_sde' if config['solver'] == 'sde_tail64' else 'native_cfg'))
            native = dict(native, strength=config['strength'], cutoff=config['cutoff'])
            disabled, _ = core.sample(rt, noise, labels, config, disable_calibration=True)
            baseline, _ = core.sample(rt, noise, labels, native)
            assert torch.equal(disabled, baseline), config['arm']
        assert rt.labels is labels and engine.parent.hook_counts(rt) == hooks
        rec = dict(arm=config['arm'], full=stats['full_calls'], prefix=stats['prefix_calls'],
            accepted=stats.get('accepted_calibrations'), max_abs=float(latent.abs().max()),
            auxiliary_decoder_images=stats.get('auxiliary_decoder_images', 0),
            auxiliary_classifier_images=stats.get('auxiliary_classifier_images', 0), trace=trace)
        trajectories.append(rec)
        print(dict(rank=rank, trajectory=rec), flush=True)
    golden = []
    for index, family in enumerate(('ig_local', 'cfg_native', 'cfg_apg')):
        if index != rank:
            continue
        pairs = [(c, r) for c, r in engine.golden_pairs(rows) if c['family'] == family]
        assert pairs, family
        config, row = pairs[0]
        latent, _ = core.sample(rt, noise, labels, config)
        path = engine.parent.ROOT/row['arm']/'rank0/batch0000.npz'
        with np.load(path) as batch:
            np.testing.assert_array_equal(latent.cpu().numpy(), batch['latents'])
        golden.append(dict(family=family, old_arm=row['arm'], sha256=sha(path), exact=True))
    repeated, _ = core.sample(rt, noise, labels, rows[0])
    assert torch.equal(original, repeated)
    result = dict(passed=True, rank=rank, limits=limits, grid_probes=probes, trajectories=trajectories,
        golden=golden, zero_ode_and_sde_exact=True, disabled_calibration_exact=True,
        native_after_interventions_exact=True, runtime_sources=rt.sources,
        source_hashes={str(path): sha(path) for path in sources()},
        assets={str(path): sha(path) for path in core.source_assets()}, no_fid_used=True)
    atomic(ROOT/'development'/f'rank{rank}.json', result)
    print(dict(rank=rank, preflight_passed=True), flush=True)


def development_check():
    configure()
    ROOT.mkdir(parents=True, exist_ok=True)
    cpu = read(ROOT/'cpu_checks.json')
    assert cpu['passed']
    for path, digest in cpu['sources'].items():
        assert sha(path) == digest, path
    assert dependency_complete(dict(phase='complete', no_further_sampling_queued=True, total_arms=140))
    for value in (dict(phase='complete'), dict(phase='complete', stage='pasted_screen_1k'),
                  dict(phase='failed', no_further_sampling_queued=True, total_arms=140)):
        assert not dependency_complete(value)
    processes, streams = [], []
    directory = ROOT/'development'
    directory.mkdir(exist_ok=True)
    try:
        for rank in range(4):
            stream = (directory/f'rank{rank}.log').open('w')
            streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON, '-u', '-m', MODULE, '--check-rank', str(rank)],
                cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT))
        atomic(directory/'processes.json', dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'GPU development check failed: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=30)
        for stream in streams:
            stream.close()
    ranks = [read(directory/f'rank{r}.json') for r in range(4)]
    assert all(r['passed'] for r in ranks)
    for record in ranks:
        for path, digest in {**record['source_hashes'], **record['assets']}.items():
            assert sha(path) == digest, path
    probes = [arm for record in ranks for arm in record['grid_probes']]
    expected = [c['arm'] for c in catalog.configurations() if c['key'] in core.SPECIAL]
    assert len(probes) == len(set(probes)) and set(probes) == set(expected)
    trajectories = [row for record in ranks for row in record['trajectories']]
    assert len(trajectories) == 59
    golden = [row for record in ranks for row in record['golden']]
    assert {r['family'] for r in golden} == {'ig_local', 'cfg_native', 'cfg_apg'}
    result = dict(passed=True, cpu_checks_sha256=sha(ROOT/'cpu_checks.json'),
        ranks=[dict(rank=r, sha256=sha(directory/f'rank{r}.json')) for r in range(4)],
        grid_probes=probes, trajectories=trajectories, golden=golden,
        source_hashes=ranks[0]['source_hashes'], assets=ranks[0]['assets'],
        runtime_sources=ranks[0]['runtime_sources'], dependency_final_marker_checks=True,
        no_fid_used=True)
    atomic(ROOT/'development_check.json', result)
    print(dict(development_check_passed=True, grid=len(probes), trajectories=len(trajectories)), flush=True)


def pipeline():
    configure()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame):
        raise RuntimeError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        check = read(ROOT/'development_check.json')
        assert check['passed']
        for path, digest in {**check['source_hashes'], **check['assets']}.items():
            assert sha(path) == digest, path
        while True:
            dependency = read(WAIT_ROOT/'status.json')
            if dependency_complete(dependency):
                state = read(WAIT_ROOT/'pasted_screen_1k/status.json')
                assert state['phase'] == 'complete' and state['completed'] == state['total'] == 140
                assert read(WAIT_ROOT/'pasted_screen_1k/analysis_audit.json')['passed']
                break
            if dependency['phase'] == 'failed':
                raise RuntimeError(f'Dependency failed: {dependency}')
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                atomic(ROOT/'status.json', dict(phase='stopped_while_waiting', controller_pid=os.getpid()))
                return
            atomic(ROOT/'status.json', dict(phase='waiting_for_fsg7_final', controller_pid=os.getpid(),
                dependency=dependency, total_arms=709, candidate_arms=636, control_arms=73))
            time.sleep(10)
        atomic(ROOT/'dependency_released.json', dict(dependency=dependency, released_unix=time.time()))
        engine.prepare_stage(STAGE, catalog.configurations())
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase'] != 'complete':
            return
        rows = read(ROOT/STAGE/'results.json')
        selected = []
        for family in dict.fromkeys(c['family'] for c in catalog.configurations()):
            valid = [r for r in rows if r['family'] == family and r['complete']]
            if valid:
                selected.append(min(valid, key=lambda r: r['fid'])['arm'])
        analysis.audit_stage(ROOT/STAGE, selected)
        write_progress(ROOT/STAGE, read(ROOT/STAGE/'request.json'), rows)
        atomic(ROOT/'status.json', dict(phase='complete', total_arms=709, candidate_arms=636, control_arms=73,
            samples_per_arm=1000, numerical_failures=sum(not r['complete'] for r in rows),
            audited_best_per_family=selected, no_further_sampling_queued=True, research_goal_achieved=False))
    except BaseException as error:
        previous = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json', dict(phase='failed', error=repr(error), previous=previous, controller_pid=os.getpid()))
        raise
    finally:
        lock.close()


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('pipeline', 'check', 'cpu-check', 'prepare', 'status', 'stop-after-current'):
        action.add_argument('--'+name, action='store_true')
    action.add_argument('--worker', type=int, choices=range(4))
    action.add_argument('--check-rank', type=int, choices=range(4))
    parser.add_argument('--stage', choices=[STAGE])
    parser.add_argument('--run-id')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.pipeline:
        pipeline()
    elif args.check:
        development_check()
    elif args.cpu_check:
        checks.cpu_checks()
    elif args.prepare:
        engine.prepare_stage(STAGE, catalog.configurations())
    elif args.status:
        print(read(ROOT/'status.json'))
    elif args.stop_after_current:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'STOP_AFTER_CURRENT').touch()
    elif args.check_rank is not None:
        check_rank(args.check_rank)
    else:
        assert args.stage and args.run_id and args.parent_pid
        engine.worker(args.stage, args.worker, args.run_id, args.parent_pid)
