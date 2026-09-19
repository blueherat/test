"""Audited seven-method follow-up, queued after the entire 20-method pipeline."""
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
from experiments import run_sit_fsg_pasted_20260910 as catalog
from experiments.sit_fsg_pasted_20260910 import core, readout
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, array_sha, atomic, read, sha

ROOT = EXPS/'sit_fsg_pasted_followup_20260910'
WAIT_ROOT = EXPS/'sit_fsg_followup_20ideas_20260910'
STAGE = 'pasted_screen_1k'
MODULE = 'experiments.run_sit_fsg_pasted_20260910'
PROTOCOL = WORK/'docs/SIT_FSG_PASTED_PROTOCOL_20260910_ZH.md'
RESEARCH = WORK/'docs/FSG_PASTED_ANALYSIS_REVIEW_20260910_ZH.md'
REPORT = WORK/'docs/SIT_FSG_PASTED_RESULTS_20260910_ZH.md'
_verify_parent = engine.verify_parent
_factory = engine.parent.operators.make_runtime
_evaluate = engine.infrastructure.evaluate
_independent = None


def sources():
    return list(dict.fromkeys([
        Path(__file__).resolve(), Path(core.__file__).resolve(), Path(readout.__file__).resolve(),
        Path(catalog.__file__).resolve(), Path(core.prior.__file__).resolve(),
        Path(core.prior.catalog.__file__).resolve(), Path(engine.__file__).resolve(),
        Path(analysis.__file__).resolve(), WORK/'experiments/sit_guidance_fusion_20260910.py',
        WORK/'experiments/sit_fsg_pasted_20260910/diagnostics.py',
        WORK/'experiments/audit_fsg_pasted_theory_20260910.py', PROTOCOL, RESEARCH,
    ]))


def verify_parent():
    value = _verify_parent()
    return dict(value, assets={**value['assets'], **{str(path): sha(path) for path in core.source_assets()}})


def make_runtime():
    rt = _factory()
    # Load and warm up auxiliary assets outside each timed sample batch.
    rt.pasted_semantic = core.SemanticReadout(rt)
    with torch.inference_mode():
        rt.pasted_semantic.probabilities(torch.zeros((2, 4, 32, 32), device='cuda'))
    return rt


def evaluate(arm, request, request_hash):
    global _independent
    row = _evaluate(arm, request, request_hash)
    if not row['complete']:
        return row
    base = ROOT/STAGE/arm
    totals = dict(auxiliary_decoder_images=0., auxiliary_classifier_images=0.,
                  accepted_calibrations=0., calibration_events=0., calibration_full_calls=0.,
                  calibration_loss_before_sum=0., calibration_loss_after_sum=0.)
    for rank in range(4):
        for rec in read(base/f'rank{rank}/summary.json')['files']:
            with np.load(base/f'rank{rank}'/rec['file']) as batch:
                for key in totals:
                    if key in batch: totals[key] += float(batch[key])
    row['auxiliary_counts'] = totals
    row['auxiliary_decoder_images_per_output'] = totals['auxiliary_decoder_images']/request['samples']
    row['auxiliary_classifier_images_per_output'] = totals['auxiliary_classifier_images']/request['samples']
    if _independent is None: _independent = readout.IndependentReadout()
    with np.load(base/'samples.npz') as data:
        pixels = data['arr_0']
    labels = np.load(Path(request['bank_root'])/'labels.npy')
    torch.cuda.synchronize(); begin = time.perf_counter()
    values = _independent.images(pixels, labels)
    torch.cuda.synchronize()
    np.savez(base/'independent_alignment.npz', values=values, labels=labels)
    row.update(independent_top1=float(values[:, 1].mean()),
               independent_target_probability=float(values[:, 0].mean()),
               independent_classifier_sha256=sha(core.INDEPENDENT_WEIGHTS),
               independent_alignment_sha256=sha(base/'independent_alignment.npz'),
               independent_alignment_seconds=time.perf_counter()-begin,
               independent_alignment_is_quality_metric=False)
    atomic(base/'result.json', row)
    return row


def configure():
    engine.ROOT = ROOT; engine.MODULE = MODULE; engine.PROTOCOL = PROTOCOL
    engine.STAGES = {STAGE: (1000, 202610090)}; engine.fusion = core
    engine.additional_sources = sources; engine.verify_parent = verify_parent
    engine.parent.operators.make_runtime = make_runtime
    engine.infrastructure.evaluate = evaluate
    analysis.write_progress = write_progress


def dependency_complete(status):
    return (status.get('phase') == 'complete' and status.get('no_further_sampling_queued') is True
            and 'stage' not in status and status.get('total_arms') == 256)


def write_progress(base, request, rows):
    valid = [row for row in rows if row['complete']]
    def best(predicate):
        pool = [row for row in valid if predicate(row)]
        return min(pool, key=lambda row: row['fid']) if pool else None
    native = best(lambda row: row['family'] == 'cfg_native')
    apg = best(lambda row: row['family'] == 'cfg_apg')
    fields = ['arm', 'idea_id', 'family', 'key', 'role', 'solver', 'strength', 'theta',
              'external_semantics', 'fid', 'sfid', 'inception_score', 'independent_top1',
              'independent_target_probability', 'full_calls_per_image', 'prefix_calls_per_image',
              'auxiliary_decoder_images_per_output', 'auxiliary_classifier_images_per_output',
              'sum_batch_gpu_seconds', 'complete']
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        value = dict(row)
        if row['complete']: value.update({key: row['metrics'][key] for key in ('sfid', 'inception_score')})
        writer.writerow(value)
    (base/'results.csv').write_text(output.getvalue())
    portable = WORK/'docs/data/sit_fsg_pasted_followup_20260910'; portable.mkdir(parents=True, exist_ok=True)
    (portable/'all_results.csv').write_text(output.getvalue())
    lines = ['# FSG 七个完整未来假设：1K 筛选\n',
             f'已完成 {len(rows)}/{len(request["configs"])} 组，数值失败 {sum(not row["complete"] for row in rows)}。七项各十组，另有七十组对照。\n',
             '本轮使用新噪声、每类10张的配对调参。最佳FID有选择偏差，尚不是独立确认。独立ResNet18的top1只衡量类别对齐。\n']
    if native: lines.append(f'原生CFG最佳 FID {native["fid"]:.6f}，`{native["arm"]}`。\n')
    if apg: lines.append(f'APG最佳 FID {apg["fid"]:.6f}，`{apg["arm"]}`。\n')
    lines += ['|方向|完成/10|最佳FID|Δ匹配对照|对照|a / 参数|成本×对照|独立top1|Full/prefix|内部解码/分类张数|',
              '|---|--:|--:|--:|---|---|--:|--:|---|---|']
    for method in catalog.METHODS:
        done = sum(row['idea_id'] == method['id'] for row in rows)
        row = best(lambda row: row['idea_id'] == method['id'])
        if row is None:
            lines.append(f'|{method["id"]} {method["title"]}|{done}/10|—|—|—|—|—|—|—|—|'); continue
        control = best(lambda other: other['source'] == 'cfg' and other['role'] != 'candidate'
                       and other['solver'] == row['solver']
                       and other['external_semantics'] == row['external_semantics'])
        delta = f'{row["fid"]-control["fid"]:+.6f}' if control else '—'
        ratio = f'{row["sum_batch_gpu_seconds"]/control["sum_batch_gpu_seconds"]:.3f}' if control else '—'
        name = control['arm'] if control else '待完成'
        lines.append(f'|{method["id"]} {method["title"]}|{done}/10|{row["fid"]:.6f}|{delta}|{name}|{row["strength"]}/{row["theta"]}|{ratio}|{row["independent_top1"]:.1%}|{row["full_calls_per_image"]:.0f}/{row["prefix_calls_per_image"]:.0f}|{row["auxiliary_decoder_images_per_output"]:.0f}/{row["auxiliary_classifier_images_per_output"]:.0f}|')
    lines += ['\nA4只与同一SDE采样器比较；B1使用额外ConvNeXt，匹配对照是相同分类器的终点目标置信度控制。其他方向与无额外语义资产的ODE对照比较。对照也分别调参，但计算预算未强制相等。\n',
              'Full/prefix为每条生成轨迹调用数；内部解码/分类为每张最终输出的额外处理张数。采样成本包含这些内部开销，后验FID和独立类别评价另记，不混入采样成本。\n',
              f'原始记录：`{base}`；请求SHA256：`{sha(base/"request.json")}`。\n',
              '[理论分析](FSG_PASTED_ANALYSIS_REVIEW_20260910_ZH.md) · [协议](SIT_FSG_PASTED_PROTOCOL_20260910_ZH.md) · [诊断](SIT_FSG_PASTED_DIAGNOSTICS_20260910_ZH.md) · [逐组CSV](data/sit_fsg_pasted_followup_20260910/all_results.csv)。\n']
    if (base/'analysis_audit.json').exists():
        audit = read(base/'analysis_audit.json')
        lines.append(f'\n全部覆盖和元数据通过；{len(audit["raw_and_metric_audits"])}个代表配置额外通过逐批哈希、实际成本和缓存特征FP64 FID/sFID复算。\n')
    text = '\n'.join(lines); (base/'report.md').write_text(text); REPORT.write_text(text)


@torch.inference_mode()
def development_check():
    configure(); ROOT.mkdir(parents=True, exist_ok=True)
    cpu = core.cpu_checks(); verify_parent()
    theory = read(ROOT/'theory_checks.json'); assert theory['passed']
    assert theory['source_sha256'] == sha(WORK/'experiments/audit_fsg_pasted_theory_20260910.py')
    assert dependency_complete(dict(phase='complete', no_further_sampling_queued=True, total_arms=256))
    for state in [dict(phase='complete'), dict(phase='complete', stage='fsg_screen_1k'),
                  dict(phase='failed', no_further_sampling_queued=True, total_arms=256)]:
        assert not dependency_complete(state)
    rt = make_runtime()
    noise = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'noise.npy')[:8].copy()).cuda()
    labels = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'labels.npy')[:8].copy()).cuda()
    limits = core.limiting_checks(rt, noise, labels); hooks = engine.parent.hook_counts(rt)
    rows = catalog.configurations(); strong = rows[0]
    original, _ = core.sample(rt, noise, labels, strong)
    probes = []
    for config in rows:
        if config['key'] not in core.SPECIAL or config['key'] in ('native_sde', 'smc_control'): continue
        rt.labels = labels[:2]; event = core.event_context(noise[:2], 8)
        value, rec = core.calibrate(rt, noise[:2], .125, config, event, config['strength'])
        assert torch.isfinite(value).all() and all(np.isfinite(v) for v in rec.values()), config['arm']
        assert rec['after'] <= rec['before']+1e-6
        assert engine.parent.hook_counts(rt) == hooks
        probes.append(config['arm'])
        if len(probes) % 10 == 0: print({'full_future_grid_probes': len(probes)}, flush=True)
    trajectories = []
    selected = [[row for row in rows if row['idea_id'] == method['id']][-1] for method in catalog.METHODS]
    selected += [[row for row in rows if row['family'] == family][-1] for family in
                 ('full_root_control', 'fsg_operator_control', 'semantic_contract_control', 'cfg_smc_control', 'cfg_sde_native')]
    sde_zero = None
    for config in selected:
        latent, stats = core.sample(rt, noise, labels, config)
        assert torch.isfinite(latent).all() and np.isfinite(stats['diagnostics']).all()
        zero, _ = core.sample(rt, noise, labels, config, zero=True)
        if config['solver'] == 'sde_tail64':
            if sde_zero is None: sde_zero = zero
            else: assert torch.equal(zero, sde_zero)
        else: assert torch.equal(zero, original), config['arm']
        if config['key'] in core.SPECIAL and config['key'] not in ('native_sde', 'smc_control'):
            native = next(row for row in rows if row['key'] == ('native_sde' if config['solver'] == 'sde_tail64' else 'native_cfg'))
            native = dict(native, strength=config['strength'], cutoff=config['cutoff'])
            disabled, _ = core.sample(rt, noise, labels, config, disable_calibration=True)
            base, _ = core.sample(rt, noise, labels, native)
            assert torch.equal(disabled, base), config['arm']
        assert engine.parent.hook_counts(rt) == hooks and rt.labels is labels
        entry = dict(arm=config['arm'], full=stats['full_calls'], prefix=stats['prefix_calls'],
                     accepted=stats.get('accepted_calibrations'), max_abs=float(latent.abs().max()),
                     auxiliary_decoder_images=stats.get('auxiliary_decoder_images', 0),
                     auxiliary_classifier_images=stats.get('auxiliary_classifier_images', 0))
        trajectories.append(entry); print(entry, flush=True)
    golden = []
    for family in ('ig_local', 'cfg_native', 'cfg_apg'):
        pairs = [(c, r) for c, r in engine.golden_pairs(rows) if c['family'] == family]
        assert pairs, family
        config, row = pairs[0]; latent, _ = core.sample(rt, noise, labels, config)
        path = engine.parent.ROOT/row['arm']/'rank0/batch0000.npz'
        with np.load(path) as batch: np.testing.assert_array_equal(latent.cpu().numpy(), batch['latents'])
        golden.append(dict(family=family, old_arm=row['arm'], sha256=sha(path), exact=True))
    repeated, _ = core.sample(rt, noise, labels, strong); assert torch.equal(original, repeated)
    independent = readout.IndependentReadout().latents(rt, original, labels)
    assert independent.shape == (8, 3) and np.isfinite(independent).all()
    result = dict(passed=True, cpu=cpu, limits=limits, grid_probes=probes, trajectories=trajectories,
                  theory_checks_sha256=sha(ROOT/'theory_checks.json'), theory_groups=list(theory['checks']),
                  golden=golden, zero_ode_and_sde_exact=True, disabled_calibration_exact=True,
                  native_after_interventions_exact=True, dependency_final_marker_checks=True,
                  source_hashes={str(path): sha(path) for path in sources()}, runtime_sources=rt.sources,
                  assets={str(path): sha(path) for path in core.source_assets()}, no_fid_used=True)
    atomic(ROOT/'development_check.json', result)
    print(dict(development_check_passed=True, grid=len(probes), trajectories=len(trajectories)), flush=True)


def run_diagnostics():
    from experiments.sit_fsg_pasted_20260910 import diagnostics
    paths = [ROOT/'diagnostics'/f'rank{rank}.json' for rank in range(4)]
    processes = []; streams = []
    try:
        for rank, path in enumerate(paths):
            if path.exists() and read(path).get('passed'): continue
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.with_suffix('.log').open('a'); streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON, '-u', '-m', MODULE, '--diagnostic-rank', str(rank)],
                cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes): raise RuntimeError('Diagnostic worker failed')
            time.sleep(2)
        assert all(process.returncode == 0 for process in processes)
        diagnostics.summarize()
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes: process.wait(timeout=30)
        for stream in streams: stream.close()


def pipeline():
    configure(); ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame): raise RuntimeError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        check = read(ROOT/'development_check.json'); assert check['passed']
        for path, digest in {**check['source_hashes'], **check['assets']}.items(): assert sha(path) == digest, path
        while True:
            dependency = read(WAIT_ROOT/'status.json')
            if dependency_complete(dependency):
                state = read(WAIT_ROOT/'fsg_screen_1k/status.json')
                assert state['phase'] == 'complete' and state['completed'] == state['total'] == 256
                assert read(WAIT_ROOT/'fsg_screen_1k/analysis_audit.json')['passed']
                break
            if dependency['phase'] == 'failed': raise RuntimeError(f'Dependency failed: {dependency}')
            atomic(ROOT/'status.json', dict(phase='waiting_for_fsg20_final', controller_pid=os.getpid(),
                dependency=dependency, total_arms=140, candidate_arms=70, control_arms=70))
            time.sleep(10)
        atomic(ROOT/'dependency_released.json', dict(dependency=dependency, released_unix=time.time()))
        atomic(ROOT/'status.json', dict(phase='diagnostics', controller_pid=os.getpid()))
        run_diagnostics()
        engine.prepare_stage(STAGE, catalog.configurations()); engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase'] != 'complete': return
        rows = read(ROOT/STAGE/'results.json'); selected = []
        for family in dict.fromkeys(row['family'] for row in catalog.configurations()):
            valid = [row for row in rows if row['family'] == family and row['complete']]
            if valid: selected.append(min(valid, key=lambda row: row['fid'])['arm'])
        analysis.audit_stage(ROOT/STAGE, selected)
        for row in rows:
            if not row['complete']: continue
            path = ROOT/STAGE/row['arm']/'independent_alignment.npz'
            assert sha(path) == row['independent_alignment_sha256']
            with np.load(path) as data:
                assert data['values'].shape == (1000, 3)
                assert float(data['values'][:, 1].mean()) == row['independent_top1']
        write_progress(ROOT/STAGE, read(ROOT/STAGE/'request.json'), rows)
        atomic(ROOT/'status.json', dict(phase='complete', total_arms=140, candidate_arms=70, control_arms=70,
            samples_per_arm=1000, numerical_failures=sum(not row['complete'] for row in rows),
            audited_best_per_family=selected, no_further_sampling_queued=True, research_goal_achieved=False))
    except BaseException as error:
        previous = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json', dict(phase='failed', error=repr(error), previous=previous, controller_pid=os.getpid()))
        raise
    finally: lock.close()


def main():
    configure(); parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('pipeline', 'check', 'status', 'stop-after-current', 'diagnostics', 'prepare'):
        action.add_argument('--'+name, action='store_true')
    action.add_argument('--worker', type=int, choices=range(4))
    action.add_argument('--diagnostic-rank', type=int, choices=range(4))
    parser.add_argument('--stage', choices=[STAGE]); parser.add_argument('--run-id'); parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.pipeline: pipeline()
    elif args.check: development_check()
    elif args.prepare: engine.prepare_stage(STAGE, catalog.configurations())
    elif args.diagnostics: run_diagnostics()
    elif args.status: print(read(ROOT/'status.json'))
    elif args.stop_after_current: ROOT.mkdir(parents=True, exist_ok=True); (ROOT/'STOP_AFTER_CURRENT').touch()
    elif args.diagnostic_rank is not None:
        from experiments.sit_fsg_pasted_20260910 import diagnostics
        diagnostics.run_rank(args.diagnostic_rank)
    else:
        assert args.stage and args.run_id and args.parent_pid
        engine.worker(args.stage, args.worker, args.run_id, args.parent_pid)
