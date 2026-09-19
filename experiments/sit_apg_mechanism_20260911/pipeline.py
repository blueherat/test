"""Append the frozen APG extension after the existing 709-arm queue finishes."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
import torch
from experiments.sit_apg_mechanism_20260911 import core, catalog, checks, study
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import WORK, atomic, read, sha, array_sha

ROOT = study.ROOT
WAIT_ROOT = study.DEPENDENCY
STAGE = 'apg_extension_screen_1k'
MODULE = 'experiments.sit_apg_mechanism_20260911.pipeline'
PROTOCOL = WORK/'docs/SIT_APG_EXTENSION_PROTOCOL_20260911_ZH.md'
RESEARCH = WORK/'docs/APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md'
REPORT = WORK/'docs/SIT_APG_EXTENSION_RESULTS_20260911_ZH.md'
SUMMARY = WORK/'docs/data/apg_mechanism_extension_20260911/summary.json'
_factory = engine.parent.operators.make_runtime
_evaluate = engine.infrastructure.evaluate


def sources():
    folder = Path(__file__).resolve().parent
    return [folder/name for name in ('__init__.py', 'common.py', 'study.py', 'analyze.py',
        'catalog.py', 'core.py', 'checks.py', 'pipeline.py', 'guarded_check.py')]


def make_runtime():
    rt = _factory()
    rt.pasted_semantic = core.previous.seven.SemanticReadout(rt)
    with torch.inference_mode():
        rt.pasted_semantic.probabilities(torch.zeros((2, 4, 32, 32), device='cuda'))
    return rt


def evaluate(arm, request, request_hash):
    result = _evaluate(arm, request, request_hash)
    if not result['complete']:
        return result
    config = next(c for c in request['configs'] if c['arm'] == arm)
    totals = dict(auxiliary_decoder_images=0., auxiliary_classifier_images=0.,
        extension_events=0., extension_event_full_calls=0., refined_sample_steps=0.,
        selected_nonzero=0., final_released_samples=0.)
    traces = []
    for rank in range(4):
        folder = ROOT/STAGE/arm/f'rank{rank}'
        for record in read(folder/'summary.json')['files']:
            with np.load(folder/record['file']) as batch:
                for key in totals:
                    if key in batch:
                        totals[key] += float(batch[key])
                traces.append(checks.audit_trace(batch, config))
    assert len(traces) == 125
    result.update(auxiliary_counts=totals, decision_batch_traces_audited=len(traces),
        auxiliary_decoder_images_per_output=totals['auxiliary_decoder_images']/1000,
        auxiliary_classifier_images_per_output=totals['auxiliary_classifier_images']/1000,
        posthoc_semantic_evaluation=False)
    atomic(ROOT/STAGE/arm/'decision_audit.json', dict(passed=True, batches=traces, totals=totals))
    atomic(ROOT/STAGE/arm/'result.json', result)
    return result


def configure():
    engine.ROOT, engine.MODULE, engine.PROTOCOL = ROOT, MODULE, PROTOCOL
    engine.STAGES = {STAGE:(1000, catalog.NOISE_SEED)}
    engine.fusion = core
    engine.additional_sources = sources
    engine.parent.operators.make_runtime = make_runtime
    engine.infrastructure.evaluate = evaluate
    analysis.write_progress = write_progress


def dependency_complete(status):
    return status.get('phase') == 'complete' and status.get('no_further_sampling_queued') is True and \
        status.get('total_arms') == 709 and 'stage' not in status


def write_progress(base, request, rows):
    fields = ['arm', 'idea_id', 'family', 'role', 'strength', 'theta', 'external_semantics',
        'fid', 'full_calls_per_image', 'prefix_calls_per_image',
        'auxiliary_decoder_images_per_output', 'auxiliary_classifier_images_per_output',
        'sum_batch_gpu_seconds', 'decision_batch_traces_audited', 'complete']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    (base/'results.csv').write_text(stream.getvalue())
    portable = WORK/'docs/data/apg_mechanism_extension_20260911'
    portable.mkdir(parents=True, exist_ok=True)
    (portable/'all_results.csv').write_text(stream.getvalue())
    valid = [r for r in rows if r['complete']]
    def best(predicate):
        pool = [r for r in valid if predicate(r)]
        return min(pool, key=lambda r:r['fid']) if pool else None
    lines = ['# APG机制延伸的1K筛选\n',
        f'已提交 {len(rows)}/201 组；60组新增候选、141组对照，数值失败 {sum(not r["complete"] for r in rows)}。\n',
        '沿用53-idea队列的同一1K噪声及标签，每类10张；机制实验的32个种子独立。不会自动追加5K。\n',
        '|方案|完成数|最低FID|对照|ΔFID|成本×对照|引导量 / 参数|内部解码、分类张数/输出|',
        '|---|--:|--:|---|--:|--:|---|---|']
    for method in catalog.IDEAS:
        done = sum(r['idea_id'] == method['id'] for r in rows)
        row = best(lambda r:r['idea_id'] == method['id'])
        if row is None:
            lines.append(f'|{method["id"]} {method["title"]}|{done}/12|—|—|—|—|—|—|')
            continue
        baseline = best(lambda r:r['role'] != 'candidate' and r['source'] == 'cfg'
                        and r['external_semantics'] == row['external_semantics'])
        assert baseline is not None
        lines.append(f'|{method["id"]} {method["title"]}|{done}/12|{row["fid"]:.6f}|{baseline["arm"]}|'
            f'{row["fid"]-baseline["fid"]:+.6f}|{row["sum_batch_gpu_seconds"]/baseline["sum_batch_gpu_seconds"]:.3f}|'
            f'{row["strength"]} / {row["theta"]}|{row["auxiliary_decoder_images_per_output"]:.0f}, '
            f'{row["auxiliary_classifier_images_per_output"]:.0f}|')
    lines += ['\n|对照family|完成数|最低FID|参数|', '|---|--:|--:|---|']
    for family in dict.fromkeys(c['family'] for c in request['configs'] if c['role'] != 'candidate'):
        count = sum(r['family'] == family for r in rows)
        row = best(lambda r:r['family'] == family)
        planned = sum(c['family'] == family for c in request['configs'])
        lines.append(f'|{family}|{count}/{planned}|{row["fid"]:.6f}|{row["strength"]} / {row["theta"]}|' if row
                     else f'|{family}|{count}/{planned}|—|—|')
    lines += ['\nΔFID为候选减已完成的同语义资产组最佳对照。另需查看逐组CSV中的计算成本和专门消融；预算未强行相等。网格最低值只是筛选结果，不代表统计显著性或独立确认。\n',
        '原APG适配与新增clean-buffer APG分别保留。主采样、全部候选探针、内部VAE/ConvNeXt和最终解码均计入采样成本；FID提取单列。逐批审计实际选择和接管记录。\n',
        '[机制与推导](APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md) · '
        '[冻结协议](SIT_APG_EXTENSION_PROTOCOL_20260911_ZH.md) · '
        '[逐组结果](data/apg_mechanism_extension_20260911/all_results.csv)。\n',
        f'请求SHA256：`{sha(base/"request.json")}`。\n']
    if (base/'analysis_audit.json').exists():
        lines.append('所有配置覆盖审计通过，各family最佳配置另完成原始批次哈希、成本及FP64 FID复算。\n')
    text = '\n'.join(lines)
    (base/'report.md').write_text(text);REPORT.write_text(text)


@torch.inference_mode()
def check_rank(rank):
    configure()
    engine.verify_parent()
    rt = make_runtime()
    noise = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'noise.npy')[:8].copy()).cuda()
    labels = torch.from_numpy(np.load(engine.parent.BANK_ROOT/'labels.npy')[:8].copy()).cuda()
    rt.labels = labels
    hooks = engine.parent.hook_counts(rt)
    limits = core.limiting_checks(rt, noise, labels)
    rows = catalog.configurations()
    native, _ = core.sample(rt, noise, labels, rows[0])
    configs = [c for c in rows if not c['parameters'].get('inherited_exact')]
    probes, trajectories, golden = [], [], []
    for index, config in enumerate(configs):
        if index % 4 != rank:
            continue
        z, stats = core.sample(rt, noise[:2], labels[:2], config)
        assert torch.isfinite(z).all() and engine.parent.hook_counts(rt) == hooks
        trace = checks.audit_trace(stats, config)
        probes.append(dict(arm=config['arm'], full_calls=stats['full_calls'], trace=trace))
        print(dict(rank=rank, configuration=config['arm'], full_calls=stats['full_calls']), flush=True)
    families = list(dict.fromkeys(c['family'] for c in configs))
    for index, family in enumerate(families):
        if index % 4 != rank:
            continue
        config = next(c for c in reversed(configs) if c['family'] == family)
        z, stats = core.sample(rt, noise, labels, config)
        trace = checks.audit_trace(stats, config)
        assert torch.isfinite(z).all() and rt.labels is labels and engine.parent.hook_counts(rt) == hooks
        zero, _ = core.sample(rt, noise, labels, config, zero=True)
        assert torch.equal(zero, native)
        reference = checks.disabled_reference(config)
        if reference:
            disabled, _ = core.sample(rt, noise, labels, config, disable_calibration=True)
            expected, _ = core.sample(rt, noise, labels, reference)
            assert torch.equal(disabled, expected), family
        trajectories.append(dict(arm=config['arm'], family=family, full_calls=stats['full_calls'],
            auxiliary_decoder_images=stats['auxiliary_decoder_images'], trace=trace,
            max_abs=float(z.abs().max()), zero_exact=True, disabled_exact=reference is not None))
    for index, family in enumerate(('ig_local', 'cfg_native', 'cfg_apg')):
        if index != rank:
            continue
        pair = next((c, r) for c, r in engine.golden_pairs(rows) if c['family'] == family)
        config, row = pair
        z, _ = core.sample(rt, noise, labels, config)
        path = engine.parent.ROOT/row['arm']/'rank0/batch0000.npz'
        with np.load(path) as batch:
            np.testing.assert_array_equal(z.cpu().numpy(), batch['latents'])
        golden.append(dict(family=family, old_arm=row['arm'], sha256=sha(path), exact=True))
    again, _ = core.sample(rt, noise, labels, rows[0])
    assert torch.equal(again, native) and rt.labels is labels
    atomic(ROOT/'development'/f'rank{rank}.json', dict(passed=True, rank=rank, limits=limits,
        configurations=probes, trajectories=trajectories, golden=golden,
        native_after_interventions_exact=True, runtime_sources=rt.sources,
        source_hashes={str(p):sha(p) for p in sources()}, no_fid=True))


def development_check():
    configure()
    cpu = read(ROOT/'cpu_checks.json')
    assert cpu['passed']
    for path, digest in cpu['source_hashes'].items():
        assert sha(path) == digest
    assert read(WAIT_ROOT/'status.json')['phase'] in ('stopped_after_current', 'complete')
    folder = ROOT/'development';folder.mkdir(exist_ok=True)
    processes, streams = [], []
    try:
        for rank in range(4):
            stream = (folder/f'rank{rank}.log').open('w');streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON, '-u', '-m', MODULE, '--check-rank', str(rank)],
                cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT))
        atomic(folder/'processes.json', dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Preflight failure: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for stream in streams:stream.close()
    ranks = [read(folder/f'rank{r}.json') for r in range(4)]
    assert all(r['passed'] for r in ranks)
    for r in ranks:
        for path, digest in r['source_hashes'].items():assert sha(path) == digest
    actual = [r['arm'] for result in ranks for r in result['configurations']]
    expected = [c['arm'] for c in catalog.configurations() if not c['parameters'].get('inherited_exact')]
    assert len(actual) == len(set(actual)) == 180 and set(actual) == set(expected)
    golden = [r for result in ranks for r in result['golden']]
    assert {r['family'] for r in golden} == {'ig_local', 'cfg_native', 'cfg_apg'}
    atomic(ROOT/'development_check.json', dict(passed=True, cpu_checks_sha256=sha(ROOT/'cpu_checks.json'),
        ranks=[dict(rank=r, sha256=sha(folder/f'rank{r}.json')) for r in range(4)],
        configurations=actual, trajectories=[r for result in ranks for r in result['trajectories']],
        golden=golden, source_hashes=ranks[0]['source_hashes'], no_fid=True))


def prepare():
    configure()
    base = ROOT/STAGE
    if (base/'request.json').exists():
        value, digest = engine.verify_stage(STAGE)
        assert value['configs'] == catalog.configurations()
        return value
    check = read(ROOT/'development_check.json');assert check['passed']
    for path, digest in check['source_hashes'].items():assert sha(path) == digest
    dependency_path = WAIT_ROOT/'control_screen_1k/request.json'
    dependency = read(dependency_path)
    for category in ('sources', 'assets'):
        for path, digest in dependency[category].items():assert sha(path) == digest
    assert read(SUMMARY)['passed'] and sha(study.STUDY/'request.json') == read(SUMMARY)['request_sha256']
    configs = catalog.configurations()
    expected_bytes = len(configs)*1000*(256*256*3*2+4*32*32*8+60000)
    assert shutil.disk_usage(ROOT).free > expected_bytes+(15<<30)
    assert not base.exists()
    base.mkdir();bank = base/'inputs';bank.mkdir()
    for name, digest in dependency['bank_files'].items():
        source = Path(dependency['bank_root'])/name
        assert sha(source) == digest
        shutil.copyfile(source, bank/name)
        assert sha(bank/name) == digest
    noise, labels = np.load(bank/'noise.npy', mmap_mode='r'), np.load(bank/'labels.npy')
    assert array_sha(noise) == dependency['bank']['noise_sha256'] and array_sha(labels) == dependency['bank']['label_sha256']
    hashes = {**dependency['sources'], **{str(p):sha(p) for p in sources()+[PROTOCOL, RESEARCH]}}
    snapshot = base/'sources';snapshot.mkdir()
    for index, path in enumerate(sorted(hashes)):
        (snapshot/f'{index:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    references = {str(p):sha(p) for p in [ROOT/'development_check.json', ROOT/'cpu_checks.json',
        dependency_path, Path(dependency['bank_root'])/'noise.npy', Path(dependency['bank_root'])/'labels.npy',
        study.STUDY/'request.json', study.STUDY/'completed.json', SUMMARY]}
    value = dict(stage=STAGE, configs=configs, arms=[c['arm'] for c in configs], samples=1000, batch=8, ranks=4,
        sources=hashes, assets=dependency['assets'], references=references,
        reference=dependency['reference'], reference_sha256=dependency['reference_sha256'],
        inception_graph_sha256=dependency['inception_graph_sha256'],
        bank_root=str(bank), bank_files={name:sha(bank/name) for name in ('noise.npy', 'labels.npy')},
        bank=dict(dependency['bank'], reuse='Exact bytes copied from 53-idea bank for paired continuation'),
        independent_confirmation=False, no_automatic_5k=True, mechanism_heldout_samples=32,
        family_definitions=core.FAMILIES, diagnostic_names=engine.parent.DIAGNOSTICS,
        python=engine.PYTHON, estimated_output_bytes=expected_bytes, prepared_unix=time.time())
    atomic(base/'request.json', value)
    atomic(base/'status.json', dict(phase='prepared', completed=0, total=len(configs)))
    write_progress(base, value, [])
    engine.verify_stage(STAGE)
    return value


def pipeline():
    configure()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'pipeline.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame):raise RuntimeError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM, interrupted);signal.signal(signal.SIGINT, interrupted)
    try:
        engine.verify_stage(STAGE)
        while True:
            dependency = read(WAIT_ROOT/'status.json')
            if dependency_complete(dependency):
                state = read(WAIT_ROOT/'control_screen_1k/status.json')
                assert state['phase'] == 'complete' and state['completed'] == state['total'] == 709
                assert read(WAIT_ROOT/'control_screen_1k/analysis_audit.json')['passed']
                break
            if dependency['phase'] == 'failed':raise RuntimeError(f'Dependency failed: {dependency}')
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                atomic(ROOT/'status.json', dict(phase='stopped_while_waiting', controller_pid=os.getpid()))
                return
            atomic(ROOT/'status.json', dict(phase='waiting_for_control53_final', controller_pid=os.getpid(),
                dependency=dependency, total_arms=201, candidate_arms=60, control_arms=141))
            time.sleep(10)
        atomic(ROOT/'dependency_released.json', dict(dependency=dependency, released_unix=time.time()))
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase'] != 'complete':return
        rows = read(ROOT/STAGE/'results.json')
        selected = []
        for family in dict.fromkeys(c['family'] for c in catalog.configurations()):
            valid = [r for r in rows if r['family'] == family and r['complete']]
            if valid:selected.append(min(valid, key=lambda r:r['fid'])['arm'])
        analysis.audit_stage(ROOT/STAGE, selected)
        write_progress(ROOT/STAGE, read(ROOT/STAGE/'request.json'), rows)
        atomic(ROOT/'status.json', dict(phase='complete', total_arms=201, candidate_arms=60,
            control_arms=141, samples_per_arm=1000, numerical_failures=sum(not r['complete'] for r in rows),
            audited_best_per_family=selected, no_further_sampling_queued=True))
    except BaseException as error:
        previous = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json', dict(phase='failed', error=repr(error), previous=previous, controller_pid=os.getpid()))
        raise
    finally:lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--pipeline', action='store_true')
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--check', action='store_true')
    action.add_argument('--check-rank', type=int, choices=range(4))
    action.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--stage', choices=[STAGE]);parser.add_argument('--run-id');parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    configure()
    if args.pipeline:pipeline()
    elif args.prepare:prepare()
    elif args.check:development_check()
    elif args.check_rank is not None:check_rank(args.check_rank)
    else:engine.worker(args.stage, args.worker, args.run_id, args.parent_pid)


if __name__ == '__main__':
    main()
