"""Run a bounded priority order against the unchanged, frozen SiT requests."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WORK = Path('/home/zhoushunyu/eqvae')
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS/'sit_refined_priority_20260911'
REPORT = WORK/'docs/SIT_REFINED_PRIORITY_EXECUTION_20260911_ZH.md'
PROTOCOL = WORK/'docs/SIT_REFINED_PRIORITY_PROTOCOL_20260911_ZH.md'
PORTABLE = WORK/'docs/data/sit_refined_priority_20260911'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
SPECS = {
    'apg': dict(root=str(EXPS/'sit_apg_mechanism_extension_20260911'),
                stage='apg_extension_screen_1k', module='experiments.sit_apg_mechanism_20260911.pipeline'),
    'control': dict(root=str(EXPS/'sit_control_output_50ideas_20260910'),
                    stage='control_screen_1k', module='experiments.sit_control_50_20260910.pipeline'),
}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def request_path(key):
    spec = SPECS[key]
    return Path(spec['root'])/spec['stage']/'request.json'


def select_orders(requests):
    configs = requests['apg']['configs']
    take = lambda predicate: [c['arm'] for c in configs if predicate(c)]
    apg = take(lambda c: c['family'] == 'semantic_direction')
    apg += take(lambda c: c['family'] == 'projection_apg' and c['theta'] == 0.)
    apg += take(lambda c: c['idea_id'] == 57)
    apg += take(lambda c: c['family'] == 'probability_null_release')
    apg += take(lambda c: c['family'] == 'fixed_null_release')
    apg += take(lambda c: c['family'] == 'projection_apg' and c['theta'] != 0.)
    apg += take(lambda c: c['idea_id'] == 56)
    apg += take(lambda c: c['family'] == 'moment_without_semantic')
    apg += take(lambda c: c['idea_id'] == 55)
    apg += take(lambda c: c['family'] == 'clean_apg' and c['theta'] == -.5
                and c['parameters']['radius'] == 5.)
    control = [c['arm'] for idea in (53, 52) for c in requests['control']['configs']
               if c['idea_id'] == idea]
    assert len(apg) == len(set(apg)) == 76
    assert len(control) == len(set(control)) == 24
    return dict(apg=apg, control=control)


def ordered_request(request, order):
    by_arm = {c['arm']: c for c in request['configs']}
    assert len(by_arm) == len(request['configs'])
    assert len(order) == len(set(order)) and set(order) <= set(by_arm)
    chosen = set(order)
    value = dict(request, configs=[by_arm[arm] for arm in order]
                 +[c for c in request['configs'] if c['arm'] not in chosen])
    # The disk request and every configuration remain exact. Only iteration order changes.
    assert {c['arm']: c for c in value['configs']} == by_arm
    return value


def bounded_stop(order, rows, manual_stop=False):
    return manual_stop or set(order) <= {r['arm'] for r in rows}


def verify_plan():
    plan = read(ROOT/'schedule.json')
    for path, digest in plan['references'].items():
        assert sha(path) == digest, path
    for block in plan['blocks']:
        request = read(request_path(block['key']))
        by_arm = {c['arm']: c for c in request['configs']}
        assert {arm: canonical(by_arm[arm]) for arm in block['arms']} == block['config_sha256']
    return plan


def prepare():
    if (ROOT/'schedule.json').exists():
        return verify_plan()
    requests = {key: read(request_path(key)) for key in SPECS}
    orders = select_orders(requests)
    assert requests['apg']['bank']['noise_sha256'] == requests['control']['bank']['noise_sha256']
    assert requests['apg']['bank']['label_sha256'] == requests['control']['bank']['label_sha256']
    references = {str(p): sha(p) for p in [Path(__file__).resolve(), PROTOCOL,
                    *(request_path(key) for key in SPECS)]}
    # Existing controls remain in their original request; no result is copied or relabelled.
    control_base = request_path('control').parent
    baselines = []
    for config in requests['control']['configs']:
        if config['role'] == 'candidate':
            continue
        folder = control_base/config['arm']
        commit = read(folder/'commit.json')
        assert commit['request_sha256'] == sha(request_path('control'))
        assert sha(folder/'result.json') == commit['files']['result.json']
        assert read(folder/'result.json')['complete']
        for name in ('commit.json', 'result.json'):
            references[str(folder/name)] = sha(folder/name)
        baselines.append(config['arm'])
    assert len(baselines) == 73
    blocks = []
    for key, order in orders.items():
        request = requests[key]
        by_arm = {c['arm']: c for c in request['configs']}
        assert all(not (request_path(key).parent/arm/'commit.json').exists() for arm in order)
        blocks.append(dict(key=key, **SPECS[key], arms=order,
            config_sha256={arm: canonical(by_arm[arm]) for arm in order},
            original_total=len(request['configs']), request_sha256=sha(request_path(key))))
    plan = dict(created_unix=time.time(), reason='User explicitly prioritizes refined ideas over the broad 50-idea sweep',
        priority_order=[57, 56, 55, 53, 52], candidate_arms=60, new_control_arms=40,
        total_new_arms=100, samples_per_arm=1000, blocks=blocks, references=references,
        existing_control_arms=baselines, existing_control_source=str(control_base),
        selection_uses_new_candidate_fid=False, independent_confirmation=False,
        no_automatic_5k=True, automatic_broad_resume=False,
        dependency_override='Only the original scheduling gate is superseded; all frozen requests, source, asset, input, and runtime checks remain in force.')
    atomic(ROOT/'schedule.json', plan)
    atomic(ROOT/'status.json', dict(phase='prepared', completed=0, total=100))
    write_progress(plan)
    return verify_plan()


def write_progress(plan):
    rows = []
    for block in plan['blocks']:
        path = Path(block['root'])/block['stage']/'results.json'
        original = read(path) if path.exists() else []
        selected = set(block['arms'])
        rows += [dict(row, priority_block=block['key']) for row in original if row['arm'] in selected]
    fields = ['priority_block', 'arm', 'idea_id', 'family', 'role', 'strength', 'theta',
              'fid', 'full_calls_per_image', 'sum_batch_gpu_seconds',
              'auxiliary_decoder_images_per_output', 'auxiliary_classifier_images_per_output', 'complete']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    PORTABLE.mkdir(parents=True, exist_ok=True)
    (PORTABLE/'all_results.csv').write_text(stream.getvalue())
    atomic(ROOT/'results.json', rows)
    text = ['# 打磨方案优先运行记录\n',
        f'重点批次已提交 {len(rows)}/100 组，数值失败 {sum(not r["complete"] for r in rows)}。'
        '五个候选方向各12组，共60组候选和40组新对照；每组同一套1K输入。\n',
        '原大筛选在300/709组安全暂停，已有73组对照直接保留为配对参照。'
        '本批新增结果写回各自原请求，随后正常恢复时会跳过已提交配置。\n',
        '|优先方向|已完成/12|本轮最低FID|', '|---|--:|--:|']
    for idea, label in [(57, '继续条件的边际价值决定撤条件'), (56, '真实续生成选择APG平行保留量'),
                        (55, '带语义约束的未来幅度投影'), (53, '有限候选中的最小干预'),
                        (52, '经粗细未来共同验证的语义写入')]:
        found = [r for r in rows if r['idea_id'] == idea]
        valid = [r for r in found if r['complete']]
        fid = f'{min(r["fid"] for r in valid):.6f}' if valid else '—'
        text.append(f'|#{idea} {label}|{len(found)}/12|{fid}|')
    text += ['\n本批完成后自动停止；原大队列保持暂停。'
             '网格最低值属于1K筛选，完整对照和实际成本须一并查看。\n',
        '[调度协议](SIT_REFINED_PRIORITY_PROTOCOL_20260911_ZH.md) · '
        '[本批逐组结果](data/sit_refined_priority_20260911/all_results.csv) · '
        '[已有对照及大队列结果](SIT_CONTROL_53_IDEAS_RESULTS_20260910_ZH.md) · '
        '[APG扩展结果](SIT_APG_EXTENSION_RESULTS_20260911_ZH.md)。\n',
        f'运行目录：`{ROOT}`；调度SHA256：`{sha(ROOT/"schedule.json")}`。\n']
    REPORT.write_text('\n'.join(text))
    return rows


def run_block(key):
    plan = verify_plan()
    block = next(b for b in plan['blocks'] if b['key'] == key)
    root = Path(block['root'])
    lock = (root/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = root/'STOP_AFTER_CURRENT'
    try:
        assert marker.exists(), 'The original queue must be paused before priority takeover.'
        assert not (ROOT/'STOP_AFTER_CURRENT').exists(), 'Priority stop requested.'
        pipeline = importlib.import_module(block['module'])
        pipeline.configure()
        engine = pipeline.engine
        original_verify = engine.verify_stage
        original_progress = pipeline.analysis.write_progress
        request, digest = original_verify(block['stage'])
        assert digest == block['request_sha256']
        before = read(root/block['stage']/'results.json') if (root/block['stage']/'results.json').exists() else []
        initial = {r['arm'] for r in before}
        if bounded_stop(block['arms'], before):
            atomic(ROOT/f'{key}_completed.json', dict(passed=True, already_complete=True))
            return
        marker_text = marker.read_text()
        atomic(ROOT/f'{key}_takeover.json', dict(controller_pid=os.getpid(), original_status=read(root/'status.json'),
            original_stop_marker=marker_text, schedule_sha256=sha(ROOT/'schedule.json'), started_unix=time.time()))

        def verify_ordered(stage):
            value, h = original_verify(stage)
            assert h == digest
            return ordered_request(value, block['arms']), h

        def progress(base, value, rows):
            # Render against the unmodified request, and stop before the first unscheduled arm.
            original_progress(base, request, rows)
            assert {r['arm'] for r in rows} - initial <= set(block['arms'])
            if bounded_stop(block['arms'], rows, (ROOT/'STOP_AFTER_CURRENT').exists()):
                marker.write_text('Bounded priority batch finished or priority stop requested.\n')
            all_rows = write_progress(plan)
            atomic(ROOT/'status.json', dict(phase='running_priority', block=key, controller_pid=os.getpid(),
                completed=len(all_rows), total=100, block_completed=sum(r['arm'] in block['arms'] for r in rows),
                block_total=len(block['arms']), updated_unix=time.time()))

        engine.verify_stage = verify_ordered
        pipeline.analysis.write_progress = progress
        def interrupted(signum, frame):
            marker.write_text(f'Priority controller interrupted by signal {signum}.\n')
            raise RuntimeError(f'Priority signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        marker.unlink()
        try:
            engine.run_stage(block['stage'])
        finally:
            marker.write_text('User reprioritization: broad queue remains paused.\n')
        rows = read(root/block['stage']/'results.json')
        assert {r['arm'] for r in rows} - initial <= set(block['arms'])
        assert sha(request_path(key)) == digest
        selected = {r['arm']: r for r in rows if r['arm'] in block['arms']}
        full = set(selected) == set(block['arms'])
        state = read(root/block['stage']/'status.json')
        assert state['phase'] == 'stopped_after_current' and state['worker_exit_codes'] == [0]*4
        # engine.run_stage has already verified old commits and committed each new artifact hash.
        receipts = {arm: sha(root/block['stage']/arm/'commit.json') for arm in selected}
        atomic(ROOT/f'{key}_completed.json', dict(passed=True, selected_complete=full,
            committed_arms=len(selected), expected=len(block['arms']), commits=receipts,
            original_request_sha256=digest, worker_exit_codes=state['worker_exit_codes'],
            no_unscheduled_arm_sampled=True, original_queue_remains_paused=marker.exists()))
        assert full or (ROOT/'STOP_AFTER_CURRENT').exists(), 'Unexpected early stop.'
    finally:
        lock.close()


def controller():
    plan = verify_plan()
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for block in plan['blocks']:
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                break
            stream = (ROOT/f'{block["key"]}.log').open('a')
            try:
                command = [PYTHON, '-u', '-m', 'experiments.run_sit_refined_priority_20260911', '--run-block', block['key']]
                process = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                atomic(ROOT/'active_process.json', dict(controller_pid=os.getpid(), child_pid=process.pid,
                    block=block['key'], command=command, launched_unix=time.time()))
                code = process.wait()
                if code:
                    raise RuntimeError(f'Priority block {block["key"]} exited {code}; see its log.')
            finally:
                stream.close()
        rows = write_progress(plan)
        complete = len(rows) == plan['total_new_arms']
        atomic(ROOT/'status.json', dict(phase='complete' if complete else 'stopped_after_current',
            completed=len(rows), total=100, numerical_failures=sum(not r['complete'] for r in rows),
            no_further_sampling_queued=True, broad_queue_remains_paused=True,
            controller_pid=os.getpid(), finished_unix=time.time()))
    except BaseException as error:
        atomic(ROOT/'status.json', dict(phase='failed', error=repr(error), controller_pid=os.getpid()))
        raise
    finally:
        lock.close()


def check():
    requests = {key: read(request_path(key)) for key in SPECS}
    original = copy.deepcopy(requests)
    orders = select_orders(requests)
    counts = {}
    for key, order in orders.items():
        value = ordered_request(requests[key], order)
        assert [c['arm'] for c in value['configs'][:len(order)]] == order
        assert requests == original
        assert len(value['configs']) == len(requests[key]['configs'])
        # Simulate resume with arbitrary prior commits. The boundary cannot spill into the broad sweep.
        prior = [dict(arm=c['arm']) for c in requests[key]['configs'] if c['arm'] not in order][:7]
        rows = prior.copy()
        visited = []
        for config in value['configs']:
            if config['arm'] in {r['arm'] for r in rows}:
                continue
            if bounded_stop(order, rows):
                break
            visited.append(config['arm'])
            rows.append(dict(arm=config['arm']))
        assert visited == order and bounded_stop(order, rows)
        assert not bounded_stop(order, rows[:-1]) and bounded_stop(order, prior, True)
        counts[key] = len(visited)
    assert counts == dict(apg=76, control=24)
    candidate_ids = [c['idea_id'] for key, order in orders.items() for c in requests[key]['configs']
                     if c['arm'] in order and c['role'] == 'candidate']
    assert {idea: candidate_ids.count(idea) for idea in set(candidate_ids)} == {52:12, 53:12, 55:12, 56:12, 57:12}
    result = dict(passed=True, exact_request_preserved=True, full_coverage_preserved=True,
        bounded_stop_and_resume_passed=True, selected_counts=counts,
        candidate_arms=len(candidate_ids), new_control_arms=100-len(candidate_ids),
        source_sha256=sha(__file__), no_gpu=True)
    atomic(ROOT/'scheduler_check.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--check', action='store_true')
    action.add_argument('--pipeline', action='store_true')
    action.add_argument('--run-block', choices=SPECS)
    action.add_argument('--stop-after-current', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    elif args.prepare:
        plan = prepare()
        print(json.dumps(dict(prepared=True, total=100, schedule_sha256=sha(ROOT/'schedule.json'))), flush=True)
    elif args.pipeline:
        controller()
    elif args.run_block:
        run_block(args.run_block)
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'STOP_AFTER_CURRENT').touch()
        for spec in SPECS.values():
            (Path(spec['root'])/'STOP_AFTER_CURRENT').touch()


if __name__ == '__main__':
    main()
