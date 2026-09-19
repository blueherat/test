"""Short, separately resumable refinement queue for two recursive guidance ideas.

The original queue's worker, image retention, per-round ADM evaluation and
commit logic are reused in memory. No original request or source is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from experiments.recursive_guidance_20260913 import pipeline as shared
from experiments.recursive_guidance_focus_20260913 import catalog, core

ROOT = shared.EXPS / 'recursive_guidance_focus_20260913'
STAGE = 'focused_screen_1k'
MODULE = 'experiments.recursive_guidance_focus_20260913.pipeline'
OLD_BASE = shared.EXPS / 'recursive_guidance_20260913/recursive_screen_1k'
REPORT = shared.WORK / 'docs/RECURSIVE_GUIDANCE_FOCUS_RESULTS_20260913_ZH.md'
PORTABLE = shared.WORK / 'docs/data/recursive_guidance_focus_20260913'
SAMPLES, BATCH, RANKS, ROUNDS = shared.SAMPLES, shared.BATCH, shared.RANKS, shared.ROUNDS
atomic, read, sha, array_sha = shared.atomic, shared.read, shared.sha, shared.array_sha


def configurations():
    configs = json.loads(json.dumps(catalog.configurations(), allow_nan=False))
    required = {'arm', 'family', 'key', 'source', 'strength', 'theta', 'solver',
                'cutoff', 'role', 'idea_id', 'parameters', 'reference'}
    assert configs and len({item['arm'] for item in configs}) == len(configs)
    for item in configs:
        assert required <= item.keys(), item
        assert re.fullmatch(r'[A-Za-z0-9_\-]+', item['arm'])
        assert item['source'] in ('cfg', 'ig') and item['strength'] >= 0
    counts = Counter(item['idea_id'] for item in configs if item['role'] == 'candidate')
    assert len(counts) == 2, counts
    assert len(configs) <= 12, 'This runner is intentionally a short refinement queue'
    return configs


def text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write_progress(base, request, rows):
    """Every row is an actual committed configuration/round, never a reused row."""
    base = Path(base)
    by_id = {(row['arm'], row['round_index']): row for row in rows}
    assert len(by_id) == len(rows)
    fields = ['arm', 'round_index', 'family', 'idea_id', 'role', 'source', 'strength',
              'theta', 'parameters', 'complete', 'fid', 'sfid', 'inception_score',
              'full_calls_per_image', 'auxiliary_full_calls_per_image',
              'evaluation_wall_seconds', 'gallery_path', 'contact_sheet_path']
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        value = dict(row, **{k: row.get('metrics', {}).get(k) for k in ('sfid', 'inception_score')})
        value['parameters'] = json.dumps(row.get('parameters', {}), sort_keys=True)
        writer.writerow(value)
    text_atomic(base / 'results.csv', output.getvalue())
    text_atomic(PORTABLE / 'all_results.csv', output.getvalue())
    successful = sum(bool(row.get('complete')) for row in rows)
    lines = ['# 两项递归 guidance 改进：短队列实时结果', '',
        f'已提交 {len(rows)}/{request["planned_config_rounds"]} 个配置轮次；成功 {successful}。', '',
        f'本次 {len(request["configs"])} 个配置，每个 R0 初始生成及 R1–R5 五轮完整回灌；'
        f'每轮分别采样、保存 {SAMPLES} 张图片及 latent，并计算 FID/sFID/IS、漂移和图集。', '',
        '沿用旧筛选的噪声和类别 bank，属于同 bank 的参数细化，不能称为独立确认。'
        '下表每行固定同一参数，保持全部六轮，不拼接各轮最优值。', '',
        '|配置|角色|α / τ|参数|R0|R1|R2|R3|R4|R5|图集|',
        '|---|---|---|---|--:|--:|--:|--:|--:|--:|---|']
    for cfg in request['configs']:
        values, contact = [], None
        for index in range(ROUNDS):
            row = by_id.get((cfg['arm'], index))
            values.append('—' if row is None else f'{row["fid"]:.4f}' if row.get('complete') else row.get('status', '失败'))
            if row and row.get('complete'):
                contact = row.get('contact_sheet_path')
        link = f'[逐轮图集](<{contact}>)' if contact else '—'
        params = json.dumps(cfg['parameters'], ensure_ascii=False, sort_keys=True)
        lines.append(f'|`{cfg["arm"]}`|{cfg["role"]}|{cfg["strength"]:g} / {cfg["theta"]:g}|'
                     f'`{params}`|' + '|'.join(values) + f'|{link}|')
    lines += ['', '旧实验已完成的外部参考配置（引用原有结果，不计为本次运行或独立验证）：', '',
              '|旧配置|α|R0|R1|R2|R3|R4|R5|原始记录|',
              '|---|--:|--:|--:|--:|--:|--:|--:|---|']
    for entry in request.get('external_baselines', []):
        values = {int(row['round_index']): row for row in entry['rounds']}
        cells = ['—' if index not in values else f'{values[index]["fid"]:.4f}' for index in range(ROUNDS)]
        lines.append(f'|`{entry["arm"]}`|{entry["strength"]:g}|' + '|'.join(cells) +
                     f'|[原始目录](<{entry["directory"]}>)|')
    if (ROOT / 'baseline_report.json').exists():
        lines += ['', f'[筛选依据与对照摘要](<{ROOT / "baseline_report.json"}>)。']
    lines += ['', f'输出：`{base}`。', '', f'[每配置每轮完整 CSV](<{PORTABLE / "all_results.csv"}>)。', '']
    document = '\n'.join(lines)
    text_atomic(base / 'report.md', document)
    text_atomic(REPORT, document)
    atomic(PORTABLE / 'progress.json', dict(committed=len(rows), successful=successful,
           planned=request['planned_config_rounds'], configurations=len(request['configs']), rounds=ROUNDS))


def configure_shared():
    shared.ROOT, shared.STAGE, shared.MODULE = ROOT, STAGE, MODULE
    shared.core, shared.catalog = core, catalog
    shared.configurations = configurations
    shared.reporting = SimpleNamespace(write_progress=write_progress)


def sources(runtime_sources):
    paths = {Path(__file__), Path(core.__file__), Path(catalog.__file__),
             Path(shared.__file__), Path(shared.gallery.__file__),
             shared.WORK / 'experiments/compute_adm_fid.py', shared.WORK / 'train_gen/evaluator.py'}
    paths.update(Path(path) for path in core.source_paths())
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value and not Path(value).name.startswith('<'):
            path = Path(value).resolve()
            if path.is_relative_to(shared.WORK) and path.suffix == '.py' and path.is_file():
                paths.add(path)
    mapping = {str(path.resolve()): sha(path) for path in paths}
    # Runtime provenance can use symlink aliases. Preserve both its exact names
    # and resolved paths, since the reused worker checks exact runtime keys.
    for path, digest in runtime_sources.items():
        assert sha(path) == digest
        mapping[str(path)] = digest
        mapping[str(Path(path).resolve())] = digest
    return mapping


def cpu_check():
    configs = configurations()
    result = dict(passed=True, configurations=len(configs),
                  candidates=sum(item['role'] == 'candidate' for item in configs),
                  ideas=len({item['idea_id'] for item in configs if item['role'] == 'candidate'}),
                  rounds=ROUNDS, generated_config_rounds=len(configs)*ROUNDS, cuda_used=False)
    if hasattr(core, 'cpu_checks'):
        result['core'] = core.cpu_checks()
    print(json.dumps(result), flush=True)
    return result


@torch.inference_mode()
def development_check():
    cpu = cpu_check()
    core.install()
    rt = core.make_runtime()
    generator = torch.Generator(device='cuda').manual_seed(shared.SEED - 17)
    noise = torch.randn((2, 4, 32, 32), device='cuda', generator=generator)
    labels = torch.tensor([0, 99], device='cuda')
    rt.labels = labels
    original_head, original_hooks = rt.head, shared.hooks(rt)
    limits = core.limiting_checks(rt, noise, labels)
    representatives = {}
    for cfg in configurations():
        if cfg['role'] == 'candidate':
            representatives.setdefault(cfg['key'], cfg)
    trajectories = []
    for cfg in representatives.values():
        previous = None
        for index in range(ROUNDS):
            generator.manual_seed(shared.SEED - 17 + index * 100)
            fresh = torch.randn(noise.shape, device='cuda', generator=generator)
            start = fresh if previous is None else shared.RENOISE_START*previous+(1-shared.RENOISE_START)*fresh
            before = dict(rt.counts)
            value, stats = core.sample(rt, start, labels, cfg,
                start_step=int(shared.RENOISE_START*int(cfg['solver'][4:])) if index else 0,
                probe_seed=int(array_sha(fresh.cpu().numpy())[:15], 16))
            assert value.shape == noise.shape and torch.isfinite(value).all()
            assert rt.head is original_head and rt.labels is labels and shared.hooks(rt) == original_hooks
            shared.verify_stats(stats, before, rt)
            trajectories.append(dict(arm=cfg['arm'], round_index=index,
                full_calls=stats['full_calls'], auxiliary_full_calls=stats['auxiliary_full_calls'],
                accepted_steps=stats['accepted_steps'], max_abs=float(value.abs().max())))
            previous = value
        pixels = rt.decode(previous)
        assert pixels.shape == (2, 256, 256, 3) and pixels.dtype == np.uint8
        print(json.dumps(dict(smoke_complete=cfg['arm'], actual_rounds=ROUNDS)), flush=True)
    old = read(OLD_BASE / 'request.json')
    shared.verify_hashes(old['assets'])
    runtime_sources = dict(rt.sources)
    result = dict(passed=True, cpu=cpu, limiting_checks=limits, trajectories=trajectories,
        source_hashes=sources(runtime_sources), assets=old['assets'], runtime_sources=runtime_sources,
        versions=shared.versions(), configs_sha256=shared.canonical_hash(configurations()),
        rounds=ROUNDS, cuda_device=torch.cuda.get_device_name(), no_fid_used=True,
        scope='Two new mechanism representatives: real R0 plus all five feedback rounds; short native and zero-correction parity',
        created_unix=time.time())
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic(ROOT / 'development_check.json', result)
    print(json.dumps(dict(development_check_passed=True, mechanisms=len(representatives),
                         real_configuration_rounds=len(trajectories))), flush=True)


def external_baselines(old):
    answer = []
    request_hash = sha(OLD_BASE / 'request.json')
    for cfg in old['configs']:
        native_reference = cfg['role'] != 'candidate' and cfg['family'] in ('cfg_native', 'ig_mlp', 'ig_mlp_adg')
        original_mechanism = cfg['role'] == 'candidate' and (
            (cfg['key'] == 'cfg_cycle_gain' and cfg['strength'] == 1.25 and cfg['theta'] == .5) or
            (cfg['key'] == 'ig_cycle_extrapolate' and cfg['strength'] == .4 and cfg['theta'] == .65))
        if not (native_reference or original_mechanism):
            continue
        rows = []
        for index in range(ROUNDS):
            directory = OLD_BASE / cfg['arm'] / f'round{index:02d}'
            if not (directory / 'commit.json').exists():
                continue
            commit = read(directory / 'commit.json')
            assert commit['request_sha256'] == request_hash
            assert commit['files']['result.json'] == sha(directory / 'result.json')
            result = read(directory / 'result.json')
            if result.get('complete'):
                rows.append(dict(round_index=index, fid=result['fid'], metrics=result['metrics'],
                                 result_path=str(directory / 'result.json'),
                                 result_sha256=sha(directory / 'result.json')))
        if rows:
            answer.append(dict(arm=cfg['arm'], family=cfg['family'], strength=cfg['strength'],
                               directory=str(OLD_BASE / cfg['arm']), rounds=rows,
                               origin='existing_old_run', independent_confirmation=False))
    return answer


def prepare():
    configs = configurations()
    ROOT.mkdir(parents=True, exist_ok=True)
    base = ROOT / STAGE
    if (base / 'request.json').exists():
        request, _ = shared.verify_stage()
        assert request['configs'] == configs
        return request
    check = read(ROOT / 'development_check.json')
    assert check['passed'] and check['rounds'] == ROUNDS
    assert check['configs_sha256'] == shared.canonical_hash(configs)
    assert check['versions'] == shared.versions()
    shared.verify_hashes(check['source_hashes'])
    shared.verify_hashes(check['assets'])
    old = read(OLD_BASE / 'request.json')
    assert old['samples'] == SAMPLES and old['rounds'] == ROUNDS
    assert old['renoise_start'] == shared.RENOISE_START
    estimated = len(configs)*ROUNDS*SAMPLES*(256*256*3+4*32*32*8+32768)
    assert shutil.disk_usage(ROOT).free > estimated+(2<<30)
    staging = ROOT / ('.prepare_'+uuid.uuid4().hex[:10])
    bank = staging / 'inputs'
    bank.mkdir(parents=True)
    for name, digest in old['bank_files'].items():
        original = OLD_BASE / 'inputs' / name
        assert sha(original) == digest, original
        try:
            os.link(original, bank / name)
        except OSError:
            shutil.copyfile(original, bank / name)
        assert sha(bank / name) == digest
    request = dict(stage=STAGE, configs=configs,
        ideas=json.loads(json.dumps(catalog.IDEAS, allow_nan=False)), arms=[cfg['arm'] for cfg in configs],
        samples=SAMPLES, batch=BATCH, ranks=RANKS, rounds=ROUNDS, feedback_rounds=ROUNDS-1,
        renoise_start=shared.RENOISE_START, round_banks=old['round_banks'],
        planned_config_rounds=len(configs)*ROUNDS, round_order='configuration-major, R0 through R5',
        retention=old['retention'], sources=check['source_hashes'], source_archive={},
        archive_deferred_by_user=True, source_recovery='Original paths remain hash-verified; source copying deferred',
        assets=check['assets'], references={str(ROOT/'development_check.json'):sha(ROOT/'development_check.json')},
        reference=old['reference'], reference_sha256=old['reference_sha256'],
        inception_graph_sha256=old['inception_graph_sha256'], bank_root=str(base/'inputs'),
        bank_files=old['bank_files'], bank=old['bank'], parent_request_path=str(OLD_BASE/'request.json'),
        parent_request_sha256=sha(OLD_BASE/'request.json'), external_baselines=external_baselines(old),
        diagnostic_names=core.DIAGNOSTICS, python=shared.PYTHON, fid_python=shared.FID_PYTHON,
        versions=check['versions'], runtime_sources=check['runtime_sources'],
        independent_confirmation=False, same_bank_refinement=True, research_goal_achieved=False,
        estimated_output_bytes=estimated, prepared_unix=time.time())
    atomic(staging/'request.json', request)
    atomic(staging/'status.json', dict(phase='prepared', completed=0, total=len(configs)*ROUNDS))
    if base.exists():
        raise RuntimeError(f'Unexpected incomplete preparation directory: {base}')
    staging.replace(base)
    atomic(ROOT/'status.json', dict(phase='prepared', completed=0, total=len(configs)*ROUNDS))
    write_progress(base, request, [])
    print(json.dumps(dict(prepared=True, configurations=len(configs), rounds=ROUNDS, root=str(base))), flush=True)
    return request


def wait_for_resources(gpus):
    while True:
        if (ROOT/'STOP_AFTER_CURRENT').exists():
            return False
        occupied = shared.gpu_processes(gpus)
        if not occupied:
            return True
        atomic(ROOT/'status.json', dict(phase='waiting_for_gpu_release', occupied=occupied,
                                      controller_pid=os.getpid(), updated_unix=time.time()))
        time.sleep(5)


def pipeline(gpus):
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(signum, frame):
        raise RuntimeError(f'Focused controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        request = prepare()
        if not wait_for_resources(gpus):
            return
        shared.run_stage(gpus)
        status = read(ROOT/STAGE/'status.json')
        if status['phase'] == 'complete':
            status.update(candidate_arms=sum(cfg['role']=='candidate' for cfg in request['configs']),
                control_arms=sum(cfg['role']!='candidate' for cfg in request['configs']),
                independent_confirmation=False, archive_deferred_by_user=True, no_automatic_5k=True)
            atomic(ROOT/'status.json', status)
    except BaseException as error:
        previous = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json', dict(phase='failed', error=repr(error), previous=previous,
                                      controller_pid=os.getpid(), updated_unix=time.time()))
        raise
    finally:
        lock.close()


def main():
    configure_shared()
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('cpu-check', 'check', 'prepare', 'pipeline', 'status', 'stop-after-current'):
        action.add_argument('--'+name, action='store_true')
    action.add_argument('--worker', type=int, choices=range(RANKS))
    parser.add_argument('--stage', choices=[STAGE], default=STAGE)
    parser.add_argument('--run-id')
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    elif args.check:
        development_check()
    elif args.prepare:
        prepare()
    elif args.pipeline:
        gpus = [value.strip() for value in args.gpus.split(',')]
        assert len(gpus) == RANKS and len(set(gpus)) == RANKS and all(value.isdigit() for value in gpus)
        pipeline(gpus)
    elif args.status:
        print(json.dumps(read(ROOT/'status.json') if (ROOT/'status.json').exists()
                         else {'phase':'not_started'}, ensure_ascii=False, indent=2))
    elif args.stop_after_current:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'STOP_AFTER_CURRENT').touch()
    else:
        assert args.run_id and args.parent_pid
        shared.worker(args.worker, args.run_id, args.parent_pid)


if __name__ == '__main__':
    main()
