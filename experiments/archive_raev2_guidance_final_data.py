#!/usr/bin/env python3
"""Export an explicit small RAEv2 result set, without reading large assets.

The default export excludes the live paired-bridge screen. A later explicit
--include-completed-screen extends this same archive only after its fixed
screen_execution.json says complete. Source data and frozen protocols are
never modified. This is an archive integrity check, not a scientific re-audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
OUTPUT = ROOT / 'docs/data/raev2_guidance_final_20260906'
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
SCHEMA = 'raev2_guidance_explicit_light_archive_v1'
ASSET_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.safetensors', '.pdf',
                  '.png', '.jpg', '.jpeg', '.webp', '.bin', '.tar', '.zip'}
SHA_RE = re.compile(r'^[0-9a-fA-F]{64}$')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def read_small(path):
    """Refuse large content before opening; detect concurrent source changes."""
    before = path.stat()
    if before.st_size > MAX_FILE_BYTES:
        raise ValueError(f'file exceeds bounded archive limit: {path}')
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or len(data) != before.st_size:
        raise RuntimeError(f'source changed while reading: {path}')
    return data


def immutable_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'refusing inconsistent archive content: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)
    if digest(path.read_bytes()) != digest(data):
        raise RuntimeError(f'copy identity failed: {path}')


def replace_metadata(path, data):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def add(paths, base, names):
    paths.update(str(Path(base) / name) for name in names.split())


def fixed_selection():
    """Explicit study/arm whitelist: no quality-, seed-, or recursive search."""
    paths = set()
    a = 'affine_reflection_v1'
    add(paths, a, '''cost_match.json cost_selection_00.json cost_selection_01.json
        evaluator_identity.json independent_review.json official_evaluation.csv
        official_evaluation.json plan.json pre_result_review.json screen_execution.json''')
    for arm in ['official100', 'reflection100', 'official200', 'official201', 'parity16']:
        add(paths, f'{a}/{arm}', 'request.json sampling_input.json summary.json')
    add(paths, f'{a}/parity16', 'candidate_formula_check.json geometry_diagnostics.json parity_result.json')

    a = 'spatial_energy_balls_v1'
    add(paths, a, '''cost_match_before_fid.json cpu_review.json fid_jobs.json
        fid_jobs_execution.json fid_results.csv fid_results.json parity_acceptance.json
        parity_jobs.json parity_jobs_execution.json plan.json result_summary.json
        sample_jobs.json sample_jobs_execution.json sample_launch_gpu.csv''')
    for arm in ['official100', 'global100', 'spatial100', 'parity16']:
        add(paths, f'{a}/{arm}', 'request.json sampling_input.json summary.json')
    add(paths, f'{a}/parity16', 'parity_result.json')
    add(paths, f'{a}/diagnostic_review_v1', 'manifest.json review.json')

    a = 'observable_potential_scale_audit_v1'
    add(paths, a, '''request.json execution_started.json execution_resumed.json
        execution_summary.json failure.json''')
    for arm in ['official100', 'potential100', 'official105']:
        add(paths, f'{a}/{arm}', 'outer_sampling.json outer_workers.json')
        merge_name = 'merge_resume.process.json' if arm == 'potential100' else 'merge.process.json'
        add(paths, f'{a}/{arm}', merge_name)
        add(paths, f'{a}/{arm}/merged', 'request.json summary.json')
        for shard in range(4):
            add(paths, f'{a}/{arm}/shard{shard}', 'request.json summary.json')
    add(paths, f'{a}/evaluation', 'evaluation.process.json metrics.csv metrics.json request.json')
    a = 'observable_potential_scale_audit_analysis_v1'
    add(paths, a, 'README_ZH.md RESULTS_ZH.md review.json toy_validation.json results_v1.process.json')
    add(paths, f'{a}/results_v1', 'request.json summary.json')

    a = 'endpoint_adjoint_response_v1'
    add(paths, a, '''driver_identity.json execution_summary.json frozen_control.json
        request.json response_summary.json rounding_diagnostic.json''')
    add(paths, f'{a}/review', '''checks.json document_identity.json manifest.json
        per_image.csv per_image_per_step.csv results.json rounding_evidence.json
        sha256_evidence.json source_evidence.json timing_evidence.json''')

    a = 'query_mean_error_compatibility_v1'
    add(paths, a, '''bridge_identity_prepare.json frozen_protocol.md paired_clean_identity.json
        per_image_risk.csv per_time_risk.csv prepare_cost.json request.json summary.json v2_parity.csv''')
    add(paths, 'query_mean_error_compatibility_review_v1', '''reconstructed_per_image.csv
        reconstructed_per_time.json request.json review.json source_and_output_hash_checks.json''')

    a = 'paired_bridge_v1'
    # Preserve the stale original parent record alongside the successful recovery;
    # it must not be silently rewritten into a completed training-driver record.
    add(paths, a, '''pipeline_created.json pipeline_status.json pipeline_resume_created.json
        pipeline_resume_status.json supplemental_source_environment.json''')
    for stage in ['pilot', 'train', 'validate', 'rollout']:
        add(paths, f'{a}/{stage}', 'request.json summary.json')
    add(paths, f'{a}/pilot', 'updates.csv')
    add(paths, f'{a}/train', 'training.csv')
    add(paths, f'{a}/validate', 'finite_map_moments.csv')
    add(paths, f'{a}/rollout', 'diagnostic_costs.csv')
    add(paths, f'{a}/analysis_v1', '''README_ZH.md summary.json finite_map_moments_recomputed.csv
        rollout_moment_descriptions.csv teacher_per_class.csv teacher_per_time.csv''')
    add(paths, f'{a}/analysis_velocity_units_v1', 'summary.json')
    return paths


def require_complete(source, relative):
    value = json.loads(read_small(source / relative))
    if value.get('complete') is not True:
        raise RuntimeError(f'not marked complete; refusing to export as completed: {relative}')
    return value


def validate_fixed_completion(source):
    gates = [
        'affine_reflection_v1/screen_execution.json',
        'spatial_energy_balls_v1/result_summary.json',
        'observable_potential_scale_audit_v1/execution_summary.json',
        'observable_potential_scale_audit_analysis_v1/results_v1/summary.json',
        'endpoint_adjoint_response_v1/execution_summary.json',
        'query_mean_error_compatibility_v1/summary.json',
        'query_mean_error_compatibility_review_v1/review.json',
        'paired_bridge_v1/pilot/summary.json',
        'paired_bridge_v1/train/summary.json',
        'paired_bridge_v1/validate/summary.json',
        'paired_bridge_v1/rollout/summary.json',
        'paired_bridge_v1/analysis_v1/summary.json',
        'paired_bridge_v1/analysis_velocity_units_v1/summary.json',
    ]
    for relative in gates:
        require_complete(source, relative)
    return gates


def completed_screen_selection(source):
    base = 'paired_bridge_v1/screen_v1'
    state = require_complete(source, f'{base}/screen_execution.json')
    if state.get('protocol') != 'raev2_paired_bridge_fixed_1k_v1':
        raise ValueError('screen protocol is not the frozen fixed first 1K')
    if not state.get('fid_started') or not state.get('sampling_results'):
        raise ValueError('completed screen lacks fixed sampling/evaluation records')
    paths = set()
    add(paths, base, '''plan.json screen_execution.json cost_match.json
        official_evaluation.csv official_evaluation.json''')
    add(paths, f'{base}/parity', 'request.json summary.json parity_result.json finite_response.json')
    for arm in state['sampling_results']:
        if not re.fullmatch(r'(official[0-9]+|candidate100|control100)', arm):
            raise ValueError(f'unexpected screen arm: {arm}')
        require_complete(source, f'{base}/{arm}/summary.json')
        add(paths, f'{base}/{arm}', 'request.json sampling_input.json summary.json')
    # Include every earlier cost-only official arm; never choose only the best FID.
    for record in state.get('cost_selections', []):
        recorded = Path(record['path'])
        relative = recorded.resolve().relative_to((source / base).resolve())
        if len(relative.parts) != 1 or not re.fullmatch(r'cost_selection_[0-9]+\.json', relative.name):
            raise ValueError(f'unexpected cost-selection record: {recorded}')
        add(paths, base, relative.name)
    return paths


def digest_records(value, provenance, pointer=''):
    """Reuse recorded identities only. Never open the referenced asset."""
    records = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = pointer + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if isinstance(child, str) and SHA_RE.fullmatch(child):
                path_key = next((k for k in ('path', 'file', 'source', 'source_path', 'absolute_path')
                                 if isinstance(value.get(k), str)), None)
                raw_path = value[path_key] if path_key else None
                if raw_path is None and ('/' in key or Path(key).suffix in ASSET_SUFFIXES):
                    raw_path = key
                if raw_path is None and str(key).endswith('_sha256'):
                    stem = str(key)[:-7]
                    raw_path = next((value[k] for k in (stem + '_path', stem)
                                     if isinstance(value.get(k), str) and '/' in value[k]), None)
                if raw_path is not None:
                    candidate = Path(raw_path)
                    # Relative path semantics stay with the original record; do not
                    # guess a base that could point at a different model or tensor.
                    resolved = str(candidate.resolve()) if candidate.is_absolute() else None
                    suffix = candidate.suffix.lower()
                    size = next((value[k] for k in ('size_bytes', 'bytes', 'size')
                                 if isinstance(value.get(k), int)), None)
                    records.append({
                        'provenance_package_relative_path': provenance,
                        'json_pointer_to_digest': child_pointer,
                        'recorded_logical_path': raw_path,
                        'resolved_path_without_content_read': resolved,
                        'recorded_sha256': child.lower(),
                        'recorded_size_bytes': size,
                        'large_or_binary_asset_by_suffix_or_recorded_size': (
                            suffix in ASSET_SUFFIXES or (size is not None and size > MAX_FILE_BYTES)),
                        'verification_this_export': 'recorded_identity_only_not_reread',
                        'digest_scope': 'as specified by original metadata; not reinterpreted as file bytes',
                    })
            records.extend(digest_records(child, provenance, child_pointer))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            records.extend(digest_records(child, provenance, pointer + '/' + str(i)))
    return records


def readme(file_count, total_bytes, screen_included):
    screen = '已显式加入已完成的固定第一轮 1K screen。' if screen_included else '当前排除 paired-bridge screen_v1；它在首版导出时仍未完成。'
    return f'''# RAEv2 guidance 最终轻量数据包

本包保存预先列明的已完成实验的小型原始记录，不重新选择方法、seed 或 FID 最好的 arm。当前包含 {file_count} 个逐字节复制文件，共 {total_bytes:,} 字节。{screen}

## 从哪里看

- [研究导航](../../RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md) 串起理论、实现、结果与大资产原址。
- [manifest.json](manifest.json) 逐文件列出源逻辑路径、解析后真实路径、字节数、SHA-256、包内相对路径。每份小文件复制后都重新检查包内 SHA。
- [recorded_asset_identities.json](recorded_asset_identities.json) 从已复制 JSON 提取已有路径/摘要记录。它们是**记录身份，未本轮重读**；某些摘要可能针对 raw tensor bytes，必须回到原始记录判断，不能当作本轮完整文件核验。
- `files/` 按原 RAEv2 实验根目录保留结构，可直接比较 request、summary、FID、成本与逐时风险。

## 已纳入的固定实验

1. affine reflection：official100、reflection100、cost-only official200、最终 official201、parity、成本选择、统一 FID、独立审核。
2. spatial energy balls：official/global/spatial100、parity、成本冻结、统一 FID 与逐样本审核摘要。
3. observable potential 5K：全部三个 arm 和四 shard、合并/恢复元数据、统一评价与独立分析。
4. endpoint adjoint：执行、冻结控制、响应、量化诊断和独立 review；没有 FID。
5. query-mean error compatibility：逐图/逐时风险、固定输入身份、独立重建；没有 FID。
6. paired bridge：pilot/train/validate/rollout、独立归一化与实际速度单位汇总、原父进程状态和后续恢复链；不把负回归结果或矩改善称为质量收益。

## 明确保留的缺口

- 不复制模型、完整样本、latent/feature bank、PDF、图像、batch 文件、大日志或大于 1 MB 的单文件。paired-bridge `teacher_regression.csv`（约 3.74 MB）与详细 draw ledger 留在原址；小型逐时/逐类汇总已纳入。
- 本轮只 hash 被复制的小文件与归档脚本，不扫描或 hash GB 级资产，不运行 GPU/模型/FID。大资产清单是已有元数据索引，不能冒充完整备份或新的数据真实性审计。
- paired-bridge 原训练父进程消失造成外层精确 wall time 缺口；原 `pipeline_status.json` 保留其历史不完整状态，后继 `pipeline_resume_status.json` 只证明后续验证/rollout 恢复。训练没有重复。
- observable-potential 的 `failure.json` 是此前中断记录；完成状态看后继 `execution_summary.json`，不删除失败历史。历史准备/数据选取成本未全部闭合，当前推理匹配不等于总成本达标。
- 本包不重导入已有 9 月 5 日 Git 数据包，也未覆盖全部早期 RAE/SiT/AdvFD/DiT 大资产。当前任务的 ≥5% 相对 FID 与独立确认目标仍未因此达成。

## 后续显式增量

```bash
python experiments/archive_raev2_guidance_final_data.py --include-completed-screen
```

该开关只允许同一固定 `paired_bridge_v1/screen_v1`，要求 `screen_execution.json` 已 complete，并保留全部 cost-only 中间 arm。没有新 seed 或扫描选择。已有复制文件必须字节相同，否则拒绝覆盖。聚合 manifest 的前版写入 `manifest_history/` 后才追加，新版记录父摘要；这不改冻结源文件。默认再次执行只验证已有包，不移除已加入的 screen。
'''.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--include-completed-screen', action='store_true')
    args = parser.parse_args()
    started, cpu_started = time.perf_counter(), time.process_time()
    source, output = args.source_root.expanduser().absolute(), args.output_dir.expanduser().absolute()
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('output must be outside the source experiment tree')
    prior_path = output / 'manifest.json'
    prior_bytes = prior_path.read_bytes() if prior_path.exists() else None
    prior = json.loads(prior_bytes) if prior_bytes else None
    if prior and (prior.get('schema') != SCHEMA or prior['source_root_logical'] != str(source)):
        raise ValueError('archive schema or source root mismatch')
    script_bytes = Path(__file__).read_bytes()
    if prior and prior['export_script']['sha256'] != digest(script_bytes):
        raise ValueError('export script changed; explicit archive migration required')
    gates = validate_fixed_completion(source)
    selection = fixed_selection()
    screen_included = args.include_completed_screen or bool(prior and prior['completed_screen_included'])
    if screen_included:
        selection.update(completed_screen_selection(source))
    payloads = {}
    records = []
    for relative in sorted(selection):
        path = source / relative
        data = read_small(path)
        package_relative = 'files/' + relative
        payloads[package_relative] = data
        records.append({'source_logical_path': str(path), 'source_resolved_path': str(path.resolve()),
                        'size_bytes': len(data), 'sha256': digest(data),
                        'package_relative_path': package_relative,
                        'verification_this_export': 'source_small_file_read_and_copied_bytes_verified'})
    total = sum(map(len, payloads.values()))
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f'explicit export exceeds {MAX_TOTAL_BYTES} byte budget: {total}')
    if prior:
        now = {r['package_relative_path']: r for r in records}
        for old in prior['copied_files']:
            if now.get(old['package_relative_path']) != old:
                raise ValueError('prior source identity changed: ' + old['package_relative_path'])
        for key, filename in [('readme_sha256', 'README.md'),
                              ('recorded_asset_identities_sha256', 'recorded_asset_identities.json')]:
            if digest((output / filename).read_bytes()) != prior[key]:
                raise ValueError('existing generated metadata changed: ' + filename)
    output.mkdir(parents=True, exist_ok=True)
    for relative, data in payloads.items():
        immutable_write(output / relative, data)
    identities = []
    for relative, data in payloads.items():
        if relative.endswith('.json'):
            identities.extend(digest_records(json.loads(data), relative))
    identity_bytes = encode({'schema': SCHEMA, 'verification': 'recorded_identity_only_not_reread',
        'large_assets_opened_or_hashed_this_export': 0, 'records': identities})
    readme_bytes = readme(len(records), total, screen_included)
    if prior and prior['copied_files'] == records and prior['completed_screen_included'] == screen_included:
        if digest(identity_bytes) != prior['recorded_asset_identities_sha256'] or digest(readme_bytes) != prior['readme_sha256']:
            raise ValueError('deterministic metadata differs from prior export')
        print(json.dumps({'status': 'existing_archive_verified_no_changes', 'files': len(records),
                          'copied_bytes': total, 'manifest_sha256': digest(prior_bytes)}))
        return
    revision = 1 if prior is None else prior['revision'] + 1
    if prior_bytes:
        immutable_write(output / f'manifest_history/manifest_{prior["revision"]:04d}.json', prior_bytes)
    metadata_writer = replace_metadata if prior else immutable_write
    metadata_writer(output / 'recorded_asset_identities.json', identity_bytes)
    metadata_writer(output / 'README.md', readme_bytes)
    manifest = {
        'schema': SCHEMA, 'revision': revision, 'created_utc': stamp(),
        'previous_manifest_sha256': digest(prior_bytes) if prior_bytes else None,
        'source_root_logical': str(source), 'source_root_resolved': str(source.resolve()),
        'output_root_logical': str(output), 'output_root_resolved': str(output.resolve()),
        'export_script': {'logical_path': str(Path(__file__).absolute()),
                          'resolved_path': str(Path(__file__).resolve()),
                          'sha256': digest(script_bytes), 'size_bytes': len(script_bytes)},
        'completion_metadata_gates': gates,
        'completed_screen_included': screen_included,
        'scope': 'explicit fixed studies; small copied byte identities, not full scientific or large-asset re-audit',
        'max_single_copied_file_bytes': MAX_FILE_BYTES, 'max_total_copied_bytes': MAX_TOTAL_BYTES,
        'copied_file_count': len(records), 'copied_total_bytes': total, 'copied_files': records,
        'recorded_asset_identity_count': len(identities),
        'recorded_asset_identities_sha256': digest(identity_bytes), 'readme_sha256': digest(readme_bytes),
        'large_assets_opened_or_hashed': 0, 'gpu_calls': 0, 'model_calls': 0,
        'historical_data_cost_gaps_closed_by_export': False, 'research_goal_complete': False,
        'wall_seconds_before_manifest_write': time.perf_counter() - started,
        'cpu_seconds_before_manifest_write': time.process_time() - cpu_started,
    }
    data = encode(manifest)
    metadata_writer(prior_path, data)
    print(json.dumps({'status': 'export_complete', 'revision': revision, 'files': len(records),
                      'copied_bytes': total, 'recorded_identity_entries': len(identities),
                      'manifest_sha256': digest(data), 'completed_screen_included': screen_included}))


if __name__ == '__main__':
    main()
