"""Atomic live summaries of fixed-configuration, six-round feedback chains."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from experiments.lifting_scale_sweep_20260909 import WORK, atomic, sha

REPORT = WORK / 'docs/RECURSIVE_GUIDANCE_RESULTS_20260913_ZH.md'
PORTABLE = WORK / 'docs/data/recursive_guidance_20260913'


def text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def _drift_columns(row):
    """Keep missing/infinite PSNR distinct from a measured finite mean."""
    drift = row.get('drift') or {}
    columns, aggregate = drift.get('columns', {}), drift.get('aggregate_psnr', {})
    answer = {}
    for reference in ('previous', 'initial'):
        for metric in ('pixel_mse', 'latent_delta_rms', 'latent_cosine'):
            answer[f'{metric}_{reference}_mean'] = columns.get(f'{metric}_{reference}', {}).get('mean')
        psnr = columns.get(f'pixel_psnr_{reference}_db', {})
        answer[f'pixel_psnr_{reference}_mean_db'] = psnr.get('mean')
        answer[f'pixel_psnr_{reference}_positive_infinity_count'] = psnr.get('positive_infinity_count')
        summary = aggregate.get(reference, {})
        answer[f'pixel_psnr_{reference}_from_mean_mse_db'] = summary.get('psnr_from_mean_mse_db')
        answer[f'exact_pixel_identity_{reference}_count'] = summary.get('exact_pixel_identity_count')
        answer[f'all_images_identical_to_{reference}'] = summary.get('all_images_identical_to_reference')
    return answer


def write_progress(base, request, rows):
    """Publish every committed round, never splice a curve across settings."""
    base = Path(base)
    configs = request['configs']
    config_by_arm = {cfg['arm']: cfg for cfg in configs}
    rounds = int(request.get('rounds', 6))
    planned = len(configs)*rounds
    assert request.get('planned_config_rounds', planned) == planned
    assert len(config_by_arm) == len(configs) and rounds > 0
    chains = {cfg['arm']: {} for cfg in configs}
    for original in rows:
        arm, index = original['arm'], int(original.get('round_index', 0))
        assert arm in chains and 0 <= index < rounds, (arm, index)
        assert index not in chains[arm], f'Duplicate committed round: {arm}/{index}'
        row = dict(config_by_arm[arm], **original)
        row['round_index'] = index
        chains[arm][index] = row
    ordered_rows = [chains[cfg['arm']][index] for cfg in configs for index in sorted(chains[cfg['arm']])]
    valid = [row for row in ordered_rows if row.get('complete')]
    numerical = [row for row in ordered_rows if not row.get('complete') and row.get('status') == 'numerical_failure']
    blocked = [row for row in ordered_rows if not row.get('complete') and row.get('status') == 'dependency_blocked']
    other_failures = [row for row in ordered_rows if not row.get('complete')
                      and row.get('status') not in ('numerical_failure', 'dependency_blocked')]
    candidates = [cfg for cfg in configs if cfg['role'] == 'candidate']

    def full_chain(cfg):
        chain = chains[cfg['arm']]
        return len(chain) == rounds and all(row.get('complete') for row in chain.values())

    def representative(choices):
        complete = [cfg for cfg in choices if full_chain(cfg)]
        if complete:
            return min(complete, key=lambda cfg: (chains[cfg['arm']][rounds-1]['fid'], cfg['arm'])), '按末轮FID选完整链'
        started = [cfg for cfg in choices if chains[cfg['arm']]]
        return (started[0], '首条已启动链；尚无完整链') if started else (None, '尚未启动')

    fields = ['arm', 'round_index', 'idea_id', 'family', 'role', 'source', 'key', 'reference', 'solver',
              'strength', 'theta', 'parameters', 'samples', 'fid', 'sfid', 'inception_score',
              'full_calls_per_image', 'prefix_calls_per_image', 'auxiliary_full_calls_per_image',
              'sum_batch_gpu_seconds', 'trajectory_gpu_seconds', 'decode_gpu_seconds',
              'evaluation_wall_seconds', 'events_per_image', 'complete', 'status', 'blocked_by_round',
              'gallery_path', 'contact_sheet_path', 'noise_sha256', 'parent_latents_sha256', 'request_sha256']
    fields += list(_drift_columns({}))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in ordered_rows:
        value, metrics = dict(row), row.get('metrics') or {}
        value.update({key: metrics.get(key) for key in ('sfid', 'inception_score')})
        value.update(_drift_columns(row))
        value['status'] = row.get('status', 'complete' if row.get('complete') else 'failed')
        value['parameters'] = json.dumps(row.get('parameters', {}), sort_keys=True, ensure_ascii=False, allow_nan=False)
        writer.writerow(value)
    for path in (base / 'results.csv', PORTABLE / 'all_results.csv'):
        text_atomic(path, output.getvalue())

    completed_chains = sum(full_chain(cfg) for cfg in configs)
    committed_chains = sum(len(chain) == rounds for chain in chains.values())
    lines = ['# 递归再生成 CFG / IG：初次生成与五轮完整回灌', '',
             f'已提交 **{len(ordered_rows)}/{planned} 个配置轮次**；成功 {len(valid)}，'
             f'数值失败 {len(numerical)}，前轮失败导致未运行 {len(blocked)}，其他失败 {len(other_failures)}。', '',
             f'成功完成整条链 **{completed_chains}/{len(configs)}**；含失败状态已归档整条链 '
             f'{committed_chains}/{len(configs)}。计划 {len(candidates)} 个候选配置和 '
             f'{len(configs)-len(candidates)} 个对照，每个固定配置 {rounds} 轮、每轮 '
             f'{request["samples"]:,} 张，共 {planned*request["samples"]:,} 张计划输出。', '',
             f'R0 是初次完整生成；R1–R{rounds-1} 分别将本配置上一轮完整 latent 加噪到 '
             f't={request.get("renoise_start", .25):g} 后运行采样后缀，每轮分别保存图像、latent、'
             'FID/sFID/IS、漂移和图集。内部再生成探针不计作完整回灌轮。', '',
             '所有配置在相同轮次共享噪声和类别；六轮具有共同样本祖先，不能视为彼此独立样本。'
             'FID 排名来自 1K 调参筛选；漂移小不等于质量或多样性好。', '',
             '下面每行始终是同一个固定配置的完整曲线。摘要仅在已成功完成的完整链中按末轮 FID '
             '选择一个配置，随后展示该配置所有轮次；尚无完整链时展示请求顺序中首条已启动链。'
             '不拼接各轮不同参数的最优值。', '',
             '|轮次|成功|数值失败|前轮失败未运行|尚未提交|', '|---|--:|--:|--:|--:|']
    round_progress = {}
    for index in range(rounds):
        subset = [row for row in ordered_rows if row['round_index'] == index]
        summary = dict(committed=len(subset), successful=sum(bool(row.get('complete')) for row in subset),
            numerical_failures=sum(row.get('status') == 'numerical_failure' for row in subset),
            dependency_blocked=sum(row.get('status') == 'dependency_blocked' for row in subset), planned=len(configs))
        round_progress[str(index)] = summary
        lines.append(f'|R{index}|{summary["successful"]}|{summary["numerical_failures"]}|'
                     f'{summary["dependency_blocked"]}|{len(configs)-len(subset)}|')

    def cell(cfg, index):
        row = chains[cfg['arm']].get(index)
        if row is None:
            return '—'
        if row.get('complete'):
            return f'{row["fid"]:.4f}'
        if row.get('status') == 'dependency_blocked':
            return '前轮失败'
        return '数值失败' if row.get('status') == 'numerical_failure' else '失败'

    def image_link(cfg):
        complete = [row for row in chains[cfg['arm']].values() if row.get('complete')]
        if not complete:
            return '—'
        last = max(complete, key=lambda row: row['round_index'])
        path = last.get('contact_sheet_path') or last.get('gallery_path')
        return f'[逐轮图集](<{path}>)' if path else '—'

    round_header = '|'.join(f'FID R{index}' for index in range(rounds))
    table_header = f'|对象|固定配置|路线/参考|α / τ|成功轮数|{round_header}|选择说明|图集|'
    table_rule = '|---|---|---|---|--:|' + '--:|'*rounds + '---|---|'

    def table_row(title, cfg, selection):
        if cfg is None:
            return f'|{title}|—|—|—|0/{rounds}|' + '—|'*rounds + f'{selection}|—|'
        successes = sum(bool(row.get('complete')) for row in chains[cfg['arm']].values())
        values = '|'.join(cell(cfg, index) for index in range(rounds))
        return (f'|{title}|`{cfg["arm"]}`|{cfg["source"].upper()}/{cfg.get("reference", "native")}|'
                f'{cfg["strength"]:g} / {cfg.get("theta", 0):g}|{successes}/{rounds}|'
                f'{values}|{selection}|{image_link(cfg)}|')

    lines += ['', '## 十项候选的固定配置曲线', '', table_header, table_rule]
    ids = list(dict.fromkeys(cfg['idea_id'] for cfg in candidates))
    names = {item['id']: item.get('title', item.get('key', str(item['id']))) for item in request.get('ideas', [])}
    selected = []
    for idea in ids:
        choices = [cfg for cfg in candidates if cfg['idea_id'] == idea]
        cfg, selection = representative(choices)
        finished = sum(full_chain(item) for item in choices)
        title = f'{idea}. {names.get(idea, choices[0]["family"])}（整链 {finished}/{len(choices)}）'
        lines.append(table_row(title, cfg, selection))
        if cfg is not None:
            selected.append(cfg)

    lines += ['', '## 对照的固定配置曲线', '', table_header, table_rule]
    control_families = list(dict.fromkeys(cfg['family'] for cfg in configs if cfg['role'] != 'candidate'))
    for family in control_families:
        choices = [cfg for cfg in configs if cfg['family'] == family and cfg['role'] != 'candidate']
        cfg, selection = representative(choices)
        lines.append(table_row(family, cfg, selection))
        if cfg is not None:
            selected.append(cfg)

    lines += ['', '## 已展示配置的逐轮质量、成本与漂移', '',
              'Full/prefix 和辅助调用按该轮每张输出计；GPU 秒是采样与解码累计 batch 时间。'
              '像素 MSE 单位为 uint8 强度平方；PSNR 由该轮平均 MSE 计算，'
              '∞ 表示所有图像与参考逐像素相同，— 表示未测或缺失。'
              '它们衡量保真/漂移，不是独立质量评分。CSV 还保留逐图 PSNR 均值及无穷计数。', '',
              '|固定配置|轮次|sFID|IS|Full/prefix（辅助）|GPU秒|MSE前轮 / R0|PSNR前轮 / R0 dB|本轮图集|',
              '|---|---|--:|--:|---|--:|---|---|---|']

    def number(value, precision=3):
        return '—' if value is None else f'{value:.{precision}f}'

    def psnr(row, reference):
        item = (row.get('drift') or {}).get('aggregate_psnr', {}).get(reference, {})
        if item.get('all_images_identical_to_reference'):
            return '∞'
        return number(item.get('psnr_from_mean_mse_db'))

    for cfg in selected:
        for index, row in sorted(chains[cfg['arm']].items()):
            if not row.get('complete'):
                continue
            metrics, drift = row.get('metrics', {}), _drift_columns(row)
            path = row.get('gallery_path')
            link = f'[R{index}](<{path}>)' if path else '—'
            lines.append(f'|`{cfg["arm"]}`|R{index}|{number(metrics.get("sfid"))}|'
                f'{number(metrics.get("inception_score"))}|{number(row.get("full_calls_per_image"), 0)}/'
                f'{number(row.get("prefix_calls_per_image"), 0)} '
                f'（{number(row.get("auxiliary_full_calls_per_image"), 0)}）|'
                f'{number(row.get("sum_batch_gpu_seconds"))}|'
                f'{number(drift["pixel_mse_previous_mean"])} / {number(drift["pixel_mse_initial_mean"])}|'
                f'{psnr(row, "previous")} / {psnr(row, "initial")}|{link}|')

    lines += ['', '## 全部固定配置的六轮记录', '', table_header, table_rule]
    for cfg in configs:
        lines.append(table_row(cfg['family'], cfg, '固定配置；不跨轮重选'))
    failures = numerical+blocked+other_failures
    if failures:
        lines += ['', '## 失败与后继未运行', '', '|配置|轮次|状态|依赖的前轮|', '|---|---|---|---|']
        for row in sorted(failures, key=lambda row: (row['arm'], row['round_index'])):
            lines.append(f'|`{row["arm"]}`|R{row["round_index"]}|{row.get("status", "failed")}|'
                         f'{row.get("blocked_by_round", "—")}|')
    audit_path = base / 'analysis_audit.json'
    if audit_path.exists():
        lines += ['', f'[覆盖、成本与缓存指标核验记录](<{audit_path}>)。']
    request_hash = sha(base / 'request.json')
    lines += ['', f'请求 SHA256：`{request_hash}`。', '', f'原始输出：`{base}`。', '',
              '[每个配置每轮的完整 CSV](data/recursive_guidance_20260913/all_results.csv)。']
    document = '\n'.join(lines)+'\n'
    text_atomic(base / 'report.md', document)
    text_atomic(REPORT, document)
    atomic(PORTABLE / 'progress.json', dict(committed=len(ordered_rows), successful=len(valid),
        numerical_failures=len(numerical), dependency_blocked=len(blocked), other_failures=len(other_failures),
        successful_complete_chains=completed_chains, committed_complete_chains=committed_chains,
        configurations=len(configs), rounds=rounds, feedback_rounds=rounds-1, planned=planned,
        planned_images=planned*request['samples'], round_progress=round_progress, request_sha256=request_hash))
