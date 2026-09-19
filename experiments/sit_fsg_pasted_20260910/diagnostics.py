"""Paired neural probes of rebound, path residuals and counterfactual bad roots.

Every measurement uses a fixed terminal time. Category predictions are alignment
proxies; this diagnostic does not compute individual-image FID or prove causality.
"""
from __future__ import annotations
import csv
import io
import time
import numpy as np
import torch
from experiments.sit_fsg_pasted_20260910 import core, readout
from experiments.sit_fsg_pasted_20260910 import pipeline as p
from experiments.lifting_scale_sweep_20260909 import array_sha, atomic, read, sha

TIME_STEPS = (8, 24, 40)
VARIANTS = ('unchanged', 'root1', 'root4', 'tube1', 'path1')
REPORT = p.WORK/'docs/SIT_FSG_PASTED_DIAGNOSTICS_20260910_ZH.md'


def identity():
    return dict(samples=64, batch=8, seed=202610088, label_seed=202610089,
                time_steps=list(TIME_STEPS), variants=list(VARIANTS), base='conditional Heun64',
                optimized_future_steps=16, validation_step=1/64,
                source_hashes={str(path): sha(path) for path in p.sources()},
                assets={str(path): sha(path) for path in core.source_assets()})


def rms(z):
    return z.flatten(1).square().mean(1).sqrt()


@torch.inference_mode()
def run_rank(rank):
    directory = p.ROOT/'diagnostics'; directory.mkdir(parents=True, exist_ok=True)
    p.verify_parent()
    check = read(p.ROOT/'development_check.json'); assert check['passed']
    for path, digest in {**check['source_hashes'], **check['assets']}.items(): assert sha(path) == digest, path
    ident = identity(); rt = p.make_runtime(); independent = readout.IndependentReadout()
    generator = torch.Generator(device='cuda').manual_seed(202610088)
    # Continuous B8 generation matches across ranks and independent reruns.
    bank = torch.cat([torch.randn((8, 4, 32, 32), device='cuda', generator=generator) for _ in range(8)])
    labels = torch.from_numpy(np.random.default_rng(202610089).permutation(100)[:64].astype(np.int64)).cuda()
    ident['noise_sha256'] = array_sha(bank.cpu().numpy()); ident['label_sha256'] = array_sha(labels.cpu().numpy())
    rows = []; begin = time.perf_counter()
    for start in range(rank*8, 64, 32):
        batch_path = directory/f'batch{start:02d}.json'
        if batch_path.exists():
            old = read(batch_path)
            assert old['identity'] == ident
            rows.extend(old['rows']); continue
        n, y = bank[start:start+8], labels[start:start+8]; rt.labels = y
        baseline = n.clone(); states = {}
        for step in range(64):
            if step in TIME_STEPS: states[step] = baseline.clone()
            t, h = step/64, 1/64
            with core.old.exact_matmul():
                first = rt.field(baseline, baseline.new_tensor(t), 'full')
                second = rt.field(baseline+h*first, baseline.new_tensor(t+h), 'full')
            baseline = baseline+(h/2)*(first+second)
        local = []
        for step in TIME_STEPS:
            original = states[step]; t = step/64
            event = core.event_context(n, step)
            reference = core.FutureProbe(rt, original, t, event, steps=64-step)
            ref_c, ref_u = reference.pair_paths(original)
            for variant in VARIANTS:
                state = original.clone(); records = []
                key = {'root1': 'full_root', 'root4': 'full_root',
                       'tube1': 'golden_tube', 'path1': 'full_path'}.get(variant)
                config = dict(arm=variant, key=key, theta=1., strength=2.25)
                for _ in range(4 if variant == 'root4' else int(key is not None)):
                    state, rec = core.calibrate(rt, state, t, config, event, 2.25)
                    records.append(rec)
                coarse = core.FutureProbe(rt, state, t, event)
                coarse_c, coarse_u = coarse.pair_paths(state)
                fine = core.FutureProbe(rt, state, t, event, steps=64-step)
                c, u = fine.pair_paths(state)
                terminal = rms(c[-1]-u[-1])
                path_values = torch.stack([rms(a-b).square() for a, b in zip(c, u)])
                weights = torch.ones(len(path_values), device='cuda'); weights[0] = weights[-1] = .5
                path_energy = (path_values*weights[:, None]).sum(0)/fine.steps
                # Restart C and U on precisely the same remaining fine grid.
                reb = []
                suffix_error = []
                for index in (1, fine.steps//4):
                    next_c = fine.rollout(u[index], 'c', start=index)[-1]
                    next_u = fine.rollout(u[index], 'u', start=index)[-1]
                    assert torch.equal(next_u, u[-1]), 'Matched discrete U semigroup changed'
                    reb.append(rms(next_c-next_u))
                    suffix_error.append(rms(next_u-u[-1]))
                rival = fine.rollout(state, fine.rival())[-1]
                negative = rms(u[-1]-rival)
                aligned_u = independent.latents(rt, u[-1], y)
                aligned_c = independent.latents(rt, c[-1], y)
                aligned_n = independent.latents(rt, rival, y)
                tensors = torch.stack([terminal, rms(coarse_c[-1]-coarse_u[-1]), path_energy,
                    reb[0], reb[1], suffix_error[0], negative, negative.square()-terminal.square(),
                    rms(state-original), rms(u[-1]-ref_u[-1]), rms(c[-1]-ref_c[-1])], 1).cpu().numpy()
                names = ('terminal_rms', 'optimized_grid_terminal_rms', 'path_mean_squared_rms',
                    'suffix_one_step_rms', 'suffix_quarter_rms', 'null_semigroup_error',
                    'negative_rms', 'squared_margin', 'state_shift_rms',
                    'null_terminal_change_rms', 'conditional_terminal_change_rms')
                for index in range(8):
                    item = dict(seed_index=start+index, label=int(y[index]), t=t, variant=variant,
                                **dict(zip(names, map(float, tensors[index]))))
                    item.update(null_target_probability=float(aligned_u[index, 0]),
                                null_target_success=bool(aligned_u[index, 1]),
                                conditional_target_probability=float(aligned_c[index, 0]),
                                conditional_target_success=bool(aligned_c[index, 1]),
                                rival_target_probability=float(aligned_n[index, 0]))
                    local.append(item)
                print(dict(rank=rank, batch=start, t=t, variant=variant,
                           terminal_rms=float(terminal.mean()),
                           suffix_one_step_rms=float(reb[0].mean()),
                           null_top1=float(aligned_u[:, 1].mean())), flush=True)
        atomic(batch_path, dict(identity=ident, rows=local, baseline_terminal_sha256=array_sha(baseline.cpu().numpy())))
        rows.extend(local)
    atomic(directory/f'rank{rank}.json', dict(passed=True, rank=rank, identity=ident, rows=rows,
                                            elapsed_seconds=time.perf_counter()-begin))


def summarize():
    directory = p.ROOT/'diagnostics'; records = [read(directory/f'rank{rank}.json') for rank in range(4)]
    assert all(record['passed'] and record['identity'] == records[0]['identity'] for record in records)
    expected = identity()
    for key, value in expected.items(): assert records[0]['identity'][key] == value, key
    rows = [row for record in records for row in record['rows']]
    assert len(rows) == 64*3*5
    assert len({(row['seed_index'], row['t'], row['variant']) for row in rows}) == len(rows)
    assert max(row['null_semigroup_error'] for row in rows) == 0.
    fields = list(rows[0]); output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows); (directory/'rows.csv').write_text(output.getvalue())
    portable = p.WORK/'docs/data/sit_fsg_pasted_followup_20260910'; portable.mkdir(parents=True, exist_ok=True)
    (portable/'diagnostics.csv').write_text(output.getvalue())
    aggregate = []
    for t in [step/64 for step in TIME_STEPS]:
        for variant in VARIANTS:
            subset = [row for row in rows if row['t'] == t and row['variant'] == variant]
            mean = lambda key: float(np.mean([row[key] for row in subset]))
            terminal = np.array([row['terminal_rms'] for row in subset])
            suffix = np.array([row['suffix_one_step_rms'] for row in subset])
            low = terminal <= np.quantile(terminal, .25)
            success = np.array([row['null_target_success'] for row in subset])
            positive = np.array([row['squared_margin'] > 0 for row in subset])
            aggregate.append(dict(t=t, variant=variant, n=len(subset),
                terminal_rms=mean('terminal_rms'), coarse_terminal_rms=mean('optimized_grid_terminal_rms'),
                path_energy=mean('path_mean_squared_rms'), suffix_one_rms=mean('suffix_one_step_rms'),
                suffix_quarter_rms=mean('suffix_quarter_rms'),
                rebound_fraction=float(np.mean(suffix > terminal*1.1+1e-8)),
                null_top1=float(success.mean()), low_residual_errors=int((low & ~success).sum()),
                low_residual_n=int(low.sum()), positive_margin_errors=int((positive & ~success).sum()),
                positive_margin_n=int(positive.sum())))
    # Within the same seed, label and t, compare variants with terminal RMS
    # within 5%. This small, selected set is a descriptive diagnostic only.
    pairs = []
    for index in range(64):
        for t in [step/64 for step in TIME_STEPS]:
            group = [row for row in rows if row['seed_index'] == index and row['t'] == t]
            for i, left in enumerate(group):
                for right in group[i+1:]:
                    relative = abs(left['terminal_rms']-right['terminal_rms'])/max(1e-8, left['terminal_rms'], right['terminal_rms'])
                    if relative > .05: continue
                    small, large = sorted((left, right), key=lambda row: row['path_mean_squared_rms'])
                    if abs(large['path_mean_squared_rms']-small['path_mean_squared_rms']) < 1e-10: continue
                    pairs.append(dict(seed_index=index, t=t, lower_path_variant=small['variant'],
                        higher_path_variant=large['variant'], relative_terminal_gap=relative,
                        target_probability_delta=small['null_target_probability']-large['null_target_probability'],
                        target_success_delta=int(small['null_target_success'])-int(large['null_target_success'])))
    matched = dict(count=len(pairs), mean_target_probability_delta=(float(np.mean([row['target_probability_delta'] for row in pairs])) if pairs else None),
                   mean_target_success_delta=(float(np.mean([row['target_success_delta'] for row in pairs])) if pairs else None),
                   interpretation='Paired descriptive alignment association; dependent pairs, no significance or quality/causality claim.')
    result = dict(passed=True, identity=records[0]['identity'], rows=len(rows),
                  null_semigroup_exact=True, aggregate=aggregate, terminal_matched_pairs=matched,
                  rank_record_sha256={str(directory/f'rank{rank}.json'): sha(directory/f'rank{rank}.json') for rank in range(4)})
    atomic(directory/'summary.json', result); atomic(directory/'terminal_matched_pairs.json', pairs)
    lines = ['# FSG 完整未来：反弹、路径与竞争条件诊断\n',
             '64个固定噪声、三个时刻、五种成对干预，共960个状态。优化未来使用16步Heun；验证使用原始1/64网格续生成至同一终点。类别对齐由未用于优化的ResNet18读取。\n',
             '**无条件后缀复合在全部状态上逐元素等于原无条件终点。** 表中的条件残差反弹不能解释成这一无条件终点遗忘了语义。\n',
             '|t|干预|验证终点RMS|优化网格RMS|下一步RMS|1/4后缀RMS|路径均方|反弹>10%|null top1|低残差错误/数|正间隔错误/数|',
             '|--:|---|--:|--:|--:|--:|--:|--:|--:|---|---|']
    for row in aggregate:
        lines.append(f'|{row["t"]}|{row["variant"]}|{row["terminal_rms"]:.5f}|{row["coarse_terminal_rms"]:.5f}|{row["suffix_one_rms"]:.5f}|{row["suffix_quarter_rms"]:.5f}|{row["path_energy"]:.5f}|{row["rebound_fraction"]:.1%}|{row["null_top1"]:.1%}|{row["low_residual_errors"]}/{row["low_residual_n"]}|{row["positive_margin_errors"]}/{row["positive_margin_n"]}|')
    lines += ['\n低残差定义为各t、干预组内部最低四分位，不等于精确黄金根。正间隔为竞争终点距离平方减目标距离平方大于0；不把它当作目标正确的证明。类别“错误”仅指该独立分类器判断。\n',
              f'在相同噪声、类别和t下，筛选终点RMS相差不超过5%的干预对，得到 {matched["count"]} 对。较低路径残差一侧的平均目标概率差为 {matched["mean_target_probability_delta"]}，top1差为 {matched["mean_target_success_delta"]}。这些对相互依赖、样本经过筛选，只作描述性关联，不报告显著性、感知质量或因果结论。\n',
              'root1/root4分别作1/4次完整终点根校准；tube1附加两个无条件后缀的条件终点约束；path1附加三个中间时距的一致性。每次写入均验证优化目标未上升，但更细网格的残差可不降。所有负结果保留。\n',
              '[逐状态CSV](data/sit_fsg_pasted_followup_20260910/diagnostics.csv) · [理论审查](FSG_PASTED_ANALYSIS_REVIEW_20260910_ZH.md) · [生成质量筛选](SIT_FSG_PASTED_RESULTS_20260910_ZH.md)。\n']
    REPORT.write_text('\n'.join(lines))
    print(dict(diagnostics_complete=True, states=len(rows), matched_pairs=matched), flush=True)
