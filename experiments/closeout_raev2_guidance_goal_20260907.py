"""Assemble the final fixed-protocol negative result only after all quality jobs finish.

This script cannot declare success, launch sampling, or change goal status. A
positive quality threshold stops it for the separately required cost review.
"""
import json
from pathlib import Path
import time
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


ORDER = [
    'official', 'piecewise', 'ancestral', 'partial', 'calibrated',
    'velocity_projection', 'noise_projection', 'stochastic_weak', 'mean_weak',
    'critic_isotropic', 'critic_exchangeable', 'two_mode', 'semantic_add',
    'semantic_orthogonal', 'paired_ratio', 'paired_ratio_calibrated',
    'prefix_ratio64k', 'conditional_variance', 'directional_variance',
]


def read(path):
    return json.loads(path.read_text())


def main():
    result_root = ROOT / 'experiments/results/raev2_guidance_20260907'
    index_path = result_root / 'research_evidence_index.json'
    index = read(index_path)
    assert index['complete'] and not index['incomplete_studies_at_snapshot']
    rows = index['quality_rows']
    assert len(rows) == 30
    assert len({(row['mode'], row['samples']) for row in rows}) == 30
    assert {row['mode'] for row in rows} == set(ORDER)
    for row in rows:
        folder = DATA / row['study']
        execution = read(folder / 'execution.json')
        assert execution['complete'] and sha(folder / 'execution.json') == row['execution_sha256']
        metrics = read(folder / 'metrics.json')
        assert sha(folder / 'metrics.json') == row['metrics_sha256']
        actual = next(value for value in metrics if value['branch'] == row['mode'])
        assert actual['fid'] == row['fid'] and actual['sample_sha256'] == row['sample']['sha256']
        assert sha(Path(row['summary']['path'])) == row['summary']['sha256']
        assert Path(row['theory_document']).is_file()
    final_audit_path = result_root / 'final_semantic_add5k_audit.json'
    final_audit = read(final_audit_path)
    assert final_audit['complete'] and final_audit['paired_inputs_and_all_merged_pixels_verified']
    assert final_audit['source_snapshots_verified'] and final_audit['unchanged_original1k_additive_formula_and_parameters']
    final_state_path = DATA / 'final_semantic_add5k_continuation/state.json'
    final_state = read(final_state_path)
    assert final_state['complete'] and final_state['audit_sha256'] == sha(final_audit_path)
    assert final_state['stage'] == 'last_candidate_complete_requires_final_review'
    last = final_audit['rows'][0]
    indexed_last = next(row for row in rows if row['mode'] == 'semantic_add' and row['samples'] == 5000)
    assert last['fid'] == indexed_last['fid']
    assert last['sample_sha256'] == indexed_last['sample']['sha256']
    summaries = {}
    for count in [1000, 5000]:
        local = [row for row in rows if row['samples'] == count]
        controls = [row for row in local if row['mode'] in ['official', 'piecewise']]
        assert len(controls) == 2
        official = next(row for row in controls if row['mode'] == 'official')
        candidates = [row for row in local if row['mode'] not in ['official', 'piecewise']]
        best = min(candidates, key=lambda row: row['fid'])
        gains = [100 * (1 - row['fid'] / official['fid']) for row in candidates]
        assert max(gains) < 3, 'A quality-positive result requires review; do not publish a negative closeout'
        summaries[count] = {
            'samples': count, 'seed': official['seed'], 'candidate_count': len(candidates),
            'official_fid': official['fid'], 'official_three_percent_threshold': .97 * official['fid'],
            'stronger_fixed_control': min(controls, key=lambda row: row['fid']),
            'best_candidate': best, 'best_improvement_vs_official_percent': max(gains),
            'best_improvement_vs_stronger_fixed_control_percent':
                100 * (1 - best['fid'] / min(row['fid'] for row in controls)),
        }
    # Inspect the independent evidence for both best candidates, not just the index flags.
    semantic1k_path = result_root / 'fid_audit_extended.json'
    semantic1k = read(semantic1k_path)
    assert semantic1k['complete']
    best1k = summaries[1000]['best_candidate']
    audited1k = next(row for row in semantic1k['new_rows'] if row['mode'] == best1k['mode'])
    assert abs(audited1k['fid'] - best1k['fid']) < 1e-3
    assert audited1k['official_fid'] == best1k['fid']
    features1k = DATA / best1k['study'] / 'official_feature_cache' / (
        f"{best1k['mode']}-{best1k['sample']['sha256'][:16]}-inception.features.pt")
    assert sha(features1k) == audited1k['feature_sha256']
    directional_path = result_root / 'directional_variance_confirm5k_audit.json'
    directional = read(directional_path)
    best5k = summaries[5000]['best_candidate']
    assert directional['complete'] and directional['source_snapshots_verified']
    assert directional['all_merged_pixels_verified_against_shards']
    if best5k['mode'] == 'directional_variance':
        assert directional['candidate']['sample_sha256'] == best5k['sample']['sha256']
        assert directional['candidate']['fid'] == best5k['fid']
        assert abs(directional['independent_reconstruction']['fid'] - best5k['fid']) < 1e-3
    else:
        assert best5k['mode'] == 'semantic_add'
        assert last['fid'] == best5k['fid'] and last['sample_sha256'] == best5k['sample']['sha256']
    assert abs(last['independent_fid']['fid'] - last['fid']) < 1e-3
    ledger = read(result_root / 'final8_rounds.json')
    assert 6 <= ledger['current_round'] <= ledger['maximum_rounds'] == 8
    quality_round = ledger['quality_completed_at_round']
    assert quality_round == 6
    final_closed = ledger.get('research_closed', False)
    if final_closed:
        assert ledger['current_round'] == ledger['research_closed_at_round'] == 8
        assert ledger['next_round'] is None and ledger['no_automatic_research_restart']
    evidence = {str(p.relative_to(ROOT)): sha(p) for p in [
        index_path, final_audit_path, semantic1k_path, directional_path,
        result_root / 'directional_variance_calibration.json',
        result_root / 'fixed_mean_nll_fid_counterexample.json',
        result_root / 'workspace_reference_review.json',
    ]}
    requirements = [
        {'requirement': 'At least3% relative FID improvement on comparable1K or5K',
         'status': 'contradicted_by_completed_results',
         'evidence': 'All26 candidate quality rows miss even the original official threshold; stronger/cost controls cannot turn this into success'},
        {'requirement': 'Theory-grounded guidance with few hyperparameters; no time-bin extrapolation sweep',
         'status': 'implemented_with_documented_limits',
         'evidence': 'Frozen protocols, one shared convex conditional-variance head and one closed-form global directional ratio; fixed old semantic .15; no new quality candidate after the latter'},
        {'requirement': 'Try5K despite slightly worse1K, including promising original settings',
         'status': 'completed', 'evidence': 'Nine fixed candidates have both1K and5K, including unchanged final semantic_add'},
        {'requirement': 'Reproducible comparable inputs and evaluation',
         'status': 'completed_with_scope_limits',
         'evidence': 'All30 current-protocol quality rows have sample/source hashes in the index; best1K,best5K,and final5K independent FID evidence inspected here; individual full audits remain separate'},
        {'requirement': 'Organize prior theory, papers, ideas, positive/negative experiments and data',
         'status': 'organized_with_eight_historical_raw_asset_availability_gaps',
         'evidence': 'Full Git-workspace navigation and per-file inventory; previous52-paper reading index; explicit raw-asset reference review'},
        {'requirement': 'Conclude within at most8 further research rounds and commit',
         'status': 'quality_research_closed_git_finalization_separate',
         'evidence': f"Quality closure at round{quality_round}; commits and final worktree state must be checked after this artifact is written"},
    ]
    git_verification = result_root / 'round7_git_commit_verification.json'
    if git_verification.exists():
        checked = read(git_verification)
        assert checked['complete'] and checked['git_archival_of_quality_closeout_verified']
        assert checked['goal_still_unmet'] and checked['no_training_sampling_or_new_parameter_changes']
        assert checked['verified_commit'] == '573e939ed26e4b530e8bdacd4bf2b6590e54e8b3'
        evidence[str(git_verification.relative_to(ROOT))] = sha(git_verification)
        requirements[-1].update(status='quality_closeout_git_archive_verified',
            evidence='Quality experiments ended in round6; round7 reads every archived file directly from commit573e939 and verifies complete tree membership and payload hashes. Later administrative changes have their own commits.')
    result = {'complete': True, 'goal_achieved': False, 'created_unix': time.time(),
              'research_round': ledger['current_round'], 'maximum_rounds': 8,
              'quality_completed_at_round': quality_round,
              'research_closed_at_user_limit': final_closed,
              'new_research_requires_new_user_instruction': final_closed,
              'quality_research_closed': True, 'no_further_quality_candidates': True,
              'quality_rows': 30, 'candidate_quality_rows': 26, 'controls_rows': 4,
              'requirements': requirements, 'summaries': summaries,
              'final_semantic_add5k': {k: last[k] for k in [
                  'fid', 'sample_sha256', 'independent_fid', 'inference_cost_ratio',
                  'improvement_vs_official_percent', 'sample_model_calls', 'extra_sample_unconditional_calls']},
              'evidence_sha256': evidence, 'final_controller_state_sha256': sha(final_state_path),
              'cost_control_decision': 'Not run: final candidate fails quality against original100-step official; no slower or worse baseline used to manufacture success',
              'timing_limit': 'Measured shared-host inference timings; GPU exclusivity was not certified, so small timing differences do not establish speedups',
              'source_sha256': sha(Path(__file__).resolve())}
    output = result_root / 'final_goal_requirement_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    b1, b5 = summaries[1000], summaries[5000]
    lines = [
        '# RAEv2 guidance：最终质量结论与归档', '',
        f"第{quality_round}/8轮完成全部质量试验，**未达到 FID 改善至少3%的目标**。"
        '最后一个原设置的 semantic_add 5K也已完成独立审核。' +
        ('第8/8轮已按用户上限完成研究收束与归档核验；不再自动开展研究，继续实验需要新的用户指令。'
         if final_closed else '质量研究在此收束，剩余轮次仅用于最终核验与Git归档，不新增方法、系数、时间窗或seed。'), '',
        '## 结果', '',
        '| 规模 | 原 official FID | 最佳候选 | 候选FID | 相对原official改善 | 推理成本比 |',
        '|---|---:|---|---:|---:|---:|',
        f"| 1K | {b1['official_fid']:.9f} | {b1['best_candidate']['mode']} | {b1['best_candidate']['fid']:.9f} | {b1['best_improvement_vs_official_percent']:.6f}% | {b1['best_candidate']['inference_cost_ratio']:.6f}× |",
        f"| 5K | {b5['official_fid']:.9f} | {b5['best_candidate']['mode']} | {b5['best_candidate']['fid']:.9f} | {b5['best_improvement_vs_official_percent']:.6f}% | {b5['best_candidate']['inference_cost_ratio']:.6f}× |", '',
        f"1K更强的历史interval控制为{b1['stronger_fixed_control']['fid']:.9f}，最佳候选相对它仅改善"
        f"{b1['best_improvement_vs_stronger_fixed_control_percent']:.6f}%。5K方向候选相对旧全局方差"
        f"只改善{directional['improvement_vs_global_variance_percent']:.6f}%。原official的3%门槛分别为"
        f"{b1['official_three_percent_threshold']:.9f}和{b5['official_three_percent_threshold']:.9f}。", '',
        f"最后semantic_add5K的FID为{last['fid']:.12f}，相对official改善"
        f"{last['improvement_vs_official_percent']:.6f}%，推理{last['inference_cost_ratio']:.6f}倍。"
        '全部5000张、625个输入batch、旧 .15 参数、原模型/decoder/统计和冻结源码均通过审核。'
        f"独立FID为{last['independent_fid']['fid']:.12f}。由于质量未过，预定201调用Heun成本控制不执行；"
        '旧补丁也没有应用。', '',
        '共17种候选完成1K，9种完成5K，连同4条控制共有30条完整质量结果。'
        '不同规模使用不同seed，不是嵌套样本；未做5K的其余候选仍标为未做。', '',
        '| 方法 | 1K FID | 5K FID |', '|---|---:|---:|',
    ]
    for mode in ORDER:
        local = {row['samples']: row for row in rows if row['mode'] == mode}
        values = [f"{local[n]['fid']:.9f}" if n in local else '未做' for n in [1000, 5000]]
        lines.append(f'| {mode} | {values[0]} | {values[1]} |')
    lines += ['', '## 保留的理论与证据', '',
        '单一固定均值 Gaussian 风险、条件方差和分歧方向有明确推导及留出信号，'
        '但并未达到目标。精确反例进一步说明：Gaussian KL下降也可伴随FID上升。'
        '实际轨迹比值、分类风险与输入梯度、decoder空间和有限样本指标之间的缺口，'
        '见[理论结论](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)。', '',
        '- [逐项目标审核与精确数值](../experiments/results/raev2_guidance_20260907/final_goal_requirement_audit.json)。',
        '- [最后semantic5K完整审核](../experiments/results/raev2_guidance_20260907/final_semantic_add5k_audit.json)。',
        '- [当前质量与历史方法入口](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)。',
        '- [全工作区理论、代码和数据清单](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md)。',
        '- [Git中实际归档的独立核验](RAEV2_GUIDANCE_ARCHIVE_VERIFICATION_20260907_ZH.md)：第6轮提交的完整Git对象与文件集合均已检查。',
        '- [历史资产缺口复核](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md)：八项旧原始资产未在记录路径找到，不能宣称全部历史数据现可重跑。',
        '- [既有52篇一手文献阅读档案](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)与[各理论家族](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)。', '',
        '大模型、latent、样本、特征及完整日志保留原址，Git保存代码、配置、理论、'
        '轻量结果与身份清单。训练和数据准备成本与在线推理成本分开；不把共享GPU的'
        'inclusive worker秒相加冒称独占算量。推理时间也在共享机器上实测，未认证GPU独占，'
        '百分之几的耗时差不能据此解释为速度改进。归档完成不等于FID目标完成。', '',
    ]
    (ROOT / 'docs/RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md').write_text('\n'.join(lines))
    print(json.dumps({'goal_achieved': False, 'quality_rows': 30,
                      'best_1k_gain': b1['best_improvement_vs_official_percent'],
                      'best_5k_gain': b5['best_improvement_vs_official_percent'],
                      'last_5k_fid': last['fid']}, indent=2))


if __name__ == '__main__':
    main()
