#!/usr/bin/env python3
"""Join completed old1K/new4K/pooled5K results with their recorded costs."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
EXT = ROOT / 'scale_extension_5k_v1'
PLAN_SHA = '55554188eeb66f2de867b58a0ef41f981a92e8aa0bb6688d89d19d7dcc39c159'
COMMIT = '19dfb4c2705333eb8b97e454fb354d47d1fe135b'


def main():
    sources = {}

    def read(path):
        path = Path(path)
        payload = path.read_bytes()
        sources[str(path)] = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                              'size_bytes': len(payload)}
        return json.loads(payload)

    plan = read(EXT / 'plan.json')
    assert sources[str(EXT / 'plan.json')]['sha256'] == PLAN_SHA
    cost = read(EXT / 'cost_review_v1/summary.json')
    release = read(EXT / 'cost_release.json')
    assert cost['complete'] and release['complete'] and release['frozen_before_fid']
    assert release['reflection_official_uniform_steps'] == 201
    assert release['reflection_pooled5k_recorded_T_and_W_covered']
    # This closeout implements the observed, completed 201-step case only.
    assert cost['reflection201_decisions']['pooled5k_protocol_decision']['both_recorded_metrics_covered']
    historical = {
        ('legacy', 'official100'): ('proximal_seed202609066/metrics_official.json', 'official', 'legacy_official100'),
        ('legacy', 'global_proximal100'): ('proximal_seed202609066/metrics_global.json', 'global_proximal', 'global_proximal100'),
        ('legacy', 'energy100'): ('energy_ball_seed202609066/metrics_energy_ball.json', 'energy_ball', 'legacy_energy100'),
        ('native_global', 'official100'): ('spatial_energy_balls_v1/fid_results.json', 'official100', 'spatial_official100'),
        ('native_global', 'global100'): ('spatial_energy_balls_v1/fid_results.json', 'global100', 'native_global100'),
        ('reflection', 'official100'): ('affine_reflection_v1/official_evaluation.json', 'official100', 'reflection_official100'),
        ('reflection', 'reflection100'): ('affine_reflection_v1/official_evaluation.json', 'reflection100', 'reflection100'),
        ('reflection', 'official201'): ('affine_reflection_v1/official_evaluation.json', 'official201', 'reflection_official201'),
    }
    metrics = {}
    for (family, arm), (filename, branch, historical_name) in historical.items():
        selected = [v for v in read(ROOT / filename) if v['branch'] == branch]
        assert len(selected) == 1
        row = selected[0]
        expected = plan['historical_1k'][historical_name]['artifacts']['samples.npz']
        assert row['sample_sha256'] == expected['sha256']
        assert Path(row['sample_path']).resolve() == Path(expected['path']).resolve()
        assert row['evaluator_commit'] == COMMIT
        metrics[family, arm, 'old1k'] = float(row['fid'])
    for family in ('legacy', 'native_global', 'reflection'):
        parent = EXT / 'evaluation_preparation_v1'
        execution = read(parent / f'{family}_evaluation_execution.json')
        assert execution['complete'] and all(j['exit_code'] == 0 for j in execution['jobs'])
        merged = read(parent / f'{family}_merged/summary.json')
        evaluated = read(parent / f'{family}_merged/feature_subsets/summary.json')
        official = read(parent / f'{family}_merged/official_evaluation.json')
        assert merged['complete'] and evaluated['complete']
        assert len(evaluated['results']) == 2 * len(merged['arms'])
        for row in evaluated['results']:
            assert row['family'] == family and row['arm'] in merged['arms']
            assert row['samples'] == {'pooled5k': 5000, 'new4k': 4000}[row['subset']]
            assert row['sample_sha256'] == merged['arms'][row['arm']][row['subset']]['samples']['sha256']
            key = (family, row['arm'], row['subset'])
            assert key not in metrics
            metrics[key] = float(row['fid'])
        for row in official:
            arm = row['branch'][len(family)+1:-len('_pooled5k')]
            assert abs(metrics[family, arm, 'pooled5k'] - row['fid']) <= 1e-6
    comparisons = [
        ('global_proximal', 'legacy', 'global_proximal100', 'official100'),
        ('legacy_energy_ball', 'legacy', 'energy100', 'official100'),
        ('native_global_ball', 'native_global', 'global100', 'official100'),
        ('affine_reflection', 'reflection', 'reflection100', 'official100'),
        ('affine_reflection', 'reflection', 'reflection100', 'official201'),
    ]
    rows = []
    for method, family, candidate, baseline in comparisons:
        for subset in ('old1k', 'new4k', 'pooled5k'):
            b, c = metrics[family, baseline, subset], metrics[family, candidate, subset]
            costs = cost['families'][method]
            cb, cc = costs[baseline][subset], costs['candidate'][subset]
            ratios = {}
            for field in ('runner_wall_seconds', 'trajectory_through_decode_wall_seconds'):
                bv, cv = cb[field]['value'], cc[field]['value']
                ratios[field + '_ratio'] = cv/bv if bv is not None and cv is not None else None
            rows.append({'method': method, 'family': family, 'candidate': candidate,
                         'baseline': baseline, 'subset': subset, 'sample_count': cb['sample_count'],
                         'baseline_fid': b, 'candidate_fid': c, 'fid_difference': c-b,
                         'relative_gain_percent': 100*(b-c)/b,
                         'candidate_main_model_nfe': cc['stage2_nfe_per_sample'],
                         'baseline_main_model_nfe': cb['stage2_nfe_per_sample'], **ratios})
    output = EXT / 'final_results_v1'
    output.mkdir(exist_ok=False)
    result = {'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'root_plan_sha256': PLAN_SHA, 'comparisons': rows,
              'any_pooled5k_point_gain_at_least_5_percent': any(r['relative_gain_percent'] >= 5 for r in rows if r['subset']=='pooled5k'),
              'fair_total_cost_goal_proven': False,
              'limits': ['Positive gain means lower FID against the named within-family baseline.',
                         'Pooled5K retains the screened old1K; new4K excludes those observations.',
                         'Interaction size remains N=1000 per cohort. No 5000-particle method was evaluated.',
                         'Separate subsets overlap; they are not independent replications of one another.',
                         'FID values at different sample counts are not directly comparable quality levels.',
                         'Timing ratios use recorded boundaries, preserve nulls and are noisy point estimates.',
                         'No sampling-selection, temporal coefficient schedule, new fit or quality search was added.'],
              'sources': list(sources.values()),
              'source': {'path': str(Path(__file__).resolve()),
                         'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (output / 'comparisons.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'output': str(output), 'complete': True, 'comparisons': rows}, indent=2))


if __name__ == '__main__':
    main()
