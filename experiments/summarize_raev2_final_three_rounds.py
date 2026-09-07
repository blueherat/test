"""Build a source-backed closeout table after the three bounded rounds finish.

This reads completed audits. It neither samples images nor selects parameters.
The older paired controls are kept separate from the last three idea rounds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {
        'quadrature': ROOT/'experiments/results/raev2_guidance_quadrature_20260907/discovery5k_audit.json',
        'spatial': ROOT/'experiments/results/raev2_spatial_guidance_covariance_20260907/discovery5k_audit.json',
        'round1': DATA/'raev2_finite_read_budget_writer_20260907/audit.json',
        'round2': DATA/'raev2_causal_reference_20260907/discovery5k/audit.json',
        'round3': DATA/'raev2_full_read_after_write_20260907/discovery5k/audit.json',
        'prelude': DATA/'raev2_finite_guidance_retention_20260907/audit.json',
    }
    audits = {key: json.loads(path.read_text()) for key, path in paths.items()}
    assert all(row['complete'] for row in audits.values()), 'All audits must finish first.'
    base = audits['quadrature']['baseline']
    for entry in [audits['spatial']['controls']['official'], audits['round2']['baseline'], audits['round3']['baseline']]:
        assert entry['sample_sha256'] == base['sample_sha256'] and entry['fid'] == base['fid']
    base_fid, base_seconds = base['fid'], base['inference_seconds']
    rows = []

    def add(raw, scope, source):
        measured = raw['inference_seconds']
        independent = raw['independent_fid']['fid']
        assert abs(independent-raw['fid']) < 1e-5
        rows.append({
            'method': raw['branch'], 'scope': scope, 'samples': 5000, 'seed': 202609072,
            'fid': raw['fid'], 'independent_fid': independent,
            'improvement_percent': 100*(1-raw['fid']/base_fid),
            'inception_score': raw['inception_score'],
            'inference_seconds_sum': measured, 'cost_ratio_to_official': measured/base_seconds,
            'main_sample_calls': raw.get('main_sample_calls', 500000),
            'extra_prefix_sample_calls': raw.get('prefix_sample_calls', 0),
            'extra_full_sample_calls': raw.get('reread_sample_calls', 0),
            'sample_path': raw['sample_path'], 'sample_sha256': raw['sample_sha256'],
            'feature_sha256': raw['feature_sha256'], 'source_audit': str(paths[source]),
        })

    add(base, 'baseline', 'quadrature')
    for raw in audits['quadrature']['rows']:
        add(raw, 'before_final_three_rounds', 'quadrature')
    add(audits['spatial']['controls']['global'], 'before_final_three_rounds', 'spatial')
    for raw in audits['spatial']['rows']:
        add(raw, 'before_final_three_rounds', 'spatial')
    for key in ['round2', 'round3']:
        audit = audits[key]
        assert audit['paired_all_batch_noise_labels_verified']
        assert audit['all_merged_pixels_and_shard_hashes_verified']
        assert len(audit['rows']) == 1 and audit['rows'][0]['samples'] == 5000
        assert audit['rows'][0]['seed'] == 202609072
        add(audit['rows'][0], key, key)
    r1 = audits['round1']
    assert r1['samples'] == 128 and r1['quality_evaluated'] is False
    best = min(rows[1:], key=lambda row: row['fid'])
    result = {
        'complete': True,
        'scope': 'Recent reopened guidance study; final three additional idea rounds, with earlier paired controls.',
        'max_additional_idea_rounds': 3, 'completed_additional_idea_rounds': 3,
        'new_idea_rounds_remaining': 0, 'stop_new_research': True,
        'baseline_fid': base_fid, 'three_percent_fid_threshold': .97*base_fid,
        'discovery_point_reaches_three_percent': best['fid'] <= .97*base_fid,
        'quality_goal_certified': False,
        'confirmation_seed': 202609073, 'confirmation_seed_used_in_these_rounds': False,
        'bank_limit': 'Repeatedly explored paired discovery bank; no independent quality confirmation or 50K SOTA claim.',
        'timing_limit': 'Sum of trajectory and decoder worker times; same hardware protocol, runs at different times; excludes preparation and model loading.',
        'round1_mechanism': r1, 'prelude_audit': audits['prelude'],
        'paired5k_rows': rows, 'best_recent_paired5k_point': best,
        'maximum_independent_fid_difference': max(abs(row['fid']-row['independent_fid']) for row in rows),
        'source_audits': {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()},
        'summarizer_sha256': sha(Path(__file__).resolve()),
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'closeout.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (out/'paired5k.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| 方法 | FID5K ↓ | 相对改善 | 推理时间比 | 范围 |', '|---|---:|---:|---:|---|']
    lines += [f"| {r['method']} | {r['fid']:.8f} | {r['improvement_percent']:+.5f}% | {r['cost_ratio_to_official']:.5f} | {r['scope']} |" for r in rows]
    (out/'paired5k.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key: result[key] for key in ['completed_additional_idea_rounds', 'new_idea_rounds_remaining', 'discovery_point_reaches_three_percent', 'quality_goal_certified', 'maximum_independent_fid_difference']}, indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
