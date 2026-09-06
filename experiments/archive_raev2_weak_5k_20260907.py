"""Archive the completed, unchanged independent 5K weak-guidance experiment."""
import json
from pathlib import Path

from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    study = DATA / 'weak_confirm5k'
    execution = json.loads((study / 'execution.json').read_text())
    if not execution['complete']:
        raise RuntimeError('independent study is incomplete')
    metrics = json.loads((study / 'metrics.json').read_text())
    if {m['branch'] for m in metrics} != {'official', 'piecewise', 'stochastic_weak'}:
        raise RuntimeError('missing a preregistered arm')
    control = json.loads((study / 'official/summary.json').read_text())
    rows = []
    for metric in metrics:
        summary = json.loads((study / metric['branch'] / 'summary.json').read_text())
        assert summary['complete'] and summary['count'] == 5000 and summary['seed'] == 202609072
        assert summary['paired_noise_labels_sha256'] == control['paired_noise_labels_sha256']
        assert sha(Path(metric['sample_path'])) == metric['sample_sha256'] == summary['sample_sha256']
        rows.append({**metric, **summary, 'inference_gpu_seconds':
                     summary['trajectory_seconds_sum'] + summary['decode_seconds_sum']})
    official = next(r for r in rows if r['branch'] == 'official')
    best = min(r['fid'] for r in rows if r['branch'] in {'official', 'piecewise'})
    for row in rows:
        row['relative_fid_improvement_percent'] = 100 * (1 - row['fid'] / official['fid'])
        row['inference_cost_ratio_to_official'] = row['inference_gpu_seconds'] / official['inference_gpu_seconds']
    candidate = next(r for r in rows if r['branch'] == 'stochastic_weak')
    record = {
        'complete': True, 'study': str(study), 'independent_seed_experiment_completed': True,
        'independent_quality_confirmation_passed': candidate['fid'] <= .97 * best,
        'goal_achieved': False,
        'conclusion': 'Unchanged stochastic weak guidance fails the independent 5K quality confirmation.',
        'cost_followup': 'No additional cost-matched samples for a candidate that fails original100 quality.',
        'best_control_fid': best, 'three_percent_target_fid': .97 * best,
        'execution_sha256': sha(study / 'execution.json'),
        'execution': execution, 'rows': rows,
        'all_blocks_diagnostic': 'weak_5k_all_balanced_blocks.json; all five blocks reported, no sample selection',
    }
    dest = ROOT / 'experiments/results/raev2_guidance_20260907/weak_confirm5k.json'
    dest.write_text(json.dumps(record, indent=2) + '\n')
    for row in rows:
        print(row['branch'], row['fid'], row['relative_fid_improvement_percent'],
              row['inference_cost_ratio_to_official'])
    print(dest)


if __name__ == '__main__':
    main()
