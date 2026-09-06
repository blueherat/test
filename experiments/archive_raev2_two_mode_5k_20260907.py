"""Verify and archive two-mode 5K against the existing paired controls."""
import json
from pathlib import Path
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    rows = []
    for name, modes in [('weak_confirm5k', {'official', 'piecewise'}),
                        ('two_mode_confirm5k', {'two_mode'})]:
        folder = DATA / name
        execution = json.loads((folder / 'execution.json').read_text())
        assert execution['complete']
        for metric in json.loads((folder / 'metrics.json').read_text()):
            if metric['branch'] not in modes:
                continue
            summary = json.loads((folder / metric['branch'] / 'summary.json').read_text())
            assert summary['complete'] and summary['seed'] == 202609072 and summary['count'] == 5000
            assert summary['sample_sha256'] == metric['sample_sha256'] == sha(Path(metric['sample_path']))
            rows.append({**metric, **summary, 'execution_sha256': sha(folder / 'execution.json'),
                         'inference_gpu_seconds': summary['trajectory_seconds_sum'] + summary['decode_seconds_sum']})
    assert len({r['paired_noise_labels_sha256'] for r in rows}) == 1
    baseline = next(r for r in rows if r['branch'] == 'official')
    for row in rows:
        row.update(relative_fid_improvement_percent=100 * (1 - row['fid'] / baseline['fid']),
                   inference_cost_ratio=row['inference_gpu_seconds'] / baseline['inference_gpu_seconds'])
    candidate = next(r for r in rows if r['branch'] == 'two_mode')
    queue = DATA / 'two_mode_5k_continuation/state.json'
    record = {'complete': True, 'goal_achieved': False, 'rows': rows,
              'three_percent_quality_passed': candidate['fid'] <= .97 * baseline['fid'],
              'conclusion': 'A small 5K gain, insufficient for the requested 3%; no parameter revision.',
              'queue_decision_sha256': sha(queue), 'queue_decision': json.loads(queue.read_text()),
              'candidate_execution': json.loads((DATA / 'two_mode_confirm5k/execution.json').read_text())}
    dest = ROOT / 'experiments/results/raev2_guidance_20260907/two_mode_confirm5k.json'
    dest.write_text(json.dumps(record, indent=2) + '\n')
    print(candidate['relative_fid_improvement_percent'], candidate['inference_cost_ratio'], dest)


if __name__ == '__main__':
    main()
