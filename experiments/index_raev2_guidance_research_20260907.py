"""Link completed quality experiments, non-image evidence, and theory sources.

This index checks identities and reports incomplete processes separately. It
does not refit, evaluate images, mutate the sampler, or declare goal success.
Historical large assets remain at their recorded locations, not copied to Git.
"""
import json
from pathlib import Path
import time
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha

DOCS = {
    'official': 'RAEV2_GUIDANCE_GOAL_20260907_ZH.md',
    'piecewise': 'RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md',
    'ancestral': 'RAEV2_GUIDANCE_GOAL_20260907_ZH.md',
    'partial': 'RAEV2_GUIDANCE_GOAL_20260907_ZH.md',
    'calibrated': 'RAEV2_GUIDED_REVERSE_VARIANCE_20260907_ZH.md',
    'velocity_projection': 'RAEV2_TRANSPORT_PROJECTION_20260907_ZH.md',
    'noise_projection': 'RAEV2_TRANSPORT_PROJECTION_20260907_ZH.md',
    'stochastic_weak': 'RAEV2_STOCHASTIC_WEAK_20260907_ZH.md',
    'mean_weak': 'RAEV2_STOCHASTIC_WEAK_20260907_ZH.md',
    'critic_isotropic': 'RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md',
    'critic_exchangeable': 'RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md',
    'two_mode': 'RAEV2_TWO_MODE_RATIO_20260907_ZH.md',
    'semantic_add': 'RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md',
    'semantic_orthogonal': 'RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md',
    'paired_ratio': 'RAEV2_PAIRED_NOISE_RATIO_20260907_ZH.md',
    'paired_ratio_calibrated': 'RAEV2_PAIRED_NOISE_RATIO_20260907_ZH.md',
    'prefix_ratio64k': 'RAEV2_PREFIX_RATIO_20260907_ZH.md',
}


def file_record(path):
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    output = ROOT/'experiments/results/raev2_guidance_20260907/research_evidence_index.json'
    controls = {}
    for n, study, seed in [(1000, 'ancestral_screen1k', 202609071), (5000, 'weak_confirm5k', 202609072)]:
        folder = DATA/study
        assert json.loads((folder/'execution.json').read_text())['complete']
        metrics = {row['branch']: row for row in json.loads((folder/'metrics.json').read_text())}
        summary = json.loads((folder/'official/summary.json').read_text())
        assert summary['count'] == n and summary['seed'] == seed
        controls[n] = {'study': study, 'seed': seed, 'official': metrics['official'],
                       'interval': metrics['piecewise'], 'noise_sha256': summary['paired_noise_labels_sha256'],
                       'inference_seconds': summary['trajectory_seconds_sum']+summary['decode_seconds_sum']}
    rows, incomplete = [], []
    for execution_path in sorted(DATA.glob('*/execution.json')):
        execution = json.loads(execution_path.read_text())
        args = execution.get('args', {})
        n = args.get('samples')
        if n not in controls or args.get('seed') != controls[n]['seed']:
            continue
        if not execution.get('complete'):
            incomplete.append({'study': execution_path.parent.name, 'execution': file_record(execution_path),
                               'complete': False, 'pid_claim_not_a_liveness_check': execution.get('pid')})
            continue
        folder = execution_path.parent
        for source, digest in execution['sources'].items():
            assert sha(folder/'frozen_source'/Path(source).name) == digest
        for metric in json.loads((folder/'metrics.json').read_text()):
            mode = metric['branch']
            assert mode in DOCS, 'new method needs an explicit theory/status mapping'
            summary_path = folder/mode/'summary.json'
            summary = json.loads(summary_path.read_text())
            assert summary['complete'] and summary['count'] == n and summary['steps'] == 100
            assert summary['seed'] == controls[n]['seed']
            assert summary['paired_noise_labels_sha256'] == controls[n]['noise_sha256']
            sample_path = Path(metric['sample_path'])
            assert sha(sample_path) == metric['sample_sha256'] == summary['sample_sha256']
            assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
            cost = summary['trajectory_seconds_sum']+summary['decode_seconds_sum']
            rows.append({'study': folder.name, 'mode': mode, 'samples': n, 'seed': summary['seed'],
                         'fid': metric['fid'], 'inception_score': metric['inception_score'],
                         'relative_improvement_percent': 100*(1-metric['fid']/controls[n]['official']['fid']),
                         'inference_seconds': cost, 'inference_cost_ratio': cost/controls[n]['inference_seconds'],
                         'main_model_calls': summary['sample_model_calls'],
                         'additional_calls': {k: v for k, v in summary.items() if k.endswith('_calls') and k != 'sample_model_calls'},
                         'sample': {'path': str(sample_path), 'bytes': sample_path.stat().st_size, 'sha256': summary['sample_sha256']},
                         'summary': file_record(summary_path), 'execution_sha256': sha(execution_path),
                         'metrics_sha256': sha(folder/'metrics.json'), 'theory_document': str(ROOT/'docs'/DOCS[mode])})
    # Complete intermediate evidence is indexed as such, never as an FID.
    intermediate = []
    for name in ['paired_ratio_fit', 'actual_ratio_fit', 'prefix_ratio_fit', 'prefix_ratio_rb_fit',
                 'actual_ratio_bank64k', 'real_ratio_bank64k', 'prefix_ratio64k_features',
                 'prefix_ratio64k_fit', 'prefix_ratio64k_gradient_audit',
                 'prefix_ratio64k_backward_precision_diagnostic', 'prefix_ratio64k_probability_calibration']:
        path = DATA/name/'execution.json'
        if not path.exists():
            intermediate.append({'name': name, 'execution_present': False})
            continue
        record = json.loads(path.read_text())
        intermediate.append({'name': name, 'execution_present': True, 'complete': record.get('complete'),
                             'execution': file_record(path), 'stage': record.get('stage'),
                             'validation': record.get('validation'), 'entry_condition_passed': record.get('entry_condition_passed'),
                             'this_is_not_quality_sampling': True})
    documents = set()
    for pattern in ['RAEV2*.md', 'PFR*.md', 'RAE_RAEV2*.md', 'INTERNAL_GUIDANCE*.md',
                    'PROJECTED_FUTURE_REFERENCE*.md', 'SEMIGROUP_CONSISTENT*.md',
                    'AUTOGUIDANCE_FORESIGHT*.md', 'AFFINE_COUNTERFACTUAL*.md', 'TELESCOPING_SCALE*.md']:
        documents.update((ROOT/'docs').glob(pattern))
    for name in ['RESEARCH_STATUS.md', 'EXPERIMENT_ARCHIVE_INDEX_ZH.md']:
        documents.add(ROOT/'docs'/name)
    document_records = [file_record(path) for path in sorted(documents)]
    prior_archive = ROOT/'docs/RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md'
    result = {'complete': True, 'goal_achieved': False, 'created_unix': time.time(),
              'scope': 'Every completed current-protocol1K/5K study, explicitly mapped intermediates and theory-document identities',
              'quality_rows': rows, 'incomplete_studies_at_snapshot': incomplete,
              'intermediate_evidence': intermediate, 'theory_documents': document_records,
              'historical_families_and_primary_paper_asset_navigation': file_record(prior_archive),
              'controls': controls,
              'full_sample_hashes_and_source_snapshots_checked_for_quality_rows': True,
              'previous_independent_metric_audits_remain_separate': True,
              'historical_large_assets_not_all_rehashed_or_copied_by_this_index': True,
              'source_sha256': sha(Path(__file__).resolve())}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'quality_rows': len(rows), 'intermediates': len(intermediate),
                      'theory_documents': len(document_records), 'incomplete': incomplete}, indent=2), flush=True)


if __name__ == '__main__':
    main()
