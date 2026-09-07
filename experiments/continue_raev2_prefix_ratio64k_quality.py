"""One pre-FID decision: unchanged prefix head gets paired1K and independent5K.

Only after the existing fit parent terminates successfully, apply the reviewed
pending integration and run the existing gradient/parity/screen sequence. Any
failure stops with original outputs retained. No resampling or tuning loop.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continue_raev2_guidance_after_5k import process_identity
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-after-reviewed-gradient-fix', action='store_true')
    args = parser.parse_args()
    recovery = args.resume_after_reviewed_gradient_fix
    out = DATA/'prefix_ratio64k_quality_continuation'
    out.mkdir(exist_ok=False)
    # This continuation decision must precede the fixed head and its first FID.
    if not recovery:
        assert not (DATA/'prefix_ratio64k_fit/head.pt').exists()
    assert not (DATA/'prefix_ratio64k_screen1k').exists()
    prior_plan = None
    if recovery:
        previous = DATA/'prefix_ratio64k_quality_continuation_failed_backward_precision'
        failed_state = json.loads((previous/'state.json').read_text())
        assert failed_state['stage'] == 'failed_requires_inspection_no_restart'
        assert process_identity(failed_state['pid']) is None
        prior_plan = json.loads((previous/'plan.json').read_text())
        assert prior_plan['decision_before_new_head_and_any_fid']
        diagnostic_path = DATA/'prefix_ratio64k_backward_precision_diagnostic/execution.json'
        diagnostic = json.loads(diagnostic_path.read_text())
        assert diagnostic['complete'] and diagnostic['forward_and_backward_protected_relative_error'] < .002
        assert sha(DATA/'prefix_ratio64k_fit/head.pt') == diagnostic['checkpoint_sha256']
    prerequisite_path = DATA/'prefix_ratio64k_features/execution.json'
    prerequisite = json.loads(prerequisite_path.read_text())
    parent_pid = prerequisite['pid']
    parent_identity = process_identity(parent_pid)
    assert parent_identity is not None or (recovery and prerequisite['complete'])
    lock = ROOT/'experiments/locks/raev2_prefix_ratio64k_20260907'
    integration = json.loads((lock/'manifest.json').read_text())
    assert sha(lock/'integration.patch') == integration['patch_sha256']
    for path, identities in integration['files'].items():
        assert sha(ROOT/path) == identities['after_sha256' if recovery else 'before_sha256'], 'reviewed integration identity changed'
    source_paths = [Path(__file__).resolve(), ROOT/'experiments/raev2_prefix_ratio_guidance.py',
        ROOT/'experiments/audit_raev2_prefix_ratio64k_gradient.py', ROOT/'experiments/run_raev2_prefix_ratio64k_screen.py',
        ROOT/'experiments/raev2_prefix_ratio_features.py', lock/'integration.patch', lock/'manifest.json']
    sources = {str(p): sha(p) for p in source_paths}
    if recovery:
        sources[str(diagnostic_path)] = sha(diagnostic_path)
    baseline = DATA/'weak_confirm5k'
    assert json.loads((baseline/'execution.json').read_text())['complete']
    baseline_summary = json.loads((baseline/'official/summary.json').read_text())
    controls = json.loads((baseline/'metrics.json').read_text())
    official = next(row for row in controls if row['branch'] == 'official')
    assert baseline_summary['count'] == 5000 and baseline_summary['seed'] == 202609072
    plan = {'created_unix': time.time(), 'decision_before_new_head_and_any_fid': True,
        'primary_requested_sample_counts': [1000, 5000],
        'screen1k': {'seed': 202609071, 'samples': 1000},
        'independent5k': {'seed': 202609072, 'samples': 5000,
                          'run_regardless_of_1k_fid_if_entry_gradient_parity_and_whole1k_pass': True},
        'rationale': 'User goal permits either1K or5K; existing all-block evidence shows finite-N ranking changes; compare both unchanged, report both',
        'head_formula_strength_and_time_schedule_unchanged': True,
        'prerequisites': 'one fixed64K fit passes original entry, numerical gradient audit and original official8 parity, finite complete1K',
        'no_new_seed_coefficient_window_layer_or_temperature_selection': True,
        'baseline5k': {'study': str(baseline), 'official_fid': official['fid'],
            'sample_sha256': baseline_summary['sample_sha256'],
            'noise_labels_sha256': baseline_summary['paired_noise_labels_sha256'],
            'summary_sha256': sha(baseline/'official/summary.json'),
            'metrics_sha256': sha(baseline/'metrics.json')},
        'cost_policy': 'reuse original official baseline; disclose all preparation/inference costs; quality success still requires independent metric audit and appropriate measured cost control',
        'no_automatic_goal_completion': True, 'sources': sources}
    if recovery:
        # Retain the original scientific decision date; this is an engineering
        # recovery after its first real-gradient preflight, before any images.
        plan['created_unix'] = prior_plan['created_unix']
        for key in ['primary_requested_sample_counts', 'screen1k', 'independent5k', 'baseline5k']:
            assert plan[key] == prior_plan[key]
        plan['recovery'] = {'created_unix': time.time(), 'previous_plan_sha256': sha(previous/'plan.json'),
                            'diagnostic_sha256': sha(diagnostic_path),
                            'only_change': 'keep FP32 and TF32-off context active during autograd backward',
                            'no_new_head_formula_data_or_tolerances': True}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_quality_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'complete': False, 'pid': os.getpid(), 'plan': plan, 'stage': 'waiting_for_existing_feature_and_fit_parent',
             'prerequisite_pid': parent_pid, 'prerequisite_process_identity': parent_identity,
             'started_unix': time.time(), 'sources': sources, 'goal_achieved': False}
    def save():
        p = out/'state.tmp'
        p.write_text(json.dumps(state, indent=2)+'\n')
        p.replace(out/'state.json')
    def check_sources():
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, 'frozen quality source changed: '+path
    def run(command, name):
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
        with (out/name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        assert code == 0, f'{name} failed ({code}); no automatic repeat'
        check_sources()
    save()
    try:
        while parent_identity is not None and process_identity(parent_pid) == parent_identity:
            time.sleep(5)
        prerequisite = json.loads(prerequisite_path.read_text())
        assert prerequisite['complete'], 'feature preparation did not complete; no restart'
        continuation = json.loads((DATA/'prefix_ratio64k_features/continuation.json').read_text())
        assert continuation['complete'] and continuation['exit_code'] == 0
        fit = json.loads((DATA/'prefix_ratio64k_fit/execution.json').read_text())
        assert fit['complete']
        check_sources()
        if not fit['validation']['entry_condition_passed']:
            state.update(complete=True, stage='fit_entry_failed_no_images_or_patch', fit_validation=fit['validation'])
            save()
            return
        for bank_name in ['actual_ratio_bank64k', 'real_ratio_bank64k']:
            assert json.loads((DATA/bank_name/'execution.json').read_text())['complete']
        if not recovery:
            for path, identities in integration['files'].items():
                assert sha(ROOT/path) == identities['before_sha256']
            state['stage'] = 'applying_reviewed_integration_after_readers_finished'
            save()
            subprocess.run(['git', 'apply', '--check', str(lock/'integration.patch')], cwd=ROOT, check=True)
            subprocess.run(['git', 'apply', str(lock/'integration.patch')], cwd=ROOT, check=True)
        for path, identities in integration['files'].items():
            assert sha(ROOT/path) == identities['after_sha256']
        state.update(stage='gradient_parity_and_fixed1k', integration_applied=True,
                     head_sha256=fit['checkpoint_sha256'])
        save()
        run([sys.executable, str(ROOT/'experiments/run_raev2_prefix_ratio64k_screen.py')], 'screen1k.log')
        screen = json.loads((DATA/'prefix_ratio64k_screen_continuation/state.json').read_text())
        assert screen['complete'] and screen['original_official8_pixel_parity']
        state.update(stage='unchanged_independent5k_regardless_of_1k_fid', screen1k_metrics=screen['metrics'])
        save()
        assert sha(baseline/'official/summary.json') == plan['baseline5k']['summary_sha256']
        assert sha(baseline/'metrics.json') == plan['baseline5k']['metrics_sha256']
        target = DATA/'prefix_ratio64k_confirm5k'
        run([sys.executable, str(ROOT/'experiments/run_raev2_ancestral_study.py'), '--output', str(target),
             '--samples', '5000', '--seed', '202609072', '--modes', 'prefix_ratio64k'], 'confirm5k.log')
        summary = json.loads((target/'prefix_ratio64k/summary.json').read_text())
        metrics = json.loads((target/'metrics.json').read_text())
        assert summary['complete'] and len(metrics) == 1
        assert summary['paired_noise_labels_sha256'] == plan['baseline5k']['noise_labels_sha256']
        assert summary['sample_model_calls'] == summary['sample_prefix_forward_calls'] == summary['sample_prefix_backward_calls'] == 500000
        assert sha(Path(metrics[0]['sample_path'])) == metrics[0]['sample_sha256'] == summary['sample_sha256']
        for path, identities in integration['files'].items():
            assert sha(ROOT/path) == identities['after_sha256']
        state.update(complete=True, stage='paired1k_and_independent5k_complete_requires_quality_and_cost_audit',
                     confirm5k_metrics=metrics, confirm5k_summary=summary, paired_official5k=official,
                     improvement5k_vs_official_percent=100*(1-metrics[0]['fid']/official['fid']),
                     inference_cost5k_ratio=(summary['trajectory_seconds_sum']+summary['decode_seconds_sum']) /
                         (baseline_summary['trajectory_seconds_sum']+baseline_summary['decode_seconds_sum']),
                     elapsed_seconds=time.time()-state['started_unix'])
        save()
        (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_confirm5k.json').write_text(json.dumps(state, indent=2)+'\n')
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
