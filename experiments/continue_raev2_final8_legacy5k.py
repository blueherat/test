"""Two fixed legacy mechanisms get 5K after the existing prefix queue exits.

The user explicitly requested 5K even for slightly negative 1K results, then
limited the remaining research to eight rounds. This is one predeclared batch,
with no retries, new coefficients, checkpoints, windows, or seed selection.
"""
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

MODES = ['paired_ratio_calibrated', 'calibrated']


def main():
    out = DATA/'final8_legacy5k_continuation'
    out.mkdir(exist_ok=False)
    target = DATA/'final8_legacy_confirm5k'
    assert not target.exists()
    prerequisite_path = DATA/'prefix_ratio64k_quality_continuation/state.json'
    prerequisite = json.loads(prerequisite_path.read_text())
    pid = prerequisite['pid']
    identity = process_identity(pid)
    assert identity is not None or prerequisite['complete']
    integration = json.loads((ROOT/'experiments/locks/raev2_prefix_ratio64k_20260907/manifest.json').read_text())
    files = [Path(__file__).resolve(), ROOT/'experiments/audit_raev2_final8_legacy5k.py',
             ROOT/'experiments/audit_raev2_prefix_ratio64k_quality.py',
             ROOT/'experiments/audit_raev2_screen_fid_20260907.py',
             ROOT/'experiments/summarize_raev2_guidance_20260907.py',
             ROOT/'experiments/continue_raev2_guidance_after_5k.py',
             ROOT/'experiments/raev2_paired_ratio_model.py',
             ROOT/'experiments/raev2_ancestral_guidance.py',
             DATA/'paired_ratio_fit/critic.pt', DATA/'paired_ratio_calibration.json',
             DATA/'guided_reverse_variance/calibration.json',
             DATA/'weak_confirm5k/official/summary.json', DATA/'weak_confirm5k/metrics.json',
             ROOT/'experiments/results/raev2_guidance_20260907/semantic_confirm5k.json',
             ROOT/'experiments/results/raev2_guidance_20260907/paired_ratio_screens.json']
    sources = {str(path): sha(path) for path in files}
    allowed_integration = {path: [record['before_sha256'], record['after_sha256']]
                           for path, record in integration['files'].items()}
    for path, allowed in allowed_integration.items():
        assert sha(ROOT/path) in allowed
    calibration = json.loads((DATA/'paired_ratio_calibration.json').read_text())
    assert calibration['entry_condition_passed'] and not calibration['fid_used_for_parameter_fit']
    baseline = json.loads((DATA/'weak_confirm5k/official/summary.json').read_text())
    assert baseline['count'] == 5000 and baseline['seed'] == 202609072
    plan = {'created_unix': time.time(), 'decision_before_either_new_5k': True,
            'research_round_started': 1, 'remaining_research_round_limit': 8,
            'modes': MODES, 'samples': 5000, 'seed': 202609072, 'steps': 100, 'batch': 8,
            'reason': 'User requested fixed 5K for slightly negative 1K; two distinct mechanisms, not a parameter sweep',
            'fixed_1k_fid': {'paired_ratio_calibrated': 38.56759907090867, 'calibrated': 38.5411998671},
            'unchanged_ratio_calibration': calibration,
            'unchanged_formulas': {'paired_ratio_calibrated': 'G + alpha*t^2*grad(f); one previous probability calibration',
                                  'calibrated': 'native Euler + ((t-s)/t)*sqrt(previous MSE_G(t))*xi'},
            'known_limitations': ['Endpoint re-noising differs from actual native marginals',
                                  'Fixed-mean Gaussian cross-entropy optimality does not imply FID improvement'],
            'no_refit_or_coefficient_window_layer_seed_search': True,
            'skip_if_prefix_quality_meets_three_percent_pending_audit': True,
            'baseline_noise_sha256': baseline['paired_noise_labels_sha256'],
            'sources': sources, 'allowed_reviewed_integration': allowed_integration,
            'no_automatic_goal_completion': True}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    record_dir = ROOT/'experiments/results/raev2_guidance_20260907'
    (record_dir/'final8_legacy5k_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'complete': False, 'pid': os.getpid(), 'stage': 'waiting_for_existing_prefix_quality',
             'prerequisite_pid': pid, 'prerequisite_process_identity': identity,
             'goal_achieved': False, 'plan': plan}

    def save():
        temp = out/'state.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'state.json')

    def check_sources():
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, 'frozen input changed: '+path

    def run(command, log_name):
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
        with (out/log_name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        assert code == 0, f'{log_name}: child failed ({code}); inspect, no automatic repeat'
        check_sources()

    save()
    try:
        while identity is not None and process_identity(pid) == identity:
            time.sleep(5)
        prerequisite = json.loads(prerequisite_path.read_text())
        assert prerequisite['complete'], 'prefix queue failed; inspect before independent GPU work'
        check_sources()
        if prerequisite['stage'] != 'fit_entry_failed_no_images_or_patch':
            assert prerequisite['stage'] == 'paired1k_and_independent5k_complete_requires_quality_and_cost_audit'
            prefix_metrics = prerequisite['screen1k_metrics']
            control1k = json.loads((DATA/'ancestral_screen1k/metrics.json').read_text())
            best1k = min(row['fid'] for row in control1k if row['branch'] in ['official', 'piecewise'])
            gain1k = 100*(1-prefix_metrics[0]['fid']/best1k)
            if max(gain1k, prerequisite['improvement5k_vs_official_percent']) >= 3:
                state.update(complete=True, stage='prefix_quality_positive_audit_first_no_legacy_sampling',
                             prefix_queue_sha256=sha(prerequisite_path))
                save()
                return
        selected_sources = {path: sha(ROOT/path) for path in allowed_integration}
        expected_key = 'after_sha256' if prerequisite.get('integration_applied') else 'before_sha256'
        for path, digest in selected_sources.items():
            assert digest == integration['files'][path][expected_key]
        state.update(stage='two_unchanged_legacy_5k', selected_integration_sources=selected_sources,
                     prefix_queue_sha256=sha(prerequisite_path), started_sampling_unix=time.time())
        save()
        run([sys.executable, str(ROOT/'experiments/run_raev2_ancestral_study.py'), '--output', str(target),
             '--samples', '5000', '--seed', '202609072', '--modes', *MODES], 'sampling.log')
        for path, digest in selected_sources.items():
            assert sha(ROOT/path) == digest
        state.update(stage='independent_metrics_pixels_and_input_audit')
        save()
        run([sys.executable, '-m', 'experiments.audit_raev2_final8_legacy5k'], 'audit.log')
        result = json.loads((record_dir/'final8_legacy5k_audit.json').read_text())
        assert result['complete']
        state.update(complete=True, stage='legacy5k_complete_results_require_review',
                     results=result['rows'], elapsed_sampling_and_audit_seconds=time.time()-state['started_sampling_unix'])
        save()
        (record_dir/'final8_legacy5k_execution.json').write_text(json.dumps(state, indent=2)+'\n')
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
