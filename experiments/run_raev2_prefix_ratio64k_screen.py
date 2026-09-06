"""Validated fixed prefix head -> gradient audit -> native parity -> paired1K."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    fit_path = DATA/'prefix_ratio64k_fit/execution.json'
    fitted = json.loads(fit_path.read_text())
    assert fitted['complete'] and fitted['optimizer_converged'] and fitted['validation']['entry_condition_passed']
    for name in ['actual_ratio_bank64k', 'real_ratio_bank64k', 'prefix_ratio64k_features']:
        assert json.loads((DATA/name/'execution.json').read_text())['complete']
    integration = json.loads((ROOT/'experiments/locks/raev2_prefix_ratio64k_20260907/manifest.json').read_text())
    for path, identity in integration['files'].items():
        assert sha(ROOT/path) == identity['after_sha256'], 'reviewed pending integration has not been applied'
    checkpoint = DATA/'prefix_ratio64k_fit/head.pt'
    assert sha(checkpoint) == fitted['checkpoint_sha256']
    out = DATA/'prefix_ratio64k_screen_continuation'
    out.mkdir(exist_ok=False)
    files = [ROOT/'experiments'/name for name in [
        'run_raev2_prefix_ratio64k_screen.py', 'sample_raev2_ancestral_guidance.py',
        'run_raev2_ancestral_study.py', 'audit_raev2_prefix_ratio64k_gradient.py',
        'raev2_prefix_ratio_guidance.py', 'raev2_prefix_ratio_features.py',
        'raev2_actual_ratio_data.py', 'evaluate_raev2_official_samples.py']]
    files += [checkpoint, fit_path, DATA/'prefix_ratio64k_fit/plan.json',
              DATA/'ancestral_smoke_v2/shard0/official/samples.npz']
    sources = {str(p): sha(p) for p in files}
    baseline = DATA/'ancestral_screen1k'
    baseline_summary = json.loads((baseline/'official/summary.json').read_text())
    controls = json.loads((baseline/'metrics.json').read_text())
    official = next(row for row in controls if row['branch'] == 'official')
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(),
             'stage': 'gradient_audit', 'sources': sources, 'fit_execution': fitted,
             'seed': 202609071, 'samples': 1000, 'steps': 100, 'strength': 1.,
             'time_interval': 'all original100', 'goal_achieved': False}
    def save():
        p = out/'state.tmp'
        p.write_text(json.dumps(state, indent=2)+'\n')
        p.replace(out/'state.json')
    def run(command, logfile, four_gpus=False):
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
        if not four_gpus:
            env['CUDA_VISIBLE_DEVICES'] = '0'
        with (out/logfile).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        assert code == 0, f'{logfile} failed with code {code}; outputs retained, no resampling'
        for path, identity in sources.items():
            assert sha(Path(path)) == identity, 'frozen source changed: '+path
    save()
    try:
        run([sys.executable, '-m', 'experiments.audit_raev2_prefix_ratio64k_gradient'], 'gradient.log')
        audit = json.loads((DATA/'prefix_ratio64k_gradient_audit/execution.json').read_text())
        assert audit['complete'] and audit['plan']['checkpoint_sha256'] == fitted['checkpoint_sha256']
        state.update(stage='original_pixel_parity_and_candidate_smoke')
        smoke = DATA/'prefix_ratio64k_smoke'
        run([sys.executable, str(ROOT/'experiments/sample_raev2_ancestral_guidance.py'), '--output', str(smoke),
             '--samples', '8', '--seed', '202609071', '--parity', '--modes', 'official', 'prefix_ratio64k'], 'smoke.log')
        with np.load(smoke/'shard0/official/samples.npz') as a, np.load(DATA/'ancestral_smoke_v2/shard0/official/samples.npz') as b:
            assert np.array_equal(a['arr_0'], b['arr_0']), 'original official8 pixels changed'
        smoke_summary = json.loads((smoke/'shard0/prefix_ratio64k/summary.json').read_text())
        assert smoke_summary['complete'] and smoke_summary['sample_model_calls'] == 800
        assert smoke_summary['sample_prefix_forward_calls'] == smoke_summary['sample_prefix_backward_calls'] == 800
        state.update(stage='fixed_paired1k', original_official8_pixel_parity=True)
        target = DATA/'prefix_ratio64k_screen1k'
        run([sys.executable, str(ROOT/'experiments/run_raev2_ancestral_study.py'), '--output', str(target),
             '--samples', '1000', '--seed', '202609071', '--modes', 'prefix_ratio64k'], 'screen.log', four_gpus=True)
        summary = json.loads((target/'prefix_ratio64k/summary.json').read_text())
        assert summary['paired_noise_labels_sha256'] == baseline_summary['paired_noise_labels_sha256']
        assert summary['sample_model_calls'] == 100000
        assert summary['sample_prefix_forward_calls'] == summary['sample_prefix_backward_calls'] == 100000
        metrics = json.loads((target/'metrics.json').read_text())
        assert len(metrics) == 1
        metric = metrics[0]
        assert sha(Path(metric['sample_path'])) == metric['sample_sha256'] == summary['sample_sha256']
        state.update(complete=True, stage='paired1k_complete_requires_quality_and_cost_review',
                     metrics=metrics, summary=summary, paired_official=official,
                     improvement_vs_official_percent=100*(1-metric['fid']/official['fid']),
                     inference_cost_ratio=(summary['trajectory_seconds_sum']+summary['decode_seconds_sum']) /
                                          (baseline_summary['trajectory_seconds_sum']+baseline_summary['decode_seconds_sum']),
                     elapsed_seconds=time.time()-state['started_unix'], independent_confirmation_pending=True)
        save()
        (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_screen1k.json').write_text(json.dumps(state, indent=2)+'\n')
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
