"""After the single fit: gate, apply reviewed patch, exact pixels, fixed1K+5K."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continue_raev2_guidance_after_5k import process_identity
from experiments.raev2_conditional_variance import PLAN
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA/'conditional_variance_continuation'
    out.mkdir(exist_ok=False)
    feature_path = DATA/'conditional_variance_features/execution.json'
    feature = json.loads(feature_path.read_text())
    pid = feature['pid']
    identity = process_identity(pid)
    assert identity is not None or feature['complete'], 'preparation stopped before completion'
    lock = ROOT/'experiments/locks/raev2_conditional_variance_20260907'
    integration = json.loads((lock/'manifest.json').read_text())
    paths = [Path(__file__).resolve(), ROOT/'experiments/raev2_conditional_variance.py',
             ROOT/'experiments/audit_raev2_conditional_variance_quality.py',
             ROOT/'experiments/audit_raev2_prefix_ratio64k_quality.py', ROOT/'experiments/audit_raev2_screen_fid_20260907.py',
             ROOT/'experiments/summarize_raev2_guidance_20260907.py', ROOT/'experiments/continue_raev2_guidance_after_5k.py',
             ROOT/'experiments/evaluate_raev2_official_samples.py', lock/'manifest.json', lock/'integration.patch']
    sources = {str(p): sha(p) for p in paths}
    for relative, hashes in integration['files'].items():
        assert sha(ROOT/relative) == hashes['before_sha256']
    state = {'complete': False, 'pid': os.getpid(), 'stage': 'waiting_for_single_fixed_fit', 'plan': PLAN,
             'preparation_pid': pid, 'preparation_process_identity': identity, 'sources': sources,
             'goal_achieved': False, 'research_round_limit': 8, 'no_automatic_restart_or_parameter_changes': True}
    def save():
        temp = out/'state.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'state.json')
    def check_sources():
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, 'source changed: '+path
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    def run(command, name, extra_env=None):
        with (out/name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**env, **(extra_env or {})}, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        assert code == 0, f'child failed: {name}, exit {code}; inspect without automatic restart'
        check_sources()
    save()
    try:
        while identity is not None and process_identity(pid) == identity:
            time.sleep(5)
        feature = json.loads(feature_path.read_text())
        assert feature['complete'], 'preparation failed; inspect before quality sampling'
        if feature['stage'] == 'legacy_quality_positive_skip_new_fit_pending_final_review':
            state.update(complete=True, stage='legacy_quality_positive_requires_final_review')
            save()
            return
        continuation = json.loads((feature_path.parent/'continuation.json').read_text())
        assert continuation['complete'] and continuation['exit_code'] == 0
        fit_path = DATA/'conditional_variance_fit/execution.json'
        fit = json.loads(fit_path.read_text())
        assert fit['complete'] and fit['plan'] == PLAN and sha(feature_path) == fit['feature_execution_sha256']
        state.update(fit_execution_sha256=sha(fit_path), checkpoint_sha256=fit['checkpoint_sha256'])
        if not fit['validation']['entry_condition_passed']:
            state.update(complete=True, stage='validation_entry_failed_no_patch_or_quality_sampling')
            save()
            return
        assert sha(DATA/'conditional_variance_fit/head.pt') == fit['checkpoint_sha256']
        check_sources()
        legacy = json.loads((DATA/'final8_legacy5k_continuation/state.json').read_text())
        assert legacy['complete'] and process_identity(legacy['pid']) is None
        for relative, hashes in integration['files'].items():
            assert sha(ROOT/relative) == hashes['before_sha256']
        subprocess.run(['git', 'apply', '--check', str(lock/'integration.patch')], cwd=ROOT, check=True)
        subprocess.run(['git', 'apply', str(lock/'integration.patch')], cwd=ROOT, check=True)
        for relative, hashes in integration['files'].items():
            assert sha(ROOT/relative) == hashes['after_sha256']
        state.update(integration_applied=True, stage='exact_8_image_native_and_zero_head_parity')
        save()
        smoke = DATA/'conditional_variance_smoke8'
        run([sys.executable, str(ROOT/'experiments/sample_raev2_ancestral_guidance.py'), '--output', str(smoke),
             '--samples', '8', '--seed', '202609071', '--steps', '100', '--parity', '--modes',
             'official', 'calibrated', 'conditional_variance_zero', 'conditional_variance'], 'smoke8.log', {'CUDA_VISIBLE_DEVICES': '0'})
        pairs = [('official', DATA/'ancestral_smoke_v2/shard0/official/samples.npz'),
                 ('calibrated', DATA/'calibrated_smoke/shard0/calibrated/samples.npz'),
                 ('conditional_variance_zero', smoke/'shard0/calibrated/samples.npz')]
        parity = []
        for mode, reference in pairs:
            path = smoke/f'shard0/{mode}/samples.npz'
            with np.load(path) as current, np.load(reference) as anchor:
                assert current['arr_0'].shape == (8, 256, 256, 3)
                assert np.array_equal(current['arr_0'], anchor['arr_0']), '8-image pixel parity failed: '+mode
                assert np.array_equal(current['ids'], anchor['ids'])
            parity.append({'mode': mode, 'sample_sha256': sha(path), 'reference': str(reference), 'reference_sha256': sha(reference), 'all_pixels_equal': True})
        state.update(parity=parity, stage='fixed_head_screen1k_then_confirm5k_regardless_1k_fid')
        save()
        for count, seed, suffix in [(1000, 202609071, 'screen1k'), (5000, 202609072, 'confirm5k')]:
            folder = DATA/f'conditional_variance_{suffix}'
            state.update(stage='sampling_'+suffix)
            save()
            run([sys.executable, str(ROOT/'experiments/run_raev2_ancestral_study.py'), '--output', str(folder),
                 '--samples', str(count), '--seed', str(seed), '--modes', 'conditional_variance'], suffix+'.log')
            state.update(stage='independent_audit_'+suffix)
            save()
            run([sys.executable, '-m', 'experiments.audit_raev2_conditional_variance_quality', '--samples', str(count)], suffix+'_audit.log',
                {'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
            record_path = ROOT/f'experiments/results/raev2_guidance_20260907/conditional_variance_{suffix}_audit.json'
            record = json.loads(record_path.read_text())
            assert record['complete'] and record['checkpoint_sha256'] == state['checkpoint_sha256']
            state[suffix] = {'audit_sha256': sha(record_path), 'fid': record['candidate']['fid'],
                             'improvement_vs_official_percent': record['relative_fid_improvement_percent'],
                             'improvement_vs_best_control_percent': record['improvement_vs_best_existing_control_percent'],
                             'inference_cost_ratio': record['inference_cost_ratio']}
            save()
        state.update(complete=True, stage='fixed1k5k_complete_requires_final_quality_cost_review')
        save()
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
