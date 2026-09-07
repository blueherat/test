"""Last quality candidate: the unchanged original semantic-add setting at5K."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continue_raev2_guidance_after_5k import process_identity
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA/'final_semantic_add5k_continuation'
    out.mkdir(exist_ok=False)
    target = DATA/'final_semantic_add_confirm5k'
    assert not target.exists()
    prerequisite_path = DATA/'directional_variance_continuation/state.json'
    prerequisite = json.loads(prerequisite_path.read_text())
    pid, identity = prerequisite['pid'], process_identity(prerequisite['pid'])
    assert prerequisite['complete'] or identity is not None
    previous = DATA/'semantic_complement_screen1k'
    original = json.loads((previous/'shard0/request.json').read_text())
    summary = json.loads((previous/'semantic_add/summary.json').read_text())
    metrics = json.loads((previous/'metrics.json').read_text())
    metric = next(row for row in metrics if row['branch'] == 'semantic_add')
    control = json.loads((DATA/'ancestral_screen1k/official/summary.json').read_text())
    ratio = (summary['trajectory_seconds_sum']+summary['decode_seconds_sum'])/(control['trajectory_seconds_sum']+control['decode_seconds_sum'])
    heun_steps = math.ceil((100*ratio+1)/2)
    plan = {'research_round': 5, 'maximum_rounds': 8, 'last_additional_quality_candidate': True,
            'mode': 'semantic_add', 'samples': 5000, 'seed': 202609072, 'steps': 100, 'batch': 8,
            'unchanged_1k_parameters': original['semantic_complement'], 'original_1k_fid': metric['fid'],
            'original_1k_sample_sha256': metric['sample_sha256'], 'original_1k_inference_cost_ratio': ratio,
            'reason': 'Strongest original1K candidate without its own5K; orthogonal variant failure is not the additive result',
            'known_limitation': 'Orthogonal1K is only .1106% better than additive and its5K fails; no assertion additive will reverse this',
            'no_new_strength_window_layer_seed_or_training': True,
            'cost_control_if_quality_passes': {'solver': 'Heun with final Euler', 'steps': heun_steps, 'main_calls_per_image': 2*heun_steps-1,
                'rule': 'ceil((100*original_additive1k_measured_cost_ratio+1)/2)',
                'fresh_source_patch_and_native8_checks_required': True, 'never_apply_stale_old_Heun_patch': True},
            'skip_if_directional_quality_meets_three_percent_pending_audit': True}
    paths = [Path(__file__).resolve(), ROOT/'experiments/audit_raev2_final_semantic_add5k.py',
        ROOT/'experiments/sample_raev2_ancestral_guidance.py', ROOT/'experiments/run_raev2_ancestral_study.py',
        ROOT/'experiments/raev2_semantic_complement.py', ROOT/'experiments/raev2_semantic_quality_guidance.py',
        ROOT/'experiments/audit_raev2_prefix_ratio64k_quality.py', ROOT/'experiments/audit_raev2_screen_fid_20260907.py',
        ROOT/'experiments/summarize_raev2_guidance_20260907.py', ROOT/'experiments/continue_raev2_guidance_after_5k.py',
        previous/'shard0/request.json', previous/'semantic_add/summary.json', previous/'metrics.json']
    sources = {str(p): sha(p) for p in paths}
    state = {'complete': False, 'pid': os.getpid(), 'stage': 'waiting_for_live_directional5k',
             'prerequisite_pid': pid, 'prerequisite_process_identity': identity, 'plan': plan, 'sources': sources,
             'goal_achieved': False, 'no_automatic_restart': True}
    def save():
        temp = out/'state.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'state.json')
    def check_sources():
        for path, expected in sources.items():
            assert sha(Path(path)) == expected, 'source changed: '+path
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    def run(command, name, extra=None):
        with (out/name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**env, **(extra or {})}, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        assert code == 0, f'{name} failed; inspect, no automatic restart'
        check_sources()
    save()
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/final_semantic_add5k_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    try:
        while identity is not None and process_identity(pid) == identity:
            time.sleep(5)
        prerequisite = json.loads(prerequisite_path.read_text())
        assert prerequisite['complete'], 'directional queue failed; inspect before new GPU work'
        check_sources()
        if max(prerequisite[s]['improvement_vs_best_control_percent'] for s in ['screen1k', 'confirm5k']) >= 3:
            state.update(complete=True, stage='directional_quality_positive_review_first_skip_last_legacy')
            save()
            return
        state.update(stage='unchanged_semantic_add8_pixel_check', prerequisite_execution_sha256=sha(prerequisite_path))
        save()
        smoke = DATA/'final_semantic_add_smoke8'
        run([sys.executable, str(ROOT/'experiments/sample_raev2_ancestral_guidance.py'), '--output', str(smoke),
             '--samples', '8', '--seed', '202609071', '--parity', '--modes', 'official', 'semantic_add'], 'smoke8.log', {'CUDA_VISIBLE_DEVICES': '0'})
        parity = []
        for mode in ['official', 'semantic_add']:
            path = smoke/f'shard0/{mode}/samples.npz'
            reference = DATA/f'semantic_complement_smoke/shard0/{mode}/samples.npz'
            with np.load(path) as current, np.load(reference) as anchor:
                assert np.array_equal(current['arr_0'], anchor['arr_0']) and np.array_equal(current['ids'], anchor['ids'])
            parity.append({'mode': mode, 'all_pixels_equal': True, 'sample_sha256': sha(path), 'reference_sha256': sha(reference)})
        state.update(stage='unchanged_final_semantic_add5k', parity=parity)
        save()
        run([sys.executable, str(ROOT/'experiments/run_raev2_ancestral_study.py'), '--output', str(target),
             '--samples', '5000', '--seed', '202609072', '--modes', 'semantic_add'], 'sampling.log')
        state.update(stage='independent_full_fid_and_input_audit')
        save()
        run([sys.executable, '-m', 'experiments.audit_raev2_final_semantic_add5k'], 'audit.log',
            {'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
        audit_path = ROOT/'experiments/results/raev2_guidance_20260907/final_semantic_add5k_audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['complete']
        state.update(complete=True, stage='last_candidate_complete_requires_final_review', results=audit['rows'], audit_sha256=sha(audit_path))
        save()
        (ROOT/'experiments/results/raev2_guidance_20260907/final_semantic_add5k_execution.json').write_text(json.dumps(state, indent=2)+'\n')
    except Exception as error:
        state.update(stage='failed_requires_inspection_no_restart', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
