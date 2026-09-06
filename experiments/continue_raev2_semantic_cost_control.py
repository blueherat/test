"""If fixed semantic 5K passes 3%, run the pre-FID Heun cost control."""
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
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA / 'semantic_cost_continuation'
    out.mkdir(exist_ok=False)
    prerequisite = DATA / 'semantic_confirm5k/execution.json'
    initial = json.loads(prerequisite.read_text())
    identities = {pid: process_identity(pid) for pid in
                  [initial['pid'], *[j['pid'] for j in initial['jobs']]]}
    assert any(identities.values()), 'confirm the original sampling job is still live'
    assert not (prerequisite.parent / 'metrics.csv').exists(), 'cost plan must precede the 5K score'
    lock = ROOT / 'experiments/locks/raev2_heun_cost_20260907'
    plan_path = DATA / 'semantic_cost_plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['fixed_before_semantic_5k_fid'] and plan['steps'] == 101
    manifest = json.loads((lock / 'manifest.json').read_text())
    sources = [lock / 'manifest.json', lock / 'integrate.patch',
               ROOT / 'experiments/raev2_heun_cost_control.py', plan_path]
    frozen = {str(p): sha(p) for p in sources}
    state = {'pid': os.getpid(), 'complete': False, 'stage': 'waiting_for_semantic_5k',
             'created_unix': time.time(), 'prerequisite_identities': identities,
             'frozen_sources': frozen, 'integration_manifest': manifest, 'cost_plan': plan}
    def save():
        temporary = out / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'state.json')
    save()
    while any(v is not None and process_identity(p) == v for p, v in identities.items()):
        time.sleep(5)
    if not json.loads(prerequisite.read_text())['complete']:
        state.update(stage='prerequisite_failed_requires_inspection', complete=True)
        save()
        raise RuntimeError('original semantic 5K failed; no resampling')
    metric = json.loads((prerequisite.parent / 'metrics.json').read_text())[0]
    controls = json.loads((DATA / 'weak_confirm5k/metrics.json').read_text())
    best = min(r['fid'] for r in controls if r['branch'] in {'official', 'piecewise'})
    candidate_summary = json.loads((prerequisite.parent / 'semantic_orthogonal/summary.json').read_text())
    baseline_summary = json.loads((DATA / 'weak_confirm5k/official/summary.json').read_text())
    assert candidate_summary['paired_noise_labels_sha256'] == baseline_summary['paired_noise_labels_sha256']
    gain = 1 - metric['fid'] / best
    state.update(semantic_5k_metric=metric, improvement_vs_best_original_control=gain)
    if gain < .03:
        state.update(stage='quality_below_three_percent_no_cost_followup', complete=True)
        save()
        return
    for p, digest in frozen.items():
        assert sha(Path(p)) == digest, 'frozen cost input changed: ' + p
    for p, digests in manifest.items():
        assert sha(ROOT / p) == digests['old_sha256'], 'source changed: ' + p
    subprocess.run(['git', 'apply', '--check', str(lock / 'integrate.patch')], cwd=ROOT, check=True)
    subprocess.run(['git', 'apply', str(lock / 'integrate.patch')], cwd=ROOT, check=True)
    for p, digests in manifest.items():
        assert sha(ROOT / p) == digests['new_sha256'], 'patch mismatch: ' + p
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    state.update(stage='integrated_running_cost_smokes')
    save()
    for name, mode, steps in [('official_parity', 'official', 100), ('heun', 'heun', 101)]:
        smoke = DATA / ('semantic_cost_smoke_' + name)
        command = [sys.executable, str(ROOT / 'experiments/sample_raev2_ancestral_guidance.py'),
                   '--output', str(smoke), '--samples', '8', '--seed', '202609071',
                   '--modes', mode, '--steps', str(steps), '--parity']
        with (out / (name + '_smoke.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': '0'},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if mode == 'official':
            with np.load(smoke / 'shard0/official/samples.npz') as a, np.load(
                    DATA / 'ancestral_smoke_v2/shard0/official/samples.npz') as b:
                assert np.array_equal(a['arr_0'], b['arr_0']), 'original official pixel parity failed'
        else:
            summary = json.loads((smoke / 'shard0/heun/summary.json').read_text())
            assert summary['complete'] and summary['sample_model_calls'] == 8 * 201
            assert summary['extra_sample_corrector_calls'] == 8 * 100
    state.update(stage='running_heun_5k_cost_control', original_official_pixel_parity=True)
    save()
    target = DATA / 'semantic_heun_cost5k'
    command = [sys.executable, str(ROOT / 'experiments/run_raev2_ancestral_study.py'),
               '--output', str(target), '--samples', '5000', '--seed', '202609072',
               '--modes', 'heun', '--steps', '101']
    with (out / 'screen.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(child_pid=child.pid, command=command)
        save()
        code = child.wait()
    if code:
        state.update(complete=True, exit_code=code, stage='heun_cost_failed_requires_inspection')
        save()
        raise RuntimeError('cost study failed; no automatic resampling')
    summary = json.loads((target / 'heun/summary.json').read_text())
    assert summary['paired_noise_labels_sha256'] == candidate_summary['paired_noise_labels_sha256']
    heun = json.loads((target / 'metrics.json').read_text())[0]
    state.update(complete=True, exit_code=0, stage='cost_complete_requires_final_goal_audit',
                 heun_metric=heun, improvement_vs_heun=1 - metric['fid'] / heun['fid'])
    save()


if __name__ == '__main__':
    main()
