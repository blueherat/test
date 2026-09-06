"""Confirm the frozen moment correction at 5K, irrespective of its 1K score.

This decision is recorded before the candidate's first 1K FID. Moment changes
can change finite-N FID rankings; the existing independent 5K controls are
reused with exactly the same input identity, without selecting images.
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


def main():
    out = DATA / 'two_mode_5k_continuation'
    out.mkdir(exist_ok=False)
    prereq = DATA / 'two_mode_screen1k/execution.json'
    initial = json.loads(prereq.read_text())
    ids = {pid: process_identity(pid) for pid in
           [initial['pid'], *[j['pid'] for j in initial['jobs']]]}
    if not any(ids.values()) or (prereq.parent / 'metrics.csv').exists():
        raise RuntimeError('this fixed 5K decision must precede the candidate 1K FID')
    sources = {**initial['sources'],
               str(DATA / 'two_mode_ratio/finite_euler_calibration.json'):
               sha(DATA / 'two_mode_ratio/finite_euler_calibration.json')}
    baseline_study = DATA / 'weak_confirm5k'
    assert json.loads((baseline_study / 'execution.json').read_text())['complete']
    baseline = json.loads((baseline_study / 'official/summary.json').read_text())
    state = {'pid': os.getpid(), 'complete': False, 'stage': 'waiting_for_two_mode_1k',
             'created_unix': time.time(), 'decision_before_candidate_1k_fid': True,
             'decision': 'run unchanged 5K regardless of finite-1K ranking; no parameter revision',
             'prerequisite_identities': ids, 'frozen_sources': sources,
             'samples': 5000, 'seed': 202609072,
             'existing_control_study': str(baseline_study),
             'expected_input_sha256': baseline['paired_noise_labels_sha256']}
    def save():
        temporary = out / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'state.json')
    save()
    while any(v is not None and process_identity(p) == v for p, v in ids.items()):
        time.sleep(5)
    assert json.loads(prereq.read_text())['complete'], '1K prerequisite failed; no resampling'
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, 'frozen source changed: ' + path
    target = DATA / 'two_mode_confirm5k'
    command = [sys.executable, str(ROOT / 'experiments/run_raev2_ancestral_study.py'),
               '--output', str(target), '--samples', '5000', '--seed', '202609072',
               '--modes', 'two_mode']
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    with (out / 'screen.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(stage='running_frozen_two_mode_5k', child_pid=child.pid, command=command)
        save()
        code = child.wait()
    state['exit_code'] = code
    if code:
        state.update(stage='5k_failed_requires_inspection', complete=True)
        save()
        raise RuntimeError('5K failed; original partial outputs preserved')
    summary = json.loads((target / 'two_mode/summary.json').read_text())
    assert summary['paired_noise_labels_sha256'] == state['expected_input_sha256']
    metrics = json.loads((target / 'metrics.json').read_text())
    controls = json.loads((baseline_study / 'metrics.json').read_text())
    control_fid = min(r['fid'] for r in controls if r['branch'] in {'official', 'piecewise'})
    state.update(stage='5k_complete_requires_quality_and_cost_review', complete=True,
                 metrics=metrics, paired_controls=controls,
                 improvement_vs_best_control=1 - metrics[0]['fid'] / control_fid)
    save()


if __name__ == '__main__':
    main()
