"""Apply the reviewed semantic patch only after the live two-mode 5K ends."""
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
    out = DATA / 'semantic_continuation'
    out.mkdir(exist_ok=False)
    prerequisite = DATA / 'two_mode_confirm5k/execution.json'
    initial = json.loads(prerequisite.read_text())
    identities = {pid: process_identity(pid) for pid in
                  [initial['pid'], *[j['pid'] for j in initial['jobs']]]}
    if not any(identities.values()) and not initial['complete']:
        raise RuntimeError('prerequisite is incomplete and stopped; no automatic restart')
    lock = ROOT / 'experiments/locks/raev2_semantic_complement_20260907'
    manifest = json.loads((lock / 'manifest.json').read_text())
    sources = [lock / 'manifest.json', lock / 'integrate.patch',
               ROOT / 'experiments/raev2_semantic_complement.py',
               ROOT / 'experiments/raev2_semantic_quality_guidance.py',
               ROOT / 'docs/RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md']
    frozen = {str(p): sha(p) for p in sources}
    state = {'pid': os.getpid(), 'complete': False, 'stage': 'waiting_for_two_mode_5k',
             'created_unix': time.time(), 'prerequisite_identities': identities,
             'frozen_sources': frozen, 'integration_manifest': manifest,
             'next_screen': {'samples': 1000, 'seed': 202609071,
                             'modes': ['semantic_add', 'semantic_orthogonal'], 'extra_cfg': .15}}
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
        raise RuntimeError('two-mode 5K failed; original artifacts preserved')
    metrics = json.loads((prerequisite.parent / 'metrics.json').read_text())
    controls = json.loads((DATA / 'weak_confirm5k/metrics.json').read_text())
    best = min(r['fid'] for r in controls if r['branch'] in {'official', 'piecewise'})
    gain = 1 - metrics[0]['fid'] / best
    state.update(two_mode_metrics=metrics, two_mode_gain=gain)
    if gain >= .03:
        state.update(stage='two_mode_quality_signal_requires_independent_confirmation', complete=True)
        save()
        return
    for p, digest in frozen.items():
        assert sha(Path(p)) == digest, 'frozen semantic input changed: ' + p
    audit = json.loads((DATA / 'semantic_complement_cpu/summary.json').read_text())
    assert audit['complete'] and len(audit['rows']) == 80 and all(r['finite'] for r in audit['rows'])
    for p, digests in manifest.items():
        assert sha(ROOT / p) == digests['old_sha256'], 'source changed: ' + p
    subprocess.run(['git', 'apply', '--check', str(lock / 'integrate.patch')], cwd=ROOT, check=True)
    subprocess.run(['git', 'apply', str(lock / 'integrate.patch')], cwd=ROOT, check=True)
    for p, digests in manifest.items():
        assert sha(ROOT / p) == digests['new_sha256'], 'patch mismatch: ' + p
    state.update(stage='integrated_running_semantic_smoke')
    save()
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    smoke = DATA / 'semantic_complement_smoke'
    command = [sys.executable, str(ROOT / 'experiments/sample_raev2_ancestral_guidance.py'),
               '--output', str(smoke), '--samples', '8', '--seed', '202609071',
               '--modes', 'official', 'semantic_add', 'semantic_orthogonal', '--parity']
    with (out / 'smoke.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': '0'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(smoke / 'shard0/official/samples.npz') as a, np.load(
            DATA / 'ancestral_smoke_v2/shard0/official/samples.npz') as b:
        assert np.array_equal(a['arr_0'], b['arr_0']), 'original official pixel parity failed'
    for mode in ('semantic_add', 'semantic_orthogonal'):
        summary = json.loads((smoke / 'shard0' / mode / 'summary.json').read_text())
        assert summary['complete'] and summary['sample_model_calls'] == 8 * 199
        assert summary['extra_sample_unconditional_calls'] == 8 * 99
    state.update(stage='running_semantic_1k', original_official_pixel_parity=True)
    save()
    command = [sys.executable, str(ROOT / 'experiments/run_raev2_ancestral_study.py'),
               '--output', str(DATA / 'semantic_complement_screen1k'), '--samples', '1000',
               '--seed', '202609071', '--modes', 'semantic_add', 'semantic_orthogonal']
    with (out / 'screen.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(child_pid=child.pid, command=command)
        save()
        code = child.wait()
    state.update(complete=True, exit_code=code,
                 stage='semantic_1k_complete' if code == 0 else 'semantic_1k_failed_requires_inspection')
    save()
    if code:
        raise RuntimeError('semantic screen failed; no automatic resampling')


if __name__ == '__main__':
    main()
