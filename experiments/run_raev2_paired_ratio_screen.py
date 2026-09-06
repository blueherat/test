"""Fixed final critic -> input-gradient audit -> native parity -> paired 1K."""
import json
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation-name', default='paired_ratio_screen_continuation')
    parser.add_argument('--mode', choices=['paired_ratio', 'paired_ratio_calibrated', 'actual_ratio'], default='paired_ratio')
    args = parser.parse_args()
    out = DATA / args.continuation_name
    out.mkdir(exist_ok=False)
    fit_name = 'actual_ratio_fit' if args.mode == 'actual_ratio' else 'paired_ratio_fit'
    training = json.loads((DATA / fit_name / 'execution.json').read_text())
    assert training['complete'] and training['validation']['entry_condition_passed']
    sources = [ROOT / 'experiments' / name for name in
               ('sample_raev2_ancestral_guidance.py', 'run_raev2_ancestral_study.py',
                'raev2_paired_ratio_model.py', 'audit_raev2_paired_ratio_gradient.py',
                'run_raev2_paired_ratio_screen.py')]
    sources.extend((DATA / fit_name / 'critic.pt', DATA / (fit_name + '_plan.json')))
    if args.mode == 'actual_ratio':
        sources.extend((ROOT / 'experiments/audit_raev2_actual_ratio_gradient.py',
                        ROOT / 'experiments/raev2_actual_ratio_data.py'))
    if args.mode == 'paired_ratio_calibrated':
        sources.append(DATA / 'paired_ratio_calibration.json')
    frozen = {str(p): sha(p) for p in sources}
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(),
             'stage': 'gradient_audit', 'sources': frozen, 'training_execution': training}
    def save():
        temp = out / 'state.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'state.json')
    def run(command, logfile, four_gpus=False):
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
        if not four_gpus:
            env['CUDA_VISIBLE_DEVICES'] = '0'
        with (out / logfile).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, command=command)
            save()
            code = child.wait()
        if code:
            state.update(stage='failed_requires_inspection_no_resampling', exit_code=code, complete=True)
            save()
            raise RuntimeError('stage failed, original outputs preserved: ' + logfile)
        for path, digest in frozen.items():
            assert sha(Path(path)) == digest, 'frozen input changed: ' + path
    save()
    if args.mode == 'actual_ratio':
        run([sys.executable, '-m', 'experiments.audit_raev2_actual_ratio_gradient'], 'gradient.log')
    elif args.mode == 'paired_ratio':
        run([sys.executable, '-m', 'experiments.audit_raev2_paired_ratio_gradient'], 'gradient.log')
    else:
        audit = ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_gradient_audit.json'
        previous = json.loads(audit.read_text())
        assert previous['complete'] and previous['checkpoint_sha256'] == sha(DATA / 'paired_ratio_fit/critic.pt')
        state['reused_unchanged_critic_gradient_audit_sha256'] = sha(audit)
    state['stage'] = 'native_and_candidate_smoke'
    smoke = DATA / (args.mode + '_smoke')
    run([sys.executable, str(ROOT / 'experiments/sample_raev2_ancestral_guidance.py'),
         '--output', str(smoke), '--samples', '8', '--seed', '202609071', '--parity',
         '--modes', 'official', args.mode], 'smoke.log')
    with np.load(smoke / 'shard0/official/samples.npz') as a, np.load(
            DATA / 'ancestral_smoke_v2/shard0/official/samples.npz') as b:
        assert np.array_equal(a['arr_0'], b['arr_0']), 'original official pixel parity failed'
    summary = json.loads((smoke / 'shard0' / args.mode / 'summary.json').read_text())
    assert summary['complete'] and summary['sample_model_calls'] == 800 and summary['sample_ratio_backward_calls'] == 800
    state.update(stage='running_fixed_1k', original_official_pixel_parity=True)
    run([sys.executable, str(ROOT / 'experiments/run_raev2_ancestral_study.py'),
         '--output', str(DATA / (args.mode + '_screen1k')), '--samples', '1000',
         '--seed', '202609071', '--modes', args.mode], 'screen.log', four_gpus=True)
    state.update(stage='screen_complete_requires_quality_inspection', complete=True,
                 metrics=json.loads((DATA / (args.mode + '_screen1k') / 'metrics.json').read_text()))
    save()


if __name__ == '__main__':
    main()
