"""Freeze the spatial-covariance hypothesis, extract paired rows, then fit once."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_spatial_guidance_covariance import PLAN
from experiments.sample_raev2_pfr_retiming import DEFAULT_CHECKPOINT
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-training', type=Path,
                        help='reuse the verified training bank from the metadata-key failure only')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(),
        ROOT/'experiments/extract_raev2_spatial_guidance_covariance.py',
        ROOT/'experiments/calibrate_raev2_spatial_guidance_covariance.py',
        ROOT/'experiments/raev2_spatial_guidance_covariance.py',
        ROOT/'experiments/raev2_directional_variance.py', ROOT/'experiments/raev2_conditional_variance.py',
        ROOT/'experiments/sample_raev2_ancestral_guidance.py',
        ROOT/'experiments/sample_raev2_pfr_retiming.py',
        ROOT/'external/RAEv2/src/stage2/models/DDT.py', ROOT/'external/RAEv2/src/stage2/models/model_utils.py',
        ROOT/'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml',
        ROOT/'docs/RAEV2_SPATIAL_GUIDANCE_COVARIANCE_20260907_ZH.md']
    sources = {str(p): sha(p) for p in source_paths}
    for p in source_paths:
        dest = out/'frozen_source'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    native = json.loads((DATA/'actual_ratio_bank64k/train/shard0/request.json').read_text())
    assert sha(DEFAULT_CHECKPOINT) == native['checkpoint_sha256']
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    state = {'complete': False, 'extraction_complete': False, 'plan': PLAN, 'sources': sources,
             'pid': os.getpid(), 'pid_starttime': Path('/proc/self/stat').read_text().split()[21],
             'checkpoint_sha256_verified': native['checkpoint_sha256'], 'splits': {}, 'jobs': []}
    began = time.perf_counter()
    def save():
        temp = out/'preparation.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'preparation.json')
    save()
    jobs = []
    try:
        if args.reuse_training is not None:
            previous = args.reuse_training.resolve()
            old = json.loads((previous/'preparation.json').read_text())
            assert old['error'] == "RuntimeError('validation extraction failed; inspect shard logs')"
            assert old['splits']['train']['complete'] and not old['extraction_complete']
            expected_old_plan = dict(PLAN)
            expected_old_plan['validation'] = expected_old_plan.pop('validation_uncertainty')
            assert old['plan'] == expected_old_plan
            geometry = ROOT/'experiments/raev2_spatial_guidance_covariance.py'
            def computation_ast(path):
                tree = ast.parse(path.read_text())
                tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'PLAN' for t in node.targets))]
                return ast.dump(tree, include_attributes=False)
            assert computation_ast(geometry) == computation_ast(previous/'frozen_source'/geometry.relative_to(ROOT))
            for source, digest in old['sources'].items():
                if Path(source) not in (Path(__file__).resolve(), geometry):
                    assert sha(Path(source)) == digest
            original = previous/'moments/train/moments.npz'
            assert sha(original) == old['splits']['train']['moments_sha256']
            target = out/'moments/train/moments.npz'
            target.parent.mkdir(parents=True)
            os.link(original, target)
            state['splits']['train'] = {**old['splits']['train'], 'reused_from': str(original),
                'computation_ast_identical_excluding_plan_metadata': True}
            save()
        for split in ['train', 'validation']:
            if split in state['splits']:
                continue
            jobs = []
            for rank in range(4):
                command = [sys.executable, str(ROOT/'experiments/extract_raev2_spatial_guidance_covariance.py'),
                           '--output', str(out/'moments'), '--split', split, '--shard', str(rank)]
                log = (out/f'{split}_shard{rank}.log').open('w')
                child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)},
                                         stdout=log, stderr=subprocess.STDOUT)
                jobs.append((child, log))
                state['jobs'].append({'split': split, 'rank': rank, 'pid': child.pid,
                    'pid_starttime': Path(f'/proc/{child.pid}/stat').read_text().split()[21], 'command': command})
                save()
            while any(child.poll() is None for child, _ in jobs):
                if any(child.poll() not in (None, 0) for child, _ in jobs):
                    raise RuntimeError(f'{split} extraction failed; inspect shard logs')
                time.sleep(2)
            if any(child.returncode for child, _ in jobs):
                raise RuntimeError(f'{split} extraction failed')
            for child, log in jobs:
                child.wait(); log.close()
            shards, summaries = [], []
            for rank in range(4):
                folder = out/'moments'/split/f'shard{rank}'
                record = json.loads((folder/'summary.json').read_text())
                assert record['complete'] and sha(folder/'moments.npz') == record['files']['moments.npz']
                with np.load(folder/'moments.npz') as data:
                    shards.append({k: data[k] for k in data.files})
                summaries.append(record)
            merged = {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}
            order = np.argsort(merged['ids'])
            merged = {k: v[order] for k, v in merged.items()}
            assert np.array_equal(merged['ids'], np.arange(PLAN[split]))
            dest = out/'moments'/split/'moments.npz'
            np.savez(dest, **merged)
            state['splits'][split] = {'complete': True, 'count': len(merged['ids']),
                'active': int(merged['active'].sum()), 'moments_sha256': sha(dest),
                'teacher_sample_main_calls': sum(s['teacher_sample_main_calls'] for s in summaries),
                'extraction_seconds_sum': sum(s['seconds_including_data_access'] for s in summaries),
                'all_cached_features_mse_scalar_and_global_projection_bitwise_equal': True}
            save()
        assert all(sha(Path(p)) == digest for p, digest in sources.items()), 'source changed during extraction'
        state['extraction_complete'] = True
        save()
        shutil.copy2(out/'preparation.json', out/'preparation_before_fit.json')
        subprocess.run([sys.executable, str(ROOT/'experiments/calibrate_raev2_spatial_guidance_covariance.py'),
                        '--folder', str(out)], cwd=ROOT, env=env, check=True)
        report = json.loads((out/'fit/calibration.json').read_text())
        state.update(complete=True, entry_condition_passed=report['entry_condition_passed'],
                     calibration_sha256=sha(out/'fit/calibration.json'), seconds=time.perf_counter()-began)
        save()
        print(json.dumps({k: v for k, v in state.items() if k not in ['sources', 'jobs', 'plan']}, indent=2), flush=True)
    except BaseException as error:
        state.update(error=repr(error), seconds=time.perf_counter()-began)
        save()
        for child, _ in jobs:
            if child.poll() is None:
                child.terminate()
        raise
    finally:
        for child, log in jobs:
            child.wait()
            log.close()


if __name__ == '__main__':
    main()
