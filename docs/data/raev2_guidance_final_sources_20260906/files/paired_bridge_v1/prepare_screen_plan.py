"""Freeze reviewed sources and identities before the paired-bridge screen."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/zhoushunyu/eqvae')
sys.path.insert(0, str(ROOT))
from experiments import sample_raev2_paired_bridge as sampler
import torch

P = Path(__file__).resolve().parent
S = P/'screen_v1'
PROTOCOL = ROOT/'docs/RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md'
EXTRA = (
    'experiments/run_raev2_paired_bridge_screen.py',
    'experiments/evaluate_raev2_official_samples.py',
    'experiments/raev2_training_core.py',
    'tests/test_raev2_paired_bridge.py',
    'docs/RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md',
    'docs/RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md',
    'docs/RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md',
    'docs/RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md',
    'docs/RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md',
    'external/RAEv2/src/stage1/__init__.py',
    'external/RAEv2/src/stage1/decoders/__init__.py',
    'external/RAEv2/src/stage1/decoders/decoder.py',
    'external/RAEv2/src/stage1/decoders/utils.py',
    'external/RAEv2/configs/decoder/ViTXL/config.json',
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    started = time.perf_counter()
    if S.exists():
        raise FileExistsError('refusing to replace screen preparation')
    config = sampler.load_config(sampler.DEFAULT_CONFIG)
    paths = [ROOT/relative for relative in dict.fromkeys((*sampler.SOURCE_FILES, *EXTRA))]
    paths += [sampler.DEFAULT_CONFIG, sampler.DEFAULT_CHECKPOINT,
              Path(config.stage_1.params['pretrained_decoder_path']),
              Path(config.stage_1.params['normalization_stat_path']),
              sampler.DEFAULT_BRIDGE_CHECKPOINT, P/'train/request.json', P/'train/summary.json',
              P/'supplemental_source_environment.json', Path(__file__).resolve(),
              Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'),
              Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')]
    frozen = {str(p.resolve()): sha(p) for p in paths}
    expected = {
        ROOT/'experiments/sample_raev2_paired_bridge.py': '93820293074739d4b794c28a8be16d2f4e1a42e1b7e7a1465f3603363895194e',
        ROOT/'tests/test_sample_raev2_paired_bridge.py': '08be2dea3d24fc748a015af3f1333f5f60efb549ed3a3d3486b47748f173744e',
        sampler.DEFAULT_BRIDGE_CHECKPOINT: sampler.BRIDGE_SHA256,
        sampler.DEFAULT_CHECKPOINT: sampler.BASELINE_SHA256,
        sampler.DEFAULT_CONFIG: sampler.CONFIG_SHA256,
    }
    for p, value in expected.items():
        if frozen[str(p.resolve())] != value:
            raise ValueError('reviewed identity changed: '+str(p))
    evaluator = Path('/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals')
    commit = subprocess.check_output(['git', '-C', str(evaluator), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(evaluator), 'status', '--porcelain'], text=True)
    if commit != '19dfb4c2705333eb8b97e454fb354d47d1fe135b' or dirty:
        raise ValueError('evaluator source changed')
    for p in (evaluator/'fd_evaluator').rglob('*.py'):
        frozen[str(p.resolve())] = sha(p)
    S.mkdir()
    archives = {}
    for p in paths:
        # Copy source/protocol identities without duplicating multi-GB weights.
        if p.is_relative_to(ROOT) and p.suffix in ('.py', '.md', '.yaml', '.json'):
            relative = p.relative_to(ROOT)
            target = S/'sources'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            archives[str(relative)] = {'path': str(target), 'sha256': sha(target)}
    value = {
        'protocol': 'raev2_paired_bridge_fixed_1k_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_freeze_complete': True,
        'protocol_document': {'path': str(PROTOCOL), 'sha256': sha(PROTOCOL), 'size_bytes': PROTOCOL.stat().st_size},
        'cohort': {'seed': 202609151, 'n': 1000, 'batch_size': 8},
        'gpu_uuid': 'GPU-7d3e4e7d-abfa-e06e-c264-796052797949',
        'arms': ['official100', 'candidate100', 'control100', 'officialK_from_pre_FID_cost_rule'],
        'frozen_files': frozen, 'source_archives': archives,
        'evaluator': {'root': str(evaluator), 'commit': commit, 'clean': True},
        'pre_gpu_checks': {'module_and_sampling_tests_passed': 27, 'pytest_wall_seconds': 8.03,
            'independent_static_review': 'paired inputs/native arithmetic/current midpoint/hooks and corrected official auxiliary loading reviewed; no remaining implementation blocker'},
        'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'python': sys.version, 'python_executable': sys.executable},
        'full_total_cost_closed': False,
        'cost_scope': 'Inference-matched 1K screen; all training/preparation/diagnostics disclosed separately. Goal cannot be completed from this screen alone.',
        'freeze_wall_seconds': time.perf_counter()-started,
    }
    (S/'plan.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'path': str(S/'plan.json'), 'sha256': sha(S/'plan.json'), 'frozen_file_count': len(frozen), 'freeze_wall_seconds': value['freeze_wall_seconds']}))

if __name__ == '__main__':
    main()
