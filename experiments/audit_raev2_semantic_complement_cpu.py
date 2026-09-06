"""All ten cached times and all eight images: condition/depth redundancy.

FP32 CPU diagnostic only, no strength/window selection and no FID.
"""
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'external/RAEv2/src'):
    sys.path.insert(0, str(p))
import torch
from experiments.raev2_semantic_complement import semantic_correction
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config
from experiments.summarize_raev2_guidance_20260907 import DATA, sha
from utils.model_utils import instantiate_from_config


@torch.no_grad()
def main():
    out = DATA / 'semantic_complement_cpu'
    out.mkdir(exist_ok=False)
    assert not torch.cuda.is_available()
    torch.set_num_threads(16)
    cfg = load_config(DEFAULT_CONFIG)
    model = instantiate_from_config(cfg.stage_2).cpu().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'], strict=True)
    del checkpoint
    cache = DATA.parent / 'raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states'
    inputs = sorted(cache.glob('step_*.pt'))
    fixed = [Path(__file__), ROOT / 'experiments/raev2_semantic_complement.py',
             ROOT / 'experiments/raev2_semantic_quality_guidance.py', DEFAULT_CONFIG,
             DEFAULT_CHECKPOINT, *inputs]
    record = {'complete': False, 'cuda_used': False, 'precision': 'FP32 CPU surrogate, not native BF16',
              'purpose': 'all 80 pre-existing records; no strength/window selection, no FID',
              'sources': {str(p): sha(p) for p in fixed}, 'rows': []}
    def save():
        (out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    began = time.perf_counter()
    for path in inputs:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload['rollout']['state'].float()
        labels = payload['labels']
        t = float(payload['t'])
        times = torch.full((len(state),), t)
        full, base = model(state, times, context=labels, attn_mask=None)
        null, _ = model(state, times, context=torch.full_like(labels, 1000), attn_mask=None)
        depth, cond = full - base, full - null
        delta = semantic_correction(full, base, null)
        orthogonal = semantic_correction(full, base, null, orthogonal=True)
        dims = tuple(range(1, state.ndim))
        depth_norm = depth.square().sum(dims).sqrt()
        cond_norm = cond.square().sum(dims).sqrt()
        dot = (depth * cond).sum(dims)
        for i, sample_id in enumerate(payload['sample_ids']):
            record['rows'].append({
                'step': int(payload['step_index']), 't': t, 'sample_id': int(sample_id),
                'label': int(labels[i]), 'cosine': float(dot[i] / (depth_norm[i] * cond_norm[i])),
                'retained_class_energy_fraction': float(orthogonal[i].square().sum() / delta[i].square().sum()),
                'add_to_ig_norm_ratio': float(delta[i].norm() / (.78 * depth_norm[i])),
                'orth_to_ig_norm_ratio': float(orthogonal[i].norm() / (.78 * depth_norm[i])),
                'normalized_orthogonality_residual': float((orthogonal[i] * depth[i]).sum().abs() /
                    (orthogonal[i].norm() * depth_norm[i])),
                'finite': bool(torch.isfinite(orthogonal[i]).all()),
            })
        save()
        print(json.dumps({'step': int(payload['step_index']), 't': t,
                          'records': len(record['rows']), 'seconds': time.perf_counter() - began}), flush=True)
    assert len(record['rows']) == 80
    record.update(complete=True, cpu_threads=16, seconds=time.perf_counter() - began)
    save()
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (ROOT / 'experiments/results/raev2_guidance_20260907/semantic_complement_cpu.json').write_text(
        json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
