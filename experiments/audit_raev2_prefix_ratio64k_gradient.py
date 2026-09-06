"""Held-out native-state gradient audit, only after fixed fit entry passes.

FP64 reference explicitly replaces RMSNorm's hard-coded float cast and creates
a float64 attention mask. The unmodified production FP32 prefix is compared
against that ideal-arithmetic reference, then the reference gets a central FD.
"""
import copy
import json
import time
from types import MethodType
import numpy as np
import torch
from torch import nn
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config
from experiments.raev2_actual_ratio_data import ActualPairs
from experiments.raev2_prefix_ratio_features import prefix_features
from experiments.raev2_prefix_ratio_guidance import PrefixRatioHead, prefix_fp32
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha
from stage2.models.model_utils import RMSNorm


class PrefixReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model_depth = model.base_model_depth
        self.s_embedder = copy.deepcopy(model.s_embedder)
        self.t_embedder = copy.deepcopy(model.t_embedder)
        self.ctx_embedder = copy.deepcopy(model.ctx_embedder)
        self.blocks = copy.deepcopy(model.blocks[:self.base_model_depth])
        self.enc_rope = copy.deepcopy(model.enc_rope)
        assert not model.use_cfg_conds
        self.double().eval().requires_grad_(False)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.forward = MethodType(reference_rms_forward, module)

    def _build_sequence(self, state, times, kwargs):
        base, time_tokens = self.t_embedder(times, return_base_embed=True)
        return torch.cat((self.s_embedder(state), time_tokens, self.ctx_embedder(kwargs['context'])), dim=1), base

    def _build_attn_mask(self, sequence, kwargs):
        assert kwargs['attn_mask'] is None
        return torch.zeros((len(sequence), 1, 1, sequence.shape[1]), device=sequence.device, dtype=sequence.dtype)


def reference_rms_forward(module, value):
    return module._norm(value)*module.weight


def main():
    out = DATA/'prefix_ratio64k_gradient_audit'
    out.mkdir(exist_ok=False)
    fitted_path = DATA/'prefix_ratio64k_fit/head.pt'
    record_path = DATA/'prefix_ratio64k_fit/execution.json'
    fit = json.loads(record_path.read_text())
    assert fit['complete'] and fit['optimizer_converged'] and fit['validation']['entry_condition_passed']
    assert sha(fitted_path) == fit['checkpoint_sha256']
    source_paths = [ROOT/'experiments/audit_raev2_prefix_ratio64k_gradient.py',
                    ROOT/'experiments/raev2_prefix_ratio_guidance.py', ROOT/'experiments/raev2_prefix_ratio_features.py']
    sources = {str(p): sha(p) for p in source_paths}
    plan = {'queries': [0, 1, 25, 75, 99], 'selection': 'first whole ascending held-out B8 at each preselected query; first native noise B8 at t1',
            'fp32_vs_fp64_gradient_relative_tolerance': .002, 'fp64_fd_relative_tolerance': .002,
            'central_difference_total_l2_step_per_image': .001,
            'zero_head_correction_must_be_exact_zero': True,
            'fp64_reference': 'same weights/buffers; preserve double RMSNorm arithmetic and use double zero attention mask',
            'fid_used': False, 'sources': sources, 'checkpoint_sha256': sha(fitted_path)}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    head = PrefixRatioHead(fitted_path).cuda().eval()
    reference = PrefixReference(model)
    real = json.loads((DATA/'real_ratio_bank64k/execution.json').read_text())
    bank = ActualPairs(DATA/'actual_ratio_bank64k/validation', real['splits']['validation'], 8000)
    request = json.loads((DATA/'actual_ratio_bank64k/validation/shard0/request.json').read_text())
    rows = []
    began = time.perf_counter()
    for index in plan['queries']:
        value = request['query_grid'][index]
        if index == 0:
            ids = np.arange(8)
            state = torch.from_numpy(np.array(bank.noise[0][:8], copy=True)).cuda()
            labels = torch.from_numpy(ids).cuda()
        else:
            ids = np.flatnonzero(bank.times == np.float32(value))[:8]
            assert len(ids) == 8 and np.array_equal(ids, np.arange(ids[0], ids[0]+8)) and ids[0]%8 == 0
            _, state, _, labels = bank.batch(ids, np.zeros(8, dtype=int), 'cuda')
        times = torch.full((8,), value, device='cuda')
        # Exercise the same nested no_grad/autocast surroundings as sampling.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            correction = head.clean_correction(model, state, times, labels)
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        gradient = correction.double()/times.double()[:, None, None, None].square()
        x64 = state.double().requires_grad_(True)
        with prefix_fp32('cuda'):
            potential64 = head.from_features(prefix_features(reference, x64, times.double(), labels))
            g64, = torch.autograd.grad(potential64.sum(), x64)
            norm = g64.flatten(1).norm(dim=1)
            assert torch.isfinite(g64).all() and (norm > 0).all()
            relative = ((gradient-g64).flatten(1).norm(dim=1)/norm).max().item()
            direction = g64/norm[:, None, None, None]
            with torch.no_grad():
                high = head.from_features(prefix_features(reference, x64+.001*direction, times.double(), labels))
                low = head.from_features(prefix_features(reference, x64-.001*direction, times.double(), labels))
                fd = (high-low)/.002
            fd_error = ((fd-norm).abs()/norm).max().item()
        row = {'query_index': index, 'time': value, 'ids': ids.tolist(),
               'fp32_vs_ideal_fp64_gradient_relative_error': relative,
               'fp64_central_difference_relative_error': fd_error,
               'correction_rms': correction.square().mean().sqrt().item(),
               'seconds_since_start': time.perf_counter()-began}
        rows.append(row)
        (out/'progress.json').write_text(json.dumps({'plan': plan, 'rows': rows}, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        assert relative < .002 and fd_error < .002, (index, relative, fd_error)
        del x64, g64, potential64, direction
    original_weight = head.weight.clone()
    head.weight.zero_()
    zero = head.clean_correction(model, state, times, labels)
    assert torch.count_nonzero(zero).item() == 0
    head.weight.copy_(original_weight)
    for path, digest in sources.items():
        assert sha(ROOT/path) == digest
    result = {'complete': True, 'plan': plan, 'rows': rows, 'zero_head_correction_exact_zero': True,
              'native_precision_flags_restored': True, 'elapsed_gpu_worker_seconds': time.perf_counter()-began,
              'fid_used': False, 'no_images_generated': True}
    (out/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_gradient_audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
