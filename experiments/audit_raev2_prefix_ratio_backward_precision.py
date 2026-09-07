"""Localize the first prefix audit failure without modifying weights or inputs."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.audit_raev2_prefix_ratio64k_gradient import PrefixReference
from experiments.raev2_prefix_ratio_features import prefix_features
from experiments.raev2_prefix_ratio_guidance import PrefixRatioHead, prefix_fp32
from experiments.sample_raev2_ancestral_guidance import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, instantiate_from_config
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    out = DATA/'prefix_ratio64k_backward_precision_diagnostic'
    out.mkdir(exist_ok=False)
    files = [ROOT/'experiments/raev2_prefix_ratio_guidance.py', Path(__file__).resolve()]
    sources = {str(path): sha(path) for path in files}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = instantiate_from_config(load_config(DEFAULT_CONFIG).stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    head_path = DATA/'prefix_ratio64k_fit/head.pt'
    head = PrefixRatioHead(head_path).cuda().eval()
    reference = PrefixReference(model)
    native = DATA/'actual_ratio_bank64k/validation/shard0'
    # Exact first whole native B8 used by the failed audit, not new noise.
    noise = np.load(native/'noise.npy', mmap_mode='r')
    state = torch.from_numpy(np.array(noise[:8], copy=True)).cuda()
    labels = torch.arange(8, device='cuda')
    times = torch.ones(8, device='cuda')
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original = head.clean_correction(model, state, times, labels)
        with prefix_fp32('cuda'):
            protected = head.clean_correction(model, state, times, labels)
    x64 = state.double().requires_grad_(True)
    with prefix_fp32('cuda'):
        f64 = head.from_features(prefix_features(reference, x64, times.double(), labels))
        gradient64, = torch.autograd.grad(f64.sum(), x64)
    norm = gradient64.flatten(1).norm(dim=1)
    original_error = ((original.double()-gradient64).flatten(1).norm(dim=1)/norm).max().item()
    protected_error = ((protected.double()-gradient64).flatten(1).norm(dim=1)/norm).max().item()
    result = {'complete': True, 'checkpoint_sha256': sha(head_path), 'sources': sources,
              'same_heldout_initial_noise_b8': True, 'original_relative_error': original_error,
              'forward_and_backward_protected_relative_error': protected_error,
              'original_threshold': .002, 'protection_restored_original_precision_flags':
                  torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32,
              'seconds': time.perf_counter()-started,
              'no_weights_data_formula_tolerances_or_samples_changed': True}
    assert original_error > .002 and protected_error < .002
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    (out/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_backward_precision_diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
