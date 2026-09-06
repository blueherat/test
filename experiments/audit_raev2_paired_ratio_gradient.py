"""Check the fitted critic's actual input gradient before image sampling."""
import json
import copy
import torch
import numpy as np
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.train_raev2_paired_ratio import Banks
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = DATA / 'paired_ratio_fit/critic.pt'
    trained = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert trained['validation']['entry_condition_passed']
    model = PairedRatioCritic(trained['state_dict']['class_features']).cuda().eval().requires_grad_(False)
    model.load_state_dict(trained['state_dict'])
    reference = copy.deepcopy(model).double()
    manifest = json.loads((DATA / 'paired_ratio_data/manifest.json').read_text())
    bank = Banks(manifest['splits']['validation'])
    labels_np = np.arange(8)
    _, fake = bank.batch(labels_np, np.zeros(8, dtype=int), np.zeros(8, dtype=int), 'cuda')
    labels = torch.from_numpy(labels_np).cuda()
    generator = torch.Generator(device='cuda').manual_seed(202609088)
    noise = torch.randn(fake.shape, device='cuda', generator=generator)
    rows = []
    for value in [1., .9987389445304871, .9, .5, .07476635277271271]:
        times = torch.full((8,), value, device='cuda')
        state = (1 - value) * fake + value * noise
        correction = model.clean_correction(state, times, labels)
        gradient = correction / value**2
        norms = gradient.flatten(1).norm(dim=1)
        assert (norms > 0).all() and torch.isfinite(gradient).all()
        direction = gradient / norms.reshape(-1, 1, 1, 1)
        with torch.no_grad():
            plus = model(state + .01 * direction, times, labels)
            minus = model(state - .01 * direction, times, labels)
        fd = (plus - minus) / .02
        relative = ((fd - norms).abs() / norms).max().item()
        # Small derivatives of an almost saturated FP32 scalar cannot be
        # resolved reliably by subtracting nearby FP32 outputs. Check the
        # identical stored weights and inputs in FP64, and separately compare
        # the actual deployed FP32 gradient with the FP64 gradient.
        x64 = state.double().requires_grad_(True)
        g64, = torch.autograd.grad(reference(x64, times.double(), labels).sum(), x64)
        norm64 = g64.flatten(1).norm(dim=1)
        gradient_error = ((gradient.double() - g64).flatten(1).norm(dim=1) / norm64).max().item()
        direction64 = g64 / norm64.reshape(-1, 1, 1, 1)
        errors64 = {}
        with torch.no_grad():
            for epsilon in (.01, .001):
                f64 = (reference(x64 + epsilon * direction64, times.double(), labels) -
                       reference(x64 - epsilon * direction64, times.double(), labels)) / (2 * epsilon)
                errors64[str(epsilon)] = ((f64 - norm64).abs() / norm64).max().item()
        assert gradient_error < .002, ('deployed gradient differs from FP64', value, gradient_error)
        assert errors64['0.001'] < .002, ('FP64 finite difference mismatch', value, errors64)
        rows.append({'time': value, 'fp32_finite_difference_relative_error': relative,
                     'fp32_gradient_vs_fp64_relative_error': gradient_error,
                     'fp64_finite_difference_relative_errors': errors64,
                     'mean_gradient_norm': norms.mean().item(),
                     'correction_rms': correction.square().mean().sqrt().item()})
    with torch.no_grad():
        model.output[-1].weight.zero_()
        model.output[-1].bias.zero_()
    assert torch.count_nonzero(model.clean_correction(noise, torch.ones(8, device='cuda'), labels)) == 0
    result = {'complete': True, 'checkpoint_sha256': sha(checkpoint), 'rows': rows,
              'original_fp32_only_preflight_failed_at_t_half': True,
              'zero_head_gives_exact_zero_correction': True, 'includes_pure_noise_endpoint': True,
              'fid_used': False}
    path = ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_gradient_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
