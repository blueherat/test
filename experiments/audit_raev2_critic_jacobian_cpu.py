"""Inspect the model Jacobian in one fixed image-critic direction on CPU.

This is a ten-state numeric diagnostic of a smooth FP32 surrogate, not a
sampler, a quality metric, or a covariance-positivity proof.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'external/RAEv2/src'):
    sys.path.insert(0, str(directory))
import torch
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.sample_raev2_pfr_retiming import load_config, DEFAULT_CONFIG, DEFAULT_CHECKPOINT
from experiments.summarize_raev2_guidance_20260907 import DATA, sha
from utils.model_utils import instantiate_from_config


def scalar(x):
    return float(x.detach())


def main():
    out = DATA / 'critic_jacobian_cpu'
    out.mkdir(exist_ok=False)
    assert not torch.cuda.is_available()
    torch.set_num_threads(16)
    os.environ['DINOV3_CKPT_DIR'] = '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3'
    os.environ['DINOV3_REPO_DIR'] = '/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo'
    install_raev2_decoder_config_compat()
    cfg = load_config(DEFAULT_CONFIG)
    decoder = instantiate_from_config(cfg.stage_1).cpu().eval().requires_grad_(False)
    model = instantiate_from_config(cfg.stage_2).cpu().eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'], strict=True)
    del checkpoint
    probe_path = DATA.parent / 'raev2_guidance_restart_20260906/posterior_cls_probe_v1/probe.pt'
    probe = torch.load(probe_path, map_location='cpu', weights_only=False)
    weight, bias = probe['weight'].double(), float(probe['bias'])
    calibration_path = DATA / 'guided_reverse_variance/calibration.json'
    calibration = json.loads(calibration_path.read_text())['rows']
    cache = DATA.parent / 'raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states'
    paths = sorted(cache.glob('step_*.pt'))
    assert len(paths) == 10
    source = Path(__file__)
    frozen = {str(p): sha(p) for p in [source, DEFAULT_CONFIG, DEFAULT_CHECKPOINT,
                                      probe_path, calibration_path, *paths]}
    (out / source.name).write_bytes(source.read_bytes())
    record = {'complete': False, 'cuda_used': False, 'cpu_threads': 16,
              'scope': 'one fixed image at all ten cached times; no timestep selection or sampler',
              'precision': 'FP32 denoiser/decoder/encoder, FP64 unit CLS; differs from native BF16 sampling',
              'sources': frozen, 'rows': []}
    def save():
        (out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    save()

    def reward(clean):
        pixels = decoder.decode(clean).clamp(0, 1)
        raw = decoder.encoder.model.forward_features(
            decoder.encoder.preprocess(pixels * 255))['x_norm_clstoken'].double()
        return (raw / raw.norm(dim=-1, keepdim=True)) @ weight + bias

    def predict(x, t, labels):
        full, base = model(x, torch.full((len(x),), t), context=labels, attn_mask=None)
        guided = base + 1.78 * (full - base) if t >= .1 else full
        return full, guided

    started = time.perf_counter()
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        t, step = float(payload['t']), int(payload['step_index'])
        state = payload['rollout']['state'][:1].float().detach().requires_grad_(True)
        labels = payload['labels'][:1]
        full, guided = predict(state, t, labels)
        # Separate decoder differentiation keeps the two denoiser VJPs explicit.
        clean = guided.detach().requires_grad_(True)
        value = reward(clean)
        gradient, = torch.autograd.grad(value.sum(), clean)
        jg, = torch.autograd.grad(guided, state, grad_outputs=gradient, retain_graph=True)
        jf, = torch.autograd.grad(full, state, grad_outputs=gradient)
        energy = scalar(gradient.square().sum())
        row = {'step': step, 't': t, 'sample_id': int(payload['sample_ids'][0]),
               'label': int(labels[0]), 'reward': scalar(value[0]),
               'gradient_rms': scalar(gradient.square().mean().sqrt()),
               'guided_vjp_rms': scalar(jg.square().mean().sqrt()),
               'full_vjp_rms': scalar(jf.square().mean().sqrt()),
               'guided_rayleigh_in_critic_direction': scalar((gradient * jg).sum()) / energy,
               'full_rayleigh_in_critic_direction': scalar((gradient * jf).sum()) / energy,
               'mse': calibration[step]['mse'],
               'finite': bool(torch.isfinite(jg).all() and torch.isfinite(jf).all())}
        if t < 1:
            factor = t * t / (1 - t)
            delta = factor * jg
            isotropic = calibration[step]['mse'] * gradient
            row.update(tweedie_factor=factor, tweedie_delta_rms=scalar(delta.square().mean().sqrt()),
                       tweedie_to_isotropic_norm_ratio=scalar(delta.norm() / isotropic.norm()),
                       tweedie_isotropic_cosine=scalar((delta * isotropic).sum() / (delta.norm() * isotropic.norm())),
                       linear_reward_gain=scalar((delta * gradient).sum()))
            with torch.no_grad():
                row['actual_reward_gain_at_corrected_clean'] = scalar(reward(clean + delta)[0] - value[0])
        else:
            row['tweedie_endpoint'] = 'undefined at t=1; nonzero finite-network Jacobian cannot be divided by zero'
        if step == 89:
            direction = jg / jg.norm()
            with torch.no_grad():
                plus = reward(predict(state + .01 * direction, t, labels)[1])
                minus = reward(predict(state - .01 * direction, t, labels)[1])
            row['input_directional_finite_difference'] = scalar((plus[0] - minus[0]) / .02)
            row['input_directional_autograd'] = scalar(jg.norm())
        row['elapsed_seconds'] = time.perf_counter() - started
        record['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
    assert sha(source) == frozen[str(source)]
    record.update(complete=True, seconds=time.perf_counter() - started)
    save()
    dest = ROOT / 'experiments/results/raev2_guidance_20260907/critic_jacobian_cpu.json'
    dest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
