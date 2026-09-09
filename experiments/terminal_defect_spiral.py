#!/usr/bin/env python3
"""Finite-suffix signed distribution defects on existing frozen spiral heads.

Restart telescoping is an identity check, not a prediction metric validation.
All expensive outputs go to an explicit new data directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import run_dual_target_closed_loop_spiral_toy as spiral
from subspace_gate_distribution import make_distribution

core = spiral.core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def defect_statistics(features):
    """features: [restart, paired sample, fixed feature]."""
    means = np.asarray(features, dtype=np.float64).mean(axis=1)
    defects = means[:-1] - means[1:]
    total = means[0] - means[-1]
    gram = defects @ defects.T
    energy = float(total @ total)
    diagonal = float(np.trace(gram))
    lengths = np.linalg.norm(defects, axis=1).sum()
    result = {
        'terminal_feature_gap_squared': energy,
        'diagonal_defect_energy': diagonal,
        'cross_time_energy': float(gram.sum() - diagonal),
        'cancellation_fraction': float(1 - np.linalg.norm(total) / lengths) if lengths else 0.,
        'telescoping_max_abs': float(np.abs(defects.sum(axis=0) - total).max()),
        'energy_identity_abs': float(abs(gram.sum() - energy)),
        'block_alignment_with_total': (defects @ total).tolist(),
        'block_norms': np.linalg.norm(defects, axis=1).tolist(),
        'gram': gram.tolist(),
    }
    # Independent pairs, not restarts, are the sampling units. Remove the
    # same-pair term; common random numbers otherwise bias cross-time signs.
    paired = np.diff(-np.asarray(features, dtype=np.float64), axis=0)
    n = paired.shape[1]
    if n > 1:
        unbiased = (n * gram - np.einsum('knf,lnf->kl', paired, paired) / n) / (n - 1)
        result['unbiased_terminal_feature_gap_squared'] = float(unbiased.sum())
        result['unbiased_diagonal_defect_energy'] = float(np.trace(unbiased))
        result['unbiased_cross_time_energy'] = float(unbiased.sum() - np.trace(unbiased))
        result['unbiased_gram'] = unbiased.tolist()
    return result


def load_setting(folder, device):
    cfg = json.loads((folder / 'config.json').read_text())
    dim, seed = cfg['ambient_dim'], cfg['seed']
    distribution = make_distribution(cfg,device)
    suite = core.build_model_suite(
        ambient_dim=dim, hidden_dim=cfg['hidden_dim'], depth=cfg['depth'],
        time_dim=cfg['time_dim'], mode_dim=cfg['mode_dim'],
        model_ids=['D0_xeps', 'D2_velocity', 'D4_safe'], lr=cfg['lr'],
        weight_decay=cfg['weight_decay'], seed=core.stable_seed(seed, dim, 113), device=device)
    checkpoint = torch.load(folder/'checkpoint.pt', map_location=device, weights_only=False)
    for name, model in suite.models.items():
        model.load_state_dict(checkpoint['models'][name], strict=True)
        model.eval().requires_grad_(False)
    return cfg, distribution, suite


def advance(state, index, *, steps, condition, distribution, suite, floor):
    def field(z, t):
        times = torch.full((len(z),), t, device=z.device)
        return core.condition_field(condition, suite=suite, distribution=distribution,
                                    state=z, time_value=times, denominator_floor=floor)[0]
    dt = 1./steps
    first = field(state, index/steps)
    second = field(state+dt*first, (index+1)/steps)
    return state + (.5*dt)*(first+second)


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, distribution, suite = load_setting(args.checkpoint_folder, device)
    dependencies = [Path(__file__), Path(spiral.__file__), Path(core.__file__),
                    Path(spiral.v10.__file__), Path(spiral.v4.__file__)]
    sources = args.output/'source'
    sources.mkdir()
    for path in dependencies:
        (sources/path.name).write_bytes(path.read_bytes())
    manifest = {
        'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256': sha(args.checkpoint_folder/'checkpoint.pt'),
        'config_sha256': sha(args.checkpoint_folder/'config.json'),
        'source_sha256': {str(p): sha(p) for p in dependencies},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
        'claim': 'exploratory signed defect attribution; no causal or paper-success claim',
    }
    core.save_json(args.output/'manifest.json', manifest)
    generator = torch.Generator(device=device).manual_seed(args.bank_seed)
    clean, intrinsic, _ = distribution.sample(args.samples, generator=generator)
    noise = torch.randn(clean.shape, generator=generator, device=device)
    _, reference, _ = distribution.sample(8192, generator=generator)
    fg = torch.Generator(device='cpu').manual_seed(73001)
    frequencies = torch.randn(3,64,2,generator=fg)/torch.tensor([.1,.3,1.])[:,None,None]
    phases = 2*np.pi*torch.rand(3,64,generator=fg)
    frequencies, phases = frequencies.reshape(-1,2).to(device), phases.reshape(-1).to(device)
    def feature(u):
        return (2/192)**.5 * torch.cos(u @ frequencies.T + phases)
    np.savez(args.output/'inputs.npz', clean=clean.cpu().numpy(), noise=noise.cpu().numpy(),
             reference=reference.cpu().numpy(), frequencies=frequencies.cpu().numpy(), phases=phases.cpu().numpy())
    bounds = np.linspace(0,args.steps,args.blocks+1,dtype=int).tolist()
    summaries = []
    for condition in args.conditions:
        feature_rows, off_rows, risk_rows, endpoints = [], [], [], []
        condition_started = time.perf_counter()
        for start in bounds:
            t = start/args.steps
            state = (1-t)*noise+t*clean
            if start < args.steps:
                mt = (start + .5*(args.steps/args.blocks))/args.steps
                teacher = (1-mt)*noise+mt*clean
                times = torch.full((len(clean),),mt,device=device)
                predicted = core.condition_field(condition,suite=suite,distribution=distribution,
                    state=teacher,time_value=times,denominator_floor=cfg['denominator_floor'])[0]
                oracle = distribution.bayes_velocity(teacher,times,denominator_floor=cfg['denominator_floor'])
                risk_rows.append({'time':mt,'bayes_mse':float((predicted-oracle).square().mean()),
                                  'pair_mse':float((predicted-(clean-noise)).square().mean())})
            for index in range(start,args.steps):
                state = advance(state,index,steps=args.steps,condition=condition,
                                distribution=distribution,suite=suite,floor=cfg['denominator_floor'])
            if not torch.isfinite(state).all():
                raise FloatingPointError(f'nonfinite {condition} restart {start}')
            u = distribution.decode_intrinsic(state)
            feature_rows.append(feature(u).cpu().numpy())
            off_rows.append(distribution.off_subspace_rms(state).cpu().numpy())
            endpoints.append(u.cpu().numpy())
            print(json.dumps({'condition':condition,'restart':start,'seconds':time.perf_counter()-condition_started}),flush=True)
        features = np.stack(feature_rows)
        np.savez(args.output/f'{condition}.npz',features=features,off_subspace=np.stack(off_rows),
                 intrinsic_endpoints=np.stack(endpoints),restart_steps=bounds)
        # The noising coupling is kept across all restart times for variance reduction.
        stats = defect_statistics(features)
        # Recompute the complete original sampler to audit faithful integration.
        direct,_ = core.sample_heun(condition,suite=suite,distribution=distribution,
            initial_noise=noise,steps=args.steps,denominator_floor=cfg['denominator_floor'],snapshot_times=[])
        parity=float(np.max(np.abs(distribution.decode_intrinsic(direct).cpu().numpy()-endpoints[0])))
        assert parity == 0., parity
        directions=core.fixed_directions(2,256,73002)
        # Equal counts needed by fixed_swd; independent reference is not the coupling clean bank.
        swd=core.fixed_swd(endpoints[0],reference[:args.samples].cpu().numpy(),directions=directions)
        stats.update(condition=condition,seed=cfg['seed'],samples=args.samples,
                     intrinsic_swd_independent_reference=swd,mean_bayes_mse=float(np.mean([r['bayes_mse'] for r in risk_rows])),
                     endpoint_off_subspace_rms=float(np.mean(off_rows[0])),
                     original_sampler_parity_max=parity,teacher_risk=risk_rows,
                     seconds=time.perf_counter()-condition_started)
        summaries.append(stats)
        core.save_json(args.output/'summary.partial.json',summaries)
    core.save_json(args.output/'summary.json',{'complete':True,'conditions':summaries,'seconds':time.perf_counter()-started})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--bank-seed',type=int,default=202609171)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--blocks',type=int,default=10)
    p.add_argument('--conditions',nargs='+',default=['D2_gate_on_D0','D3_oracle_bayes_gate','D4_gate_on_D0'])
    args=p.parse_args()
    if args.steps%args.blocks or args.samples>8192 or args.samples<2:
        p.error('steps must divide into blocks; samples must be in [2,8192]')
    run(args)
