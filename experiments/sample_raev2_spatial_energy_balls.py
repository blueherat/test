#!/usr/bin/env python3
"""Frozen 1K cohort spatial-energy balls, global-ball ablation, and native baseline.

Run ``--mode parity`` on 16 images first; sampling requires that matching record.
Official sampling is production batch-major. Balls use one complete GPU cohort
and only two (or one) shared contraction coefficients after actual Euler steps.
No gain, time-window, noise or channel search is implemented.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import sample_raev2_observable_potential as production
from experiments.sample_raev2_observable_potential import (
    BASELINE_SHA256, CONFIG_SHA256, DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
    LATENT_SHAPE, SOURCE_FILES as PRODUCTION_SOURCES, T_EPS, euler_update,
    native_uint8, paired_noise, save_npz, tensor_sha256,
)
from experiments.train_raev2_observable_potential import artifact, atomic_json, clean_forward, sha256_file
from experiments.sample_raev2_pfr_retiming import load_config, shifted_time_grid

PROTOCOL = 'raev2_fixed_cohort_spatial_energy_balls_v1'
SEED = 202609121
COUNT = 1000
BATCH_SIZE = 8
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
CALIBRATION = RESTART/'spectral_energy_audit_v1/seed20260801_energies.npz'
CALIBRATION_SHA256 = '5fbb3db3eca2adf0193896ebfd4657ee06014c88c630cb9f4445b77db92cf774'
CALIBRATION_SUMMARY_SHA256 = '9f98830570684fc0f70a5718c34e280d8d125962f5d521c9200f2a13bc2804cc'
CALIBRATION_REQUEST_SHA256 = 'd7a4860fe131a5075ccfc3ec9a9294747359f3b6af56f4223e54b69c66a03f17'
PLAN = RESTART/'spatial_energy_balls_v1/plan.json'
PLAN_SHA256 = '9c7972ac459b5511f293553e86e43671d0bca8eda7d5e5359f5ebcb03ef62025'
SOURCE_FILES = tuple(dict.fromkeys((
    'experiments/sample_raev2_spatial_energy_balls.py',
    'tests/test_raev2_spatial_energy_balls.py', *PRODUCTION_SOURCES,
)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('parity', 'official', 'global_ball', 'spatial_balls'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parity-dir', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args(argv)
    args.command = 'parity' if args.mode == 'parity' else 'sample'
    expected_count = 16 if args.command == 'parity' else COUNT
    if args.num_samples is None:
        args.num_samples = expected_count
    if args.num_samples != expected_count:
        parser.error(f'{args.command} fixes exactly {expected_count} samples')
    if args.seed != SEED:
        parser.error(f'this request fixes seed {SEED}; no seed search')
    if args.num_steps <= 0:
        parser.error('--num-steps must be positive')
    if args.command == 'parity':
        if args.parity_dir is not None or args.num_steps != 100:
            parser.error('parity fixes 16 images and 100 steps; no prior parity')
    else:
        if args.mode is None or args.parity_dir is None:
            parser.error('sample requires --mode and the matching completed --parity-dir')
        if args.mode != 'official' and args.num_steps != 100:
            parser.error('only official may change num-steps for a later explicit cost comparison')
    return args


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class Timer:
    """Wall time plus CUDA event span; event spans include stream idle gaps."""
    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        synchronize(self.device)
        self.started = time.perf_counter()
        if self.device.type == 'cuda':
            self.begin = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.begin.record()
        return self

    def __exit__(self, *exc):
        if self.device.type == 'cuda':
            self.end.record()
        synchronize(self.device)
        self.wall = time.perf_counter() - self.started
        self.cuda_span = self.begin.elapsed_time(self.end)/1000 if self.device.type == 'cuda' else None


def split_spatial(value):
    """Orthogonal DC/AC formula, evaluated in the input dtype."""
    if value.ndim != 4 or any(n <= 0 for n in value.shape):
        raise ValueError('expected nonempty NCHW tensor')
    dc = value.mean(dim=(-2, -1), keepdim=True).expand_as(value)
    return dc, value-dc


def energy_sums_fp64(value):
    """Unnormalized projected squared norms, with only a microbatch FP64 copy."""
    x = value.double()
    mu = x.mean(dim=(-2, -1), keepdim=True)
    spatial = x.shape[-2]*x.shape[-1]
    return torch.stack((mu.square().sum()*spatial, (x-mu).square().sum()))


def cohort_energies(state, *, microbatch=BATCH_SIZE):
    if state.ndim != 4 or any(n <= 0 for n in state.shape) or microbatch <= 0:
        raise ValueError('expected nonempty NCHW cohort and positive microbatch')
    sums = torch.zeros(2, dtype=torch.float64, device=state.device)
    for start in range(0, len(state), microbatch):
        sums += energy_sums_fp64(state[start:start+microbatch])
    result = (sums/state.numel()).cpu().tolist()
    if any(not math.isfinite(a) or a < 0 for a in result):
        raise FloatingPointError('nonfinite cohort energy')
    return result


def bridge_budgets(following, clean_moments, *, spatial_size=256):
    if not 0 <= following <= 1 or spatial_size <= 0:
        raise ValueError('invalid bridge time or spatial size')
    if len(clean_moments) != 2 or any(not math.isfinite(m) or m < 0 for m in clean_moments):
        raise ValueError('two finite nonnegative clean moments required')
    return [(1-following)**2*clean_moments[0]+following**2/spatial_size,
            (1-following)**2*clean_moments[1]+following**2*(spatial_size-1)/spatial_size]


def contraction(energy, budget):
    if not math.isfinite(energy) or not math.isfinite(budget) or min(energy, budget) < 0:
        raise ValueError('finite nonnegative energy and budget required')
    return 1. if energy == 0 or energy <= budget else math.sqrt(budget/energy)


@torch.no_grad()
def project_cohort(state, following, clean_moments, *, mode, microbatch=BATCH_SIZE):
    """Modify the entire actual Euler successor, with globally shared coefficients.

    FP64 moments use mathematical spatial projections of FP32 values. Application
    uses FP32 means, coefficients and arithmetic; measured residuals report that
    finite-precision difference without silently tightening or reprojecting.
    """
    if mode == 'official':
        return None  # Exact bypass: no energy pass, decomposition or timer.
    if mode not in ('global_ball', 'spatial_balls') or state.dtype != torch.float32:
        raise ValueError('balls require FP32 states and a valid ball mode')
    started = time.perf_counter()
    before = cohort_energies(state, microbatch=microbatch)
    moment_wall = time.perf_counter()-started
    budgets = bridge_budgets(following, clean_moments, spatial_size=state.shape[-2]*state.shape[-1])
    lambdas = ([contraction(sum(before), sum(budgets))]*2 if mode == 'global_ball'
               else [contraction(a, b) for a, b in zip(before, budgets)])
    applied = [float(np.float32(x)) for x in lambdas]
    ideal_after = [a*l*l for a, l in zip(before, lambdas)]
    started = time.perf_counter()
    ideal_identical = lambdas == [1., 1.]
    identical = applied == [1., 1.]
    squared_float_error = torch.zeros((), dtype=torch.float64, device=state.device)
    maximum_float_error = torch.zeros_like(squared_float_error)
    if ideal_identical:
        after = before.copy()
    else:
        after_sums = torch.zeros(2, dtype=torch.float64, device=state.device)
        for start in range(0, len(state), microbatch):
            x = state[start:start+microbatch]
            x64 = x.double()
            if mode == 'global_ball':
                ideal = x64*lambdas[0]
                replacement = x if identical else x*applied[0]
            else:
                dc64, ac64 = split_spatial(x64)
                ideal = lambdas[0]*dc64+lambdas[1]*ac64
                if identical:
                    replacement = x
                else:
                    dc32, ac32 = split_spatial(x)
                    replacement = applied[0]*dc32+applied[1]*ac32
            float_error = replacement.double()-ideal
            squared_float_error += float_error.square().sum()
            maximum_float_error = torch.maximum(maximum_float_error, float_error.abs().max())
            after_sums += energy_sums_fp64(replacement)
            if not identical:
                x.copy_(replacement)
        after = (after_sums/state.numel()).cpu().tolist()
    rms = float((squared_float_error/state.numel()).sqrt().cpu())
    maximum = float(maximum_float_error.cpu())
    if any(not math.isfinite(a) or a < 0 for a in after) or not math.isfinite(rms):
        raise FloatingPointError('nonfinite projected cohort')
    return {'mode': mode, 'following': following, 'cohort_count': len(state),
            'coefficients_are_shared_over_all_images_and_channels': True,
            'energy_before_dc_ac': before, 'bridge_budget_dc_ac': budgets,
            'lambda_fp64': lambdas, 'lambda_applied_fp32': applied,
            'ideal_energy_after_dc_ac': ideal_after, 'energy_after_dc_ac': after,
            'rounding_energy_residual_dc_ac': [a-b for a, b in zip(after, ideal_after)],
            'budget_residual_dc_ac': [a-b for a, b in zip(after, budgets)],
            'global_budget_residual': sum(after)-sum(budgets),
            'float_state_residual_rms': rms, 'float_state_residual_max_abs': maximum,
            'exact_all_one_bypass': identical,
            'ideal_active': not ideal_identical, 'actual_applied_active': not identical,
            'moment_before_wall_seconds': moment_wall,
            'application_and_after_diagnostics_wall_seconds': time.perf_counter()-started,
            'constraint_note': 'global_ball constrains only summed energy; component residuals may be positive',
            'rounding_policy': 'record FP32 residual; no additional projection, margin, threshold or fitted epsilon'}


@torch.no_grad()
def successor(model, state, labels, current, following):
    times = torch.full((len(state),), current, dtype=torch.float32, device=state.device)
    guided = clean_forward(model, state, times, labels)
    return euler_update(state, guided, None, current, following)


@torch.no_grad()
def time_major(model, state, labels, grid, *, mode, clean_moments, callback=None):
    """Every image receives its Euler update before any cohort projection."""
    if state.dtype != torch.float32 or len(state) % BATCH_SIZE or labels.shape != (len(state),):
        raise ValueError('complete FP32 cohort and labels, partitioned into B8 batches, required')
    if mode not in ('official', 'global_ball', 'spatial_balls'):
        raise ValueError('invalid mode')
    records = []
    with Timer(state.device) as total_timer:
        for step, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
            with Timer(state.device) as forward_timer:
                for start in range(0, len(state), BATCH_SIZE):
                    block = slice(start, start+BATCH_SIZE)
                    state[block].copy_(successor(model, state[block], labels[block], current, following))
            if mode == 'official':
                projection = None
                projection_wall, projection_span = 0., None
            else:
                with Timer(state.device) as projection_timer:
                    projection = project_cohort(state, following, clean_moments, mode=mode)
                projection_wall, projection_span = projection_timer.wall, projection_timer.cuda_span
            record = {'step_index': step, 'current': current, 'following': following,
                      'stage2_forward_calls': len(state)//BATCH_SIZE,
                      'model_and_euler_wall_seconds': forward_timer.wall,
                      'model_and_euler_cuda_span_seconds': forward_timer.cuda_span,
                      'projection_and_diagnostics_wall_seconds': projection_wall,
                      'projection_and_diagnostics_cuda_span_seconds': projection_span,
                      'projection': projection}
            records.append(record)
            if callback is not None:
                callback(record)
    return state, records, {'trajectory_wall_seconds': total_timer.wall,
                           'trajectory_cuda_event_span_seconds': total_timer.cuda_span,
                           'stage2_forward_calls': (len(state)//BATCH_SIZE)*(len(grid)-1),
                           'stage2_sample_forwards': len(state)*(len(grid)-1)}


def batch_major(model, noise, labels, grid, device):
    """Production baseline: existing official step, one complete B8 trajectory."""
    for start in range(0, len(noise), BATCH_SIZE):
        prep = time.perf_counter()
        state = noise[start:start+BATCH_SIZE].to(device, copy=True)
        local_labels = labels[start:start+BATCH_SIZE].to(device)
        synchronize(device)
        transfer_wall = time.perf_counter()-prep
        with Timer(device) as timer, torch.no_grad():
            for current, following in zip(grid[:-1], grid[1:]):
                state = production.sampling_step(model, None, state, local_labels, current, following, use_potential=False)
        yield start, state, {'trajectory_wall_seconds': timer.wall,
                            'trajectory_cuda_event_span_seconds': timer.cuda_span,
                            'noise_and_labels_transfer_wall_seconds': transfer_wall}


def decode(decoder, state):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return native_uint8(decoder.decode(state))


def load_calibration(path, *, expected_sha=CALIBRATION_SHA256):
    identity_record = artifact(path)
    if identity_record['sha256'] != expected_sha:
        raise ValueError('calibration must be the frozen historical real seed20260801 energy artifact')
    with np.load(path, allow_pickle=False) as z:
        if (z['views'].tolist() != ['all_5000', 'exclude_21_shared_sources']
                or z['arms'].tolist() != ['real', 'historical_scale1', 'ig_1p78']
                or z['components'].tolist() != ['spatial_DC', 'spatial_AC']
                or not np.array_equal(z['channels'], np.arange(1024))):
            raise ValueError('calibration axes differ from the fixed protocol')
        energy = z['global_equal_class_energy']
        if energy.shape != (2, 3, 2, 1024) or energy.dtype != np.float64:
            raise ValueError('unexpected calibration energy shape/dtype')
        moments = energy[0, 0].sum(axis=-1, dtype=np.float64)/1024.
    if np.any(~np.isfinite(moments)) or np.any(moments <= 0):
        raise ValueError('invalid real DC/AC energy calibration')
    return moments.tolist(), identity_record


def archive_sources(output):
    records = {}
    for relative in SOURCE_FILES:
        target = output/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
        records[relative] = artifact(target)
    return records


def verify_parity(directory, identities, source_hashes):
    summary_path, request_path = directory/'summary.json', directory/'request.json'
    summary, request = json.loads(summary_path.read_text()), json.loads(request_path.read_text())
    if (not summary.get('complete') or summary.get('command') != 'parity'
            or not summary.get('endpoint_bitwise') or not summary.get('pixel_bitwise')
            or summary.get('samples_per_loop') != 16 or request.get('num_steps') != 100
            or request.get('protocol') != PROTOCOL or request.get('seed') != SEED
            or summary.get('request', {}).get('sha256') != sha256_file(request_path)):
        raise ValueError('matching completed 16-image official parity is required')
    if request['identities'] != identities:
        raise ValueError('parity weights, configuration or calibration identities differ')
    if {name: value['sha256'] for name, value in request['sources'].items()} != source_hashes:
        raise ValueError('source changed since parity; repeat parity before sampling')
    return {'request': artifact(request_path), 'summary': artifact(summary_path)}


def make_request(args, output):
    sources = archive_sources(output)
    config_identity, baseline_identity = artifact(args.config), artifact(args.checkpoint)
    if config_identity['sha256'] != CONFIG_SHA256 or baseline_identity['sha256'] != BASELINE_SHA256:
        raise ValueError('requires frozen official configuration and EMA weights')
    config = load_config(args.config)
    if (tuple(config.misc.latent_size) != LATENT_SHAPE or config.misc.num_classes != 1000
            or config.transport.prediction != 'x' or float(config.transport.t_eps) != T_EPS
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != .1
            or config.guidance.ig.t_max != 1 or config.guidance.cfg.scale != 1):
        raise ValueError('not the frozen official latent/transport/guidance protocol')
    moments, calibration = load_calibration(args.calibration)
    calibration_request = artifact(args.calibration.parent/'request.json')
    calibration_summary = artifact(args.calibration.parent/'summary.json')
    if (calibration_request['sha256'] != CALIBRATION_REQUEST_SHA256
            or calibration_summary['sha256'] != CALIBRATION_SUMMARY_SHA256):
        raise ValueError('calibration lineage differs from the frozen audit')
    plan_identity = artifact(PLAN)
    if plan_identity['sha256'] != PLAN_SHA256:
        raise ValueError('root study plan changed')
    plan = json.loads(PLAN.read_text())
    if (moments != [plan['reference']['m_dc'], plan['reference']['m_ac']]
            or plan['cohort']['seed'] != SEED or plan['cohort']['n'] != COUNT):
        raise ValueError('calibration/cohort differs from frozen root plan')
    document_identity = artifact(Path(plan['protocol_document']['path']))
    if document_identity['sha256'] != plan['protocol_document']['sha256']:
        raise ValueError('root frozen protocol document changed')
    identities = {'config': config_identity, 'baseline_checkpoint': baseline_identity,
                  'decoder_checkpoint': artifact(Path(config.stage_1.params['pretrained_decoder_path'])),
                  'normalization_stats': artifact(Path(config.stage_1.params['normalization_stat_path'])),
                  'calibration': calibration, 'calibration_request': calibration_request,
                  'calibration_summary': calibration_summary, 'root_plan': plan_identity,
                  'protocol_document': document_identity}
    prior = (verify_parity(args.parity_dir, identities, {k:v['sha256'] for k,v in sources.items()})
             if args.command == 'sample' else None)
    grid = shifted_time_grid(args.num_steps, 8, torch.device('cpu')).tolist()
    if not all(0 <= b < a <= 1 for a,b in zip(grid[:-1],grid[1:])):
        raise ValueError('invalid FP32 decreasing grid')
    request = {'protocol': PROTOCOL, 'command': args.command, 'mode': args.mode,
               'seed': SEED, 'sample_count': 16 if args.command == 'parity' else COUNT,
               'batch_size': BATCH_SIZE, 'num_steps': args.num_steps, 'time_grid': grid,
               'time_shift': 8, 'transport_t_eps': T_EPS, 'identities': identities,
               'sources': sources, 'parity': prior, 'clean_moments_dc_ac': moments,
               'calibration_scope': 'seed20260801 historical real, all5000/all1000classes, sumChannels/1024. BF16 encoder autocast -> FP16 storage; not new FP32 encoder calibration.',
               'bridge_budgets': 'b0=(1-t_next)^2*m0+t_next^2/256; b1=(1-t_next)^2*m1+t_next^2*255/256',
               'projection': 'after all actual Euler successors Y: aj=||Pj Y||²/(N*D), lambda_j=min(1,sqrt(bj/aj)), aj=0 ->1; shared across all images AND channels; FP32 lambda0*P0Y+lambda1*P1Y',
               'global_ball': 'one lambda=min(1,sqrt((b0+b1)/(a0+a1))); same empirical m0+m1; apply FP32 lambda*Y',
               'all_one_behavior': 'exact state bypass when both applied FP32 lambdas equal1, even if ideal FP64 lambdas slightly below1; record both activity flags and roundoff without margin',
               'control_scope': 'complete 1000-image single-GPU cohort; only model/moment arithmetic is microbatched B8; no microbatch-specific lambdas',
               'precision': 'FP32 state and Euler; native BF16 B+1.78*(F-B) then float; FP64 energy moments, FP32 applied coefficients/state; TF32 off',
               'pixel_arithmetic': 'native BF16 decoder -> clamp -> BF16 multiply255 -> uint8 NHWC',
               'noise_schema': 'one full CUDA Generator(seed) randn([N,1024,16,16]) via production.paired_noise, no rank/batch seed offset; actual noise/RNG/label SHA frozen in sampling_input.json before first forward',
               'labels': 'ascending IDs0..N-1, label=id%1000; one image per class for real1K',
               'official_loop': 'production batch-major B8 complete trajectory then decode; no extra per-step energy reduction, decomposition, timing or synchronization',
               'ball_loop': 'time-major B8 model forwards then full-cohort moment/projection at every successor including terminal; no new interval',
               'parity_reference': 'time-major official versus production.sampling_step(use_potential=False) batch-major; 16 same-noise images,100steps; both endpoints and native pixels bitwise',
               'diagnostics': 'every ball step records lambda, before/after energy, bridge bounds, FP32 state/energy residuals; included in candidate cost, no corrective second projection',
               'cost_scope': 'all loads, hashing/noise preparation, complete trajectories, model/Euler, moment/projection diagnostics, decoder/uint8, transfers, output writing and peaks recorded; CUDA events are stream spans, not summed kernel busy time',
               'torch_version': str(torch.__version__), 'automatic_step_selection': False,
               'fid_performed': False, 'no_parameter_or_channel_search': True}
    atomic_json(output/'request.json', request)  # Never rewritten after loading or sampling.
    return request, config


def load_models(config, checkpoint_path, device):
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from utils.model_utils import instantiate_from_config
    os.environ.setdefault('DINOV3_CKPT_DIR', '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3')
    install_raev2_decoder_config_compat()
    with Timer(device) as decoder_timer:
        decoder = instantiate_from_config(config.stage_1)
        del decoder.encoder
        decoder = decoder.to(device).eval().requires_grad_(False)
    with Timer(device) as backbone_timer:
        model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(checkpoint['ema'], strict=True)
        step = int(checkpoint['step'])
        del checkpoint
    gc.collect()
    return model, decoder, {'decoder_load_wall_seconds': decoder_timer.wall,
                           'backbone_load_wall_seconds': backbone_timer.wall, 'checkpoint_step': step}


def run_parity(model, decoder, noise, labels, grid, output, device):
    state = noise.to(device, copy=True)
    time_state, records, time_cost = time_major(model, state, labels.to(device), grid,
                                               mode='official', clean_moments=None)
    time_cpu = time_state.cpu()
    time_pixels, direct_pixels, direct_states = [], [], []
    decoder_wall = 0.; direct_wall = 0.; direct_spans = []
    for start in range(0,16,BATCH_SIZE):
        with Timer(device) as timer:
            time_pixels.append(decode(decoder, time_state[start:start+BATCH_SIZE]))
        decoder_wall += timer.wall
    del state, time_state
    for start, direct, timing in batch_major(model, noise, labels, grid, device):
        direct_states.append(direct.cpu())
        direct_wall += timing['trajectory_wall_seconds']
        direct_spans.append(timing['trajectory_cuda_event_span_seconds'])
        with Timer(device) as timer:
            direct_pixels.append(decode(decoder, direct))
        decoder_wall += timer.wall
    direct_cpu = torch.cat(direct_states)
    image_a, image_b = np.concatenate(time_pixels), np.concatenate(direct_pixels)
    endpoint_equal, pixel_equal = torch.equal(time_cpu,direct_cpu), np.array_equal(image_a,image_b)
    result = {'complete': bool(endpoint_equal and pixel_equal), 'samples_per_loop':16,
              'endpoint_bitwise':endpoint_equal, 'pixel_bitwise':pixel_equal,
              'max_endpoint_abs_difference':float((time_cpu.double()-direct_cpu.double()).abs().max()),
              'pixel_mismatches':int(np.count_nonzero(image_a!=image_b)),
              'time_major_endpoint_sha256':tensor_sha256(time_cpu),
              'direct_batch_major_endpoint_sha256':tensor_sha256(direct_cpu),
              'time_major_pixels_sha256':hashlib.sha256(image_a.tobytes()).hexdigest(),
              'direct_pixels_sha256':hashlib.sha256(image_b.tobytes()).hexdigest(),
              'time_major_cost':time_cost, 'direct_batch_major_trajectory_wall_seconds':direct_wall,
              'direct_batch_major_trajectory_cuda_span_seconds':sum(direct_spans) if all(x is not None for x in direct_spans) else None,
              'stage2_forward_calls':400,'stage2_sample_forwards':3200,
              'decoder_forward_calls':4,'decoder_sample_forwards':32,
              'decode_and_uint8_wall_seconds':decoder_wall,
              'parity_is_not_quality_or_cost_comparison':True}
    atomic_json(output/'parity_result.json',result)
    if not result['complete']:
        raise AssertionError('official loop parity failed; no sampling admission')
    return result


def sample(model, decoder, noise, labels, grid, clean_moments, args, output, device):
    if noise.shape != (COUNT,*LATENT_SHAPE) or noise.dtype != torch.float32:
        raise ValueError('one complete1000-image FP32 noise cohort is required')
    if not torch.equal(labels.cpu(),torch.arange(COUNT)):
        raise ValueError('fixed one-per-class ascending cohort required')
    started=time.perf_counter(); writing=0.; hash_and_validation=0.; decode_wall=0.; decode_spans=[]
    trajectory_wall=0.; trajectory_spans=[]; transfer_wall=0.; manifest=[]; images_all=[]
    endpoint_hasher=hashlib.sha256()
    (output/'batches').mkdir()
    state=None; steps=[]
    if args.mode != 'official':
        prep=time.perf_counter(); state=noise.to(device, copy=True); gpu_labels=labels.to(device)
        synchronize(device); transfer_wall+=time.perf_counter()-prep
        def callback(record):
            nonlocal writing
            t=time.perf_counter()
            with (output/'step_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n')
            atomic_json(output/'progress.json',{'complete':False,'phase':'trajectory','completed_steps':record['step_index']+1,
                        'stage2_forward_calls':(record['step_index']+1)*(COUNT//BATCH_SIZE),
                        'elapsed_seconds':time.perf_counter()-started})
            writing+=time.perf_counter()-t
            if record['step_index']%10==0 or record['step_index']==len(grid)-2:
                print(json.dumps({'step':record['step_index'],'mode':args.mode,'lambda':record['projection']['lambda_fp64']}),flush=True)
        state,steps,cost=time_major(model,state,gpu_labels,grid,mode=args.mode,clean_moments=clean_moments,callback=callback)
        trajectory_wall=cost['trajectory_wall_seconds'];trajectory_spans=[cost['trajectory_cuda_event_span_seconds']]
        batches=((start,state[start:start+BATCH_SIZE],None) for start in range(0,COUNT,BATCH_SIZE))
    else:
        batches=batch_major(model,noise,labels,grid,device)
    for start,endpoint,timing in batches:
        if timing is not None:
            trajectory_wall+=timing['trajectory_wall_seconds'];trajectory_spans.append(timing['trajectory_cuda_event_span_seconds'])
            transfer_wall+=timing['noise_and_labels_transfer_wall_seconds']
        prep=time.perf_counter()
        if endpoint.dtype!=torch.float32 or not bool(torch.isfinite(endpoint).all()):
            raise FloatingPointError('nonfinite or non-FP32 endpoint')
        endpoint_cpu=endpoint.detach().cpu().contiguous().numpy()
        endpoint_hasher.update(memoryview(endpoint_cpu).cast('B'))
        endpoint_hash=hashlib.sha256(memoryview(endpoint_cpu).cast('B')).hexdigest()
        noise_hash=tensor_sha256(noise[start:start+BATCH_SIZE])
        ids=np.arange(start,start+BATCH_SIZE,dtype=np.int64)
        hash_and_validation+=time.perf_counter()-prep
        with Timer(device) as timer:
            pixels=decode(decoder,endpoint)
        decode_wall+=timer.wall;decode_spans.append(timer.cuda_span)
        t=time.perf_counter();path=output/'batches'/f'{start:06d}_{start+BATCH_SIZE:06d}.npz'
        save_npz(path,pixels,ids=ids,labels=ids)
        manifest.append({'global_ids':ids.tolist(),'labels_sha256':tensor_sha256(labels[start:start+BATCH_SIZE]),
                         'noise_sha256':noise_hash,'endpoint_sha256':endpoint_hash,'archive':artifact(path),
                         'trajectory':timing,'decode_and_uint8_wall_seconds':timer.wall})
        images_all.append(pixels)
        atomic_json(output/'batch_manifest.json',{'batches':manifest})
        atomic_json(output/'progress.json',{'complete':False,'phase':'decode','completed':start+BATCH_SIZE,
                    'stage2_forward_calls':COUNT//BATCH_SIZE*(len(grid)-1) if args.mode!='official' else len(manifest)*(len(grid)-1),
                    'decoder_forward_calls':len(manifest),'elapsed_seconds':time.perf_counter()-started})
        writing+=time.perf_counter()-t
        if len(manifest)%16==0 or start+BATCH_SIZE==COUNT:
            print(json.dumps({'mode':args.mode,'decoded':start+BATCH_SIZE}),flush=True)
    t=time.perf_counter();path=output/'samples.npz'
    save_npz(path,np.concatenate(images_all),ids=np.arange(COUNT,dtype=np.int64),labels=np.arange(COUNT,dtype=np.int64))
    sample_record=artifact(path);manifest_record=artifact(output/'batch_manifest.json')
    writing+=time.perf_counter()-t
    return {'complete':True,'samples':COUNT,'global_cohort_size':COUNT,'global_ids':list(range(COUNT)),
            'sample_archive':sample_record,'archive_sha256':sample_record['sha256'],'batch_manifest':manifest_record,
            'global_endpoint_sha256':endpoint_hasher.hexdigest(),
            'stage2_forward_calls':COUNT//BATCH_SIZE*(len(grid)-1),'stage2_sample_forwards':COUNT*(len(grid)-1),
            'stage2_nfe_per_sample':len(grid)-1,'decoder_forward_calls':COUNT//BATCH_SIZE,'decoder_sample_forwards':COUNT,
            'trajectory_wall_seconds':trajectory_wall,
            'trajectory_cuda_event_span_seconds':sum(trajectory_spans) if all(x is not None for x in trajectory_spans) else None,
            'decode_and_uint8_wall_seconds':decode_wall,
            'decode_and_uint8_cuda_event_span_seconds':sum(decode_spans) if all(x is not None for x in decode_spans) else None,
            'model_and_euler_wall_seconds':sum(x['model_and_euler_wall_seconds'] for x in steps) if steps else trajectory_wall,
            'projection_including_diagnostics_wall_seconds':sum(x['projection_and_diagnostics_wall_seconds'] for x in steps),
            'noise_labels_transfer_wall_seconds':transfer_wall,'endpoint_hash_and_validation_wall_seconds':hash_and_validation,
            'output_write_hash_and_progress_wall_seconds':writing,
            'sampling_wall_including_output_seconds':time.perf_counter()-started,
            'diagnostic_steps':len(steps),'energy_reductions_in_official':0,
            'step_diagnostics':artifact(output/'step_diagnostics.jsonl') if steps else None,
            'accounting_overlap':'time-major trajectory wall includes step diagnostics/progress writes; write field is a breakdown, not additive to trajectory total',
            'output_format':'arr_0 uint8 NHWC, ascending ids/labels; directly readable by ADM evaluator',
            'base_head_note':'Full/Base returned in one stage2 forward, not two NFE',
            'image_sampling_performed':True,'fid_performed':False}


def main():
    started,cpu_started=time.perf_counter(),time.process_time()
    args=parse_args()
    for name in ('output_dir','config','checkpoint','calibration','parity_dir'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.expanduser().resolve())
    output=args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite an existing run')
    output.mkdir(parents=True,exist_ok=True)
    request,config=make_request(args,output)
    preprocessing_wall=time.perf_counter()-started
    atomic_json(output/'progress.json',{'complete':False,'phase':'loading','request':artifact(output/'request.json')})
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('run with exactly one visible CUDA GPU')
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats(device)
        model,decoder,loading=load_models(config,args.checkpoint,device)
        peak_loading={'allocated_bytes':torch.cuda.max_memory_allocated(device),'reserved_bytes':torch.cuda.max_memory_reserved(device)}
        t=time.perf_counter();count=request['sample_count']
        noise,rng_state=paired_noise(SEED,device,count=count)
        synchronize(device)
        noise_generation_wall=time.perf_counter()-t
        t=time.perf_counter();labels=torch.arange(count,dtype=torch.long)%1000
        save_npz(output/'paired_noise_audit.npz',first_noise=noise[0].numpy(),rng_state=rng_state.numpy())
        inputs={'request':artifact(output/'request.json'),'seed':SEED,'noise_shape':list(noise.shape),
                'global_noise_sha256':tensor_sha256(noise),'noise_rng_state_sha256':tensor_sha256(rng_state),
                'global_labels_sha256':tensor_sha256(labels),'paired_noise_audit':artifact(output/'paired_noise_audit.npz'),
                'cuda_device':torch.cuda.get_device_name(device),'frozen_before_first_model_forward':True}
        atomic_json(output/'sampling_input.json',inputs)
        noise_identity_and_save_wall=time.perf_counter()-t
        peak_preparation={'allocated_bytes':torch.cuda.max_memory_allocated(device),'reserved_bytes':torch.cuda.max_memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
        if args.command=='parity':
            result=run_parity(model,decoder,noise,labels,request['time_grid'],output,device)
        else:
            result=sample(model,decoder,noise,labels,request['time_grid'],request['clean_moments_dc_ac'],args,output,device)
        result.update(protocol=PROTOCOL,command=args.command,mode=args.mode,request=artifact(output/'request.json'),
            sampling_input=artifact(output/'sampling_input.json'),seed=SEED,num_steps=args.num_steps,
            global_noise_sha256=inputs['global_noise_sha256'],noise_rng_state_sha256=inputs['noise_rng_state_sha256'],
            global_labels_sha256=inputs['global_labels_sha256'],preprocessing_sources_calibration_weight_hash_wall_seconds=preprocessing_wall,
            loading=loading,noise_generation_wall_seconds=noise_generation_wall,
            noise_identity_and_audit_save_wall_seconds=noise_identity_and_save_wall,
            peak_loading=peak_loading,peak_loading_and_noise_preparation=peak_preparation,
            peak_sampling={'allocated_bytes':torch.cuda.max_memory_allocated(device),
                                                   'reserved_bytes':torch.cuda.max_memory_reserved(device)},
            total_wall_seconds_before_summary=time.perf_counter()-started,cpu_seconds_before_summary=time.process_time()-cpu_started,
            cost_not_yet_matched='No automated official step selection; compare complete recorded costs before calling any run cost matched.')
        atomic_json(output/'summary.json',result)
        atomic_json(output/'progress.json',{'complete':True,'summary':artifact(output/'summary.json')})
        print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(output/'failure.json',{'complete':False,'error':f'{type(error).__name__}: {error}',
                    'wall_seconds':time.perf_counter()-started,'partial_progress':str(output/'progress.json')})
        raise


if __name__=='__main__':
    main()
