#!/usr/bin/env python3
"""Fixed native RAEv2 guidance averaged over its known affine reflection.

Only the reflection of all normal coordinates together is averaged (group C2).
This is not integration over all normal noise. Run the fixed 16-image parity
before the three root-controlled 1K arms. This runner never chooses K or FID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
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

PROTOCOL = 'raev2_affine_reflection_sampling_v1'
SEED = 202609131
COUNT = 1000
BATCH_SIZE = 8
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
PLAN = RESTART/'affine_reflection_v1/plan.json'
PROTOCOL_DOCUMENT = ROOT/'docs/RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md'
STATS_SHA256 = '40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77'
SOURCE_FILES = tuple(dict.fromkeys((
    'experiments/sample_raev2_affine_reflection.py',
    'tests/test_raev2_affine_reflection.py', *PRODUCTION_SOURCES,
    'external/RAEv2/src/encoders/vision_encoder.py',
)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('parity', 'official', 'reflection'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parity-dir', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args(argv)
    args.command = 'parity' if args.mode == 'parity' else 'sample'
    expected = 16 if args.command == 'parity' else COUNT
    if args.num_samples is None:
        args.num_samples = expected
    if args.seed != SEED or args.num_samples != expected:
        parser.error(f'fixed seed {SEED} and {expected} samples required; no search')
    if args.num_steps <= 0 or (args.mode != 'official' and args.num_steps != 100):
        parser.error('positive --num-steps may differ from100 only in official mode')
    if (args.command == 'parity') != (args.parity_dir is None):
        parser.error('parity takes no prior parity; sampling requires --parity-dir')
    return args


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class Timer:
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
        self.wall = time.perf_counter()-self.started
        self.cuda_span = self.begin.elapsed_time(self.end)/1000 if self.device.type == 'cuda' else None


class ForwardAudit:
    """Observed calls only: no per-step events, reductions or synchronization.

    Operator costs are included in the enclosing synchronized trajectory timer.
    Root-module hooks count actual forwards; explicit geometry contexts count
    the executed reflection/projection operations, without pretending to time
    asynchronous CUDA operations using host dispatch intervals.
    """
    def __init__(self, model, decoder, device):
        self.device = torch.device(device)
        self.counts = {key: 0 for key in ('stage2_forward_calls', 'stage2_sample_forwards',
                                        'decoder_forward_calls', 'decoder_sample_forwards')}
        self.operations, self.handles = {}, []
        for name, module in (('stage2', model), ('decoder', decoder.decoder)):
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))

    def record(self, name):
        self.operations[name] = self.operations.get(name, 0)+1

    @contextmanager
    def measure(self, name):
        self.record(name)
        yield

    def _pre(self, name):
        def hook(module, inputs):
            n = len(inputs[0])
            if n != BATCH_SIZE:
                raise ValueError(f'{name} requires separate B8 calls, observed B{n}')
            self.counts[name+'_forward_calls'] += 1
            self.counts[name+'_sample_forwards'] += n
            self.record(name+'_forward')
        return hook

    def snapshot(self):
        return dict(self.counts)

    def flush_operations(self):
        report = dict(self.operations)
        self.operations.clear()
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()


def affine_geometry(mean, variance):
    """The known channel-sum constraint in normalized latent coordinates."""
    if mean.ndim != 3 or mean.shape != variance.shape or any(n <= 0 for n in mean.shape):
        raise ValueError('matching nonempty mean/variance [C,H,W] required')
    if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
        raise ValueError('nonfinite normalization statistics')
    if bool((variance < 0).any()):
        raise ValueError('negative normalization variance')
    sigma = (variance.double()+1e-5).sqrt()
    norm = sigma.square().sum(0, keepdim=True).sqrt()
    unit = sigma/norm
    anchor = -mean.double().sum(0, keepdim=True)/norm*unit
    return unit, anchor


def normal_part(value, unit):
    return (value*unit).sum(1, keepdim=True)*unit


def project_clean(value, unit, anchor):
    return value-normal_part(value-anchor, unit)


def mirror_normal_noise(state, current, unit, anchor):
    if not 0 <= current <= 1:
        raise ValueError('reflection time outside [0,1]')
    return state-2*normal_part(state-(1-current)*anchor, unit)


@torch.no_grad()
def reflection_guided(model, state, labels, current, unit, anchor, *, audit=None):
    if (state.dtype != torch.float32 or unit.dtype != torch.float32 or anchor.dtype != torch.float32
            or unit.shape != state.shape[1:] or anchor.shape != unit.shape):
        raise ValueError('FP32 state, unit and anchor with matching latent shape required')
    times = torch.full((len(state),), current, dtype=torch.float32, device=state.device)
    original = clean_forward(model, state, times, labels)
    with audit.measure('reflection') if audit is not None else nullcontext():
        reflected = mirror_normal_noise(state, current, unit, anchor)
    reflected_guided = clean_forward(model, reflected, times, labels)
    with audit.measure('average_projection') if audit is not None else nullcontext():
        guided = project_clean((original+reflected_guided)*.5, unit, anchor)
    return guided


@torch.no_grad()
def successor(model, state, labels, current, following, *, mode, unit=None, anchor=None, audit=None):
    if mode == 'official':
        # Independent spelling of the official step for the parity comparison.
        times = torch.full((len(state),), current, dtype=torch.float32, device=state.device)
        guided = clean_forward(model, state, times, labels)
    elif mode == 'reflection':
        guided = reflection_guided(model, state, labels, current, unit, anchor, audit=audit)
    else:
        raise ValueError('invalid sampling mode')
    return euler_update(state, guided, None, current, following)


def decode(decoder, state):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return native_uint8(decoder.decode(state))


def observed_delta(audit, before, *, calls, samples, decoder_calls=0, decoder_samples=0):
    actual = {k: audit.counts[k]-v for k, v in before.items()}
    expected = {'stage2_forward_calls': calls, 'stage2_sample_forwards': samples,
                'decoder_forward_calls': decoder_calls, 'decoder_sample_forwards': decoder_samples}
    if actual != expected:
        raise AssertionError(f'observed forward counts {actual} != expected {expected}')
    return {'observed': actual, 'expected': expected}


def trajectory(model, state, labels, grid, *, mode, unit=None, anchor=None, audit=None,
               reference=False):
    with Timer(state.device) as timer, torch.no_grad():
        for current, following in zip(grid[:-1], grid[1:]):
            if reference:
                state = production.sampling_step(model, None, state, labels, current, following,
                                                 use_potential=False)
            else:
                state = successor(model, state, labels, current, following, mode=mode,
                                  unit=unit, anchor=anchor, audit=audit)
    return state, {'trajectory_wall_seconds': timer.wall,
                   'trajectory_cuda_event_span_seconds': timer.cuda_span}


def geometry_diagnostics(noise, unit64, anchor64):
    """Actual FP32 operations compared against FP64 geometry; no fitted tolerance."""
    started = time.perf_counter()
    unit, anchor = unit64.float(), anchor64.float()
    rows = []
    # Endpoints plus fixed interior time: numerical checks, not sampling windows.
    for current in (1., .5, 0.):
        reflected = mirror_normal_noise(noise, current, unit, anchor)
        twice = mirror_normal_noise(reflected, current, unit, anchor)
        projected = project_clean(noise, unit, anchor)
        errors = {
            'involution': twice.double()-noise.double(),
            'projection_idempotence': project_clean(projected, unit, anchor).double()-projected.double(),
            'projected_affine_normal': normal_part(projected.double()-anchor64, unit64),
            'reflection_tangent_change': (reflected.double()-noise.double())-
                normal_part(reflected.double()-noise.double(), unit64),
        }
        rows.append({'t': current, 'sample_count': len(noise),
                     'residuals': {k: {'max_abs': float(v.abs().max()),
                                       'rms': float(v.square().mean().sqrt())} for k, v in errors.items()}})
    return {'rows': rows, 'unit64_norm_squared_max_error': float((unit64.square().sum(0)-1).abs().max()),
            'unit32_norm_squared_max_error_fp64': float((unit.double().square().sum(0)-1).abs().max()),
            'unit32_sha256': tensor_sha256(unit), 'anchor32_sha256': tensor_sha256(anchor),
            'wall_seconds': time.perf_counter()-started,
            'scope': 'CPU FP32 formula with actual stats and parity CUDA noise; FP64 reference reductions. Residuals are descriptive, not an extra projection or parameter.'}


def archive_sources(output):
    records = {}
    for relative in SOURCE_FILES:
        target = output/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
        records[relative] = artifact(target)
    return records


def verify_frozen_files(plan, identities, sources):
    required = {str((ROOT/name).resolve()): record['sha256'] for name, record in sources.items()}
    required.update({record['path']: record['sha256'] for key, record in identities.items()
                     if key not in ('root_plan',)})
    frozen = {}
    for name, expected in plan['frozen_files'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT/path
        path = path.resolve()
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f'root frozen source/artifact changed: {path}')
        frozen[str(path)] = expected
    for path, expected in required.items():
        if frozen.get(path) != expected:
            raise ValueError(f'required source/artifact absent or different in root frozen_files: {path}')


def verify_parity(directory, identities, sources):
    summary_path, request_path = directory/'summary.json', directory/'request.json'
    summary, request = json.loads(summary_path.read_text()), json.loads(request_path.read_text())
    if (summary.get('complete') is not True or summary.get('command') != 'parity'
            or not summary.get('endpoint_bitwise') or not summary.get('pixel_bitwise')
            or summary.get('samples_per_loop') != 16 or request.get('num_steps') != 100
            or request.get('protocol') != PROTOCOL or request.get('seed') != SEED
            or summary.get('request', {}).get('sha256') != sha256_file(request_path)):
        raise ValueError('matching completed 16-image native official parity required')
    if request['identities'] != identities:
        raise ValueError('weights, configuration, stats, plan or protocol changed since parity')
    if {k: v['sha256'] for k, v in request['sources'].items()} != {k: v['sha256'] for k, v in sources.items()}:
        raise ValueError('source changed since parity')
    for name in ('geometry_diagnostics', 'candidate_formula_check', 'parity_result'):
        record = summary[name]
        if sha256_file(Path(record['path'])) != record['sha256']:
            raise ValueError(f'parity output identity changed: {name}')
    return {'request': artifact(request_path), 'summary': artifact(summary_path)}


def make_request(args, output):
    sources = archive_sources(output)
    config_id, weights_id = artifact(args.config), artifact(args.checkpoint)
    if config_id['sha256'] != CONFIG_SHA256 or weights_id['sha256'] != BASELINE_SHA256:
        raise ValueError('frozen official configuration and EMA checkpoint required')
    config = load_config(args.config)
    if (tuple(config.misc.latent_size) != LATENT_SHAPE or config.misc.num_classes != COUNT
            or config.transport.prediction != 'x' or float(config.transport.t_eps) != T_EPS
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != .1
            or config.guidance.ig.t_max != 1 or config.guidance.cfg.scale != 1):
        raise ValueError('official latent, transport or guidance settings changed')
    identities = {'config': config_id, 'baseline_checkpoint': weights_id,
                  'decoder_checkpoint': artifact(Path(config.stage_1.params['pretrained_decoder_path'])),
                  'normalization_stats': artifact(Path(config.stage_1.params['normalization_stat_path'])),
                  'root_plan': artifact(PLAN), 'protocol_document': artifact(PROTOCOL_DOCUMENT)}
    if identities['normalization_stats']['sha256'] != STATS_SHA256:
        raise ValueError('normalization statistics changed')
    plan = json.loads(PLAN.read_text())
    if plan.get('source_freeze_complete') is not True:
        raise ValueError('root source freeze is not complete')
    if (plan['cohort']['seed'] != SEED or plan['cohort']['n'] != COUNT
            or plan['cohort']['batch_size'] != BATCH_SIZE
            or Path(plan['protocol_document']['path']).resolve() != PROTOCOL_DOCUMENT.resolve()
            or plan['protocol_document']['sha256'] != identities['protocol_document']['sha256']):
        raise ValueError('root protocol/cohort differs from runner')
    verify_frozen_files(plan, identities, sources)
    stats = torch.load(Path(identities['normalization_stats']['path']), map_location='cpu', weights_only=False)
    if tuple(stats['mean'].shape) != LATENT_SHAPE:
        raise ValueError('normalization stats shape differs from latent')
    unit64, anchor64 = affine_geometry(stats['mean'], stats['var'])
    geometry = {'definition': 'sigma=sqrt(var+1e-5); u=sigma/||sigma||_channels; c=-(sum_channels mean)/||sigma||*u',
                'shape': list(unit64.shape), 'unit64_sha256': tensor_sha256(unit64),
                'anchor64_sha256': tensor_sha256(anchor64), 'unit32_sha256': tensor_sha256(unit64.float()),
                'anchor32_sha256': tensor_sha256(anchor64.float()),
                'stat_mean_dtype': str(stats['mean'].dtype), 'stat_variance_dtype': str(stats['var'].dtype),
                'construction_dtype': 'float64 on CPU, then one cast to float32',
                'source_is_known_encoder_affine_structure_not_fitted_reference': True}
    del stats
    prior = verify_parity(args.parity_dir, identities, sources) if args.command == 'sample' else None
    grid = shifted_time_grid(args.num_steps, 8, torch.device('cpu')).tolist()
    if not all(0 <= s < t <= 1 for t, s in zip(grid[:-1], grid[1:])):
        raise ValueError('invalid decreasing FP32 time grid')
    request = {'protocol': PROTOCOL, 'command': args.command, 'mode': args.mode, 'seed': SEED,
               'sample_count': args.num_samples, 'batch_size': BATCH_SIZE, 'num_steps': args.num_steps,
               'time_grid': grid, 'time_shift': 8, 'transport_t_eps': T_EPS,
               'identities': identities, 'sources': sources, 'parity': prior, 'geometry': geometry,
               'reflection': 'R_t z=z-2 P_N(z-(1-t)c); all normal coordinates reflected together, C2 group',
               'candidate': 'FP32 Pi_H[(G_native(z,t)+G_native(R_t z,t))*.5], at every query including outside IG window; FP32 Euler successor',
               'native_guidance': 'BF16 B+1.78*(F-B) inside [.1,1]; native Full outside; promote only final guided output toFP32',
               'pixel_arithmetic': 'native BF16 decoder, clamp and multiply255 before uint8 NHWC',
               'loop': 'both arms batch-major B8; candidate two separate B8 forwards perstep, never concatenate B16',
               'noise_schema': 'one full CUDA Generator(seed) draw[N,1024,16,16]; exact payload/RNG/labels SHA saved before first forward',
               'labels': 'ascending IDs0..N-1, labels=id%1000; all1000 classes once in sample',
               'precision': 'FP32 state/reflection/average/projection; BF16 native model/IG/decoder; TF32off',
               'cost_scope': 'synchronized full trajectory including all native forwards/reflection/projection, trajectory through decode, decoder, transfers, hashes, loads and writes recorded; no subtraction of operator costs',
               'instrumentation': 'same actual root-module FW/decoder count hooks for both arms; CUDA events only around whole trajectory/decode, no perstep events or synchronization',
               'parity_scope': '16 same-noise images: new official successor versus direct production.sampling_step, endpoints/pixels bitwise. Additionally firstB8 at t=1: Gsym(z) and Gsym(Rz), exactly4 native forwards; finite geometry/invariance/first-successor residuals, no tuned threshold or quality selection.',
               'theory_scope': 'known C2 symmetry and H-valued output; Euler equivariance/normal bridge requires t>=t_eps, true at all candidate100 query times; no FID guarantee',
               'automatic_step_selection': False, 'new_gain_schedule_or_projection_selection': False,
               'fid_performed': False, 'torch_version': str(torch.__version__)}
    atomic_json(output/'request.json', request)
    return request, config, unit64, anchor64


def load_models(config, checkpoint_path, device):
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from utils.model_utils import instantiate_from_config
    os.environ.setdefault('DINOV3_CKPT_DIR', '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3')
    install_raev2_decoder_config_compat()
    with Timer(device) as decoder_timer:
        decoder = instantiate_from_config(config.stage_1)
        del decoder.encoder
        decoder = decoder.to(device).eval().requires_grad_(False)
    with Timer(device) as model_timer:
        model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(checkpoint['ema'], strict=True)
        step = int(checkpoint['step'])
        del checkpoint
    gc.collect()
    return model, decoder, {'decoder_load_wall_seconds': decoder_timer.wall,
                           'backbone_load_wall_seconds': model_timer.wall, 'checkpoint_step': step}


def run_parity(model, decoder, noise, labels, grid, unit64, anchor64, output, device, audit):
    geometry = geometry_diagnostics(noise, unit64, anchor64)
    atomic_json(output/'geometry_diagnostics.json', geometry)
    endpoints, pixels, records = [], [], []
    for reference in (False, True):
        local_endpoints, local_pixels = [], []
        for start in range(0, 16, BATCH_SIZE):
            before = audit.snapshot()
            state = noise[start:start+BATCH_SIZE].to(device, copy=True)
            y = labels[start:start+BATCH_SIZE].to(device)
            with Timer(device) as complete:
                state, timing = trajectory(model, state, y, grid, mode='official', reference=reference)
                with Timer(device) as decoder_timer:
                    image = decode(decoder, state)
            local_endpoints.append(state.cpu()); local_pixels.append(image)
            counts = observed_delta(audit, before, calls=len(grid)-1, samples=BATCH_SIZE*(len(grid)-1),
                                    decoder_calls=1, decoder_samples=BATCH_SIZE)
            records.append({'reference': reference, 'start': start, **timing, 'counts': counts,
                            'trajectory_through_decode_wall_seconds': complete.wall,
                            'decode_and_uint8_wall_seconds': decoder_timer.wall, 'operator_counts': audit.flush_operations()})
        endpoints.append(torch.cat(local_endpoints)); pixels.append(np.concatenate(local_pixels))
    endpoint_equal = torch.equal(*endpoints)
    pixel_equal = np.array_equal(*pixels)
    off_counts = audit.snapshot()
    before = audit.snapshot()
    with Timer(device) as check_timer:
        z = noise[:BATCH_SIZE].to(device, copy=True)
        y = labels[:BATCH_SIZE].to(device)
        u64, c64 = unit64.to(device), anchor64.to(device)
        u, c = u64.float(), c64.float()
        reflected = mirror_normal_noise(z, 1., u, c)
        g = reflection_guided(model, z, y, 1., u, c, audit=audit)
        gr = reflection_guided(model, reflected, y, 1., u, c, audit=audit)
        following = grid[1]
        first_successor = euler_update(z, g, None, 1., following)
        errors = {'reflection_involution': mirror_normal_noise(reflected, 1., u, c).double()-z.double(),
                  'guided_affine_normal': normal_part(g.double()-c64, u64),
                  'guided_C2_invariance': gr.double()-g.double(),
                  'first_successor_normal_bridge': normal_part(first_successor.double(), u64)-
                    ((1-following)*c64+following*normal_part(z.double(), u64))}
        if not all(bool(torch.isfinite(value).all()) for value in (*errors.values(), g, gr, first_successor)):
            raise FloatingPointError('nonfinite actual candidate formula check')
        residuals = {name: {'max_abs': float(value.abs().max()),
                            'rms': float(value.square().mean().sqrt())} for name, value in errors.items()}
    check_counts = observed_delta(audit, before, calls=4, samples=4*BATCH_SIZE)
    check = {'sample_ids': list(range(BATCH_SIZE)), 't': 1., 'following': following,
             'residuals': residuals, 'all_finite': True,
             'counts': check_counts, 'operator_counts': audit.flush_operations(),
             'wall_seconds': check_timer.wall, 'cuda_event_span_seconds': check_timer.cuda_span,
             'guided_sha256': tensor_sha256(g), 'reflected_guided_sha256': tensor_sha256(gr),
             'scope': 'fixed first B8, exactly4 native model calls; descriptive FP32/BF16 residuals, no selected threshold or re-projection; no decode/FID'}
    atomic_json(output/'candidate_formula_check.json', check)
    result = {'complete': bool(endpoint_equal and pixel_equal), 'samples_per_loop': 16,
              'endpoint_bitwise': endpoint_equal, 'pixel_bitwise': pixel_equal,
              'max_endpoint_abs_difference': float((endpoints[0].double()-endpoints[1].double()).abs().max()),
              'pixel_mismatches': int(np.count_nonzero(pixels[0] != pixels[1])),
              'new_official_endpoint_sha256': tensor_sha256(endpoints[0]),
              'production_endpoint_sha256': tensor_sha256(endpoints[1]),
              'new_official_pixels_sha256': hashlib.sha256(pixels[0].tobytes()).hexdigest(),
              'production_pixels_sha256': hashlib.sha256(pixels[1].tobytes()).hexdigest(),
              'batches': records, 'forward_counts_observed': audit.snapshot(),
              'off_branch_forward_counts_observed': off_counts,
              'candidate_formula_check': artifact(output/'candidate_formula_check.json'),
              'candidate_formula_check_forward_counts': check_counts,
              'geometry_diagnostics': artifact(output/'geometry_diagnostics.json'),
              'parity_is_not_quality_or_cost_comparison': True}
    atomic_json(output/'parity_result.json', result)
    if not result['complete']:
        raise AssertionError('official native parity failed')
    result['parity_result'] = artifact(output/'parity_result.json')
    return result


def sample(model, decoder, noise, labels, grid, unit, anchor, args, output, device, audit):
    if noise.shape != (COUNT, *LATENT_SHAPE) or noise.dtype != torch.float32:
        raise ValueError('complete fixed1000 FP32 noise cohort required')
    if not torch.equal(labels.cpu(), torch.arange(COUNT)):
        raise ValueError('all1000 classes required once, in ascending order')
    started = time.perf_counter()
    writing = validation = transfer_wall = 0.
    manifest, all_pixels = [], []
    endpoint_hasher = hashlib.sha256()
    (output/'batches').mkdir()
    for start in range(0, COUNT, BATCH_SIZE):
        prep = time.perf_counter()
        state = noise[start:start+BATCH_SIZE].to(device, copy=True)
        y = labels[start:start+BATCH_SIZE].to(device)
        synchronize(device)
        transfer = time.perf_counter()-prep
        transfer_wall += transfer
        before = audit.snapshot()
        with Timer(device) as complete:
            state, timing = trajectory(model, state, y, grid, mode=args.mode,
                                       unit=unit, anchor=anchor, audit=audit)
            with Timer(device) as decoder_timer:
                image = decode(decoder, state)
        timing.update(trajectory_through_decode_wall_seconds=complete.wall,
                      trajectory_through_decode_cuda_event_span_seconds=complete.cuda_span,
                      decode_and_uint8_wall_seconds=decoder_timer.wall,
                      decode_and_uint8_cuda_event_span_seconds=decoder_timer.cuda_span,
                      noise_and_labels_transfer_wall_seconds=transfer)
        prep = time.perf_counter()
        factor = 2 if args.mode == 'reflection' else 1
        counts = observed_delta(audit, before, calls=factor*(len(grid)-1), samples=factor*BATCH_SIZE*(len(grid)-1),
                                decoder_calls=1, decoder_samples=BATCH_SIZE)
        operations = audit.flush_operations()
        if args.mode == 'reflection':
            if any(operations[name] != len(grid)-1 for name in ('reflection', 'average_projection')):
                raise AssertionError('every query requires exactly one reflection and projection')
        elif 'reflection' in operations or 'average_projection' in operations:
            raise AssertionError('official performed geometry work')
        if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
            raise FloatingPointError('nonfinite or non-FP32 endpoint')
        payload = state.detach().cpu().contiguous().numpy()
        endpoint_hasher.update(memoryview(payload).cast('B'))
        endpoint_hash = hashlib.sha256(memoryview(payload).cast('B')).hexdigest()
        noise_hash = tensor_sha256(noise[start:start+BATCH_SIZE])
        ids = np.arange(start, start+BATCH_SIZE, dtype=np.int64)
        validation += time.perf_counter()-prep
        prep = time.perf_counter()
        path = output/'batches'/f'{start:06d}_{start+BATCH_SIZE:06d}.npz'
        save_npz(path, image, ids=ids, labels=ids)
        manifest.append({'global_ids': ids.tolist(), 'noise_sha256': noise_hash,
                         'labels_sha256': tensor_sha256(labels[start:start+BATCH_SIZE]),
                         'endpoint_sha256': endpoint_hash, 'archive': artifact(path),
                         'trajectory': timing, 'forward_counts': counts, 'operator_counts': operations})
        all_pixels.append(image)
        atomic_json(output/'batch_manifest.json', {'batches': manifest})
        atomic_json(output/'progress.json', {'complete': False, 'phase': 'sampling', 'completed': start+BATCH_SIZE,
                    'forward_counts_observed': audit.snapshot(), 'elapsed_seconds': time.perf_counter()-started})
        writing += time.perf_counter()-prep
        if len(manifest) % 16 == 0 or start+BATCH_SIZE == COUNT:
            print(json.dumps({'mode': args.mode, 'decoded': start+BATCH_SIZE}), flush=True)
    prep = time.perf_counter()
    path = output/'samples.npz'
    save_npz(path, np.concatenate(all_pixels), ids=np.arange(COUNT, dtype=np.int64), labels=np.arange(COUNT, dtype=np.int64))
    archive = artifact(path)
    manifest_record = artifact(output/'batch_manifest.json')
    writing += time.perf_counter()-prep
    timing_names = manifest[0]['trajectory'].keys()
    timing_totals = {name: sum(row['trajectory'][name] for row in manifest)
                     if all(row['trajectory'][name] is not None for row in manifest) else None for name in timing_names}
    operation_totals = {}
    for batch in manifest:
        for name, value in batch['operator_counts'].items():
            operation_totals[name] = operation_totals.get(name, 0)+value
    expected = {'stage2_forward_calls': COUNT//BATCH_SIZE*(len(grid)-1)*factor,
                'stage2_sample_forwards': COUNT*(len(grid)-1)*factor,
                'decoder_forward_calls': COUNT//BATCH_SIZE, 'decoder_sample_forwards': COUNT}
    if audit.snapshot() != expected:
        raise AssertionError('total observed calls differ from fixed protocol')
    return {'complete': True, 'samples': COUNT, 'global_cohort_size': COUNT, 'global_ids': list(range(COUNT)),
            'sample_archive': archive, 'archive_sha256': archive['sha256'], 'batch_manifest': manifest_record,
            'global_endpoint_sha256': endpoint_hasher.hexdigest(), **audit.snapshot(),
            'forward_counts_observed': audit.snapshot(), 'forward_counts_expected': expected,
            'stage2_nfe_per_sample': (len(grid)-1)*factor, **timing_totals, 'operator_counts': operation_totals,
            'inference_trajectory_plus_decode_wall_seconds': timing_totals['trajectory_wall_seconds']+timing_totals['decode_and_uint8_wall_seconds'],
            'endpoint_hash_validation_and_count_collection_wall_seconds': validation,
            'output_write_hash_and_progress_wall_seconds': writing,
            'sampling_wall_including_output_seconds': time.perf_counter()-started,
            'reflection_calls': operation_totals.get('reflection', 0),
            'average_projection_calls': operation_totals.get('average_projection', 0),
            'accounting_overlap': 'trajectory_through_decode contains trajectory+decode; native forwards and all reflection/projection costs are included in trajectory, not separately timed. Full sampling wall also includes transfers, validation/hash/count collection and writing.',
            'output_format': 'arr_0 uint8 NHWC, ascending ids/labels; official ADM evaluator readable',
            'base_head_note': 'Full/Base share one stage2 forward at each queried input',
            'image_sampling_performed': True, 'fid_performed': False}


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    for name in ('output_dir', 'config', 'checkpoint', 'parity_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite existing run')
    output.mkdir(parents=True, exist_ok=True)
    audit = None
    try:
        request, config, unit64, anchor64 = make_request(args, output)
        preprocessing_wall = time.perf_counter()-started
        atomic_json(output/'progress.json', {'complete': False, 'phase': 'loading', 'run_started_utc': run_started_utc,
                    'request': artifact(output/'request.json')})
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one visible CUDA GPU required')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats(device)
        model, decoder, loading = load_models(config, args.checkpoint, device)
        peak_loading = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'reserved_bytes': torch.cuda.max_memory_reserved(device)}
        prep = time.perf_counter()
        unit, anchor = unit64.float().to(device), anchor64.float().to(device)
        synchronize(device)
        geometry_transfer_wall = time.perf_counter()-prep
        prep = time.perf_counter()
        noise, rng_state = paired_noise(SEED, device, count=request['sample_count'])
        synchronize(device)
        noise_wall = time.perf_counter()-prep
        prep = time.perf_counter()
        labels = torch.arange(request['sample_count'], dtype=torch.long)%1000
        save_npz(output/'paired_noise_audit.npz', first_noise=noise[0].numpy(), rng_state=rng_state.numpy())
        inputs = {'request': artifact(output/'request.json'), 'seed': SEED, 'noise_shape': list(noise.shape),
                  'global_noise_sha256': tensor_sha256(noise), 'noise_rng_state_sha256': tensor_sha256(rng_state),
                  'global_labels_sha256': tensor_sha256(labels), 'paired_noise_audit': artifact(output/'paired_noise_audit.npz'),
                  'cuda_device': torch.cuda.get_device_name(device), 'frozen_before_first_model_forward': True}
        atomic_json(output/'sampling_input.json', inputs)
        noise_identity_wall = time.perf_counter()-prep
        peak_preparation = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                            'reserved_bytes': torch.cuda.max_memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
        audit = ForwardAudit(model, decoder, device)
        if args.command == 'parity':
            result = run_parity(model, decoder, noise, labels, request['time_grid'], unit64, anchor64, output, device, audit)
        else:
            result = sample(model, decoder, noise, labels, request['time_grid'], unit, anchor, args, output, device, audit)
        result.update(protocol=PROTOCOL, command=args.command, mode=args.mode, run_started_utc=run_started_utc,
            request=artifact(output/'request.json'), sampling_input=artifact(output/'sampling_input.json'),
            seed=SEED, num_steps=args.num_steps, global_noise_sha256=inputs['global_noise_sha256'],
            noise_rng_state_sha256=inputs['noise_rng_state_sha256'], global_labels_sha256=inputs['global_labels_sha256'],
            preprocessing_source_weight_stats_hash_and_geometry_wall_seconds=preprocessing_wall,
            geometry_transfer_wall_seconds=geometry_transfer_wall, loading=loading,
            noise_generation_wall_seconds=noise_wall, noise_identity_and_audit_save_wall_seconds=noise_identity_wall,
            peak_loading=peak_loading, peak_loading_and_noise_preparation=peak_preparation,
            peak_sampling={'allocated_bytes': torch.cuda.max_memory_allocated(device),
                           'reserved_bytes': torch.cuda.max_memory_reserved(device)},
            total_wall_seconds_before_summary=time.perf_counter()-started,
            cpu_seconds_before_summary=time.process_time()-cpu_started,
            cost_not_yet_matched='No automated official step choice. Root must compare observed full costs before any cost-matched claim.')
        atomic_json(output/'summary.json', result)
        atomic_json(output/'progress.json', {'complete': True, 'summary': artifact(output/'summary.json')})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(output/'failure.json', {'complete': False, 'run_started_utc': run_started_utc,
                    'error': f'{type(error).__name__}: {error}', 'wall_seconds': time.perf_counter()-started,
                    'forward_counts_observed': audit.snapshot() if audit is not None else None})
        raise
    finally:
        if audit is not None:
            audit.close()


if __name__ == '__main__':
    main()
