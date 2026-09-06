#!/usr/bin/env python3
"""Fixed paired-bridge 1K sampling; root chooses the cost baseline before FID.

The candidate and mean-only control each apply the frozen FP32 two-call bridge
after every native Euler step. This runner never selects gains, weights or K.
"""
from __future__ import annotations

import argparse
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
from experiments.raev2_paired_bridge import PairedBridgeField, bridge_midpoint

PROTOCOL = 'raev2_paired_bridge_sampling_v1'
TRAINING_PROTOCOL = 'raev2_paired_bridge_fixed_mechanism_pilot_v1'
SEED, COUNT, BATCH_SIZE = 202609151, 1000, 8
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
DEFAULT_BRIDGE_CHECKPOINT = RESTART/'paired_bridge_v1/train/final.pt'
BRIDGE_SHA256 = '5013cbf075ddffa5c2ae021fc916be0615ca41d9817d3af91e6b3e46c86e1983'
STATS_SHA256 = '40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77'
DECODER_SHA256 = '779871dc7e81d4c31cc5dc80824760c79b40e4a8b90e0d743d1e79c10d515d87'
SOURCE_FILES = tuple(dict.fromkeys((
    'experiments/sample_raev2_paired_bridge.py',
    'tests/test_sample_raev2_paired_bridge.py',
    'experiments/raev2_paired_bridge.py',
    'experiments/run_raev2_paired_bridge_pilot.py',
    *PRODUCTION_SOURCES,
)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('parity', 'official', 'candidate', 'control'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parity-dir', type=Path)
    parser.add_argument('--bridge-checkpoint', type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
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
        parser.error(f'fixed seed {SEED} and {expected} samples required')
    if args.num_steps <= 0 or (args.mode != 'official' and args.num_steps != 100):
        parser.error('positive --num-steps may differ from 100 only in official mode')
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
    """Root-module hooks count actual calls and examples, without per-call timing."""
    def __init__(self, modules):
        self.counts, self.handles = {}, []
        for name, module in modules.items():
            self.counts[name+'_forward_calls'] = 0
            self.counts[name+'_sample_forwards'] = 0
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))

    def _pre(self, name):
        def hook(module, inputs):
            n = len(inputs[0])
            if n != BATCH_SIZE:
                raise ValueError(f'{name} requires separate B8 forwards, observed B{n}')
            self.counts[name+'_forward_calls'] += 1
            self.counts[name+'_sample_forwards'] += n
        return hook

    def snapshot(self):
        return dict(self.counts)

    def check_delta(self, before, calls):
        actual = {key: value-before[key] for key, value in self.counts.items()}
        expected = {key: 0 for key in actual}
        for name, count in calls.items():
            expected[name+'_forward_calls'] = count
            expected[name+'_sample_forwards'] = count*BATCH_SIZE
        if actual != expected:
            raise AssertionError(f'observed forwards {actual} != expected {expected}')
        return {'observed': actual, 'expected': expected}

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def successor(model, state, labels, current, following, *, mode, field=None):
    if mode not in ('official', 'candidate', 'control'):
        raise ValueError('invalid sampling mode')
    if (mode == 'official') != (field is None):
        raise ValueError('only candidate/control require a field')
    times = torch.full((len(state),), current, dtype=torch.float32, device=state.device)
    guided = clean_forward(model, state, times, labels)
    y = euler_update(state, guided, None, current, following)
    if mode == 'official':
        return y
    # Only native heads/IG used autocast, inside clean_forward. The bridge is FP32.
    return bridge_midpoint(field, y, current, following, labels,
                           tau_locked_zero=(mode == 'control'))


def trajectory(model, state, labels, grid, *, mode, field=None):
    with Timer(state.device) as timer, torch.no_grad():
        for current, following in zip(grid[:-1], grid[1:]):
            state = successor(model, state, labels, current, following, mode=mode, field=field)
    return state, {'trajectory_wall_seconds': timer.wall,
                   'trajectory_cuda_event_span_seconds': timer.cuda_span}


def decode(decoder, state):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return native_uint8(decoder.decode(state))


def verify_record(record):
    if artifact(Path(record['path'])) != record:
        raise ValueError(f'artifact binding changed: {record["path"]}')


def archive_sources(output):
    records = {}
    for relative in SOURCE_FILES:
        target = output/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
        records[relative] = artifact(target)
    return records


def verify_training(checkpoint_path, identities, *, load_weights=True):
    """Bind only this paired-bridge fit, never the old potential training chain."""
    record = artifact(checkpoint_path)
    if (record['sha256'] != BRIDGE_SHA256
            or checkpoint_path.resolve() != DEFAULT_BRIDGE_CHECKPOINT.resolve()):
        raise ValueError('fixed paired-bridge final checkpoint required')
    summary_path, request_path = checkpoint_path.parent/'summary.json', checkpoint_path.parent/'request.json'
    summary, request = json.loads(summary_path.read_text()), json.loads(request_path.read_text())
    request_id = artifact(request_path)
    if (not summary.get('complete') or summary.get('protocol') != TRAINING_PROTOCOL
            or summary.get('mode') != 'train' or summary.get('updates') != 2048
            or summary.get('checkpoint') != record or summary.get('request') != request_id):
        raise ValueError('completed fixed paired-bridge training summary required')
    if (request.get('protocol') != TRAINING_PROTOCOL or request.get('mode') != 'train'
            or request['config'] != identities['config']
            or request['baseline_checkpoint'] != identities['baseline_checkpoint']
            or request.get('baseline_checkpoint_step') != 100080):
        raise ValueError('training model/configuration differs from sampling')
    for relative, item in request['sources'].items():
        verify_record(item)
        if sha256_file(ROOT/relative) != item['sha256']:
            raise ValueError(f'paired-bridge training source changed: {relative}')
    supplement_path = checkpoint_path.parent.parent/'supplemental_source_environment.json'
    supplement = json.loads(supplement_path.read_text())
    for item in supplement['sources']:
        if sha256_file(ROOT/item['path']) != item['sha256']:
            raise ValueError(f'supplemental model source changed: {item["path"]}')
    weights = None
    if load_weights:
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if (weights.get('protocol') != TRAINING_PROTOCOL or weights.get('updates') != 2048
                or Path(weights['request']).resolve() != request_path.resolve()
                or weights['request_sha256'] != request_id['sha256']):
            raise ValueError('checkpoint internal training binding differs')
        for name in ('candidate', 'control'):
            if not all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                       for value in weights[name].values()):
                raise ValueError(f'nonfinite or non-FP32 {name} weights')
    return {'checkpoint': record, 'request': request_id, 'summary': artifact(summary_path),
            'supplemental_sources': artifact(supplement_path),
            'cost_boundary': 'Fit, data creation and mechanism diagnostics are additional costs disclosed by root; inference timing alone is not a total-cost success claim.'}, weights


def verify_parity(directory, identities, training, sources):
    summary_path, request_path = directory/'summary.json', directory/'request.json'
    summary, request = json.loads(summary_path.read_text()), json.loads(request_path.read_text())
    if (summary.get('complete') is not True or summary.get('command') != 'parity'
            or not all(summary.get(key) for key in ('stepwise_bitwise', 'endpoint_bitwise', 'pixel_bitwise'))
            or summary.get('samples_per_loop') != 16 or request.get('num_steps') != 100
            or request.get('protocol') != PROTOCOL or request.get('seed') != SEED
            or summary.get('request') != artifact(request_path)):
        raise ValueError('completed 16-image production/official/zero-bridge parity required')
    if request['identities'] != identities or request['training'] != training:
        raise ValueError('model or paired-bridge training identity changed since parity')
    if {k: v['sha256'] for k, v in request['sources'].items()} != {k: v['sha256'] for k, v in sources.items()}:
        raise ValueError('source changed since parity')
    verify_record(summary['parity_result'])
    verify_record(summary['finite_response'])
    return {'request': artifact(request_path), 'summary': artifact(summary_path)}


def make_request(args, output):
    sources = archive_sources(output)
    config_id, baseline_id = artifact(args.config), artifact(args.checkpoint)
    if config_id['sha256'] != CONFIG_SHA256 or baseline_id['sha256'] != BASELINE_SHA256:
        raise ValueError('frozen official configuration and EMA checkpoint required')
    config = load_config(args.config)
    if (tuple(config.misc.latent_size) != LATENT_SHAPE or config.misc.num_classes != COUNT
            or config.transport.prediction != 'x' or float(config.transport.t_eps) != T_EPS
            or config.guidance.ig.scale != 1.78 or config.guidance.ig.t_min != .1
            or config.guidance.ig.t_max != 1 or config.guidance.cfg.scale != 1):
        raise ValueError('official latent, transport or guidance settings changed')
    identities = {'config': config_id, 'baseline_checkpoint': baseline_id,
                  'decoder_checkpoint': artifact(Path(config.stage_1.params['pretrained_decoder_path'])),
                  'normalization_stats': artifact(Path(config.stage_1.params['normalization_stat_path']))}
    if (identities['normalization_stats']['sha256'] != STATS_SHA256
            or identities['decoder_checkpoint']['sha256'] != DECODER_SHA256):
        raise ValueError('frozen decoder or latent normalization changed')
    training, weights = verify_training(args.bridge_checkpoint, identities, load_weights=(args.mode != 'official'))
    parity = verify_parity(args.parity_dir, identities, training, sources) if args.command == 'sample' else None
    grid = shifted_time_grid(args.num_steps, 8, torch.device('cpu')).tolist()
    if not all(0 <= s < t <= 1 for t, s in zip(grid[:-1], grid[1:])):
        raise ValueError('invalid decreasing FP32 time grid')
    request = {'protocol': PROTOCOL, 'command': args.command, 'mode': args.mode,
               'seed': SEED, 'sample_count': args.num_samples, 'batch_size': BATCH_SIZE,
               'num_steps': args.num_steps, 'time_grid': grid, 'time_shift': 8,
               'transport_t_eps': T_EPS, 'identities': identities, 'training': training,
               'sources': sources, 'parity': parity,
               'candidate': 'native Euler Y then frozen bridge_midpoint(candidate,Y,t,s,labels): tau0 and tau.5, two current-state FP32 field calls, beta applied once each',
               'control': 'same midpoint with frozen control field, both tau inputs0, current midpoint state retained',
               'native_guidance': 'BF16 B+1.78*(F-B) inside [.1,1], Full outside; final guided output promoted toFP32',
               'noise_schema': 'one full CUDA Generator(seed) draw[N,1024,16,16] before batching; payload/RNG SHA saved before first model forward',
               'labels': 'ascending IDs0..N-1, labels=id%1000',
               'precision': 'native model/IG/decoder BF16; state/Euler/auxiliary field FP32; TF32off',
               'pixel_arithmetic': 'BF16 decoder, clamp and multiply255 before uint8 NHWC',
               'resident_models': 'stage2 and decoder in all arms; only the used candidate/control field is instantiated in sampling; parity additionally uses fresh zero-readout fields',
               'auxiliary_checkpoint_deserialized': args.mode != 'official',
               'auxiliary_verification_boundary': 'all modes verify file/source SHA and completed training summary; official skips auxiliary tensor deserialization and tensor checks, and instantiates no auxiliary field',
               'cost_scope': 'T=trajectory_through_decode_wall_seconds; W=total_wall_seconds_before_summary, from main entry after imports, includes source/checkpoint hashing, load, noise, transfers, serialization. Root separately records process wall and inherited costs.',
               'instrumentation': 'actual root-module forward hooks; candidate/control calls reported separately, never equated to stage2 NFE; whole trajectory/decode CUDA events only',
               'automatic_step_selection': False, 'fid_performed': False,
               'torch_version': str(torch.__version__), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    atomic_json(output/'request.json', request)
    return request, config, weights


def load_models(config, checkpoint_path, weights, device, *, mode):
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
        if step != 100080:
            raise ValueError('unexpected baseline checkpoint step')
        del checkpoint
    with Timer(device) as auxiliary_timer:
        fields = {}
        field_names = ('candidate', 'control') if mode == 'parity' else (() if mode == 'official' else (mode,))
        for name in field_names:
            field = PairedBridgeField()
            field.load_state_dict(weights[name], strict=True)
            fields[name] = field.to(device).eval().requires_grad_(False)
    gc.collect()
    return model, decoder, fields, {'decoder_load_wall_seconds': decoder_timer.wall,
                                   'backbone_load_wall_seconds': model_timer.wall,
                                   'auxiliary_load_wall_seconds': auxiliary_timer.wall,
                                   'checkpoint_step': step}


@torch.no_grad()
def run_parity(model, decoder, fields, noise, labels, grid, output, device, audit):
    modes = ('production', 'official', 'zero_candidate', 'zero_control')
    endpoint_hashers = {name: hashlib.sha256() for name in modes}
    pixel_hashers = {name: hashlib.sha256() for name in modes}
    records = []
    for start in range(0, 16, BATCH_SIZE):
        before = audit.snapshot()
        states = {name: noise[start:start+BATCH_SIZE].to(device, copy=True) for name in modes}
        y = labels[start:start+BATCH_SIZE].to(device)
        with Timer(device) as timer:
            for index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                states['production'] = production.sampling_step(model, None, states['production'], y,
                                                                 current, following, use_potential=False)
                for name in modes[1:]:
                    mode = name.removeprefix('zero_')
                    states[name] = successor(model, states[name], y, current, following, mode=mode,
                                             field=fields.get(name))
                    if not torch.equal(states[name], states['production']):
                        raise AssertionError(f'{name} differs at parity batch{start}, step{index}')
            pixels = {name: decode(decoder, state) for name, state in states.items()}
        for name in modes:
            if not np.array_equal(pixels[name], pixels['production']):
                raise AssertionError(f'{name} decoded parity differs')
            endpoint_hashers[name].update(states[name].cpu().contiguous().numpy().tobytes())
            pixel_hashers[name].update(pixels[name].tobytes())
        counts = audit.check_delta(before, {'stage2': 4*(len(grid)-1), 'decoder': 4,
                                           'zero_candidate': 2*(len(grid)-1), 'zero_control': 2*(len(grid)-1)})
        records.append({'start': start, 'steps_compared': len(grid)-1,
                        'stepwise_bitwise': True, 'pixel_bitwise': True,
                        'complete_loop_wall_seconds': timer.wall, 'forward_counts': counts})
    before = audit.snapshot()
    with Timer(device) as timer:
        z, y = noise[:BATCH_SIZE].to(device), labels[:BATCH_SIZE].to(device)
        current, following = grid[:2]
        native = successor(model, z, y, current, following, mode='official')
        outputs = {name: bridge_midpoint(fields[name], native, current, following, y,
                                        tau_locked_zero=(name == 'control')) for name in ('candidate', 'control')}
        if not all(bool(torch.isfinite(value).all()) for value in (native, *outputs.values())):
            raise FloatingPointError('nonfinite first-step bridge response')
        response = {name: {'rms_change': float((value-native).double().square().mean().sqrt()),
                           'max_abs_change': float((value-native).abs().max()),
                           'endpoint_sha256': tensor_sha256(value)} for name, value in outputs.items()}
    counts = audit.check_delta(before, {'stage2': 1, 'candidate': 2, 'control': 2})
    finite = {'complete': True, 'sample_ids': list(range(BATCH_SIZE)), 't': current, 's': following,
              'responses': response, 'forward_counts': counts, 'wall_seconds': timer.wall,
              'scope': 'fixed first B8 response only, no fitted threshold or quality evaluation'}
    atomic_json(output/'finite_response.json', finite)
    result = {'complete': True, 'samples_per_loop': 16, 'stepwise_bitwise': True,
              'endpoint_bitwise': True, 'pixel_bitwise': True,
              'endpoint_sha256': {name: h.hexdigest() for name, h in endpoint_hashers.items()},
              'pixels_sha256': {name: h.hexdigest() for name, h in pixel_hashers.items()},
              'batches': records, 'forward_counts_observed': audit.snapshot(),
              'finite_response': artifact(output/'finite_response.json'),
              'parity_is_not_quality_or_cost_comparison': True}
    atomic_json(output/'parity_result.json', result)
    result['parity_result'] = artifact(output/'parity_result.json')
    return result


def sample(model, decoder, fields, noise, labels, grid, args, output, device, audit):
    if noise.shape != (COUNT, *LATENT_SHAPE) or noise.dtype != torch.float32:
        raise ValueError('complete fixed1000 FP32 noise cohort required')
    if not torch.equal(labels.cpu(), torch.arange(COUNT)):
        raise ValueError('all1000 classes required once, ascending')
    started = time.perf_counter()
    writing = validation = 0.
    manifest, all_pixels = [], []
    endpoint_hasher = hashlib.sha256()
    (output/'batches').mkdir()
    steps = len(grid)-1
    expected_calls = {'stage2': steps, 'decoder': 1}
    if args.mode != 'official':
        expected_calls[args.mode] = 2*steps
    for start in range(0, COUNT, BATCH_SIZE):
        prep = time.perf_counter()
        state = noise[start:start+BATCH_SIZE].to(device, copy=True)
        y = labels[start:start+BATCH_SIZE].to(device)
        synchronize(device)
        transfer = time.perf_counter()-prep
        before = audit.snapshot()
        with Timer(device) as complete:
            state, timing = trajectory(model, state, y, grid, mode=args.mode, field=fields.get(args.mode))
            with Timer(device) as decoder_timer:
                image = decode(decoder, state)
        timing.update(trajectory_through_decode_wall_seconds=complete.wall,
                      trajectory_through_decode_cuda_event_span_seconds=complete.cuda_span,
                      decode_and_uint8_wall_seconds=decoder_timer.wall,
                      decode_and_uint8_cuda_event_span_seconds=decoder_timer.cuda_span,
                      noise_and_labels_transfer_wall_seconds=transfer)
        prep = time.perf_counter()
        counts = audit.check_delta(before, expected_calls)
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
                         'trajectory': timing, 'forward_counts': counts})
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
    timing_totals = {name: sum(row['trajectory'][name] for row in manifest)
                     if all(row['trajectory'][name] is not None for row in manifest) else None
                     for name in manifest[0]['trajectory']}
    total = audit.check_delta({key: 0 for key in audit.snapshot()},
                              {key: value*COUNT//BATCH_SIZE for key, value in expected_calls.items()})
    return {'complete': True, 'samples': COUNT, 'global_cohort_size': COUNT, 'global_ids': list(range(COUNT)),
            'sample_archive': archive, 'samples_npz': archive, 'archive_sha256': archive['sha256'], 'batch_manifest': manifest_record,
            'global_endpoint_sha256': endpoint_hasher.hexdigest(), **audit.snapshot(),
            'forward_counts_observed': total['observed'], 'forward_counts_expected': total['expected'],
            'stage2_nfe_per_sample': steps,
            'auxiliary_forwards_per_sample': {name: 2*steps if args.mode == name else 0 for name in ('candidate', 'control')},
            **timing_totals,
            'inference_trajectory_plus_decode_wall_seconds': timing_totals['trajectory_wall_seconds']+timing_totals['decode_and_uint8_wall_seconds'],
            'endpoint_hash_validation_and_count_collection_wall_seconds': validation,
            'output_write_hash_and_progress_wall_seconds': writing,
            'sampling_wall_including_output_seconds': time.perf_counter()-started,
            'accounting_overlap': 'trajectory_through_decode contains trajectory+decode. Auxiliary forwards, validation inside the fixed field, and bridge arithmetic are inside trajectory. Complete main wall includes transfers, hashes, loading and writes.',
            'output_format': 'arr_0 uint8 NHWC, ascending ids/labels; official ADM evaluator readable',
            'base_head_note': 'Full/Base share one stage2 forward; small auxiliary forwards are a distinct model cost, not stage2 NFE',
            'image_sampling_performed': True, 'fid_performed': False}


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    for name in ('output_dir', 'config', 'checkpoint', 'parity_dir', 'bridge_checkpoint'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite existing run')
    output.mkdir(parents=True, exist_ok=True)
    audit = None
    try:
        request, config, weights = make_request(args, output)
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
        model, decoder, fields, loading = load_models(config, args.checkpoint, weights, device, mode=args.mode)
        del weights
        if args.command == 'parity':
            # Fresh full architectures with zero readouts; loaded final weights remain untouched.
            with Timer(device) as zero_timer, torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                for name in ('zero_candidate', 'zero_control'):
                    fields[name] = PairedBridgeField().to(device).eval().requires_grad_(False)
            loading['parity_zero_fields_load_wall_seconds'] = zero_timer.wall
        peak_loading = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'reserved_bytes': torch.cuda.max_memory_reserved(device)}
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
                  'cuda_device': torch.cuda.get_device_name(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'frozen_before_first_model_forward': True}
        atomic_json(output/'sampling_input.json', inputs)
        noise_identity_wall = time.perf_counter()-prep
        peak_preparation = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                            'reserved_bytes': torch.cuda.max_memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
        audit = ForwardAudit({'stage2': model, 'decoder': decoder.decoder, **fields})
        if args.command == 'parity':
            result = run_parity(model, decoder, fields, noise, labels, request['time_grid'], output, device, audit)
        else:
            result = sample(model, decoder, fields, noise, labels, request['time_grid'], args, output, device, audit)
        result.update(protocol=PROTOCOL, command=args.command, mode=args.mode, run_started_utc=run_started_utc,
            request=artifact(output/'request.json'), sampling_input=artifact(output/'sampling_input.json'),
            seed=SEED, num_steps=args.num_steps, global_noise_sha256=inputs['global_noise_sha256'],
            noise_rng_state_sha256=inputs['noise_rng_state_sha256'], global_labels_sha256=inputs['global_labels_sha256'],
            preprocessing_source_and_weight_hash_wall_seconds=preprocessing_wall, loading=loading,
            noise_generation_wall_seconds=noise_wall, noise_identity_and_audit_save_wall_seconds=noise_identity_wall,
            peak_loading=peak_loading, peak_loading_and_noise_preparation=peak_preparation,
            peak_sampling={'allocated_bytes': torch.cuda.max_memory_allocated(device),
                           'reserved_bytes': torch.cuda.max_memory_reserved(device)},
            total_wall_seconds_before_summary=time.perf_counter()-started,
            cpu_seconds_before_summary=time.process_time()-cpu_started,
            cost_not_yet_matched='Root freezes the inference-cost baseline before FID; training/data/diagnostics remain additional disclosed costs.')
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
