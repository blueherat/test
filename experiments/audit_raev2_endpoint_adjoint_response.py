#!/usr/bin/env python3
"""Frozen eight-image, full-suffix endpoint-response mechanism audit.

prepare/finalize/summarize are CPU-only. collect/replay require CUDA, use B1, and keep
every image. This is an expensive open-loop sensitivity audit, not a deployed
guidance method or a FID experiment. FP32 derivatives belong to the explicitly
defined continuous surrogate, never to BF16 rounding or uint8 quantization.
"""
from __future__ import annotations

import time

START_WALL, START_CPU = time.perf_counter(), time.process_time()

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae')
RESTART = DATA / 'experiments/raev2_guidance_restart_20260906'
PROTOCOL = 'raev2_endpoint_adjoint_response_v1'
IDS = (0, 142, 285, 428, 570, 713, 856, 999)
SEED, STEPS, SHIFT, T_EPS = 202609111, 100, 8, .05
DELTA = 0.0018097258402418833
LATENT_SHAPE = (1024, 16, 16)
CONFIG = ROOT / 'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml'
CHECKPOINT = DATA / 'models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt'
PROTOTYPES = RESTART / 'endpoint_crossprototype_witness_v1'
EXPECTED = {
    'config': '3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342',
    'stage2': '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a',
    'decoder': '779871dc7e81d4c31cc5dc80824760c79b40e4a8b90e0d743d1e79c10d515d87',
    'stats': '40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77',
    'inception': '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2',
    'prototype_request': '65fc7ebac7eaec85bcbfbfdd6b4df3c73c54f1443438c61836502ba82d0f96fe',
    'prototype_values': '59a542a42b1fda5248dede846d6e12adb803faefb3317300ad737887e04339b1',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def verify(rec):
    path = Path(rec['path'])
    if path.stat().st_size != rec['bytes'] or sha256(path) != rec['sha256']:
        raise ValueError(f'Artifact changed: {path}')
    return path


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def write_json(path, value):
    # Every durable protocol/result is append-only, including the frozen lambda.
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def parse_ids(text):
    try:
        values = tuple(int(value) for value in text.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('IDs must be comma-separated integers') from error
    if not values or len(set(values)) != len(values) or any(value not in IDS for value in values):
        raise argparse.ArgumentTypeError(f'Use a nonempty, nonduplicated subset of {IDS}')
    return tuple(sorted(values))


def time_grid():
    base = torch.linspace(1., 0., STEPS + 1, dtype=torch.float32)
    return (SHIFT * base / (1. + (SHIFT - 1.) * base)).tolist()


def protocol_fields():
    return {'protocol': PROTOCOL, 'seed': SEED, 'global_ids_and_labels': list(IDS),
            'noise_shape': [len(IDS), *LATENT_SHAPE], 'batch_size': 1,
            'num_steps': STEPS, 'time_grid': time_grid(), 'delta': DELTA,
            'prototype_seed': 20260801, 'reference_seed': 20260802,
            'prototype_fold': 'train20260801_eval20260802', 't_eps': T_EPS,
            'ig_scale': 1.78, 'ig_interval': [.1, 1.],
            'control_location': 'z[k+1] = T[k](z[k]) + h[k] * u[k]',
            'control': 'u[k] = lambda * grad_successor(psi after full suffix); lambda = delta / mean_i(sum_k h[k]*sum_coordinates(a[i,k]^2))',
            'derivative': 'FP32 backbone/decoder/Inception continuous surrogate, clamp retained, no uint8, TF32 false; not a quantized/native Jacobian',
            'native': 'same FP32 resident weights under BF16 backbone/decoder autocast; native BF16 head arithmetic, clamp and multiply255 then uint8; FP32 Euler',
            'noise': 'one CUDA Generator(seed), one full-shape randn([8,1024,16,16]) per process; select by position in fixed IDs, never seed by shard or ID',
            'scope': 'Eight-image open-loop mechanism audit only; no FID, quality guarantee, deployment or fair-cost gain claim; every output retained',
            'delta_boundary': 'Historical scale_response IG mixing/pixel arithmetic differ from the current native protocol. Delta is only a fixed perturbation scale; it is not a measured deficit of this pilot or of the current native population. No proxy result automatically authorizes training.',
            'no_tuning': 'No ridge, clipping, lambda sweep, step window, output ranking, or retry with modified gains'}


def source_paths():
    relative = (
        'experiments/audit_raev2_endpoint_adjoint_response.py',
        'docs/RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md',
        'experiments/advfd_cleanroom/feature_extractors.py',
        'experiments/raev2_stage1_compat.py',
        'experiments/sample_raev2_pfr_retiming.py',
        'external/RAEv2/src/stage1/rae.py',
        'external/RAEv2/src/stage2/models/DDT.py',
        'external/RAEv2/src/stage2/models/model_utils.py',
        'external/RAEv2/src/utils/model_utils.py',
        'external/RAEv2/src/configs/stage2.py',
        'external/RAEv2/src/encoders/vision_encoder.py',
        'external/RAEv2/src/encoders/models/dinov3_loader.py',
    )
    result = {name: ROOT / name for name in relative}
    for path in sorted((ROOT / 'external/RAEv2/src/stage1/decoders').rglob('*.py')):
        result[str(path.relative_to(ROOT))] = path
    from torch_fidelity import feature_extractor_inceptionv3, interpolate_compat_tensorflow
    result['dependency/feature_extractor_inceptionv3.py'] = Path(feature_extractor_inceptionv3.__file__)
    result['dependency/interpolate_compat_tensorflow.py'] = Path(interpolate_compat_tensorflow.__file__)
    return result


def config_object():
    for path in (ROOT, ROOT / 'external/RAEv2/src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiments.sample_raev2_pfr_retiming import load_config
    return load_config(CONFIG)


def process_cost():
    return {'process_wall_seconds_since_initial_time_import': time.perf_counter() - START_WALL,
            'process_cpu_seconds_since_initial_time_import': time.process_time() - START_CPU,
            'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'excludes': 'final JSON write, interpreter exit, parent shell startup'}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    config = config_object()
    weights = Path(torch.hub.get_dir()) / 'checkpoints/weights-inception-2015-12-05-6726825d.pth'
    paths = {'config': CONFIG, 'stage2': CHECKPOINT,
             'decoder': Path(config.stage_1.params['pretrained_decoder_path']),
             'stats': Path(config.stage_1.params['normalization_stat_path']),
             'unused_encoder_constructed_by_RAE': Path(os.environ.get('DINOV3_CKPT_DIR', str(DATA / 'models/RAEv2/encoders/dinov3'))) / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
             'decoder_config': Path(config.stage_1.params['decoder_config_path']) / 'config.json',
             'inception': weights, 'prototype_request': PROTOTYPES / 'request.json',
             'prototype_summary': PROTOTYPES / 'summary.json',
             'prototype_values': PROTOTYPES / 'class_observables.npz'}
    identities = {key: record(path) for key, path in paths.items()}
    for key, expected in EXPECTED.items():
        if identities[key]['sha256'] != expected:
            raise ValueError(f'Frozen identity differs: {key}')
    summary = json.loads(paths['prototype_summary'].read_text())
    if summary['request']['sha256'] != EXPECTED['prototype_request']:
        raise ValueError('Prototype request link differs')
    if summary['artifacts']['class_observables']['sha256'] != EXPECTED['prototype_values']:
        raise ValueError('Prototype artifact link differs')
    fold = next(row for row in summary['results'] if row['name'] == 'train20260801_eval20260802')
    if -fold['contrasts']['ig_minus_source']['class_equal_weighted_mean'] != DELTA:
        raise ValueError('Fixed historical delta differs')
    with np.load(paths['prototype_values'], allow_pickle=False) as values:
        vectors = values['seed20260801_prototype_vectors']
        center = values['seed20260801_source_global_mean']
        if vectors.shape != (1000, 2048) or center.shape != (2048,):
            raise ValueError('Wrong prototype shapes')
        if not np.isfinite(vectors).all() or not np.isfinite(center).all():
            raise ValueError('Nonfinite prototype')
        np.savez(out / 'fixed_prototypes.npz', ids=np.asarray(IDS),
                 directions=vectors[list(IDS)].astype(np.float64) / 2048., center=center.astype(np.float64))
    sources = {}
    for name, path in source_paths().items():
        target = out / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[name] = {'current': record(path), 'archive': record(target)}
    request = {**protocol_fields(), 'identities': identities, 'sources': sources,
               'prototypes': record(out / 'fixed_prototypes.npz'),
               'torch_version': str(torch.__version__), 'prepare_cost': process_cost(),
               'unused_encoder_note': 'The standard RAE constructor loads an encoder; it is deleted before decoder use. Its weight identity and loading time are included, and encoder forward calls are zero.',
               'dinov3_repo_dir': os.environ.get('DINOV3_REPO_DIR'),
               'model_calls': 0, 'gpu_calls': 0}
    write_json(out / 'request.json', request)
    print(json.dumps({'prepared': str(out), 'request': record(out / 'request.json')}), flush=True)


def load_request(out, *, weights=False):
    request = json.loads((out / 'request.json').read_text())
    for key, expected in protocol_fields().items():
        if request.get(key) != expected:
            raise ValueError(f'Frozen protocol differs: {key}')
    if request['torch_version'] != str(torch.__version__):
        raise ValueError('Torch version differs')
    if request['dinov3_repo_dir'] != os.environ.get('DINOV3_REPO_DIR'):
        raise ValueError('DINO source override differs')
    for sources in request['sources'].values():
        verify(sources['current'])
        verify(sources['archive'])
    verify(request['prototypes'])
    if weights:
        for identity in request['identities'].values():
            verify(identity)
    return request


class Timing:
    def __init__(self, device=None):
        self.device = device

    def __enter__(self):
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *_):
        if self.device is not None:
            self.end.record()
            torch.cuda.synchronize(self.device)
        self.result = {'wall_seconds': time.perf_counter()-self.wall,
                       'cpu_seconds': time.process_time()-self.cpu}
        if self.device is not None:
            self.result['cuda_stream_elapsed_seconds'] = self.start.elapsed_time(self.end)/1000.
            self.result['cuda_timing_scope'] = 'CUDA event span; includes stream idle gaps, not summed kernel busy time'


def official_heads(full, base, times):
    mask = ((times >= .1) & (times <= 1)).reshape((-1,) + (1,)*(full.ndim-1))
    return torch.where(mask, base + 1.78*(full-base), full)


def euler_from_clean(state, clean, current, following):
    if not 0 <= following < current <= 1:
        raise ValueError('Invalid decreasing Euler grid')
    return state - (current-following)*((state-clean)/max(current, T_EPS))


def unit_step(model, state, labels, current, following, *, native, counts):
    counts['stage2_forward_calls'] += 1
    counts['stage2_sample_forwards'] += len(state)
    times = torch.full((len(state),), current, device=state.device, dtype=torch.float32)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if native else nullcontext()
    with context:
        full, base = model(state, times, context=labels, attn_mask=None)
        clean = official_heads(full, base, times)
    return euler_from_clean(state, clean.float(), current, following)


def suffix_adjoint(states, step, terminal_gradient, h, save, counts):
    """B_k^T at successor z[k+1], requiring A[K-1] ... A[1], never A[0]."""
    a = terminal_gradient.detach()
    gram = np.empty(len(h), dtype=np.float64)
    for k in range(len(h)-1, -1, -1):
        if not bool(torch.isfinite(a).all()):
            raise FloatingPointError(f'Nonfinite adjoint at step {k}')
        save(k, a)
        gram[k] = float(a.double().square().sum())
        if k > 0:
            state = torch.as_tensor(np.array(states[k], copy=True), device=a.device).unsqueeze(0)
            state.requires_grad_(True)
            with torch.enable_grad():
                following = step(state, k)
                counts['stage2_input_vjp_calls'] += 1
                counts['stage2_sample_input_vjps'] += len(state)
                a, = torch.autograd.grad(following, state, grad_outputs=a, create_graph=False)
            a = a.detach()
    return gram, float(np.dot(np.asarray(h, dtype=np.float64), gram))


def solve_cohort_gram(per_image_gram, *, expected_count=len(IDS), delta=DELTA):
    values = np.asarray(per_image_gram, dtype=np.float64)
    if values.shape != (expected_count,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Need all finite nonnegative per-image integrated Grams')
    gram = float(values.mean())
    if gram <= 0:
        raise ValueError('Positive delta is outside the zero Gram range; no ridge permitted')
    lam = delta / gram
    if not math.isfinite(lam):
        raise FloatingPointError('Nonfinite fixed lambda')
    return gram, lam


def native_pixels(decoded):
    if decoded.dtype != torch.bfloat16:
        raise ValueError('Native decoder did not return BF16')
    return decoded.clamp(0, 1).mul(255).to(torch.uint8)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'Nonfinite {name}; no clipping or gain adjustment permitted')


def observable(features, direction, center):
    return ((features.double()-center)*direction).sum(dim=1)


def load_models(device, request):
    config = config_object()
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    from utils.model_utils import instantiate_from_config
    os.environ.setdefault('DINOV3_CKPT_DIR', str(DATA / 'models/RAEv2/encoders/dinov3'))
    install_raev2_decoder_config_compat()
    actual_encoder = Path(os.environ['DINOV3_CKPT_DIR']) / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    if actual_encoder.resolve() != Path(request['identities']['unused_encoder_constructed_by_RAE']['path']).resolve():
        raise ValueError('RAE constructor would load a different unused encoder')
    costs = {}
    with Timing(device) as timing:
        decoder = instantiate_from_config(config.stage_1)
        del decoder.encoder
        decoder = decoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    costs['decoder_load_including_unused_encoder_construction'] = timing.result
    with Timing(device) as timing:
        model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
        if int(checkpoint['step']) != 100080:
            raise ValueError('Wrong EMA step')
        model.load_state_dict(checkpoint['ema'], strict=True)
        del checkpoint
    costs['stage2_load'] = timing.result
    with Timing(device) as timing:
        extractor = DifferentiableInception2048().to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    costs['inception_load'] = timing.result
    for module in (model, decoder, extractor):
        if any(parameter.dtype != torch.float32 for parameter in module.parameters()):
            raise ValueError('One FP32 resident parameter copy is required')
    actual_weights = Path(torch.hub.get_dir()) / 'checkpoints/weights-inception-2015-12-05-6726825d.pth'
    if sha256(actual_weights) != request['identities']['inception']['sha256']:
        raise ValueError('Inception loaded from a different cache')
    gc.collect()
    return model, decoder, extractor, costs


def endpoint(model_decoder, extractor, state, direction, center, *, native, counts, gradient=False):
    value = state.detach().requires_grad_(gradient)
    with torch.set_grad_enabled(gradient):
        context = torch.autocast('cuda', dtype=torch.bfloat16) if native else nullcontext()
        counts['decoder_forward_calls'] += 1
        counts['decoder_sample_forwards'] += len(value)
        with context:
            decoded = model_decoder.decode(value)
        finite(decoded, 'decoder output')
        pixels = native_pixels(decoded) if native else decoded.clamp(0, 1)
        counts['inception_forward_calls'] += 1
        counts['inception_sample_forwards'] += len(value)
        features = extractor.extractor(pixels)[0] if native else extractor(pixels)
        finite(features, 'Inception feature')
        psi = observable(features, direction, center)
        a = None
        if gradient:
            if native:
                raise ValueError('Native quantized endpoint has no claimed gradient')
            counts['decoder_input_vjp_calls'] += 1
            counts['inception_input_vjp_calls'] += 1
            a, = torch.autograd.grad(psi.sum(), value, create_graph=False)
            finite(a, 'terminal input gradient')
    return {'psi': float(psi.detach()[0]), 'features': features.detach().cpu().numpy(),
            'pixels': pixels.detach().cpu().numpy(), 'gradient': None if a is None else a.detach(),
            'clamp_fraction': float(((decoded.detach() < 0) | (decoded.detach() > 1)).float().mean())}


def save_endpoint(out, name, state, result):
    np.save(out / f'{name}_latent.npy', state.detach().cpu().numpy())
    np.save(out / f'{name}_pixels.npy', result['pixels'])
    np.save(out / f'{name}_features.npy', result['features'])
    return {'psi': result['psi'], 'clamp_fraction': result['clamp_fraction'],
            'state_sha256': tensor_hash(state),
            'artifacts': {kind: record(out / f'{name}_{kind}.npy') for kind in ('latent', 'pixels', 'features')}}


def check_feature_coordinates(decoder, extractor, state, out, counts):
    # This is an input-coordinate check at a native BF16 decode, not its gradient.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        counts['decoder_forward_calls'] += 1
        counts['decoder_sample_forwards'] += 1
        decoded = decoder.decode(state)
        finite(decoded, 'feature check decoder')
        pixels = native_pixels(decoded)
    with torch.no_grad():
        counts['inception_forward_calls'] += 1
        counts['inception_sample_forwards'] += 1
        expected = extractor.extractor(pixels)[0]
        counts['inception_forward_calls'] += 1
        counts['inception_sample_forwards'] += 1
        actual = extractor(pixels.float()/255.)
    maximum = float((expected-actual).abs().max())
    passed = bool(torch.allclose(expected, actual, rtol=2e-5, atol=2e-5))
    np.savez(out / 'feature_coordinate_check.npz', pixels=pixels.cpu().numpy(),
             expected=expected.cpu().numpy(), actual=actual.cpu().numpy())
    result = {'passed': passed, 'maximum_absolute_error': maximum, 'rtol': 2e-5, 'atol': 2e-5,
              'artifact': record(out / 'feature_coordinate_check.npz'),
              'meaning': 'same torch-fidelity feature coordinates on uint8/255; no claim about a quantization derivative'}
    write_json(out / 'feature_coordinate_check.json', result)
    if not passed:
        raise ValueError('Continuous/native Inception coordinate check failed')
    return result


def collect_image(out, image_id, noise, model, decoder, extractor, direction, center, request_hash, counts, global_noise_hash):
    out.mkdir(parents=True, exist_ok=False)
    grid = time_grid()
    h = np.asarray(grid[:-1])-np.asarray(grid[1:])
    state = noise.clone()
    labels = torch.tensor([image_id], device=state.device, dtype=torch.long)
    states = np.lib.format.open_memmap(out / 'states.npy', mode='w+', dtype=np.float32,
                                      shape=(STEPS+1, *LATENT_SHAPE))
    states[0] = state[0].cpu().numpy()
    state_hashes = [tensor_hash(state)]
    with Timing(state.device) as trajectory_time:
        with torch.no_grad():
            for k, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                state = unit_step(model, state, labels, current, following, native=False, counts=counts)
                finite(state, f'FP32 baseline state {k+1}')
                states[k+1] = state[0].cpu().numpy()
                state_hashes.append(tensor_hash(state))
        states.flush()
    with Timing(state.device) as endpoint_time:
        result = endpoint(decoder, extractor, state, direction, center, native=False, counts=counts, gradient=True)
        saved_endpoint = save_endpoint(out, 'fp32_baseline', state, result)
    coordinate_check = None
    if image_id == IDS[0]:
        with Timing(state.device) as coordinate_time:
            coordinate_check = check_feature_coordinates(decoder, extractor, state, out, counts)
        coordinate_check['timing'] = coordinate_time.result
    adjoints = np.lib.format.open_memmap(out / 'post_state_adjoint.npy', mode='w+', dtype=np.float32,
                                        shape=(STEPS, *LATENT_SHAPE))
    def save(k, value):
        adjoints[k] = value[0].detach().cpu().numpy()
    def step(value, k):
        return unit_step(model, value, labels, grid[k], grid[k+1], native=False, counts=counts)
    with Timing(state.device) as adjoint_time:
        per_step, gram = suffix_adjoint(states, step, result['gradient'], h, save, counts)
        adjoints.flush()
    del states, adjoints
    summary = {'protocol': PROTOCOL, 'phase': 'collect', 'complete': True, 'global_id': image_id,
               'request_sha256': request_hash, 'noise_sha256': tensor_hash(noise),
               'global_noise_sha256': global_noise_hash,
               'state_sha256_by_step': state_hashes, 'fp32_baseline': saved_endpoint,
               'post_state_squared_norms': per_step.tolist(), 'integrated_gram': gram,
               'states': record(out / 'states.npy'), 'adjoints': record(out / 'post_state_adjoint.npy'),
               'coordinate_check': coordinate_check,
               'timing': {'baseline_rollout': trajectory_time.result, 'terminal_decoder_feature_vjp': endpoint_time.result,
                          'full_suffix_adjoint': adjoint_time.result}}
    write_json(out / 'summary.json', summary)
    return summary


def collect_record(out, image_id, request_hash):
    path = out / 'collect' / f'id{image_id:04d}' / 'summary.json'
    summary = json.loads(path.read_text())
    if (summary.get('protocol') != PROTOCOL or summary.get('phase') != 'collect'
            or summary.get('complete') is not True or summary.get('global_id') != image_id
            or summary.get('request_sha256') != request_hash):
        raise ValueError(f'Invalid collect result: {path}')
    for key in ('states', 'adjoints'):
        verify(summary[key])
    for rec in summary['fp32_baseline']['artifacts'].values():
        verify(rec)
    return summary, record(path)


def finalize(out):
    load_request(out)
    request_hash = sha256(out / 'request.json')
    h = np.diff(-np.asarray(time_grid(), dtype=np.float64))
    summaries, identities = [], []
    for image_id in IDS:
        summary, identity = collect_record(out, image_id, request_hash)
        adjoints = np.load(summary['adjoints']['path'], mmap_mode='r', allow_pickle=False)
        if adjoints.shape != (STEPS, *LATENT_SHAPE) or adjoints.dtype != np.float32:
            raise ValueError('Wrong stored adjoint shape/dtype')
        norms = np.asarray([np.square(np.asarray(row, dtype=np.float64)).sum() for row in adjoints])
        if not np.isfinite(norms).all():
            raise ValueError('Nonfinite stored adjoints')
        np.testing.assert_allclose(norms, summary['post_state_squared_norms'], rtol=1e-10, atol=0)
        integrated = float(h @ norms)
        if not math.isclose(integrated, summary['integrated_gram'], rel_tol=1e-10, abs_tol=0):
            raise ValueError('Stored adjoint Gram disagrees with collect')
        summaries.append(summary)
        identities.append(identity)
    check = summaries[0]['coordinate_check']
    if len({row['global_noise_sha256'] for row in summaries}) != 1:
        raise ValueError('Collect processes did not use identical full-shape noise')
    if not check or check.get('passed') is not True:
        raise ValueError('Missing successful native/continuous feature-coordinate check')
    verify(check['artifact'])
    gram, lam = solve_cohort_gram([row['integrated_gram'] for row in summaries])
    frozen = {'protocol': PROTOCOL, 'phase': 'finalize', 'complete': True,
              'request_sha256': request_hash, 'global_ids': list(IDS),
              'global_noise_sha256': summaries[0]['global_noise_sha256'],
              'delta': DELTA, 'cohort_gram': gram, 'lambda': lam,
              'collect_summaries': identities,
              'predicted_mean_response': lam * gram,
              'minimum_surrogate_energy': DELTA * lam,
              'per_image_gram': [row['integrated_gram'] for row in summaries],
              'cost': process_cost(), 'model_calls': 0, 'gpu_calls': 0,
              'scope': 'Pilot empirical Gram; same eight trajectories, not independent population validation'}
    write_json(out / 'frozen_control.json', frozen)
    print(json.dumps(frozen), flush=True)


def load_frozen(out, request_hash):
    frozen = json.loads((out / 'frozen_control.json').read_text())
    if (frozen.get('protocol') != PROTOCOL or frozen.get('phase') != 'finalize'
            or frozen.get('complete') is not True or frozen.get('request_sha256') != request_hash
            or frozen.get('global_ids') != list(IDS) or frozen.get('delta') != DELTA):
        raise ValueError('Invalid frozen whole-cohort control')
    values = []
    if len(frozen['collect_summaries']) != len(IDS):
        raise ValueError('Missing collect summary identities')
    for image_id, rec in zip(IDS, frozen['collect_summaries']):
        path = verify(rec)
        summary = json.loads(path.read_text())
        if summary['global_id'] != image_id or summary['request_sha256'] != request_hash or not summary['complete']:
            raise ValueError('Collect identity/order differs')
        values.append(summary['integrated_gram'])
    gram, lam = solve_cohort_gram(values)
    if gram != frozen['cohort_gram'] or lam != frozen['lambda']:
        raise ValueError('Frozen lambda does not equal the complete cohort solution')
    return frozen


def replay_image(out, image_id, noise, model, decoder, extractor, direction, center, collected, frozen, counts):
    out.mkdir(parents=True, exist_ok=False)
    if tensor_hash(noise) != collected['noise_sha256']:
        raise ValueError('Replay noise differs from collect')
    adjoints = np.load(verify(collected['adjoints']), mmap_mode='r', allow_pickle=False)
    grid = time_grid()
    labels = torch.tensor([image_id], device=noise.device, dtype=torch.long)
    results = {}
    for name, native, controlled in (('fp32_controlled', False, True),
                                     ('native_baseline', True, False),
                                     ('native_controlled', True, True)):
        state, hashes = noise.clone(), [tensor_hash(noise)]
        predicted, energy = 0., 0.
        with Timing(state.device) as trajectory_time:
            with torch.no_grad():
                for k, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
                    state = unit_step(model, state, labels, current, following, native=native, counts=counts)
                    if controlled:
                        a = torch.from_numpy(np.array(adjoints[k], copy=True)).unsqueeze(0).to(state.device)
                        u = a * frozen['lambda']
                        finite(u, f'control {k}')
                        state = state + (current-following)*u
                        predicted += (current-following)*float((a.double()*u.double()).sum())
                        energy += (current-following)*float(u.double().square().sum())
                    finite(state, f'{name} state {k+1}')
                    hashes.append(tensor_hash(state))
        with Timing(state.device) as endpoint_time:
            result = endpoint(decoder, extractor, state, direction, center, native=native, counts=counts)
            saved = save_endpoint(out, name, state, result)
        results[name] = {**saved, 'state_sha256_by_step': hashes,
                         'linear_response_from_stored_fp32_control': predicted if controlled else None,
                         'implemented_control_energy': energy if controlled else 0.,
                         'timing': {'rollout': trajectory_time.result, 'endpoint': endpoint_time.result}}
    predicted = frozen['lambda']*collected['integrated_gram']
    fp32_change = results['fp32_controlled']['psi']-collected['fp32_baseline']['psi']
    native_change = results['native_controlled']['psi']-results['native_baseline']['psi']
    summary = {'protocol': PROTOCOL, 'phase': 'replay', 'complete': True, 'global_id': image_id,
               'request_sha256': frozen['request_sha256'], 'noise_sha256': tensor_hash(noise),
               'lambda': frozen['lambda'], 'results': results,
               'fp32_baseline_psi': collected['fp32_baseline']['psi'], 'predicted_response': predicted,
               'fp32_actual_response': fp32_change, 'native_actual_response': native_change,
               'fp32_linearization_error': fp32_change-predicted,
               'fp32_response_over_fixed_delta': fp32_change/DELTA,
               'native_response_over_fixed_delta': native_change/DELTA,
               'native_minus_fp32_response': native_change-fp32_change,
               'native_minus_fp32_baseline': results['native_baseline']['psi']-collected['fp32_baseline']['psi'],
               'scope': protocol_fields()['scope']}
    write_json(out / 'summary.json', summary)
    return summary


def gpu_phase(out, phase, ids):
    worker = out / 'workers' / (phase + '_' + '_'.join(f'{value:04d}' for value in ids))
    worker.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    device = torch.device('cuda:0')
    timings = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('collect/replay require CUDA; prepare/finalize are CPU-only')
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats(device)
        with Timing() as verification_time:
            request = load_request(out, weights=True)
        timings['source_and_weight_verification'] = verification_time.result
        request_hash = sha256(out / 'request.json')
        frozen = load_frozen(out, request_hash) if phase == 'replay' else None
        model, decoder, extractor, timings['model_loads'] = load_models(device, request)
        with Timing(device) as noise_time:
            generator = torch.Generator(device=device).manual_seed(SEED)
            all_noise = torch.randn((len(IDS), *LATENT_SHAPE), generator=generator,
                                    device=device, dtype=torch.float32).cpu()
            noise_hash = tensor_hash(all_noise)
            np.savez(worker / 'noise_audit.npz', noise=all_noise.numpy(), rng_state=generator.get_state().cpu().numpy(), ids=np.asarray(IDS))
        timings['full_noise_draw_copy_and_save'] = noise_time.result
        with np.load(verify(request['prototypes']), allow_pickle=False) as arrays:
            directions = torch.from_numpy(arrays['directions'].copy()).to(device)
            center = torch.from_numpy(arrays['center'].copy()).to(device)
        records = []
        with Timing(device) as phase_time:
            for image_id in ids:
                index = IDS.index(image_id)
                noise = all_noise[index:index+1].to(device)
                image_out = out / phase / f'id{image_id:04d}'
                before = counts.copy()
                with Timing(device) as image_time:
                    if phase == 'collect':
                        result = collect_image(image_out, image_id, noise, model, decoder, extractor,
                                               directions[index], center, request_hash, counts, noise_hash)
                    else:
                        collected, _ = collect_record(out, image_id, request_hash)
                        if noise_hash != collected['global_noise_sha256'] or noise_hash != frozen['global_noise_sha256']:
                            raise ValueError('Replay full-shape noise differs from collect/finalize')
                        result = replay_image(image_out, image_id, noise, model, decoder, extractor,
                                              directions[index], center, collected, frozen, counts)
                record_result = {'global_id': image_id, 'summary': record(image_out / 'summary.json'),
                                 'timing': image_time.result, 'calls': dict(counts-before)}
                write_json(image_out / 'execution.json', record_result)
                records.append(record_result)
                print(json.dumps({'phase': phase, 'completed_id': image_id, 'completed_ids': [r['global_id'] for r in records],
                                  'integrated_gram': result.get('integrated_gram'), 'calls': dict(counts)}), flush=True)
        timings['phase'] = phase_time.result
        report = {'protocol': PROTOCOL, 'phase': phase, 'complete': True,
                  'ids': list(ids), 'request': record(out / 'request.json'), 'global_noise_sha256': noise_hash,
                  'noise_artifact': record(worker / 'noise_audit.npz'), 'images': records,
                  'timing': timings, 'process_cost': process_cost(), 'calls': dict(counts),
                  'call_scope': 'Attempted calls, including coordinate checks; successful process means all completed. Base head shares stage2 forward. No parameter gradients.',
                  'cuda_device': torch.cuda.get_device_name(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'peak_gpu_allocated_bytes_including_model_load': torch.cuda.max_memory_allocated(device),
                  'peak_gpu_reserved_bytes_including_model_load': torch.cuda.max_memory_reserved(device),
                  'fid_calls': 0, 'training_updates': 0,
                  'frozen_control': record(out / 'frozen_control.json') if phase == 'replay' else None}
        write_json(worker / 'summary.json', report)
    except BaseException as error:
        failure = {'protocol': PROTOCOL, 'phase': phase, 'complete': False, 'ids': list(ids),
                   'error': f'{type(error).__name__}: {error}', 'calls_attempted': dict(counts),
                   'completed_timing': timings, 'cost': process_cost(), 'retry_with_changed_gains': False}
        if torch.cuda.is_initialized():
            failure['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        write_json(worker / 'failure.json', failure)
        raise


def summarize(out):
    """Report all eight finite responses, without a quality/automatic-training gate."""
    load_request(out)
    request_hash = sha256(out / 'request.json')
    frozen = load_frozen(out, request_hash)
    rows, costs, identities = [], Counter(), []
    for image_id in IDS:
        collected, _ = collect_record(out, image_id, request_hash)
        path = out / 'replay' / f'id{image_id:04d}' / 'summary.json'
        result = json.loads(path.read_text())
        if (result.get('protocol') != PROTOCOL or result.get('phase') != 'replay'
                or result.get('complete') is not True or result.get('global_id') != image_id
                or result.get('request_sha256') != request_hash or result.get('lambda') != frozen['lambda']
                or result.get('noise_sha256') != collected['noise_sha256']):
            raise ValueError(f'Invalid replay result {path}')
        for arm in result['results'].values():
            for artifact in arm['artifacts'].values():
                verify(artifact)
        identities.append(record(path))
        rows.append({'global_id': image_id, 'fp32_baseline_psi': collected['fp32_baseline']['psi'],
                     **{name+'_psi': arm['psi'] for name, arm in result['results'].items()},
                     'predicted_response': result['predicted_response'],
                     'fp32_actual_response': result['fp32_actual_response'],
                     'native_actual_response': result['native_actual_response'],
                     'fp32_response_over_fixed_delta': result['fp32_actual_response']/DELTA,
                     'native_response_over_fixed_delta': result['native_actual_response']/DELTA,
                     'fp32_minus_prediction': result['fp32_actual_response']-result['predicted_response'],
                     'integrated_gram': collected['integrated_gram'],
                     'post_state_squared_norms': collected['post_state_squared_norms'],
                     'implemented_control_energy': result['results']['fp32_controlled']['implemented_control_energy'],
                     'implemented_linear_response': result['results']['fp32_controlled']['linear_response_from_stored_fp32_control']})
        for phase in ('collect', 'replay'):
            execution_path = out / phase / f'id{image_id:04d}' / 'execution.json'
            execution = json.loads(execution_path.read_text())
            verify(execution['summary'])
            identities.append(record(execution_path))
            costs.update(execution['calls'])
    metric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    means = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    workers = []
    for path in sorted((out / 'workers').glob('*/*.json')):
        if path.name not in ('summary.json', 'failure.json'):
            continue
        workers.append({'artifact': record(path), 'report': json.loads(path.read_text())})
    report = {'protocol': PROTOCOL, 'phase': 'summarize', 'complete': True,
              'request': record(out / 'request.json'), 'frozen_control': record(out / 'frozen_control.json'),
              'global_ids': list(IDS), 'delta': DELTA, 'lambda': frozen['lambda'],
              'per_image': rows, 'eight_image_means': means,
              'successful_image_calls': dict(costs), 'worker_cost_reports_including_failures': workers,
              'call_cost_boundary': 'Image-call totals exclude model loading/noise and failed extra work. Worker reports retain these costs separately; CUDA event spans are not kernel busy time.',
              'inputs': identities, 'cpu_summary_cost': process_cost(),
              'fid_calls': 0, 'training_updates': 0,
              'scope': protocol_fields()['scope'], 'delta_boundary': protocol_fields()['delta_boundary'],
              'training_gate': 'None. This report never starts or authorizes training.'}
    write_json(out / 'response_summary.json', report)
    print(json.dumps({'eight_image_means': means, 'successful_image_calls': dict(costs),
                      'summary': record(out / 'response_summary.json')}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'collect', 'finalize', 'replay', 'summarize'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--ids', type=parse_ids, default=IDS)
    args = parser.parse_args(argv)
    out = args.output_root.expanduser().resolve()
    if args.phase == 'prepare':
        prepare(out)
    elif args.phase == 'finalize':
        finalize(out)
    elif args.phase == 'summarize':
        summarize(out)
    else:
        gpu_phase(out, args.phase, args.ids)


if __name__ == '__main__':
    main()
