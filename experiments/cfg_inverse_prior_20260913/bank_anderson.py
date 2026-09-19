"""Real-train preimages of the actual frozen CFG64 Heun *discrete* map.

Run as ``python -m experiments.cfg_inverse_prior_20260913.bank_anderson PHASE``.
Phases: prepare, preflight, invert --shard {0,1,2,3}, collect.
Preparation freezes 100 classes x (20 fit + 10 held-out) real source images
and one VAE posterior draw. No decoder, validation bank or FID is loaded.
The parent must review this source before prepare, then inspect preflight.json
before authorizing inversion with --preflight-sha256 SHA256_OF_THAT_REPORT.

Inverse step: reverse-Heun initializer, then Anderson-accelerate the map
f(x)=y-(F_k(x)-x), at most 16 updates, checking the actual F_k residual.
Memory is five and resets each physical step. FP64 per-source constrained
least squares minimizes the history residual with sum(weights)=1, using
trace-scaled 1e-6 regularization. With one history pair, the next state is f(x). Both stages
use the FORWARD left-endpoint CFG switch. Accepted RMS is <= 1e-5 for EVERY
source at EVERY step. This is a numerical local preimage search, not a proof
of global invertibility. No unconverged batch is silently accepted.

Batch files are immutable; resume validates their metadata, contents and hash.
The final bank.npz has noise/labels/split/source_ids (sourceids is an alias),
plus per-source step residuals, counts and Anderson update counts. inputs.npz
contains the matching clean images. No training or GPU run occurs on import.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

WORK = Path(__file__).resolve().parents[2]
BASE = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow')
CACHE = BASE / 'imagenet100_cmc_sdvae'
CHECKPOINT = BASE / 'runs/sit-s-2_seed0/checkpoints/step_00800000.pt'
OFFICIAL = Path('/home/zhoushunyu/data/research_repos/SiT/models.py')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson')
SEED = 2026091471
CONFIG = dict(kind='cfg', alpha=1.25, steps=64, cutoff=.75)
PREFLIGHT_ROWS = np.arange(8, dtype=np.int64) * 30
SOURCE_FILES = (
    Path(__file__).resolve(),
    WORK / 'experiments/cfg_transport_search_20260913/baselines.py',
    WORK / 'experiments/guidance_pasted_20260912/common.py',
    WORK / 'experiments/lifting_scale_sweep_20260909.py',
    WORK / 'experiments/sample_imagenet100_sit_foresight_fixed_point.py',
    WORK / 'experiments/imagenet100_sit_multiscale_models.py',
    WORK / 'experiments/imagenet100_sit_static_pair.py',
    WORK / 'experiments/imagenet100_sit_prediction_targets.py',
    WORK / 'experiments/train_imagenet100_sit_flow.py',
    OFFICIAL,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def source_hashes():
    return {str(path): sha(path) for path in SOURCE_FILES}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, *, arrays=None, document=None):
    """Atomic publish without overwriting a completed file, including races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if arrays is not None:
                np.savez(stream, **arrays)
            else:
                stream.write((json.dumps(document, indent=2, ensure_ascii=False,
                                         allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args):
    for name in ('inputs.npz', 'request.json'):
        if (args.out / name).exists():
            raise FileExistsError(args.out / name)
    if not 0 < args.tol <= 1e-5 or not 1 <= args.max_updates <= 16:
        raise ValueError('This pilot permits tol <= 1e-5 and <= 16 Anderson updates.')
    labels = np.load(CACHE / 'train_labels.npy', mmap_mode='r')
    ids = np.load(CACHE / 'train_source_indices.npy', mmap_mode='r')
    moments = np.load(CACHE / 'train_moments.npy', mmap_mode='r')
    manifest = read(CACHE / 'manifest.json')
    if manifest['source']['posterior_layout'] != 'channels 0:4 mean, channels 4:8 standard deviation':
        raise ValueError('Unexpected VAE posterior layout')
    rng = np.random.default_rng(SEED)
    indices, split = [], []
    for category in range(100):
        choices = np.flatnonzero(labels == category)
        if len(choices) < 30:
            raise ValueError(f'Insufficient class {category}: {len(choices)}')
        indices.extend(rng.permutation(choices)[:30].tolist())
        split.extend([0] * 20 + [1] * 10)
    indices = np.asarray(indices, dtype=np.int64)
    selected = np.array(moments[indices], dtype=np.float32)
    posterior_noise = rng.standard_normal((3000, 4, 32, 32), dtype=np.float32)
    clean = (selected[:, :4] + selected[:, 4:] * posterior_noise) * np.float32(.18215)
    arrays = dict(clean=clean, labels=np.array(labels[indices], dtype=np.int64),
                  source_ids=np.array(ids[indices], dtype=np.int64), indices=indices,
                  split=np.asarray(split, dtype=np.int8), posterior_noise=posterior_noise)
    validate_inputs(arrays)
    request = dict(format='cfg64_discrete_inverse_realtrain_anderson_v1', seed=SEED,
        sources=3000, fit_sources=2000, holdout_sources=1000, source_split='train',
        classes=100, per_class=30, fit_per_class=20, holdout_per_class=10,
        cache=str(CACHE), cache_manifest_sha256=sha(CACHE / 'manifest.json'),
        selected_posterior_moments_sha256=array_sha(selected),
        cache_full_moments_rehashed=False,
        label_cache_sha256=sha(CACHE / 'train_labels.npy'),
        source_id_cache_sha256=sha(CACHE / 'train_source_indices.npy'),
        no_fid_validation_loaded=True, posterior_draws_per_source=1,
        posterior_scaling_factor=.18215, new_noise_during_inverse=False,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT), weights='ema',
        config=CONFIG, dtype='float32', autocast=False, tf32=False,
        inverse=dict(method='reverse-Heun seed then Anderson on actual forward step',
                     anderson_memory=5, anderson_trace_regularization=1e-6,
                     anderson_solve_dtype='float64 per source',
                     anderson_first_update='ordinary fixed-point update',
                     rms_tolerance=args.tol, max_anderson_updates=args.max_updates,
                     convergence='every source, every step; residual evaluated after update',
                     state_abs_cap=1e6, rejected_batches_never_enter_bank=True),
        shards=4, batch_size=16, preflight_rows=PREFLIGHT_ROWS.tolist(),
        sources_sha256=source_hashes(), created_unix=time.time(),
        limitation='local numerical inverse certification only; no quality or global invertibility claim')
    write_new(args.out / 'inputs.npz', arrays=arrays)
    request['inputs_sha256'] = sha(args.out / 'inputs.npz')
    write_new(args.out / 'request.json', document=request)
    print(json.dumps(request), flush=True)


def validate_inputs(data):
    if data['clean'].shape != (3000, 4, 32, 32) or data['clean'].dtype != np.float32:
        raise ValueError('Expected frozen float32[3000,4,32,32] clean bank')
    if not np.isfinite(data['clean']).all():
        raise ValueError('Nonfinite input')
    for key, dtype in (('labels', np.int64), ('source_ids', np.int64),
                       ('indices', np.int64), ('split', np.int8)):
        if data[key].shape != (3000,) or data[key].dtype != dtype:
            raise ValueError((key, data[key].shape, data[key].dtype))
    if len(np.unique(data['source_ids'])) != 3000 or len(np.unique(data['indices'])) != 3000:
        raise ValueError('Source IDs / cache rows must be globally unique')
    if not np.array_equal(data['labels'], np.repeat(np.arange(100), 30)):
        raise ValueError('Unexpected class layout')
    if not np.array_equal(data['split'], np.tile([0] * 20 + [1] * 10, 100)):
        raise ValueError('Unexpected source-level split')


def load_frozen(root):
    request = read(root / 'request.json')
    if request['sources_sha256'] != source_hashes():
        raise RuntimeError('Frozen source hash mismatch; preserve this bank and use a new output directory')
    if request['inputs_sha256'] != sha(root / 'inputs.npz'):
        raise RuntimeError('Frozen inputs hash mismatch')
    if request['config'] != CONFIG or request['checkpoint_sha256'] != sha(CHECKPOINT):
        raise RuntimeError('Frozen CFG configuration or checkpoint mismatch')
    with np.load(root / 'inputs.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_inputs(data)
    return request, data


class NativeRuntime:
    """Only S800 EMA; field backend equals Runtime.field's SiT-small branch."""
    name = 'sit_small'

    def __init__(self, device):
        import torch
        from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
        from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO
        from experiments.sample_imagenet100_sit_foresight_fixed_point import _model_velocity
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
        self.model, self.semantics, self.metadata = load_sit_field_model(
            checkpoint_path=CHECKPOINT, weights='ema', sit_module=module,
            source_metadata=source, device=torch.device(device))
        if self.semantics.prediction_target != 'velocity':
            raise ValueError('Expected native velocity checkpoint')
        self.model.float().eval().requires_grad_(False)
        self.backend = _model_velocity
        self.labels = None
        self.counts = dict(full=0, prefix=0)
        self.branch_image_evaluations = 0

    def context(self):
        return nullcontext()

    def field(self, z, t, kind):
        import torch
        if kind != 'full' or z.dtype != torch.float32:
            raise ValueError('This inverse bank is FP32 full-model only')
        self.counts['full'] += 1
        self.branch_image_evaluations += len(z)
        return self.backend(self.model, self.semantics, z, t, self.labels, autocast_dtype=None)


def velocity(rt, z, t, labels, active):
    """Operation order mirrors frozen baselines.field(kind='cfg')."""
    import torch
    previous = rt.labels
    try:
        rt.labels = labels
        scalar = z.new_tensor(t)
        conditional = rt.field(z, scalar, 'full')
        if not active:
            return conditional
        rt.labels = torch.full_like(labels, 100)
        unconditional = rt.field(z, scalar, 'full')
    finally:
        rt.labels = previous
    gap = conditional - unconditional
    return conditional + CONFIG['alpha'] * gap


def check_state(z, location):
    import torch
    if not torch.isfinite(z).all() or z.abs().max() > 1e6:
        raise FloatingPointError(f'Invalid state at {location}')


def forward_step(rt, x, labels, k):
    t, h = k / 64, 1 / 64
    active = t < CONFIG['cutoff']
    first = velocity(rt, x, t, labels, active)
    second = velocity(rt, x + h * first, t + h, labels, active)
    return x + (h / 2) * (first + second)


def forward(rt, noise, labels, save_states=False):
    z = noise.clone()
    states = [z.detach().cpu().numpy().copy()] if save_states else None
    before = rt.counts['full']
    for k in range(64):
        z = forward_step(rt, z, labels, k)
        check_state(z, f'forward step {k}')
        if save_states:
            states.append(z.detach().cpu().numpy().copy())
    if rt.counts['full'] - before != 224:
        raise RuntimeError('Forward NFE differs from native CFG64')
    return z, np.stack(states) if save_states else None


class InverseFailure(RuntimeError):
    def __init__(self, message, arrays):
        super().__init__(message)
        self.arrays = arrays


def anderson_update(x, fixed_point_value, history):
    """Five-history Anderson mixing; no model queries or cross-source mixing.

    Solve (R R^T + 1e-6 mean(diag(R R^T)) I) b = 1, then a=b/sum(b).
    Dividing the Gram matrix by its trace scale is algebraically equivalent
    and avoids under/overflow for nearly converged sources. If all residuals
    of one source are zero, the normalized system yields uniform weights.
    """
    import torch
    history.append((x.detach().clone(), fixed_point_value.detach().clone()))
    del history[:-5]
    if len(history) == 1:
        return fixed_point_value
    residual = torch.stack([f.double() - old.double() for old, f in history], 1).flatten(2)
    values = torch.stack([f.double() for _, f in history], 1).flatten(2)
    gram = torch.bmm(residual, residual.transpose(1, 2))
    scale = gram.diagonal(dim1=1, dim2=2).mean(1)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = gram / divisor[:, None, None]
    identity = torch.eye(len(history), dtype=torch.float64, device=x.device)
    regularized = normalized + 1e-6 * identity[None]
    ones = torch.ones((len(x), len(history), 1), dtype=torch.float64, device=x.device)
    coefficients = torch.linalg.solve(regularized, ones)
    coefficients = coefficients / coefficients.sum(dim=1, keepdim=True)
    result = torch.bmm(coefficients.transpose(1, 2), values).squeeze(1)
    return result.reshape_as(x).to(x.dtype)


def inverse(rt, clean, labels, *, tol=1e-5, max_updates=16, save_states=False):
    import torch
    z = clean.clone()
    batch = len(z)
    residuals = np.full((batch, 64), np.nan, dtype=np.float64)
    initial_residuals = np.full_like(residuals, np.nan)
    maximum_seen = np.full(64, np.nan, dtype=np.float64)
    updates = np.full(64, -1, dtype=np.int16)
    calls = np.zeros(64, dtype=np.int64)
    states = np.full((65,) + tuple(z.shape), np.nan, dtype=np.float32) if save_states else None
    if save_states:
        states[64] = z.cpu().numpy()
    for k in reversed(range(64)):
        before = rt.counts['full']
        t, h = k / 64, 1 / 64
        active = t < CONFIG['cutoff']
        y = z
        first = velocity(rt, y, t + h, labels, active)
        second = velocity(rt, y - h * first, t, labels, active)
        x = y - (h / 2) * (first + second)
        history = []
        anderson_history = []
        for update in range(max_updates + 1):
            try:
                check_state(x, f'inverse candidate step {k}, update {update}')
                fx = forward_step(rt, x, labels, k)
                check_state(fx, f'inverse residual step {k}, update {update}')
            except FloatingPointError as error:
                raise InverseFailure(str(error), dict(step=np.array(k), candidate=x.cpu().numpy(),
                    target=y.cpu().numpy(), completed_residual_rms=residuals,
                    branch_calls=calls, failing_step_branch_calls=np.array(rt.counts['full'] - before))) from error
            # FP64 accumulation measures the residual of the FP32 map itself.
            per_source = (fx.double() - y.double()).square().flatten(1).mean(1).sqrt()
            history.append(per_source.cpu().numpy())
            if update == 0:
                initial_residuals[:, k] = history[-1]
            if bool((per_source <= tol).all()):
                residuals[:, k] = history[-1]
                updates[k] = update
                calls[k] = rt.counts['full'] - before
                maximum_seen[k] = np.max(history)
                break
            if update == max_updates:
                payload = dict(step=np.array(k), candidate=x.cpu().numpy(), target=y.cpu().numpy(),
                    residual_history_rms=np.stack(history), completed_residual_rms=residuals,
                    completed_updates=updates, branch_calls=calls,
                    failing_step_branch_calls=np.array(rt.counts['full'] - before))
                if save_states:
                    payload['partial_states'] = states
                raise InverseFailure(f'Step {k}: residual RMS {float(per_source.max()):.8g} '
                    f'exceeds {tol:g} after {max_updates} Anderson updates; batch rejected', payload)
            fixed_point_value = y - (fx - x)
            x = anderson_update(x, fixed_point_value, anderson_history)
        z = x
        if save_states:
            states[k] = z.cpu().numpy()
    output = dict(noise=z.cpu().numpy(), step_residual_rms=residuals,
        initial_step_residual_rms=initial_residuals, max_observed_rms_by_step=maximum_seen,
        anderson_updates=updates, branch_calls_by_step=calls)
    if save_states:
        output['states'] = states
    return output


def error_stats(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rms = np.sqrt(np.mean(delta.reshape(len(delta), -1) ** 2, axis=1))
    return dict(per_source_rms=rms.tolist(), mean_rms=float(rms.mean()),
                max_rms=float(rms.max()), max_abs=float(np.abs(delta).max()))


def inverse_stats(result):
    return dict(max_accepted_residual_rms=float(result['step_residual_rms'].max()),
        max_initializer_residual_rms=float(result['initial_step_residual_rms'].max()),
        max_observed_iteration_residual_rms=float(result['max_observed_rms_by_step'].max()),
        per_source_max_accepted_residual_rms=result['step_residual_rms'].max(axis=1).tolist(),
        anderson_updates_by_step=result['anderson_updates'].tolist(),
        branch_calls_by_step=result['branch_calls_by_step'].tolist(),
        branch_calls_per_source=int(result['branch_calls_by_step'].sum()),
        branch_image_evaluations=int(result['branch_calls_by_step'].sum() * len(result['noise'])))


def save_failure(directory, error, report):
    if isinstance(error, InverseFailure):
        write_new(directory / 'failure.npz', arrays=error.arrays)
    write_new(directory / 'failure.json', document={**report, 'passed': False,
        'error': str(error), 'exception': type(error).__name__})


def preflight(args):
    import torch
    from experiments.cfg_transport_search_20260913 import baselines
    request, data = load_frozen(args.out)
    directory = args.out / 'preflight'
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(request_sha256=sha(args.out / 'request.json'),
                  rows=PREFLIGHT_ROWS.tolist(), labels=data['labels'][PREFLIGHT_ROWS].tolist(),
                  sources_sha256=source_hashes(), runtime_dtype='float32', tf32=False,
                  no_autocast=True, preflight_seed=SEED + 1, passed=False)
    rt = NativeRuntime(args.device)
    report['model'] = rt.metadata
    rows = PREFLIGHT_ROWS
    clean = torch.from_numpy(data['clean'][rows]).to(args.device)
    labels = torch.from_numpy(data['labels'][rows]).to(args.device)
    noise_np = np.random.default_rng(SEED + 1).standard_normal(clean.shape, dtype=np.float32)
    noise = torch.from_numpy(noise_np).to(args.device)
    try:
        with torch.inference_mode(), rt.context():
            generated, known_forward_states = forward(rt, noise, labels, save_states=True)
            native = baselines.sample(rt, noise, labels, CONFIG, snapshots=True)
            equal = bool(torch.equal(generated, native['latents']))
            state_equal = {key: bool(np.array_equal(known_forward_states[int(key[5:])],
                                                    value.cpu().numpy()))
                           for key, value in native['snapshots'].items()}
            report['native_forward_bitwise_equal'] = equal and all(state_equal.values())
            report['native_snapshot_bitwise_equal'] = state_equal
            report['native_forward_error'] = error_stats(generated.cpu().numpy(), native['latents'].cpu().numpy())
            write_new(directory / 'native_forward.npz', arrays=dict(noise=noise_np, labels=data['labels'][rows],
                output=generated.cpu().numpy(), baseline_output=native['latents'].cpu().numpy(),
                states=known_forward_states))
            if not report['native_forward_bitwise_equal']:
                raise RuntimeError('Own forward differs bitwise from frozen native baselines.sample')
            kwargs = dict(tol=request['inverse']['rms_tolerance'],
                          max_updates=request['inverse']['max_anderson_updates'], save_states=True)
            recovered = inverse(rt, generated, labels, **kwargs)
            reconstructed, reconstruction_states = forward(rt,
                torch.from_numpy(recovered['noise']).to(args.device), labels, save_states=True)
            report['known_noise_inverse'] = inverse_stats(recovered)
            report['known_noise_preimage_error'] = error_stats(recovered['noise'], noise_np)
            report['known_noise_roundtrip_error'] = error_stats(reconstructed.cpu().numpy(), generated.cpu().numpy())
            write_new(directory / 'known_noise_inverse.npz', arrays={**recovered,
                'original_noise': noise_np, 'generated': generated.cpu().numpy(),
                'reconstructed': reconstructed.cpu().numpy(), 'forward_states': reconstruction_states})
            real = inverse(rt, clean, labels, **kwargs)
            copied, real_forward_states = forward(rt,
                torch.from_numpy(real['noise']).to(args.device), labels, save_states=True)
            report['real_inverse'] = inverse_stats(real)
            report['real_roundtrip_error'] = error_stats(copied.cpu().numpy(), clean.cpu().numpy())
            write_new(directory / 'real_inverse.npz', arrays={**real, 'clean': clean.cpu().numpy(),
                'reconstructed': copied.cpu().numpy(), 'forward_states': real_forward_states,
                'rows': rows, 'labels': data['labels'][rows], 'source_ids': data['source_ids'][rows]})
        report.update(passed=True, seconds=time.monotonic() - start,
            branch_image_evaluations=rt.branch_image_evaluations,
            pass_scope='bitwise forward equality and every local residual <= tolerance; '
                       'parent must inspect accumulated preimage/roundtrip errors before full-bank approval',
            files_sha256={p.name: sha(p) for p in directory.glob('*.npz')})
    except BaseException as error:
        report.update(seconds=time.monotonic() - start, branch_image_evaluations=rt.branch_image_evaluations)
        save_failure(directory, error, report)
        write_new(args.out / 'preflight.json', document={**report, 'passed': False, 'error': str(error)})
        raise
    write_new(args.out / 'preflight.json', document=report)
    print(json.dumps(report), flush=True)
    print('REVIEW_REQUIRED --preflight-sha256 ' + sha(args.out / 'preflight.json'), flush=True)


def approved_request(args):
    request, data = load_frozen(args.out)
    path = args.out / 'preflight.json'
    if not args.preflight_sha256 or args.preflight_sha256 != sha(path):
        raise ValueError('Supply the exact reviewed preflight.json SHA256 before inversion/collection')
    report = read(path)
    if not report['passed'] or report['request_sha256'] != sha(args.out / 'request.json'):
        raise ValueError('Preflight failed or belongs to a different frozen request')
    for filename, digest in report['files_sha256'].items():
        if sha(args.out / 'preflight' / filename) != digest:
            raise ValueError('Preflight artifact hash mismatch')
    return request, data


def batch_paths(root, begin):
    shard = (begin // 16) % 4
    stem = root / 'shards' / str(shard) / f'batch_{begin:06d}'
    return stem.with_suffix('.npz'), stem.with_suffix('.json')


def validate_batch(root, begin, request, data, preflight_sha256):
    npz_path, json_path = batch_paths(root, begin)
    if not npz_path.exists() and not json_path.exists():
        return None
    if not npz_path.exists() or not json_path.exists():
        raise RuntimeError(f'Unvalidated partial batch preserved: {npz_path}')
    metadata = read(json_path)
    stop = min(begin + 16, 3000)
    expected = dict(begin=begin, stop=stop, request_sha256=sha(root / 'request.json'),
                    preflight_sha256=preflight_sha256,
                    clean_slice_sha256=array_sha(data['clean'][begin:stop]))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Batch metadata mismatch: {npz_path}')
    if metadata.get('npz_sha256') != sha(npz_path) or not metadata.get('validated'):
        raise ValueError(f'Unvalidated/corrupted batch: {npz_path}')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays['noise'].dtype != np.float32 or arrays['noise'].shape != (stop - begin, 4, 32, 32):
        raise ValueError('Malformed inverse noise')
    if not np.isfinite(arrays['noise']).all() or np.abs(arrays['noise']).max() > 1e6:
        raise ValueError('Invalid inverse noise')
    for key in ('labels', 'split', 'source_ids', 'indices'):
        if not np.array_equal(arrays[key], data[key][begin:stop]):
            raise ValueError(f'Mismatched batch {key}')
    residuals = arrays['step_residual_rms']
    if residuals.shape != (stop - begin, 64) or not np.isfinite(residuals).all() or (residuals < 0).any():
        raise ValueError('Invalid per-source step residuals')
    if residuals.max() > request['inverse']['rms_tolerance']:
        raise ValueError('Batch failed local inverse tolerance')
    updates = arrays['anderson_updates']
    if updates.shape != (64,) or (updates < 0).any() or (updates > request['inverse']['max_anderson_updates']).any():
        raise ValueError('Invalid Anderson update counts')
    expected_calls = np.where(np.arange(64) < 48, 4, 2) * (2 + updates)
    if not np.array_equal(arrays['branch_calls_by_step'], expected_calls):
        raise ValueError('Inconsistent branch evaluation accounting')
    return arrays, metadata


def invert(args):
    import torch
    if args.shard is None:
        raise ValueError('invert requires --shard 0, 1, 2 or 3')
    request, data = approved_request(args)
    begins = [i for i in range(0, 3000, 16) if (i // 16) % 4 == args.shard]
    directory = args.out / 'shards' / str(args.shard)
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for begin in begins:
        if validate_batch(args.out, begin, request, data, args.preflight_sha256) is None:
            if (directory / f'failure_{begin:06d}').exists():
                raise RuntimeError(f'Previous failed batch {begin} requires parent review; no silent retry')
            pending.append(begin)
    rt = NativeRuntime(args.device) if pending else None
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for begin in pending:
                stop = min(begin + 16, 3000)
                clean = torch.from_numpy(data['clean'][begin:stop]).to(args.device)
                labels = torch.from_numpy(data['labels'][begin:stop]).to(args.device)
                before = rt.branch_image_evaluations
                result = inverse(rt, clean, labels, tol=request['inverse']['rms_tolerance'],
                    max_updates=request['inverse']['max_anderson_updates'])
                counts = rt.branch_image_evaluations - before
                if counts != int(result['branch_calls_by_step'].sum()) * (stop - begin):
                    raise RuntimeError('Inverse cost accounting mismatch')
                arrays = {**result, **{key: data[key][begin:stop]
                    for key in ('labels', 'split', 'source_ids', 'indices')}}
                metadata = dict(begin=begin, stop=stop, shard=args.shard, validated=True,
                    request_sha256=sha(args.out / 'request.json'), preflight_sha256=args.preflight_sha256,
                    clean_slice_sha256=array_sha(data['clean'][begin:stop]),
                    inverse_stats=inverse_stats(result), branch_image_evaluations=counts)
                npz_path, json_path = batch_paths(args.out, begin)
                write_new(npz_path, arrays=arrays)
                metadata['npz_sha256'] = sha(npz_path)
                write_new(json_path, document=metadata)
                validate_batch(args.out, begin, request, data, args.preflight_sha256)
                print(json.dumps(dict(shard=args.shard, begin=begin, stop=stop,
                    seconds=time.monotonic() - start, branch_image_evaluations=counts,
                    max_residual_rms=float(result['step_residual_rms'].max()))), flush=True)
    except BaseException as error:
        failure = directory / f'failure_{begin:06d}'
        failure.mkdir(exist_ok=False)
        save_failure(failure, error, dict(begin=begin, shard=args.shard,
            total_branch_image_evaluations_this_process=rt.branch_image_evaluations,
            request_sha256=sha(args.out / 'request.json')))
        raise
    all_metadata = [validate_batch(args.out, begin, request, data, args.preflight_sha256)[1]
                    for begin in begins]
    summary = dict(complete=True, shard=args.shard, sources=sum(x['stop'] - x['begin'] for x in all_metadata),
        batches=len(begins), request_sha256=sha(args.out / 'request.json'),
        preflight_sha256=args.preflight_sha256,
        branch_image_evaluations=sum(x['branch_image_evaluations'] for x in all_metadata),
        max_accepted_residual_rms=max(x['inverse_stats']['max_accepted_residual_rms'] for x in all_metadata),
        seconds_this_process=time.monotonic() - start)
    summary_path = directory / 'summary.json'
    if not summary_path.exists():
        write_new(summary_path, document=summary)
    print(json.dumps(summary), flush=True)


def collect(args):
    """CPU-only merge. Every row must belong to a validated, immutable batch."""
    request, data = approved_request(args)
    if (args.out / 'bank.npz').exists() or (args.out / 'bank_summary.json').exists():
        raise FileExistsError('Final bank already exists; refusing to overwrite')
    batches, metadata = [], []
    for begin in range(0, 3000, 16):
        result = validate_batch(args.out, begin, request, data, args.preflight_sha256)
        if result is None:
            raise FileNotFoundError(f'Missing batch at row {begin}')
        arrays, record = result
        batches.append(arrays)
        metadata.append(record)
    output = {key: np.concatenate([x[key] for x in batches]) for key in
              ('noise', 'labels', 'split', 'source_ids', 'indices', 'step_residual_rms',
               'initial_step_residual_rms')}
    output['sourceids'] = output['source_ids']
    output['anderson_updates'] = np.concatenate([np.tile(x['anderson_updates'], (len(x['noise']), 1)) for x in batches])
    output['branch_calls_per_source'] = np.concatenate([
        np.full(len(x['noise']), x['branch_calls_by_step'].sum(), dtype=np.int64) for x in batches])
    if output['noise'].shape != (3000, 4, 32, 32):
        raise RuntimeError('Incomplete collected bank')
    for key in ('labels', 'split', 'source_ids', 'indices'):
        if not np.array_equal(output[key], data[key]):
            raise RuntimeError(f'Collected {key} does not match frozen input order')
    write_new(args.out / 'bank.npz', arrays=output)
    summary = dict(complete=True, sources=3000, fit_sources=2000, holdout_sources=1000,
        request_sha256=sha(args.out / 'request.json'), preflight_sha256=args.preflight_sha256,
        bank_sha256=sha(args.out / 'bank.npz'),
        max_accepted_residual_rms=float(output['step_residual_rms'].max()),
        per_source_max_accepted_residual_rms=output['step_residual_rms'].max(axis=1).tolist(),
        production_branch_image_evaluations=sum(x['branch_image_evaluations'] for x in metadata),
        preflight_branch_image_evaluations=read(args.out / 'preflight.json')['branch_image_evaluations'],
        first8_real_roundtrip='preflight/real_inverse.npz, bank rows 0,30,...,210; separate B8 preflight',
        full_bank_roundtrip_checked=False, local_discrete_residuals_checked_for_every_source=True,
        preflight_not_quality_evidence=True)
    write_new(args.out / 'bank_summary.json', document=summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_source_max_accepted_residual_rms'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'invert', 'collect'))
    parser.add_argument('--out', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, choices=range(4))
    parser.add_argument('--preflight-sha256', default=None)
    parser.add_argument('--tol', type=float, default=1e-5, help='prepare only; frozen thereafter')
    parser.add_argument('--max-updates', type=int, default=16, help='prepare only; frozen thereafter')
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
