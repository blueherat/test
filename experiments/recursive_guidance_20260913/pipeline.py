"""Frozen queue: each setting generates round 0 and five full feedback rounds.

--cpu-check and --prepare never use CUDA. Run --check on one available GPU,
then --pipeline waits for the explicitly named existing job and idle GPUs.
Sources are archived and checked in their original checkout; changing a frozen
source stops the queue rather than silently mixing implementations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import torch

from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, array_sha, atomic, read, sha
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.small_sit_carrier_flow_20260909 import REFERENCE
from experiments.recursive_guidance_20260913 import core, catalog, reporting, gallery

ROOT = EXPS / 'recursive_guidance_20260913'
STAGE = 'recursive_screen_1k'
MODULE = 'experiments.recursive_guidance_20260913.pipeline'
PROTOCOL = WORK / 'docs/RECURSIVE_GUIDANCE_PROTOCOL_20260913_ZH.md'
WAIT_ROOT = EXPS / 'jit_readout_cfg_application_20260913'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
FID_PYTHON = '/data/shared/envs/adm-fid/bin/python'
GRAPH = Path('/data/shared/adm_refs/classify_image_graph_def.pb')
SAMPLES, BATCH, RANKS, SEED = 1000, 8, 4, 2026131301
ROUNDS = 6
RENOISE_START = .25
DIAGNOSTICS = ('correction_to_native_norm', 'off_affine_energy', 'clean_norm_ratio',
               'constraint_fraction', 'zero_gap_fraction')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False).encode()).hexdigest()


def configurations():
    values = json.loads(json.dumps(catalog.configurations(), allow_nan=False))
    required = {'arm', 'family', 'key', 'source', 'strength', 'theta', 'solver',
                'cutoff', 'role', 'idea_id', 'parameters'}
    assert values and len({cfg['arm'] for cfg in values}) == len(values)
    for cfg in values:
        assert required <= cfg.keys(), (cfg.get('arm'), required - cfg.keys())
        assert re.fullmatch(r'[A-Za-z0-9_\-]+', cfg['arm']), cfg['arm']
        assert cfg['source'] in ('cfg', 'ig') and cfg['strength'] >= 0
    counts = Counter(cfg['idea_id'] for cfg in values if cfg['role'] == 'candidate')
    assert len(counts) == 10 and set(counts.values()) == {20}, counts
    assert len([cfg for cfg in values if cfg['role'] != 'candidate']) > 0
    return values


def sources(extra=()):
    paths = {Path(__file__).resolve(), Path(core.__file__).resolve(),
             Path(catalog.__file__).resolve(), Path(reporting.__file__).resolve(),
             Path(gallery.__file__).resolve(),
             WORK / 'experiments/compute_adm_fid.py', WORK / 'train_gen/evaluator.py',
             WORK / 'docs/RECURSIVE_GUIDANCE_DESIGN_REVIEW_20260913_ZH.md',
             *map(Path, extra)}
    paths.update(map(Path, core.source_paths()))
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if not value or Path(value).name.startswith('<'):
            continue
        path = Path(value).resolve()
        if path.is_relative_to(WORK) and path.suffix == '.py' and path.is_file():
            paths.add(path)
    if PROTOCOL.exists():
        paths.add(PROTOCOL)
    return sorted(path.resolve() for path in paths)


def assets():
    from huggingface_hub import snapshot_download
    paths = {Path(path).resolve() for path in core.asset_paths()}
    snapshot = Path(snapshot_download('stabilityai/sd-vae-ft-mse', local_files_only=True))
    vae_files = [snapshot / name for name in ('config.json', 'diffusion_pytorch_model.bin',
                                            'diffusion_pytorch_model.safetensors')
                 if (snapshot / name).exists()]
    assert (snapshot / 'config.json').exists() and len(vae_files) >= 2
    # Keep snapshot symlink paths: changed blob contents or targets are detected.
    paths.update(vae_files)
    main_reference = snapshot.parent.parent / 'refs/main'
    if main_reference.exists():
        paths.add(main_reference)
    return sorted(paths)


def versions():
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'torchdiffeq', 'diffusers', 'numpy', 'scipy')}


def verify_hashes(mapping):
    for path, digest in mapping.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise RuntimeError(f'Frozen file changed or missing: {path}; archived sources are retained. '
                               'Use the matching source version to resume this request.')


def cpu_check():
    configs = configurations()
    result = dict(passed=True, core=core.cpu_checks(), configurations=len(configs),
                  candidates=200, candidate_settings_per_idea=20, cuda_used=False,
                  feedback_rounds=ROUNDS-1, generated_config_rounds=len(configs)*ROUNDS)
    # Exercise our resume identities without allocating images or a GPU.
    with tempfile.TemporaryDirectory(prefix='recursive_queue_cpu_') as temporary:
        base = Path(temporary)
        cfg = configs[0]
        directory = round_directory(base, cfg['arm'], 0)
        directory.mkdir(parents=True)
        row = dict(arm=cfg['arm'], round_index=0, request_sha256='test', complete=False, fid=None)
        atomic(directory / 'result.json', row)
        atomic(directory / 'commit.json', dict(request_sha256='test',
               config_sha256=canonical_hash(cfg),
               files={'result.json': sha(directory / 'result.json')}))
        assert committed(base, cfg, 'test') == row
        atomic(directory / 'result.json', dict(row, altered=True))
        try:
            committed(base, cfg, 'test')
        except AssertionError:
            pass
        else:
            raise AssertionError('Tampered committed output was accepted')
    result['resume_tamper_rejected'] = True
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def hooks(rt):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in rt.model.modules()]


def verify_stats(stats, before, rt):
    assert int(stats['full_calls']) == rt.counts['full'] - before['full']
    assert int(stats['prefix_calls']) == rt.counts['prefix'] - before['prefix']
    assert 0 <= int(stats['auxiliary_full_calls']) <= int(stats['full_calls'])
    assert np.isfinite(stats['diagnostics']).all()


@torch.inference_mode()
def development_check():
    cpu = cpu_check()
    configs = configurations()
    core.install()
    rt = core.make_runtime()
    generator = torch.Generator(device='cuda').manual_seed(SEED - 1)
    noise = torch.randn((2, 4, 32, 32), device='cuda', generator=generator)
    labels = torch.tensor([0, 99], device='cuda')
    rt.labels = labels
    original_hooks = hooks(rt)
    limits = core.limiting_checks(rt, noise, labels)
    strong = next(cfg for cfg in configs if cfg['strength'] == 0)
    baseline, _ = core.sample(rt, noise, labels, strong)
    baselines = {strong['solver']: baseline}
    assert hasattr(core, 'preflight_grid'), 'core.preflight_grid must cover every candidate setting'
    grid = core.preflight_grid(rt, noise, labels, configs)
    trajectories = []
    groups = list(dict.fromkeys(cfg['family'] for cfg in configs))
    for family in groups:
        cfg = [cfg for cfg in configs if cfg['family'] == family][-1]
        before = dict(rt.counts)
        z, stats = core.sample(rt, noise, labels, cfg)
        assert z.shape == noise.shape and torch.isfinite(z).all(), cfg['arm']
        verify_stats(stats, before, rt)
        zero, _ = core.sample(rt, noise, labels, cfg, zero=True)
        if cfg['solver'] not in baselines:
            baselines[cfg['solver']], _ = core.sample(rt, noise, labels, dict(strong, solver=cfg['solver']))
        assert torch.equal(zero, baselines[cfg['solver']]), f'Zero-strength trajectory mismatch: {cfg["arm"]}'
        assert hooks(rt) == original_hooks and rt.labels is labels, cfg['arm']
        trajectories.append(dict(arm=cfg['arm'], family=family,
            full_calls=int(stats['full_calls']), prefix_calls=int(stats['prefix_calls']),
            max_abs=float(z.abs().max()), zero_exact=True))
        print(json.dumps(dict(preflight_trajectory=cfg['arm'], **trajectories[-1])), flush=True)
    repeat, _ = core.sample(rt, noise, labels, strong)
    assert torch.equal(repeat, baseline)
    pixels = rt.decode(baseline)
    assert pixels.shape == (2, 256, 256, 3) and pixels.dtype == np.uint8
    recursive_checks = core.feedback_checks(rt, noise, labels)
    runtime_sources = dict(rt.sources)
    source_hashes = {str(path): sha(path) for path in sources(runtime_sources)}
    result = dict(passed=True, cpu=cpu, limiting_checks=limits, grid=grid,
                  trajectories=trajectories, source_hashes=source_hashes,
                  assets={str(path): sha(path) for path in assets()},
                  runtime_sources=runtime_sources, versions=versions(),
                  configs_sha256=canonical_hash(configs), cuda_device=torch.cuda.get_device_name(),
                  native_after_candidates_exact=True, decode_passed=True,
                  recursive_checks=recursive_checks, rounds=ROUNDS,
                  no_fid_used=True, created_unix=time.time())
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic(ROOT / 'development_check.json', result)
    print(json.dumps(dict(development_check_passed=True, configurations=len(configs),
                         trajectories=len(trajectories))), flush=True)


def prepare():
    configs = configurations()
    ROOT.mkdir(parents=True, exist_ok=True)
    base = ROOT / STAGE
    if (base / 'request.json').exists():
        request, _ = verify_stage()
        assert request['configs'] == configs
        return request
    check = read(ROOT / 'development_check.json')
    assert check['passed'] and check['configs_sha256'] == canonical_hash(configs)
    assert check.get('rounds') == ROUNDS, 'A six-round GPU development check is required'
    assert check['versions'] == versions()
    verify_hashes(check['source_hashes'])
    verify_hashes(check['assets'])
    estimated = len(configs) * ROUNDS * SAMPLES * (256 * 256 * 3 + 4 * 32 * 32 * 8 + 32768)
    assert shutil.disk_usage(ROOT).free > estimated + (10 << 30), 'Insufficient disk for complete screen'
    if base.exists():
        orphan = ROOT / 'orphans' / (STAGE + '_' + uuid.uuid4().hex[:10])
        orphan.parent.mkdir(exist_ok=True)
        base.replace(orphan)
    staging = ROOT / ('.prepare_' + uuid.uuid4().hex[:10])
    bank = staging / 'inputs'
    bank.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((SAMPLES, 4, 32, 32), dtype=np.float32)
    labels = np.random.default_rng(SEED + 1).permutation(np.repeat(np.arange(100, dtype=np.int64), 10))
    np.save(bank / 'noise.npy', noise)
    round_banks = [dict(round_index=0, file='noise.npy', seed=SEED, array_sha256=array_sha(noise))]
    for round_index in range(1, ROUNDS):
        seed = SEED+100*round_index
        fresh = np.random.default_rng(seed).standard_normal(noise.shape, dtype=np.float32)
        name = f'renoise{round_index}.npy'
        np.save(bank / name, fresh)
        round_banks.append(dict(round_index=round_index, file=name, seed=seed, array_sha256=array_sha(fresh)))
    np.save(bank / 'labels.npy', labels)
    source_manifest = dict(check['source_hashes'])
    archive = {}
    for index, (path, digest) in enumerate(sorted(source_manifest.items())):
        original = Path(path)
        relative = Path('sources') / (original.relative_to(WORK) if original.is_relative_to(WORK)
                                     else Path('_external') / f'{index:04d}_{original.name}')
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        assert sha(destination) == digest
        archive[path] = str(relative)
    ideas = json.loads(json.dumps(catalog.IDEAS, allow_nan=False))
    request = dict(stage=STAGE, configs=configs, ideas=ideas, arms=[cfg['arm'] for cfg in configs],
        samples=SAMPLES, batch=BATCH, ranks=RANKS, sources=source_manifest,
        rounds=ROUNDS, feedback_rounds=ROUNDS-1, renoise_start=RENOISE_START,
        round_banks=round_banks, planned_config_rounds=len(configs)*ROUNDS,
        round_order='configuration-major: initial complete generation then five full feedback generations',
        retention='Raw pixel+latent shards, aggregate latent, features, metrics and galleries retained; aggregate pixel NPZ transient',
        source_archive=archive, assets=check['assets'],
        references={str(ROOT / 'development_check.json'): sha(ROOT / 'development_check.json')},
        reference=str(REFERENCE), reference_sha256=sha(REFERENCE),
        inception_graph_sha256=sha(GRAPH), bank_root=str(base / 'inputs'),
        bank_files={name: sha(bank / name) for name in ('noise.npy', 'labels.npy',
                    *(f'renoise{index}.npy' for index in range(1, ROUNDS)))},
        bank=dict(samples=SAMPLES, batch=BATCH, noise_seed=SEED, label_seed=SEED + 1,
            noise_sha256=array_sha(noise), label_sha256=array_sha(labels), classes_balanced=True,
            noise_generation='NumPy default_rng PCG64 standard_normal float32; fixed complete bank'),
        diagnostic_names=getattr(core, 'DIAGNOSTICS', DIAGNOSTICS),
        python=PYTHON, fid_python=FID_PYTHON, versions=versions(),
        independent_confirmation=False, research_goal_achieved=False,
        estimated_output_bytes=estimated, prepared_unix=time.time(),
        source_recovery='Original paths verified against frozen hashes and archived copies; fail closed on edits',
        runtime_sources=check['runtime_sources'])
    atomic(staging / 'request.json', request)
    atomic(staging / 'status.json', dict(phase='prepared', completed=0, total=len(configs)*ROUNDS))
    staging.replace(base)
    atomic(ROOT / 'status.json', dict(phase='prepared', completed=0, total=len(configs)*ROUNDS, stage=STAGE))
    reporting.write_progress(base, request, [])
    print(json.dumps(dict(prepared=True, arms=len(configs), samples=SAMPLES,
                         root=str(base), request_sha256=sha(base / 'request.json'))), flush=True)
    return request


def verify_stage():
    base = ROOT / STAGE
    request = read(base / 'request.json')
    assert request['stage'] == STAGE and request['samples'] == SAMPLES and request['rounds'] == ROUNDS
    assert request['batch'] == BATCH and request['ranks'] == RANKS
    for category in ('sources', 'assets', 'references'):
        verify_hashes(request[category])
    for original, relative in request['source_archive'].items():
        assert sha(base / relative) == request['sources'][original], relative
    for name, digest in request['bank_files'].items():
        assert sha(base / 'inputs' / name) == digest, name
    assert sha(REFERENCE) == request['reference_sha256'] and sha(GRAPH) == request['inception_graph_sha256']
    assert request['versions'] == versions()
    return request, sha(base / 'request.json')


def round_directory(base, arm, round_index):
    return Path(base) / arm / f'round{round_index:02d}'


def committed(base, config, request_hash, round_index=0):
    directory = round_directory(base, config['arm'], round_index)
    if not (directory / 'commit.json').exists():
        return None
    receipt = read(directory / 'commit.json')
    assert receipt['request_sha256'] == request_hash
    assert receipt['config_sha256'] == canonical_hash(config)
    assert receipt.get('round_index', 0) == round_index
    for name, digest in receipt['files'].items():
        assert sha(directory / name) == digest, directory / name
    row = read(directory / 'result.json')
    assert row['request_sha256'] == request_hash and row['round_index'] == round_index
    return row


def commit_result(base, config, row, request_hash):
    out = round_directory(base, config['arm'], row['round_index'])
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / 'result.json', row)
    names = ['result.json']
    if row['complete']:
        names += ['latents.npy', 'activations.npz', 'fid.json', 'drift.json',
                  'drift_metrics.npz', 'gallery.png', 'gallery.json']
        names += [f'rank{rank}/summary.json' for rank in range(RANKS)]
    elif row['status'] == 'numerical_failure':
        names += ['numerical_failure.json']
    atomic(out / 'commit.json', dict(request_sha256=request_hash,
        config_sha256=canonical_hash(config), round_index=row['round_index'],
        files={name: sha(out / name) for name in names}))
    # Pixel aggregates are scratch space for ADM only, never a second retained copy.
    (out / 'samples.npz').unlink(missing_ok=True)


def verify_batch(path, receipt, config, start, request_hash, noise, labels,
                 round_index=0, previous=None):
    record = read(receipt)
    assert record['request_sha256'] == request_hash and record['config_sha256'] == canonical_hash(config)
    assert record['start'] == start and record['round_index'] == round_index and record['sha256'] == sha(path), path
    fresh = np.array(noise[start:start+BATCH])
    expected = fresh if previous is None else RENOISE_START*np.array(previous[start:start+BATCH])+(1-RENOISE_START)*fresh
    with np.load(path) as batch:
        assert str(batch['request_sha256']) == request_hash and int(batch['round_index']) == round_index
        assert str(batch['noise_sha256']) == array_sha(fresh)
        assert str(batch['input_state_sha256']) == array_sha(expected)
        if previous is not None:
            assert str(batch['parent_latent_sha256']) == array_sha(previous[start:start+BATCH])
        np.testing.assert_array_equal(batch['labels'], labels[start:start+BATCH])
        assert batch['latents'].shape == (BATCH, 4, 32, 32) and np.isfinite(batch['latents']).all()
        assert batch['arr_0'].shape == (BATCH, 256, 256, 3) and batch['arr_0'].dtype == np.uint8
    return dict(start=start, file=path.name, sha256=record['sha256'])


def ensure_parent(pid):
    if os.getppid() != pid:
        raise RuntimeError('Controller exited; stopping this worker')


@torch.inference_mode()
def worker(rank, run_id, parent_pid):
    request, request_hash = verify_stage()
    base = ROOT / STAGE
    directory = base / 'runs' / run_id
    configs = {cfg['arm']: cfg for cfg in request['configs']}
    banks = [np.load(base / 'inputs' / item['file'], mmap_mode='r') for item in request['round_banks']]
    labels = np.load(base / 'inputs/labels.npy', mmap_mode='r')
    core.install()
    rt = core.make_runtime()
    cuda = lambda value: torch.from_numpy(np.array(value)).cuda()
    n = cuda(banks[0][rank*BATCH:(rank+1)*BATCH])
    y = cuda(labels[rank*BATCH:(rank+1)*BATCH])
    rt.labels = y
    original_hooks = hooks(rt)
    for path, digest in rt.sources.items():
        assert request['sources'].get(path) == digest, path
    for time_value in (0., .375):
        t = n.new_tensor(time_value)
        strong, weak = rt.pair(n, t)
        assert torch.equal(strong, rt.field(n, t, 'full'))
        assert torch.equal(weak, rt.field(n, t, 'base'))
    limits = core.limiting_checks(rt, n, y)
    atomic(directory / f'ready{rank}.json', dict(passed=True, rank=rank, request_sha256=request_hash,
        run_id=run_id, pid=os.getpid(), native_pair_prefix_exact=True, limits=limits,
        cuda_device=torch.cuda.get_device_name()))
    last_command = None
    while True:
        ensure_parent(parent_pid)
        command_path = directory / 'command.json'
        command = read(command_path) if command_path.exists() else {}
        if command.get('command') == 'complete':
            return
        identity = (command.get('arm'), command.get('round_index'))
        if command.get('command') != 'sample' or identity == last_command:
            time.sleep(.5)
            continue
        verify_hashes(request['sources'])
        config = configs[command['arm']]
        arm, round_index = config['arm'], int(command['round_index'])
        round_out = round_directory(base, arm, round_index)
        noise = banks[round_index]
        previous = (np.load(round_directory(base, arm, round_index-1)/'latents.npy', mmap_mode='r')
                    if round_index else None)
        out = round_out / f'rank{rank}'
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for start in range(rank*BATCH, SAMPLES, RANKS*BATCH):
            ensure_parent(parent_pid)
            if (round_out / 'numerical_failure.json').exists():
                break
            path = out / f'batch{start:04d}.npz'
            receipt = path.with_suffix('.receipt.json')
            if path.exists() and receipt.exists():
                files.append(verify_batch(path, receipt, config, start, request_hash, noise, labels,
                                          round_index, previous))
                continue
            if path.exists() or receipt.exists():
                orphan = out / 'orphans' / run_id
                orphan.mkdir(parents=True, exist_ok=True)
                for fragment in (path, receipt):
                    if fragment.exists():
                        fragment.replace(orphan / fragment.name)
            fresh = np.array(noise[start:start+BATCH])
            parent_latent = np.array(previous[start:start+BATCH]) if previous is not None else None
            initial = fresh if parent_latent is None else RENOISE_START*parent_latent+(1-RENOISE_START)*fresh
            n, y = cuda(initial), cuda(labels[start:start+BATCH])
            rt.labels = y
            before = dict(rt.counts)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            try:
                z, stats = core.sample(rt, n, y, config,
                    start_step=int(RENOISE_START*int(config['solver'][4:])) if round_index else 0,
                    probe_seed=int(array_sha(fresh)[:15], 16))
                if not torch.isfinite(z).all():
                    raise FloatingPointError('Nonfinite output latent')
                verify_stats(stats, before, rt)
            except (FloatingPointError, AssertionError) as error:
                if not isinstance(error, FloatingPointError) and 'underflow in dt' not in str(error):
                    raise
                failure = dict(arm=arm, round_index=round_index, rank=rank, start=start,
                               error=repr(error), request_sha256=request_hash)
                atomic(out / 'numerical_failure.json', failure)
                try:
                    os.link(out / 'numerical_failure.json', round_out / 'numerical_failure.json')
                except FileExistsError:
                    pass
                break
            torch.cuda.synchronize()
            trajectory = time.perf_counter()-begin
            begin = time.perf_counter()
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            decode = time.perf_counter()-begin
            assert pixels.shape == (BATCH, 256, 256, 3) and pixels.dtype == np.uint8
            assert hooks(rt) == original_hooks and rt.labels is y
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, arr_0=pixels, latents=z.cpu().numpy(), labels=np.array(labels[start:start+BATCH]),
                    request_sha256=request_hash, noise_sha256=array_sha(fresh),
                    input_state_sha256=array_sha(initial),
                    parent_latent_sha256=array_sha(parent_latent) if parent_latent is not None else '',
                    round_index=round_index, start=start,
                    trajectory_seconds=trajectory, decode_seconds=decode, **stats)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            atomic(receipt, dict(request_sha256=request_hash, config_sha256=canonical_hash(config),
                                 start=start, round_index=round_index, sha256=sha(path)))
            files.append(verify_batch(path, receipt, config, start, request_hash, noise, labels,
                                      round_index, previous))
            progress = dict(arm=arm, round_index=round_index, rank=rank, images=len(files)*BATCH,
                            run_id=run_id, last_start=start, updated_unix=time.time())
            atomic(out / 'progress.json', progress)
            if len(files) == 1 or len(files) % 16 == 0:
                print(json.dumps(progress), flush=True)
        atomic(out / 'summary.json', dict(complete=not (round_out / 'numerical_failure.json').exists(),
            rank=rank, round_index=round_index, files=files, request_sha256=request_hash, run_id=run_id,
            noise_sha256=request['round_banks'][round_index]['array_sha256'],
            label_sha256=request['bank']['label_sha256']))
        last_command = identity


def evaluate(base, config, request, request_hash, gpu, round_index):
    """ADM evaluates every saved complete endpoint; its pixel NPZ is temporary."""
    arm = config['arm']
    out = round_directory(base, arm, round_index)
    if (out / 'numerical_failure.json').exists():
        return dict(arm=arm, round_index=round_index, complete=False, fid=None, status='numerical_failure',
                    failure=read(out / 'numerical_failure.json'), request_sha256=request_hash)
    noise = np.load(base / 'inputs' / request['round_banks'][round_index]['file'], mmap_mode='r')
    labels = np.load(base / 'inputs/labels.npy', mmap_mode='r')
    previous_dir = round_directory(base, arm, round_index-1) if round_index else out
    initial_dir = round_directory(base, arm, 0)
    previous = np.load(previous_dir / 'latents.npy', mmap_mode='r') if round_index else None
    images = np.empty((SAMPLES, 256, 256, 3), np.uint8)
    latents = np.empty_like(noise)
    drift_values = {}
    seen = set()
    paths = []
    calls = np.zeros(3, dtype=np.float64)
    trajectory = decode = events = 0.
    for rank in range(RANKS):
        shard = out / f'rank{rank}'
        summary = read(shard / 'summary.json')
        assert summary['complete'] and summary['request_sha256'] == request_hash
        assert summary['round_index'] == round_index
        assert summary['noise_sha256'] == request['round_banks'][round_index]['array_sha256']
        assert summary['label_sha256'] == request['bank']['label_sha256']
        for entry in summary['files']:
            start = entry['start']
            path = shard / entry['file']
            paths.append(path)
            assert start not in seen and (start // BATCH) % RANKS == rank
            seen.add(start)
            checked = verify_batch(path, path.with_suffix('.receipt.json'), config, start,
                                   request_hash, noise, labels, round_index, previous)
            assert checked['sha256'] == entry['sha256']
            with np.load(path) as batch:
                current_pixels, current_latents = batch['arr_0'], batch['latents']
                images[start:start+BATCH] = current_pixels
                latents[start:start+BATCH] = current_latents
                if round_index:
                    previous_path = previous_dir / f'rank{rank}' / entry['file']
                    initial_path = initial_dir / f'rank{rank}' / entry['file']
                    for parent_path in (previous_path, initial_path):
                        assert sha(parent_path) == read(parent_path.with_suffix('.receipt.json'))['sha256']
                    with np.load(previous_path) as before:
                        previous_pixels, previous_latents = before['arr_0'], before['latents']
                        np.testing.assert_array_equal(before['labels'], labels[start:start+BATCH])
                        np.testing.assert_array_equal(previous_latents, previous[start:start+BATCH])
                    with np.load(initial_path) as initial:
                        initial_pixels, initial_latents = initial['arr_0'], initial['latents']
                        np.testing.assert_array_equal(initial['labels'], labels[start:start+BATCH])
                else:
                    previous_pixels = initial_pixels = current_pixels
                    previous_latents = initial_latents = current_latents
                comparison = gallery.compare_round(current_latents, current_pixels,
                    previous_latents, previous_pixels, initial_latents, initial_pixels)
                for name, values in comparison.items():
                    if name not in drift_values:
                        drift_values[name] = np.empty(SAMPLES, np.float64)
                    drift_values[name][start:start+BATCH] = values
                trajectory += float(batch['trajectory_seconds'])
                decode += float(batch['decode_seconds'])
                calls += [int(batch[key]) for key in ('full_calls', 'prefix_calls', 'auxiliary_full_calls')]
                events += float(batch['probe_events'])
    assert seen == set(range(0, SAMPLES, BATCH))
    for name, value in (('samples.npz', images), ('latents.npy', latents)):
        temporary = out / (name + '.tmp')
        with temporary.open('wb') as stream:
            if name.endswith('.npz'):
                np.savez(stream, arr_0=value)
            else:
                np.save(stream, value)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(out / name)
    with (out / 'drift_metrics.npz.tmp').open('wb') as stream:
        np.savez(stream, **drift_values)
    (out / 'drift_metrics.npz.tmp').replace(out / 'drift_metrics.npz')
    drift = gallery.summarize_comparison(drift_values)
    atomic(out / 'drift.json', drift)
    gallery_record = gallery.render_round_gallery(paths, out / 'gallery.png', count=16)
    atomic(out / 'gallery.json', gallery_record)
    round_paths = {index: sorted(round_directory(base, arm, index).glob('rank*/batch*.npz'))
                   for index in range(round_index+1)}
    gallery.render_round_contact_sheet(round_paths, out.parent / 'rounds.png', count=8)
    command = [FID_PYTHON, str(WORK / 'experiments/compute_adm_fid.py'),
        '--reference', str(REFERENCE), '--samples', str(out / 'samples.npz'),
        '--output', str(out / '.fid.pending.json'), '--activations-output', str(out / 'activations.npz'),
        '--batch-size', '8', '--gpu-memory-fraction', '.30']
    with (out / 'evaluation.log').open('w') as stream:
        subprocess.run(command, cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    metrics = read(out / '.fid.pending.json')
    assert metrics['sample_count'] == SAMPLES and all(np.isfinite(metrics[key])
           for key in ('fid', 'sfid', 'inception_score'))
    (out / '.fid.pending.json').replace(out / 'fid.json')
    return dict(arm=arm, round_index=round_index, complete=True, samples=SAMPLES,
        fid=metrics['fid'], metrics=metrics, drift=drift,
        noise_sha256=request['round_banks'][round_index]['array_sha256'],
        label_sha256=request['bank']['label_sha256'],
        parent_latents_sha256=sha(previous_dir / 'latents.npy') if round_index else None,
        pixel_array_sha256=array_sha(images), sample_sha256=sha(out / 'samples.npz'),
        sample_archive_retained=False, pixel_storage='rank*/batch*.npz:arr_0',
        latents_sha256=sha(out / 'latents.npy'), activations_sha256=sha(out / 'activations.npz'),
        trajectory_gpu_seconds=trajectory, decode_gpu_seconds=decode,
        sum_batch_gpu_seconds=trajectory+decode, full_calls_per_image=float(calls[0]/(SAMPLES/BATCH)),
        prefix_calls_per_image=float(calls[1]/(SAMPLES/BATCH)),
        auxiliary_full_calls_per_image=float(calls[2]/(SAMPLES/BATCH)),
        events_per_image=events/(SAMPLES/BATCH), request_sha256=request_hash, coverage_verified=True,
        gallery_path=str(out / 'gallery.png'), contact_sheet_path=str(out.parent / 'rounds.png'))


def audit_stage(base, selected_arms):
    """Check all round coverage; independently recompute selected complete chains."""
    request, request_hash = verify_stage()
    rows = read(base / 'results.json')
    configs = {cfg['arm']: cfg for cfg in request['configs']}
    expected = {(arm, index) for arm in configs for index in range(ROUNDS)}
    assert len(rows) == len(expected) and {(row['arm'], row['round_index']) for row in rows} == expected
    assert read(base / 'status.json')['phase'] == 'complete'
    records, coverage = [], 0
    selected = set(selected_arms)
    for row in rows:
        arm, index = row['arm'], row['round_index']
        out = round_directory(base, arm, index)
        assert committed(base, configs[arm], request_hash, index) == row
        assert not (out / 'samples.npz').exists(), 'Duplicate pixel aggregate should have been removed'
        if not row['complete']:
            assert row['fid'] is None
            if row['status'] == 'dependency_blocked':
                assert index > 0 and not read(round_directory(base, arm, index-1)/'result.json')['complete']
            continue
        assert row['samples'] == SAMPLES and row['coverage_verified']
        assert row['noise_sha256'] == request['round_banks'][index]['array_sha256']
        assert row['metrics'] == read(out / 'fid.json')
        assert row['fid'] == row['metrics']['fid'] and row['metrics']['sample_count'] == SAMPLES
        if index:
            assert row['parent_latents_sha256'] == sha(round_directory(base, arm, index-1)/'latents.npy')
        seen = set()
        detail = arm in selected
        pixels = np.empty((SAMPLES, 256, 256, 3), np.uint8) if detail else None
        seconds, calls = 0., np.zeros(3)
        for rank in range(RANKS):
            shard = out / f'rank{rank}'
            summary = read(shard / 'summary.json')
            assert summary['complete'] and summary['request_sha256'] == request_hash
            assert summary['round_index'] == index
            for entry in summary['files']:
                start = entry['start']
                assert start not in seen and (start//BATCH)%RANKS == rank
                seen.add(start)
                path = shard / entry['file']
                assert path.exists()
                if detail:
                    assert sha(path) == entry['sha256']
                    with np.load(path) as batch:
                        pixels[start:start+BATCH] = batch['arr_0']
                        seconds += float(batch['trajectory_seconds'])+float(batch['decode_seconds'])
                        calls += [int(batch[key]) for key in ('full_calls','prefix_calls','auxiliary_full_calls')]
        assert seen == set(range(0, SAMPLES, BATCH))
        coverage += len(seen)
        if not detail:
            continue
        assert array_sha(pixels) == row['pixel_array_sha256']
        del pixels
        assert abs(seconds-row['sum_batch_gpu_seconds']) < 1e-7
        for value, key in zip(calls/(SAMPLES/BATCH), ('full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image')):
            assert value == row[key]
        audit_path = out / 'metric_audit.json'
        identity = dict(activations_sha256=row['activations_sha256'], reference_sha256=request['reference_sha256'])
        if audit_path.exists():
            checked = read(audit_path)
            assert checked['identity'] == identity
        else:
            values = {}
            with np.load(out / 'activations.npz') as data:
                for key, (mu, cov, _) in analysis.references(request).items():
                    values[key] = sum(analysis.fid_components(data[key].astype(float), mu, cov))
            checked = dict(identity=identity, independent_fid=values['pool_3'], independent_sfid=values['spatial'],
                fid_absolute_error=abs(values['pool_3']-row['fid']),
                sfid_absolute_error=abs(values['spatial']-row['metrics']['sfid']))
            assert checked['fid_absolute_error'] < .001 and checked['sfid_absolute_error'] < .001
            atomic(audit_path, checked)
        records.append(dict(arm=arm, round_index=index, pixel_array_reconstructed_from_retained_shards=True,
                            costs_reconciled=True, **checked))
        print(json.dumps(dict(audited=arm, round_index=index)), flush=True)
    result = dict(passed=True, request_sha256=request_hash, configurations=len(configs), rounds=ROUNDS,
                  results=len(rows), covered_batch_indices=coverage,
                  metadata_and_coverage_all_complete_rounds=True,
                  raw_and_metric_audits=records, feature_extraction_repeated=False)
    atomic(base / 'analysis_audit.json', result)
    return result


def gpu_processes(gpus):
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    selected = {parts[1].strip() for line in listing.splitlines()
                if len(parts := line.split(',')) == 2 and parts[0].strip() in gpus}
    assert len(selected) == RANKS, f'Cannot find all requested GPUs: {gpus}'
    listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    return [dict(pid=int(parts[0].strip()), gpu_uuid=parts[1].strip())
            for line in listing.splitlines() if len(parts := line.split(',')) == 2
            and parts[1].strip() in selected]


def wait_for_resources(gpus, dependency):
    while True:
        if (ROOT / 'STOP_AFTER_CURRENT').exists():
            atomic(ROOT / 'status.json', dict(phase='stopped_before_sampling', controller_pid=os.getpid()))
            return False
        path = Path(dependency)
        previous = read(path) if path.exists() else {'phase': 'missing'}
        if previous.get('phase') == 'failed':
            raise RuntimeError(f'Named dependency failed: {path}; {previous.get("error")}')
        occupied = gpu_processes(gpus) if previous.get('phase') == 'complete' else []
        if previous.get('phase') == 'complete' and not occupied:
            return True
        phase = 'waiting_for_gpu_release' if previous.get('phase') == 'complete' else 'waiting_for_dependency'
        atomic(ROOT / 'status.json', dict(phase=phase, controller_pid=os.getpid(),
            dependency=str(path), dependency_phase=previous.get('phase'), occupied=occupied,
            updated_unix=time.time()))
        time.sleep(10)


def run_stage(gpus):
    base = ROOT / STAGE
    request, request_hash = verify_stage()
    run_id = time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory = base / 'runs' / run_id
    directory.mkdir(parents=True)
    rows, workers, streams = [], [], []
    begin = time.perf_counter()
    total = len(request['configs'])*ROUNDS

    def status(phase, **extra):
        value = dict(phase=phase, stage=STAGE, completed=len(rows), total=total,
            configurations=len(request['configs']), rounds=ROUNDS,
            run_id=run_id, controller_pid=os.getpid(), worker_pids=[child.pid for child in workers],
            run_wall_seconds=time.perf_counter()-begin, updated_unix=time.time(), **extra)
        atomic(base / 'status.json', value); atomic(ROOT / 'status.json', value)

    def wait_files(paths, phase, **details):
        heartbeat = 0.
        while True:
            codes = [child.poll() for child in workers]
            if any(code is not None for code in codes):
                raise RuntimeError(f'Worker exited before completion: {codes}; logs: {directory}')
            if all(path.exists() and read(path).get('run_id') == run_id for path in paths):
                return
            if time.monotonic()-heartbeat > 15:
                status(phase, **details); heartbeat = time.monotonic()
            time.sleep(.5)

    def publish(cfg, result):
        result.update(cfg)
        commit_result(base, cfg, result, request_hash)
        rows.append(result)
        completed.add((cfg['arm'], result['round_index']))
        by_identity[(cfg['arm'], result['round_index'])] = result
        atomic(base / 'results.json', rows)
        reporting.write_progress(base, request, rows)
        print(json.dumps(dict(arm=cfg['arm'], round_index=result['round_index'], fid=result['fid'],
                              complete=result['complete'], completed=len(rows), total=total)), flush=True)

    try:
        for cfg in request['configs']:
            for index in range(ROUNDS):
                row = committed(base, cfg, request_hash, index)
                if row is not None:
                    rows.append(row)
                    (round_directory(base, cfg['arm'], index)/'samples.npz').unlink(missing_ok=True)
                elif any(committed(base, cfg, request_hash, later) is not None for later in range(index+1,ROUNDS)):
                    raise RuntimeError(f'Noncontiguous committed rounds for {cfg["arm"]}')
        completed = {(row['arm'],row['round_index']) for row in rows}
        by_identity = {(row['arm'],row['round_index']):row for row in rows}
        atomic(base / 'results.json', rows)
        reporting.write_progress(base, request, rows)
        if len(completed) == total:
            status('complete'); return
        for rank, gpu in enumerate(gpus):
            stream = (directory / f'worker{rank}.log').open('a'); streams.append(stream)
            workers.append(subprocess.Popen([PYTHON, '-u', '-m', MODULE, '--worker', str(rank),
                '--stage', STAGE, '--run-id', run_id, '--parent-pid', str(os.getpid())], cwd=WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        status('preflight')
        wait_files([directory / f'ready{rank}.json' for rank in range(RANKS)], 'preflight')
        checks = [read(directory / f'ready{rank}.json') for rank in range(RANKS)]
        assert all(check['passed'] and check['request_sha256'] == request_hash for check in checks)
        atomic(directory / 'preflight_passed.json', dict(passed=True, checks=checks, request_sha256=request_hash))
        for cfg in request['configs']:
            arm = cfg['arm']
            for index in range(ROUNDS):
                if (arm,index) in completed:
                    continue
                if (ROOT / 'STOP_AFTER_CURRENT').exists():
                    break
                if index and not by_identity[(arm,index-1)]['complete']:
                    publish(cfg, dict(arm=arm, round_index=index, complete=False, fid=None,
                        status='dependency_blocked', blocked_by_round=index-1,
                        request_sha256=request_hash, evaluation_wall_seconds=0.))
                    continue
                assert shutil.disk_usage(base).free > (2<<30), 'Less than 2 GiB free; preserving output'
                verify_hashes(request['sources'])
                status('sampling', arm=arm, round_index=index, family=cfg['family'])
                atomic(directory / 'command.json', dict(command='sample',arm=arm,round_index=index))
                out = round_directory(base, arm, index)
                wait_files([out / f'rank{rank}/summary.json' for rank in range(RANKS)],
                           'sampling', arm=arm, round_index=index, family=cfg['family'])
                status('evaluating',arm=arm,round_index=index,family=cfg['family'])
                evaluation_begin = time.perf_counter()
                result = evaluate(base,cfg,request,request_hash,gpus[0],index)
                result['evaluation_wall_seconds'] = time.perf_counter()-evaluation_begin
                publish(cfg,result)
            if (ROOT / 'STOP_AFTER_CURRENT').exists():
                break
        atomic(directory / 'command.json', dict(command='complete'))
        codes = [child.wait(timeout=45) for child in workers]
        assert codes == [0]*RANKS,codes
        status('complete' if len(rows)==total else 'stopped_after_current',worker_exit_codes=codes,
               numerical_failures=sum(row.get('status')=='numerical_failure' for row in rows),
               dependency_blocked=sum(row.get('status')=='dependency_blocked' for row in rows))
    except BaseException as error:
        status('failed',error=repr(error)); raise
    finally:
        for child in workers:
            if child.poll() is None: child.terminate()
        for child in workers:
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in streams: stream.close()


def pipeline(gpus, dependency):
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(signum, frame):
        raise RuntimeError(f'Controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        request = prepare()
        if not wait_for_resources(gpus, dependency):
            return
        run_stage(gpus)
        base = ROOT / STAGE
        if read(base / 'status.json')['phase'] != 'complete':
            return
        rows = read(base / 'results.json')
        selected = []
        # Choose a complete chain by final-round FID, then audit all six rounds
        # of that fixed setting. Do not splice different settings across rounds.
        for family in dict.fromkeys(cfg['family'] for cfg in request['configs']):
            valid = [row for row in rows if row['family']==family and row['round_index']==ROUNDS-1 and row['complete']]
            if valid:
                selected.append(min(valid,key=lambda row:row['fid'])['arm'])
        atomic(ROOT / 'status.json',dict(phase='auditing',completed=len(rows),
               total=len(request['configs'])*ROUNDS,controller_pid=os.getpid()))
        audit = audit_stage(base,selected)
        reporting.write_progress(base,request,rows)
        atomic(ROOT / 'status.json',dict(phase='complete',completed=len(rows),
            total=len(request['configs'])*ROUNDS,configurations=len(request['configs']),rounds=ROUNDS,
            samples_per_round=SAMPLES,candidate_arms=200,control_arms=len(request['configs'])-200,
            numerical_failures=sum(row.get('status')=='numerical_failure' for row in rows),
            dependency_blocked=sum(row.get('status')=='dependency_blocked' for row in rows),
            final_audit_passed=audit['passed'],no_automatic_5k=True,
            research_goal_achieved=False,updated_unix=time.time()))
    except BaseException as error:
        previous = read(ROOT / 'status.json') if (ROOT / 'status.json').exists() else {}
        atomic(ROOT / 'status.json', dict(phase='failed', error=repr(error), previous=previous,
                                         controller_pid=os.getpid(), updated_unix=time.time()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('cpu-check', 'check', 'prepare', 'pipeline', 'status', 'stop-after-current'):
        action.add_argument('--' + name, action='store_true')
    action.add_argument('--worker', type=int, choices=range(RANKS))
    parser.add_argument('--stage', choices=[STAGE], default=STAGE)
    parser.add_argument('--run-id')
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--wait-for', default=str(WAIT_ROOT / 'status.json'))
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    elif args.check:
        development_check()
    elif args.prepare:
        prepare()
    elif args.pipeline:
        gpus = [value.strip() for value in args.gpus.split(',')]
        assert len(gpus) == RANKS and len(set(gpus)) == RANKS and all(value.isdigit() for value in gpus)
        pipeline(gpus, args.wait_for)
    elif args.status:
        path = ROOT / 'status.json'
        print(json.dumps(read(path) if path.exists() else {'phase': 'not_started'}, ensure_ascii=False, indent=2))
    elif args.stop_after_current:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / 'STOP_AFTER_CURRENT').touch()
    else:
        assert args.run_id and args.parent_pid
        worker(args.worker, args.run_id, args.parent_pid)


if __name__ == '__main__':
    main()
