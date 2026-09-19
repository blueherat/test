"""A cheap, learned readout of cached SiT predictions and their internal gap."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments.lifting_scale_sweep_20260909 import (
    EXPS, WORK, SMALL_DATA, Runtime, array_sha, atomic, read, sha,
)
from experiments.train_imagenet100_sit_flow import sample_sdvae_posterior

OLD_ROOT = infrastructure.ROOT
ROOT = EXPS / 'small_sit_feedback_reader_20260909'
PROTOCOL = WORK / 'docs/SMALL_SIT_FEEDBACK_READER_PROTOCOL_20260909_ZH.md'
DATA_ROOT = SMALL_DATA / 'imagenet100_cmc_sdvae'
ARMS = ('strong_only', 'strong_gap')
GRID, ALPHAS = infrastructure.GRID, infrastructure.ALPHAS
BATCH, SAMPLES, RANKS = infrastructure.BATCH, infrastructure.SAMPLES, infrastructure.RANKS
TRAIN_IMAGES, VALIDATION_IMAGES, COLLECTION_BATCH, PATCHES = 8192, 1024, 64, 16
WIDTH, TRAIN_STEPS, TOKEN_BATCH = 64, 1000, 2048
SEED, TRAIN_SEED = 202609977, 202609978
LEARNING_RATE, WEIGHT_DECAY, EMA_DECAY = .001, .0001, .99


def patchify(x):
    b, c, h, w = x.shape
    assert h == w == 32 and c == 4
    return x.reshape(b, c, 16, 2, 16, 2).permute(0, 2, 4, 3, 5, 1).reshape(b, 256, 16)


def unpatchify(x):
    return x.reshape(len(x), 16, 16, 2, 2, 4).permute(0, 5, 1, 3, 2, 4).reshape(len(x), 4, 32, 32)


class Capture:
    """Observe the original, conditioned pre-linear features without changing them."""
    def __init__(self, rt):
        self.values = {}
        def hook(key):
            def save(module, inputs):
                self.values[key] = inputs[0].detach()
            return save
        self.handles = [rt.model.final_layer.linear.register_forward_pre_hook(hook('hs')),
                        rt.head.module.linear.register_forward_pre_hook(hook('hw'))]

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_tokens(rt, z, t, strong, weak):
    count = len(z)
    ts = t.expand(count)
    sigma = (1. - ts).reshape(-1, 1, 1, 1)
    values = dict(hs=rt.capture.values['hs'], hw=rt.capture.values['hw'],
        z=patchify(z), a=patchify(z + sigma * strong), d=patchify(sigma * (strong - weak)),
        s=patchify(strong), w=patchify(weak))
    values['time'] = torch.stack((ts, 1. - ts, torch.sin(math.pi * ts), torch.cos(math.pi * ts)), -1)
    values['time'] = values['time'][:, None, :].expand(-1, 256, -1)
    return values


class Reader(nn.Module):
    def __init__(self, hidden=384):
        super().__init__()
        self.layers = nn.ModuleDict({key: nn.Sequential(nn.Linear(hidden + 52, WIDTH), nn.SiLU(),
            nn.Linear(WIDTH, 16)) for key in ('s', 'w')})
        for layer in self.layers.values():
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def forward(self, tokens, arm, null_mask=None):
        assert arm in ARMS
        a = tokens['a']
        d = tokens['d'] if arm == 'strong_gap' else torch.zeros_like(tokens['d'])
        if null_mask is not None:
            a, d = a * null_mask, d * null_mask
        common = (tokens['z'], a, d, tokens['time'])
        result = []
        for key, feature in (('s', 'hs'), ('w', 'hw')):
            h = tokens[feature]
            inputs = torch.cat((F.layer_norm(h, (h.shape[-1],)), *common), dim=-1)
            result.append(tokens[key] + self.layers[key](inputs))
        return tuple(result)


def verify_training():
    request = read(ROOT / 'training_request.json')
    for group in ('sources', 'assets', 'data_files', 'index_files'):
        for p, h in request[group].items():
            assert sha(p) == h, (group, p)
    return request


def prepare_training():
    parent = read(OLD_ROOT / 'request.json')
    for group in ('sources', 'assets'):
        for p, h in parent[group].items():
            assert sha(p) == h, p
    ROOT.mkdir(parents=True, exist_ok=False)
    labels = np.load(DATA_ROOT / 'train_labels.npy', mmap_mode='r')
    perm = np.random.default_rng(SEED).permutation(len(labels))[:TRAIN_IMAGES + VALIDATION_IMAGES]
    np.save(ROOT / 'train_indices.npy', perm[:TRAIN_IMAGES])
    np.save(ROOT / 'validation_indices.npy', perm[TRAIN_IMAGES:])
    assert len(set(perm.tolist())) == len(perm)
    sources = parent['sources'].copy()
    for p in (Path(__file__), PROTOCOL, WORK / 'train_gen/evaluator.py'):
        sources[str(p)] = sha(p)
    snapshot = ROOT / 'training_sources'
    snapshot.mkdir()
    for i, p in enumerate(sorted(sources)):
        (snapshot / f'{i:02d}_{Path(p).name}').write_bytes(Path(p).read_bytes())
    request = dict(sources=sources, assets=parent['assets'], parent_request_sha256=sha(OLD_ROOT / 'request.json'),
        data_files={str(DATA_ROOT / name): sha(DATA_ROOT / name)
            for name in ('manifest.json', 'train_moments.npy', 'train_labels.npy')},
        index_files={str(ROOT / name): sha(ROOT / name)
            for name in ('train_indices.npy', 'validation_indices.npy')},
        train_images=TRAIN_IMAGES, validation_images=VALIDATION_IMAGES, collection_batch=COLLECTION_BATCH,
        patches_per_image=PATCHES, width=WIDTH, steps=TRAIN_STEPS, token_batch=TOKEN_BATCH,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        collection_seed=SEED, training_seed=TRAIN_SEED, time_range=[0., .5], null_probability=.5,
        training_target='native linear-flow velocity, independent supervision for both heads',
        sampler_baseline_bank_used_for_training=False,
        inception_graph_sha256=sha('/data/shared/adm_refs/classify_image_graph_def.pb'))
    atomic(ROOT / 'training_request.json', request)
    atomic(ROOT / 'status.json', dict(phase='training_prepared', research_goal_achieved=False))
    print(json.dumps(dict(training_prepared=True, root=str(ROOT), request_sha256=sha(ROOT / 'training_request.json'))), flush=True)


@torch.no_grad()
def collect():
    verify_training()
    rt = Runtime('sit_small')
    assert rt.semantics.prediction_target == 'velocity' and rt.head.prediction_target == 'velocity'
    rt.capture = Capture(rt)
    gen = torch.Generator(device='cuda').manual_seed(SEED)
    moments = np.load(DATA_ROOT / 'train_moments.npy', mmap_mode='r')
    labels = np.load(DATA_ROOT / 'train_labels.npy', mmap_mode='r')
    records = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    parity = None
    for split in ('train', 'validation'):
        indices = np.load(ROOT / f'{split}_indices.npy')
        cache = {}
        per_batch = []
        for start in range(0, len(indices), COLLECTION_BATCH):
            ids = indices[start:start + COLLECTION_BATCH]
            moment = torch.from_numpy(np.array(moments[ids])).cuda()
            target = sample_sdvae_posterior(moment, torch.randn((len(ids), 4, 32, 32), generator=gen, device='cuda'))
            noise = torch.randn(target.shape, generator=gen, device='cuda')
            ts = .5 * torch.rand(len(ids), generator=gen, device='cuda')
            z = (1. - ts[:, None, None, None]) * noise + ts[:, None, None, None] * target
            rt.labels = torch.from_numpy(np.array(labels[ids])).to(device='cuda', dtype=torch.long)
            s, w = rt.pair(z, ts)
            tokens = make_tokens(rt, z, ts, s, w)
            tokens['target'] = patchify(target - noise)
            if parity is None:
                assert torch.equal(unpatchify(patchify(z)), z)
                ps = rt.model.unpatchify(rt.model.final_layer.linear(tokens['hs']))[:, :4]
                pw = unpatchify(rt.head.module.linear(tokens['hw']))
                assert torch.equal(ps, s) and torch.equal(pw, w)
                torch.manual_seed(TRAIN_SEED)
                empty = Reader(tokens['hs'].shape[-1]).cuda()
                for arm in ARMS:
                    rs, rw = empty(tokens, arm)
                    assert torch.equal(rs, tokens['s']) and torch.equal(rw, tokens['w'])
                parity = dict(native_linear_readouts_exact=True, zero_adapter_exact=True,
                    patch_roundtrip_exact=True, hidden=tokens['hs'].shape[-1],
                    parameters=sum(p.numel() for p in empty.parameters()),
                    source_parameters=sum(p.numel() for p in rt.model.parameters()),
                    source_semantics=str(rt.semantics), runtime_sources=rt.sources)
                del empty
            chosen = torch.stack([torch.randperm(256, generator=gen, device='cuda')[:PATCHES] for _ in ids])
            batch_ids = torch.arange(len(ids), device='cuda')[:, None]
            for key, value in tokens.items():
                cache.setdefault(key, []).append(value[batch_ids, chosen].reshape(-1, value.shape[-1]).cpu())
            per_batch.append(dict(start=start, state_sha256=array_sha(z.cpu().numpy()),
                clean_sha256=array_sha(target.cpu().numpy()), noise_sha256=array_sha(noise.cpu().numpy()),
                time_sha256=array_sha(ts.cpu().numpy()), patch_indices_sha256=array_sha(chosen.cpu().numpy())))
            if start % (16 * COLLECTION_BATCH) == 0:
                print(json.dumps(dict(collecting=split, images=start + len(ids))), flush=True)
        cache = {key: torch.cat(value) for key, value in cache.items()}
        path = ROOT / f'{split}_cache.pt'
        torch.save(cache, path)
        records.append(dict(split=split, images=len(indices), tokens=len(cache['s']), path=str(path),
            sha256=sha(path), batches=per_batch, shapes={k: list(v.shape) for k, v in cache.items()}))
        del cache
    torch.cuda.synchronize()
    record = dict(seconds=time.perf_counter() - begin, full_batch_calls=rt.counts['full'],
        records=records, preflight=parity, training_request_sha256=sha(ROOT / 'training_request.json'))
    atomic(ROOT / 'collection.json', record)
    rt.capture.close()
    del rt
    gc.collect()
    torch.cuda.empty_cache()
    return record


@torch.no_grad()
def validation(reader, data, arm):
    sums = torch.zeros(8, dtype=torch.float64, device='cuda')
    total = 0
    for i in range(0, len(data['s']), TOKEN_BATCH):
        chunk = {key: value[i:i + TOKEN_BATCH] for key, value in data.items()}
        s, w = reader(chunk, arm)
        target, bs, bw = chunk['target'], chunk['s'], chunk['w']
        alpha = torch.where(chunk['time'][:, :1] < .25, .6, .7)
        g, bg = s + alpha * (s - w), bs + alpha * (bs - bw)
        for k, value in enumerate((s-target, w-target, g-target, bs-target, bw-target, bg-target, s-w, bs-bw)):
            sums[k] += value.double().square().sum()
        total += target.numel()
    values = (sums / total).cpu().tolist()
    return dict(zip(('strong_mse', 'weak_mse', 'guided_mse', 'native_strong_mse', 'native_weak_mse',
                     'native_guided_mse', 'gap_mean_square', 'native_gap_mean_square'), values))


def fit_arm(arm):
    assert arm in ARMS
    request = verify_training()
    collection = read(ROOT / 'collection.json')
    for rec in collection['records']:
        assert sha(rec['path']) == rec['sha256']
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(TRAIN_SEED)
    train = torch.load(ROOT / 'train_cache.pt', map_location='cuda', weights_only=True)
    val = torch.load(ROOT / 'validation_cache.pt', map_location='cuda', weights_only=True)
    reader = Reader(train['hs'].shape[-1]).cuda()
    ema = copy.deepcopy(reader).requires_grad_(False)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device='cuda').manual_seed(TRAIN_SEED)
    log = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        ids = torch.randint(len(train['s']), (TOKEN_BATCH,), device='cuda', generator=gen)
        tokens = {key: value[ids] for key, value in train.items()}
        mask = (torch.rand((TOKEN_BATCH, 1), device='cuda', generator=gen) >= .5).float()
        warmup = min(step / 100., 1.)
        cosine = .01 + .99 * .5 * (1. + math.cos(math.pi * max(step - 100, 0) / (TRAIN_STEPS - 100)))
        optimizer.param_groups[0]['lr'] = LEARNING_RATE * warmup * cosine
        strong, weak = reader(tokens, arm, mask)
        loss = .5 * (F.mse_loss(strong, tokens['target']) + F.mse_loss(weak, tokens['target']))
        assert torch.isfinite(loss), (arm, step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), reader.parameters()):
                ep.lerp_(p, 1. - EMA_DECAY)
        if step == 1 or step % 250 == 0:
            row = dict(step=step, train_minibatch_loss=float(loss), gradient_norm=float(grad_norm),
                learning_rate=optimizer.param_groups[0]['lr'], validation=validation(ema, val, arm))
            log.append(row)
            atomic(ROOT / f'training_{arm}.json', dict(complete=False, rows=log))
            print(json.dumps(dict(arm=arm, **row)), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    output = ROOT / f'{arm}.pt'
    torch.save(dict(arm=arm, step=TRAIN_STEPS, ema=ema.state_dict(), reader=reader.state_dict(),
        hidden=train['hs'].shape[-1], training_request_sha256=sha(ROOT / 'training_request.json'),
        train_cache_sha256=sha(ROOT / 'train_cache.pt'), validation_cache_sha256=sha(ROOT / 'validation_cache.pt')), output)
    record = dict(complete=True, arm=arm, rows=log, seconds=seconds, checkpoint=str(output), sha256=sha(output),
        parameters=sum(p.numel() for p in reader.parameters()),
        steps=TRAIN_STEPS, checkpoint_selection='fixed final-step EMA, no validation selection')
    atomic(ROOT / f'training_{arm}.json', record)
    print(json.dumps(dict(trained=True, arm=arm, seconds=seconds, sha256=record['sha256'])), flush=True)


def train():
    assert read(ROOT / 'status.json')['phase'] == 'training_prepared'
    atomic(ROOT / 'status.json', dict(phase='collecting_training_features', controller_pid=os.getpid()))
    collect()
    workers, streams = [], []
    try:
        for arm, gpu in zip(ARMS, (1, 2)):
            stream = (ROOT / f'fit_{arm}.log').open('w')
            streams.append(stream)
            workers.append(subprocess.Popen([sys.executable, '-u', '-m',
                'experiments.small_sit_feedback_reader_20260909', '--fit-arm', arm], cwd=WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        atomic(ROOT / 'status.json', dict(phase='fitting_readers', controller_pid=os.getpid(), worker_pids=[p.pid for p in workers]))
        codes = [p.wait() for p in workers]
        assert codes == [0, 0], codes
        verify_training()
        atomic(ROOT / 'status.json', dict(phase='trained', worker_exit_codes=codes, research_goal_achieved=False))
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            p.wait()
        for stream in streams:
            stream.close()


def make_runtime(name):
    rt = Runtime(name)
    rt.capture = Capture(rt)
    rt.readers = {}
    for arm in ARMS:
        state = torch.load(ROOT / f'{arm}.pt', map_location='cuda', weights_only=True)
        reader = Reader(state['hidden']).cuda().eval().requires_grad_(False)
        reader.load_state_dict(state['ema'], strict=True)
        rt.readers[arm] = reader
    return rt


def rms(x):
    return x.double().flatten(1).square().mean(1).sqrt()


@torch.inference_mode()
def feedback_field(rt, z, t, alpha, arm, *, disable=False):
    strong, weak = rt.pair(z, t)
    ordinary = strong + alpha * (strong - weak)
    if disable:
        return ordinary, None
    tokens = make_tokens(rt, z, t, strong, weak)
    sp, wp = rt.readers[arm](tokens, arm)
    s, w = unpatchify(sp), unpatchify(wp)
    value = s + alpha * (s - w)
    diag = torch.stack((rms(value - ordinary) / rms(ordinary).clamp_min(1e-12),
        rms(s - w) / rms(strong - weak).clamp_min(1e-12), rms(s - strong), rms(w - weak)), -1)
    return value, diag


@torch.inference_mode()
def sample(rt, noise, labels, arm, *, zero=False, disable=False):
    assert arm in ARMS
    rt.labels = labels
    z = noise.clone()
    grid = z.new_tensor(GRID)
    before = rt.counts.copy()
    block_calls, block_reader, states, diagnostics = [], [], [], []
    for k, amount in enumerate(ALPHAS):
        alpha = 0. if zero else amount
        calls = rt.counts['full']
        local = []
        reader_calls = 0
        def field(t, x):
            nonlocal reader_calls
            if alpha == 0.:
                return rt.field(x, t, 'full')
            value, diag = feedback_field(rt, x, t, alpha, arm, disable=disable)
            if diag is not None:
                local.append(diag)
                reader_calls += 1
            return value
        z = odeint(field, z, grid[k:k + 2], method='dopri5', rtol=.001, atol=1e-6)[-1]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{arm}: nonfinite block {k}')
        block_calls.append(rt.counts['full'] - calls)
        block_reader.append(reader_calls)
        states.append(rms(z).cpu().numpy())
        diagnostics.append(torch.stack(local).mean(0).cpu().numpy() if local else np.zeros((len(z), 4)))
    full = rt.counts['full'] - before['full']
    assert rt.counts['prefix'] == before['prefix'] and full == sum(block_calls)
    return z, dict(full_calls=full, prefix_calls=0, auxiliary_full_calls=0,
        reader_calls=sum(block_reader), block_reader_calls=np.asarray(block_reader),
        block_full_calls=np.asarray(block_calls), block_state_rms=np.asarray(states),
        block_reader_diagnostics=np.asarray(diagnostics))


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    rt.labels = labels
    checks = []
    for tvalue in (0., .125, .375, .875):
        t = noise.new_tensor(tvalue)
        full, weak = rt.pair(noise, t)
        tokens = make_tokens(rt, noise, t, full, weak)
        assert torch.equal(full, rt.field(noise, t, 'full'))
        assert torch.equal(weak, rt.field(noise, t, 'base'))
        # A fresh zero reader must exactly preserve both original predictions.
        reader = Reader(tokens['hs'].shape[-1]).cuda()
        for arm in ARMS:
            s, w = reader(tokens, arm)
            assert torch.equal(s, tokens['s']) and torch.equal(w, tokens['w'])
        checks.append(tvalue)
    z, _ = sample(rt, noise, labels, ARMS[0], disable=True)
    golden = OLD_ROOT / 'ig_restarted' / f'rank{rank}' / f'batch{rank * BATCH:04d}.npz'
    with np.load(golden) as data:
        np.testing.assert_array_equal(z.cpu().numpy(), data['latents'])
    zeros = [sample(rt, noise, labels, arm, zero=True)[0] for arm in ARMS]
    assert torch.equal(zeros[0], zeros[1])
    # A matched batch-state timing measures only the per-query reader overhead.
    t = noise.new_tensor(.25)
    measurements = {}
    for mode in ('native', *ARMS):
        for _ in range(5):
            rt.guided(noise, t, .6) if mode == 'native' else feedback_field(rt, noise, t, .6, mode)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(30):
            rt.guided(noise, t, .6) if mode == 'native' else feedback_field(rt, noise, t, .6, mode)
        torch.cuda.synchronize()
        measurements[mode] = (time.perf_counter() - begin) / 30
    for p, h in rt.sources.items():
        assert request['sources'].get(p) == h, p
    return dict(passed=True, rank=rank, pair_prefix_exact_at=checks, zero_readers_exact=True,
        ordinary_ig_latents_exact=True, zero_alpha_arms_exact=True, golden_sha256=sha(golden),
        per_query_seconds=measurements, runtime_sources=rt.sources,
        parameters={arm: sum(p.numel() for p in rt.readers[arm].parameters()) for arm in ARMS})


def install_infrastructure():
    infrastructure.ROOT = ROOT
    infrastructure.ARMS = ARMS
    infrastructure.Runtime = make_runtime
    infrastructure.sample = sample
    infrastructure.preflight = preflight


def prepare():
    assert read(ROOT / 'status.json')['phase'] == 'trained'
    training = verify_training()
    parent = read(OLD_ROOT / 'request.json')
    baseline = read(OLD_ROOT / 'ig_restarted/result.json')
    assert baseline['complete'] and sha(baseline['sample_path']) == baseline['sample_sha256']
    assets = parent['assets'].copy()
    summaries = [read(ROOT / f'training_{arm}.json') for arm in ARMS]
    for rec in summaries:
        assert rec['complete'] and sha(rec['checkpoint']) == rec['sha256']
        assets[rec['checkpoint']] = rec['sha256']
    request = dict(parent, arms=ARMS, sources=training['sources'], assets=assets, baseline=baseline,
        parent_request_sha256=sha(OLD_ROOT / 'request.json'), training_request_sha256=sha(ROOT / 'training_request.json'),
        training_summaries=summaries, collection=read(ROOT / 'collection.json'),
        feedback_iterations=1, feedback_location='two readouts using frozen cached pre-linear features',
        backbone_evaluations_per_rhs=1, inception_graph_sha256=training['inception_graph_sha256'])
    request.pop('heun_steps')
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', research_goal_achieved=False))
    print(json.dumps(dict(prepared=True, request_sha256=sha(ROOT / 'request.json'))), flush=True)


def run():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request, request_hash = infrastructure.verify_request()
    assert read(ROOT / 'status.json')['phase'] == 'prepared'
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
    workers, streams, results = [], [], []
    begin = time.perf_counter()
    def interrupted(signum, frame):
        raise RuntimeError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank in range(RANKS):
            stream = (ROOT / f'worker{rank}.log').open('w')
            streams.append(stream)
            workers.append(subprocess.Popen([sys.executable, '-u', '-m',
                'experiments.small_sit_feedback_reader_20260909', '--rank', str(rank)], cwd=WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        status = dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in workers], research_goal_achieved=False)
        atomic(ROOT / 'status.json', dict(phase='preflight', **status))
        infrastructure.wait_files([ROOT / f'preflight_rank{r}.json' for r in range(RANKS)], workers)
        checks = [read(ROOT / f'preflight_rank{r}.json') for r in range(RANKS)]
        assert all(c['passed'] for c in checks)
        assert all(c['runtime_sources'] == checks[0]['runtime_sources'] for c in checks)
        atomic(ROOT / 'preflight_passed.json', dict(passed=True, checks=checks, request_sha256=request_hash))
        for arm in ARMS:
            atomic(ROOT / 'status.json', dict(phase='sampling', arm=arm, **status))
            infrastructure.wait_files([ROOT / arm / f'rank{r}/summary.json' for r in range(RANKS)], workers)
            atomic(ROOT / 'status.json', dict(phase='evaluating', arm=arm, **status))
            result = infrastructure.evaluate(arm, request, request_hash)
            results.append(result)
            atomic(ROOT / 'results.json', results)
            print(json.dumps(result), flush=True)
            atomic(ROOT / arm / 'advance.json', dict(complete=True, request_sha256=request_hash))
        codes = [p.wait() for p in workers]
        assert codes == [0] * RANKS, codes
        infrastructure.verify_request()
        atomic(ROOT / 'status.json', dict(phase='complete', results=len(results),
            wall_seconds=time.perf_counter() - begin, numerical_failures=sum(not r['complete'] for r in results),
            worker_exit_codes=codes, **status))
    except BaseException as error:
        atomic(ROOT / 'status.json', dict(phase='failed', error=repr(error), controller_pid=os.getpid(),
            worker_pids=[p.pid for p in workers], research_goal_achieved=False))
        raise
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            p.wait()
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-training', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fit-arm', choices=ARMS)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(RANKS))
    args = parser.parse_args()
    assert sum((args.prepare_training, args.train, args.fit_arm is not None,
                args.prepare, args.run_prepared, args.rank is not None)) == 1
    if args.prepare_training:
        prepare_training()
    elif args.train:
        train()
    elif args.fit_arm is not None:
        fit_arm(args.fit_arm)
    elif args.prepare:
        prepare()
    else:
        install_infrastructure()
        run() if args.run_prepared else infrastructure.worker(args.rank)
