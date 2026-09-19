"""Train matched weak distributions, with immutable requests and held-out losses."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import data, theory
from .data import ROOT, WORK, DATA, METHODS
from experiments.lifting_scale_sweep_20260909 import Runtime, SMALL_CKPT, atomic, read, sha


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'training_request.json').exists():
        verify()
        return
    for name in ('sit_refined_priority_20260911', 'sit_control_output_50ideas_20260910'):
        status = read(ROOT.parent/name/'status.json')
        assert status['phase'] in ('stopped_after_current', 'complete'), status
    theory.run()
    previous = read(ROOT.parent/'small_sit_feedback_reader_20260909/training_request.json')
    for path, digest in previous['data_files'].items():
        assert sha(path) == digest, path
    labels = np.load(DATA/'train_labels.npy')
    moments = np.load(DATA/'train_moments.npy', mmap_mode='r')
    rng = np.random.default_rng(data.SEED)
    indices = np.stack([rng.permutation(np.flatnonzero(labels == c))[
        :data.TRAIN_PER_CLASS+data.VALID_PER_CLASS] for c in range(100)])
    assert len(set(indices.flatten())) == indices.size
    means, variances = [], []
    files = {}
    for split, subset in [('train', indices[:, :data.TRAIN_PER_CLASS]),
                          ('validation', indices[:, data.TRAIN_PER_CLASS:])]:
        np.save(ROOT/f'{split}_indices.npy', subset)
        out = np.lib.format.open_memmap(ROOT/f'{split}_moments.npy', mode='w+',
            dtype=np.float32, shape=(*subset.shape, 8, 32, 32))
        for c in range(100):
            block = np.asarray(moments[subset[c]], dtype=np.float32)
            out[c] = block
            if split == 'train':
                center = block[:, :4].astype(np.float64).mean(0)*.18215
                variance = ((block[:, :4].astype(np.float64)*.18215-center)**2
                            +(block[:, 4:].astype(np.float64)*.18215)**2).mean()
                means.append(center.astype(np.float32))
                variances.append(variance)
        out.flush()
        del out
    np.save(ROOT/'class_means.npy', np.stack(means))
    np.save(ROOT/'class_variances.npy', np.array(variances))
    np.save(ROOT/'heat_sigma.npy', np.sqrt(np.mean(variances)))
    state = torch.load(SMALL_CKPT, map_location='cpu', mmap=True, weights_only=False)
    weights = state['ema']
    embedding = weights['y_embedder.embedding_table.weight'][:100].float().numpy()
    local = data.pair_classes(embedding)
    order = np.random.default_rng(data.SEED+1).permutation(100)
    random = np.empty(100, dtype=np.int64)
    random[order[::2]], random[order[1::2]] = order[1::2], order[::2]
    assert np.array_equal(random[random], np.arange(100))
    for key, partner in [('local', local), ('random', random)]:
        np.save(ROOT/f'{key}_partners.npy', partner)
    sources = dict(previous['sources'])
    for path in [Path(__file__), Path(data.__file__), Path(theory.__file__),
                 Path(__file__).parent/'train_batch.py',
                 Path(__file__).parent/'__init__.py', data.PROTOCOL]:
        sources[str(path.resolve())] = sha(path)
    for path in ROOT.glob('*.npy'):
        files[str(path)] = sha(path)
    request = dict(sources=sources, assets={str(SMALL_CKPT):sha(SMALL_CKPT)},
        original_data_files=previous['data_files'], input_files=files,
        methods=METHODS, initial_weights='800K EMA', full_backbone_training=True,
        steps=data.STEPS, batch=data.BATCH, learning_rate=data.LR, ema=data.EMA,
        adamw_betas=[.9, .999], weight_decay=.01, gradient_clip=1.,
        train_seed=data.TRAIN_SEED, time_range=[.01, .99], null_probability=.1,
        no_validation_selection=True, no_fid_used=True,
        heldout_times=[.15, .5, .85], heldout_draws_per_class=4,
        created_unix=time.time())
    atomic(ROOT/'training_request.json', request)
    snapshot = ROOT/'training_sources'
    snapshot.mkdir(exist_ok=True)
    for i, path in enumerate(sorted(sources)):
        (snapshot/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    atomic(ROOT/'training_status.json', dict(phase='prepared', methods=METHODS))
    print(json.dumps(dict(prepared=True, methods=METHODS, training_images=25600,
                         validation_images=3200, sigma=float(np.sqrt(np.mean(variances))))), flush=True)


def verify():
    request = read(ROOT/'training_request.json')
    for category in ('sources', 'assets', 'input_files'):
        for path, digest in request[category].items():
            assert sha(path) == digest, path
    return request


@torch.no_grad()
def validate(model, pool, method):
    generator = torch.Generator(device='cuda').manual_seed(data.TRAIN_SEED+100)
    rows = []
    for tv in (.15, .5, .85):
        total, size = 0., 0
        for start in range(0, 400, 20):
            labels = torch.arange(start, start+20, device='cuda') % 100
            clean, labels = pool.draw(method, generator, 20, labels=labels)
            noise = torch.randn(clean.shape, device='cuda', generator=generator)
            t = torch.full((len(clean),), tv, device='cuda')
            z = tv*clean+(1-tv)*noise
            prediction = model(z, t, labels)
            assert prediction.shape == clean.shape
            total += float((prediction.double()-(clean-noise).double()).square().sum())
            size += clean.numel()
        rows.append(dict(time=tv, mse=total/size, images=400))
    return rows


def train(method):
    request = verify()
    folder = ROOT/'training'/method
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'complete.json').exists():
            done = read(folder/'complete.json')
            assert done['request_sha256'] == sha(ROOT/'training_request.json')
            assert done['checkpoint_sha256'] == sha(folder/'model.pt')
            return
        rt = Runtime('sit_small')
        model = rt.model.eval().requires_grad_(True)
        model.pos_embed.requires_grad_(False)
        # Official learn_sigma models declare eight raw channels but forward()
        # returns only the four velocity channels.
        assert model.in_channels == 4 and rt.semantics.prediction_target == 'velocity'
        torch.set_num_threads(4)
        torch.manual_seed(data.TRAIN_SEED)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        pool = data.Pool('train')
        validation_pool = data.Pool('validation')
        before = validate(ema, validation_pool, method)
        optimizer = torch.optim.AdamW(model.parameters(), lr=data.LR,
                                       betas=(.9, .999), weight_decay=.01)
        generator = torch.Generator(device='cuda').manual_seed(data.TRAIN_SEED)
        start_step = 0
        if (folder/'resume.pt').exists():
            saved = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
            assert saved['request_sha256'] == sha(ROOT/'training_request.json')
            model.load_state_dict(saved['model'])
            ema.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['generator'])
            start_step = saved['step']
        begin = time.perf_counter()
        losses = []
        for step in range(start_step+1, data.STEPS+1):
            clean, labels = pool.draw(method, generator)
            noise = torch.randn(clean.shape, device='cuda', generator=generator)
            t = .01+.98*torch.rand(len(clean), device='cuda', generator=generator)
            dropped = torch.rand(len(clean), device='cuda', generator=generator) < .1
            labels = torch.where(dropped, torch.full_like(labels, 100), labels)
            z = t[:, None, None, None]*clean+(1-t[:, None, None, None])*noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(z, t, labels)
                loss = (prediction.float()-(clean-noise)).square().mean()
            assert torch.isfinite(loss), (method, step)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1-data.EMA)
            losses.append(float(loss.detach()))
            if step % 50 == 0 or step == 1:
                torch.cuda.synchronize()
                record = dict(phase='training', method=method, step=step, total=data.STEPS,
                    loss_mean=float(np.mean(losses[-50:])), gradient_norm=float(grad),
                    elapsed_seconds=time.perf_counter()-begin, pid=os.getpid())
                atomic(folder/'status.json', record)
                print(json.dumps(record), flush=True)
            if step % 250 == 0 or step == data.STEPS or (ROOT/'STOP_AFTER_CURRENT').exists():
                temporary = folder/'resume.tmp.pt'
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(), generator=generator.get_state(),
                    step=step, request_sha256=sha(ROOT/'training_request.json')), temporary)
                temporary.replace(folder/'resume.pt')
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                atomic(folder/'status.json', dict(phase='stopped_after_step', step=step))
                return
        torch.cuda.synchronize()
        training_seconds = time.perf_counter()-begin
        after_target = validate(ema, validation_pool, method)
        after_native = validate(ema, validation_pool, 'native')
        temp = folder/'model.tmp.pt'
        torch.save(dict(ema={k:v.cpu() for k,v in ema.state_dict().items()},
            step=data.STEPS, method=method, request_sha256=sha(ROOT/'training_request.json')), temp)
        temp.replace(folder/'model.pt')
        result = dict(passed=True, method=method, step=data.STEPS, full_training_images=data.STEPS*data.BATCH,
            request_sha256=sha(ROOT/'training_request.json'), checkpoint_sha256=sha(folder/'model.pt'),
            target_mse_before=before, target_mse_after=after_target, native_mse_after=after_native,
            training_seconds=training_seconds, resumed_from_step=start_step,
            no_fid_used=True, selected_by_validation=False,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            created_unix=time.time())
        atomic(folder/'complete.json', result)
        atomic(folder/'status.json', dict(phase='complete', **result))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--train', choices=METHODS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        train(args.train)
