from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from experiments.raev2_training_core import DeterministicImageNetPacked
from . import common as c


def prepare():
    c.ROOT.mkdir(parents=True, exist_ok=True)
    if (c.ROOT / 'training_request.json').exists():
        return c.verify_request(c.ROOT / 'training_request.json')
    dataset = DeterministicImageNetPacked(c.DATA, image_size=256, horizontal_flip=False)
    labels = np.concatenate(dataset._labels)
    validation = np.load(c.OLD_HEAD_ROOT / 'validation_ids.npy')
    assert np.array_equal(labels[validation], np.arange(1000))
    assert np.array_equal(validation, [np.flatnonzero(labels == i)[0] for i in range(1000)])
    rng = np.random.default_rng(c.SEED)
    train = np.concatenate([rng.choice(np.flatnonzero(labels == i)[1:], 48, replace=False) for i in range(1000)])
    train = rng.permutation(train).astype(np.int64)
    assert len(train) == c.STEPS * c.BATCH and len(set(train) & set(validation)) == 0
    assert len(np.unique(train)) == len(train)
    np.save(c.ROOT / 'training_ids.npy', train)
    np.save(c.ROOT / 'validation_ids.npy', validation)
    sources = [Path(__file__).resolve(), Path(c.__file__).resolve(), Path(__file__).resolve().parent / '__init__.py',
        c.PROTOCOL, Path(c.jig.__file__).resolve(), c.jig.REPO / 'model_jit.py',
        c.jig.REPO / 'util/model_util.py', c.jig.REPO / 'denoiser.py',
        c.WORK / 'experiments/raev2_training_core.py', c.WORK / 'experiments/train_jit_internal_readouts.py']
    old = c.read(c.OLD_HEAD_ROOT / 'request.json')
    assert old['checkpoint_sha256'] == c.sha(c.jig.CHECKPOINT)
    assert old['dataset_manifest_sha256'] == c.sha(c.DATA / 'manifest.json')
    for name, digest in old['sources'].items():
        assert c.sha(name) == digest, name
    request = dict(model='JiT-B/16', methods=c.METHODS, steps=c.STEPS, batch=c.BATCH,
        learning_rate=1e-4, adamw_betas=[.9, .999], weight_decay=0., ema=.995, gradient_clip=1.,
        seed=c.SEED, time_distribution='sigmoid(N(-.8,.8^2))', t_eps=.05,
        label_dropout=.1, flip_probability=.5, training_examples=len(train),
        target='true clean-image velocity loss; no teacher targets', selected_by_validation=False,
        sources={str(p): c.sha(p) for p in sources},
        assets={str(p): c.sha(p) for p in (c.jig.CHECKPOINT, c.HEAD)},
        inputs={str(p): c.sha(p) for p in (c.DATA / 'manifest.json', c.ROOT / 'training_ids.npy',
            c.ROOT / 'validation_ids.npy', c.OLD_HEAD_ROOT / 'request.json')},
        old_stop_markers={str(p): c.sha(p) for p in c.OLD_STOP}, created_unix=time.time())
    c.atomic(c.ROOT / 'training_request.json', request)
    snapshot = c.ROOT / 'training_sources'
    snapshot.mkdir(exist_ok=True)
    for i, path in enumerate(sources):
        (snapshot / f'{i:02d}_{path.name}').write_bytes(path.read_bytes())
    print('Training request prepared', flush=True)
    return request


@torch.no_grad()
def validate(strong, weak, dataset):
    ids = np.load(c.ROOT / 'validation_ids.npy')
    loader = DataLoader(Subset(dataset, ids.tolist()), batch_size=16, num_workers=2, pin_memory=True)
    rng = torch.Generator(device='cuda').manual_seed(c.SEED + 100)
    total = torch.zeros(3, dtype=torch.float64, device='cuda')
    for images, labels, _ in loader:
        x, y = images.cuda().mul(2).sub(1), labels.cuda()
        t = torch.sigmoid(torch.randn(len(x), device='cuda', generator=rng) * .8 - .8)
        tt = t[:, None, None, None]
        eps = torch.randn(x.shape, device='cuda', generator=rng)
        z = tt * x + (1 - tt) * eps
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted, full = weak(z, t, y), strong(z, t, y)
        den = (1 - tt).clamp_min(.05)
        total[0] += ((predicted.float() - x) / den).double().square().flatten(1).mean(1).sum()
        total[1] += ((full.float() - x) / den).double().square().flatten(1).mean(1).sum()
        total[2] += len(x)
    return dict(samples=int(total[2]), weak=float(total[0] / total[2]), strong=float(total[1] / total[2]))


def train(method):
    request = c.verify_request(c.ROOT / 'training_request.json')
    rh = c.sha(c.ROOT / 'training_request.json')
    out = c.ROOT / 'training' / method
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists():
        done = c.read(out / 'complete.json')
        assert done['request_sha256'] == rh and done['checkpoint_sha256'] == c.sha(out / 'model.pt')
        return
    c.setup()
    strong, readout = c.load_models()
    weak = c.WeakPrefix(strong, readout).configure_training(method)
    ema = copy.deepcopy(weak).requires_grad_(False)
    strong_hash = c.state_sha(strong)
    fixed_initial = {key: value.detach().cpu().clone() for key, value in weak.state_dict().items()
        if not key.startswith('readout.')}
    rng = torch.Generator(device='cuda').manual_seed(c.SEED)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = torch.randn(4, 3, 256, 256, device='cuda', generator=rng)
        t = torch.full((4,), .35, device='cuda')
        y = torch.arange(4, device='cuda')
        feats, context = c.jig.features(strong, z, t, y, depths=(4,))
        old = c.jig.unpatchify(readout(feats['4'], context))
        torch.testing.assert_close(weak(z, t, y), old, rtol=0, atol=0)
    dataset = DeterministicImageNetPacked(c.DATA, image_size=256, horizontal_flip=False)
    before = validate(strong, ema, dataset)
    optimizer = torch.optim.AdamW([p for p in weak.parameters() if p.requires_grad],
        lr=1e-4, betas=(.9, .999), weight_decay=0.)
    rng.manual_seed(c.SEED)
    step0, elapsed0, fingerprints = 0, 0., []
    if (out / 'resume.pt').exists():
        saved = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        assert saved['request_sha256'] == rh
        weak.load_state_dict(saved['weak']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); rng.set_state(saved['rng'])
        step0, elapsed0, fingerprints = saved['step'], saved['elapsed_seconds'], saved['fingerprints']
    ids = np.load(c.ROOT / 'training_ids.npy')[step0 * c.BATCH:]
    loader = DataLoader(Subset(dataset, ids.tolist()), batch_size=c.BATCH, num_workers=4,
        pin_memory=True, drop_last=True, persistent_workers=True)
    losses = []
    started = time.perf_counter()
    trainable = [p for p in weak.parameters() if p.requires_grad]
    for step, (images, labels, source_ids) in enumerate(loader, start=step0 + 1):
        x, y = images.cuda(non_blocking=True).mul(2).sub(1), labels.cuda(non_blocking=True)
        flip = torch.rand(len(x), device='cuda', generator=rng) < .5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        y = torch.where(torch.rand(len(x), device='cuda', generator=rng) < .1, 1000, y)
        t = torch.sigmoid(torch.randn(len(x), device='cuda', generator=rng) * .8 - .8)
        tt = t[:, None, None, None]
        eps = torch.randn(x.shape, device='cuda', generator=rng)
        z = tt * x + (1 - tt) * eps
        # Compact RNG fingerprint, explicitly not a hash of every noise element.
        digest = hashlib.sha256()
        for tensor in (source_ids, flip, y, t, eps.flatten(1)[:, :64]):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        fingerprints.append(digest.hexdigest())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = weak(z, t, y)
        loss = ((prediction.float() - x) / (1 - tt).clamp_min(.05)).square().mean()
        assert torch.isfinite(loss), (method, step)
        loss.backward()
        assert all(p.grad is not None for p in trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), weak.parameters()):
                if source.requires_grad:
                    target.lerp_(source, .005)
        losses.append(float(loss.detach()))
        elapsed = elapsed0 + time.perf_counter() - started
        if step == 1 or step % 50 == 0:
            record = dict(phase='training', method=method, step=step, steps=c.STEPS,
                loss=float(np.mean(losses[-50:])), gradient_norm=float(norm), elapsed_seconds=elapsed, pid=os.getpid())
            c.atomic(out / 'status.json', record)
            print(record, flush=True)
        stopping = (c.ROOT / 'STOP_AFTER_CURRENT').exists()
        if step % 250 == 0 or stopping:
            c.atomic_torch(out / 'resume.pt', dict(weak=weak.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng.get_state(), step=step,
                elapsed_seconds=elapsed, fingerprints=fingerprints, request_sha256=rh))
        if stopping:
            c.atomic(out / 'status.json', dict(phase='paused', step=step))
            return
    torch.cuda.synchronize()
    elapsed = elapsed0 + time.perf_counter() - started
    assert step == c.STEPS and len(fingerprints) == c.STEPS
    assert all(p.grad is None for p in strong.parameters()) and c.state_sha(strong) == strong_hash
    if method == 'head_only':
        for key, original in fixed_initial.items():
            assert torch.equal(ema.state_dict()[key].cpu(), original), key
    after = validate(strong, ema, dataset)
    assert before['strong'] == after['strong']
    c.atomic_torch(out / 'model.pt', dict(ema={key: value.cpu() for key, value in ema.state_dict().items()},
        step=c.STEPS, method=method, request_sha256=rh))
    c.atomic(out / 'random_fingerprints.json', fingerprints)
    done = dict(passed=True, method=method, step=c.STEPS, training_seconds=elapsed,
        trainable_parameters=sum(p.numel() for p in trainable), strong_state_sha256=strong_hash,
        strong_unchanged=True, strong_gradients_absent=True, initial_weak_exact=True,
        frozen_prefix_unchanged=method == 'head_only', validation_before=before, validation_after=after,
        random_fingerprints_sha256=c.sha(out / 'random_fingerprints.json'),
        request_sha256=rh, checkpoint_sha256=c.sha(out / 'model.pt'), selected_by_validation=False)
    c.atomic(out / 'complete.json', done)
    c.atomic(out / 'status.json', dict(phase='complete', **done))
    print(done, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', action='store_true')
    choice.add_argument('--train', choices=c.METHODS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        train(args.train)
