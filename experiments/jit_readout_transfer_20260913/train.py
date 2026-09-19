import copy
import fcntl
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from experiments.raev2_training_core import DeterministicImageNetPacked
from . import common as c


def prepare():
    c.TRAIN.mkdir(parents=True, exist_ok=True)
    path = c.TRAIN / 'request.json'
    if path.exists():
        return c.verify(path)
    data = DeterministicImageNetPacked(c.DATA, image_size=256, horizontal_flip=False)
    labels = np.concatenate(data._labels)
    validation = np.load(c.OLD / 'validation_ids.npy')
    assert np.array_equal(labels[validation], np.arange(1000))
    rng = np.random.default_rng(c.SEED)
    pool = np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(labels == i), validation[i:i + 1]),
                           48, replace=False) for i in range(1000)])
    updates = np.concatenate([rng.permutation(pool) for _ in range(2)])
    statistics = rng.permutation(pool)[:32 * c.BATCH]
    stream = np.concatenate([statistics, updates]).astype(np.int64)
    assert len(updates) == c.STEPS * c.BATCH and not np.isin(stream, validation).any()
    np.save(c.TRAIN / 'training_ids.npy', stream)
    np.save(c.TRAIN / 'validation_ids.npy', validation)
    old = c.read(c.OLD / 'request.json')
    assert c.sha(c.jig.CHECKPOINT) == old['checkpoint_sha256']
    assert c.sha(c.DATA / 'manifest.json') == old['dataset_manifest_sha256']
    for p, digest in old['sources'].items():
        assert c.sha(p) == digest
    request = dict(model='JiT-B/16', depth=4, steps=c.STEPS, batch=c.BATCH, seed=c.SEED,
        lr=.0003, weight_decay=.0001, ema=.995, native_prediction='clean', statistics_batches=32,
        time_distribution='sigmoid(N(-.8,.8^2))', loss='true clean-output velocity MSE with denominator floor .05',
        label_dropout=.1, flip_probability=.5, distinct_real_training_images=len(pool),
        training_image_presentations=len(updates), statistics_image_presentations=len(statistics),
        sources={str(p.resolve()): c.sha(p) for p in c.training_sources(__file__)},
        assets={str(p): c.sha(p) for p in (c.jig.CHECKPOINT, c.OLD / 'last.pt')},
        data={str(p): c.sha(p) for p in (c.DATA / 'manifest.json', c.OLD / 'request.json',
              c.TRAIN / 'training_ids.npy', c.TRAIN / 'validation_ids.npy')})
    c.atomic(path, request)
    return request


def batch(images, labels, rng):
    x, y = images.cuda(non_blocking=True).mul(2).sub(1), labels.cuda(non_blocking=True)
    flip = torch.rand(len(x), device='cuda', generator=rng) < .5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(torch.rand(len(x), device='cuda', generator=rng) < .1, 1000, y)
    t = torch.sigmoid(torch.randn(len(x), device='cuda', generator=rng) * .8 - .8)
    tt = t[:, None, None, None]
    eps = torch.randn(x.shape, device='cuda', generator=rng)
    return x, y, t, tt * x + (1 - tt) * eps


def main():
    prepare()
    lock = (c.TRAIN / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (c.TRAIN / 'summary.json').exists():
        done = c.read(c.TRAIN / 'summary.json')
        assert done['complete'] and c.sha(c.TRAIN / 'head.pt') == done['head_sha256']
        return
    c.setup()
    model = c.jig.load_source('cuda')
    initial_hash = c.state_sha(model)
    heads = c.make_heads()
    assert torch.equal(heads['native_fresh'].norm_final.weight, torch.ones_like(heads['native_fresh'].norm_final.weight))
    data = DeterministicImageNetPacked(c.DATA, image_size=256, horizontal_flip=False)
    stream = np.load(c.TRAIN / 'training_ids.npy')
    loader = DataLoader(Subset(data, stream.tolist()), batch_size=c.BATCH, num_workers=4,
                        pin_memory=True, persistent_workers=True, drop_last=True)
    generator = iter(loader)
    rng = torch.Generator(device='cuda').manual_seed(c.SEED)
    sums = {k: torch.zeros(768, dtype=torch.float64, device='cuda') for k in ('token', 'condition')}
    squares = {k: v.clone() for k, v in sums.items()}
    counts = {k: 0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            images, labels, _ = next(generator)
            x, y, t, z = batch(images, labels, rng)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat, condition = c.jig.features(model, z, t, y, depths=(4,))
            for key, values in (('token', feat['4']), ('condition', condition)):
                values = values.reshape(-1, 768).double()
                sums[key] += values.sum(0)
                squares[key] += values.square().sum(0)
                counts[key] += len(values)
        for key in sums:
            mean = sums[key] / counts[key]
            std = (squares[key] / counts[key] - mean.square()).clamp_min(1e-8).sqrt()
            getattr(heads['mlp'], key + '_mean').copy_(mean.float())
            getattr(heads['mlp'], key + '_std').copy_(std.float())
    ema = {k: copy.deepcopy(h).eval().requires_grad_(False) for k, h in heads.items()}
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=.0003, weight_decay=.0001) for k, h in heads.items()}
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, c.STEPS + 1):
        images, labels, _ = next(generator)
        x, y, t, z = batch(images, labels, rng)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            feat, condition = c.jig.features(model, z, t, y, depths=(4,))
        losses = {}
        for key, head in heads.items():
            optimizers[key].zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = c.jig.unpatchify(head(feat['4'], condition))
            loss = ((prediction.float() - x) / (1 - t[:, None, None, None]).clamp_min(.05)).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizers[key].step()
            with torch.no_grad():
                for target, source in zip(ema[key].parameters(), head.parameters()):
                    target.lerp_(source, .005)
            losses[key] = float(loss.detach())
        if step == 1 or step % 100 == 0:
            row = dict(step=step, **losses)
            history.append(row)
            c.atomic(c.TRAIN / 'progress.json', dict(pid=os.getpid(), phase='training', **row))
            print(row, flush=True)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert c.state_sha(model) == initial_hash
    val = np.load(c.TRAIN / 'validation_ids.npy')
    vd = DataLoader(Subset(data, val.tolist()), batch_size=16, num_workers=2, pin_memory=True)
    vg = torch.Generator(device='cuda').manual_seed(c.SEED + 7)
    validation = {k: 0. for k in (*heads, 'native_base')}
    old_head = c.original_head()
    with torch.no_grad():
        for images, labels, _ in vd:
            x, y = images.cuda().mul(2).sub(1), labels.cuda()
            t = torch.sigmoid(torch.randn(len(x), device='cuda', generator=vg) * .8 - .8)
            tt = t[:, None, None, None]
            z = tt * x + (1 - tt) * torch.randn(x.shape, device='cuda', generator=vg)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat, condition = c.jig.features(model, z, t, y, depths=(4,))
                for key, head in dict(**ema, native_base=old_head).items():
                    pred = c.jig.unpatchify(head(feat['4'], condition))
                    error = ((pred.float() - x) / (1 - tt).clamp_min(.05)).double().square().flatten(1).mean(1)
                    validation[key] += float(error.sum())
    c.atomic_torch(c.TRAIN / 'head.pt', dict(ema={k: h.cpu().state_dict() for k, h in ema.items()},
        steps=c.STEPS, request_sha256=c.sha(c.TRAIN / 'request.json')))
    summary = dict(complete=True, steps=c.STEPS, batch=c.BATCH, training_seconds=training_seconds,
        validation_samples=len(val), validation_mse={k: v / len(val) for k, v in validation.items()},
        parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()},
        native_parameters=sum(p.numel() for p in old_head.parameters()), strong_state_sha256=initial_hash,
        strong_unchanged=True, strong_gradients_absent=True, history=history,
        head_sha256=c.sha(c.TRAIN / 'head.pt'), request_sha256=c.sha(c.TRAIN / 'request.json'))
    c.atomic(c.TRAIN / 'summary.json', summary)
    print('Training complete', summary['validation_mse'], flush=True)


if __name__ == '__main__':
    main()
