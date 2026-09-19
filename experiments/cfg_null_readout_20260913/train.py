"""Three null readouts on one frozen full backbone and one real-data training stream."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local
from experiments.imagenet100_sit_internal_v_head import extract_internal_features, unpatchify_channels

ROOT = c.EXPS / 'cfg_null_readout_20260913'
TRAIN = ROOT / 'sit_small/training'
PROTOCOL = c.WORK / 'docs/CFG_NULL_READOUT_PROTOCOL_20260913_ZH.md'
MODES = ('mlp', 'adaln_fresh', 'adaln_warm')
SEED = 2026121351


def prepare():
    request = dict(model='sit_small', modes=list(MODES), steps=3000, batch=32, seed=SEED,
        lr=.0003, weight_decay=.0001, ema=.995, normalization_batches=32, real_data_only=True,
        sources=c.source_manifest([Path(__file__), Path(local.__file__), PROTOCOL,
            c.WORK / 'experiments/sit_measure_guidance_20260912/data.py']),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        data={str(p): c.sha(p) for p in local.data_paths('sit_small')})
    path = TRAIN / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


@torch.no_grad()
def features(rt, z, t):
    labels = torch.full((len(z),), 100, device=z.device, dtype=torch.long)
    return extract_internal_features(rt.model, z, t, labels, internal_depth=len(rt.model.blocks))


def project(rt, head, tokens, condition):
    value = head(tokens, condition)
    channels = value.shape[-1] // (int(rt.model.x_embedder.patch_size[0]) ** 2)
    return unpatchify_channels(rt.model, value, channels=channels)[:, :4]


def batch(rt, data, split, generator):
    clean, _ = data.draw(split, generator, 32)
    times = .01 + .98 * torch.rand(32, device='cuda', generator=generator)
    noise = torch.randn(clean.shape, device='cuda', generator=generator)
    t = times[:, None, None, None]
    z = t * clean + (1 - t) * noise
    tokens, condition = features(rt, z, times)
    return tokens, condition, clean - noise


def make_heads(rt):
    width = int(rt.model.pos_embed.shape[-1])
    side = 32 // int(rt.model.x_embedder.patch_size[0])
    output = rt.model.final_layer.linear.out_features
    mlp = local.Head(width, output, side).cuda()
    warm = copy.deepcopy(rt.model.final_layer).requires_grad_(True)
    fresh = copy.deepcopy(warm)
    with torch.no_grad():
        for p in fresh.parameters():
            p.zero_()
    return dict(mlp=mlp, adaln_fresh=fresh, adaln_warm=warm)


def train(parent=0):
    path = prepare()
    local.verify_request(path)
    if (TRAIN / 'summary.json').exists():
        return
    torch.manual_seed(SEED)
    rt = c.runtime('sit_small')
    data = local.RealData('sit_small')
    heads = make_heads(rt)
    generator = torch.Generator(device='cuda').manual_seed(SEED)
    sums = {k: torch.zeros(384, device='cuda', dtype=torch.float64) for k in ('token', 'condition')}
    squares = {k: v.clone() for k, v in sums.items()}
    counts = {k: 0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            tokens, condition, _ = batch(rt, data, 'train', generator)
            for k, value in [('token', tokens), ('condition', condition)]:
                value = value.reshape(-1, 384).double()
                sums[k] += value.sum(0)
                squares[k] += value.square().sum(0)
                counts[k] += len(value)
        for k in sums:
            mean = sums[k] / counts[k]
            std = (squares[k] / counts[k] - mean.square()).clamp_min(1e-8).sqrt()
            getattr(heads['mlp'], k + '_mean').copy_(mean.float())
            getattr(heads['mlp'], k + '_std').copy_(std.float())
    ema = {k: copy.deepcopy(h).eval().requires_grad_(False) for k, h in heads.items()}
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=.0003, weight_decay=.0001) for k, h in heads.items()}
    history = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for step in range(1, 3001):
        tokens, condition, target = batch(rt, data, 'train', generator)
        losses = {}
        for mode, head in heads.items():
            optimizers[mode].zero_grad(set_to_none=True)
            prediction = project(rt, head, tokens, condition)
            loss = (prediction.float() - target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizers[mode].step()
            with torch.no_grad():
                for p, q in zip(ema[mode].parameters(), head.parameters()):
                    p.lerp_(q, .005)
            losses[mode] = float(loss.detach())
        if step == 1 or step % 100 == 0:
            c.check_parent(parent)
            row = dict(step=step, **losses)
            history.append(row)
            c.atomic(TRAIN / 'progress.json', row)
            print(row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    validation = {k: [] for k in heads}
    vg = torch.Generator(device='cuda').manual_seed(SEED + 7)
    with torch.no_grad():
        for _ in range(32):
            tokens, condition, target = batch(rt, data, 'validation', vg)
            for mode, head in ema.items():
                validation[mode].append(float((project(rt, head, tokens, condition).float() - target).square().mean()))
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    checkpoint = TRAIN / 'head.pt'
    temporary = checkpoint.with_suffix('.tmp')
    torch.save(dict(ema={k: h.state_dict() for k, h in ema.items()}, steps=3000, request_sha256=c.sha(path)), temporary)
    temporary.replace(checkpoint)
    c.atomic(TRAIN / 'summary.json', dict(complete=True, steps=3000, shared_training_seconds=seconds,
        validation_mse={k: float(np.mean(v)) for k, v in validation.items()}, history=history,
        parameters={k: sum(p.numel() for p in h.parameters()) for k, h in ema.items()},
        backbone_frozen=True, original_conditional_head_frozen=True, unused_output_channels_not_supervised=True,
        request_sha256=c.sha(path), head_sha256=c.sha(checkpoint)))


def load_heads(rt):
    local.verify_request(TRAIN / 'request.json')
    summary = c.read(TRAIN / 'summary.json')
    assert summary['complete'] and c.sha(TRAIN / 'head.pt') == summary['head_sha256']
    state = torch.load(TRAIN / 'head.pt', map_location='cpu', weights_only=True)
    assert state['steps'] == 3000 and state['request_sha256'] == c.sha(TRAIN / 'request.json')
    heads = make_heads(rt)
    for k, h in heads.items():
        h.load_state_dict(state['ema'][k], strict=True)
        h.eval().requires_grad_(False)
    return heads


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=int, default=0)
    args = parser.parse_args()
    train(args.parent)
