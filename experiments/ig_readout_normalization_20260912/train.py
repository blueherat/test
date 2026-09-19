"""Three matched real-data heads; the only intervention is readout input representation."""
import copy
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as local

ROOT = c.EXPS / 'ig_readout_normalization_20260912'
PROTOCOL = c.WORK / 'docs/IG_READOUT_NORMALIZATION_PROTOCOL_20260912_ZH.md'
TRAIN = ROOT / 'sit_small/training'
MODES = ('raw', 'ln', 'ln_stats')
EPS = 1e-6


def representation(tokens, mode):
    tokens = tokens.float()
    if mode == 'raw':
        return tokens, None
    mu = tokens.mean(-1, keepdim=True)
    radius = ((tokens - mu).square().mean(-1, keepdim=True) + EPS).sqrt()
    norm = F.layer_norm(tokens, (tokens.shape[-1],), eps=EPS)
    return norm, torch.cat((mu, radius), -1)


class Head(nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        self.base = base
        self.mode = mode
        if mode == 'ln_stats':
            width = base.token.in_features
            self.stats = nn.Linear(2, width, bias=False, device=base.token.weight.device)
            nn.init.zeros_(self.stats.weight)
            self.register_buffer('stats_mean', torch.zeros(2, device=base.token.weight.device))
            self.register_buffer('stats_std', torch.ones(2, device=base.token.weight.device))

    def forward(self, tokens, condition):
        tokens, statistics = representation(tokens, self.mode)
        if self.mode != 'ln_stats':
            return self.base(tokens, condition)
        b = self.base
        x = (tokens - b.token_mean) / b.token_std
        y = (condition.float() - b.condition_mean) / b.condition_std
        stats = (statistics - self.stats_mean) / self.stats_std
        value = b.token(x) + b.condition(y)[:, None] + b.position(b.positions) + self.stats(stats)
        return b.output(F.silu(value))


def prepare():
    old = local.ROOT / 'sit_small/input_local_training'
    local.verify_request(old / 'request.json')
    files = [Path(__file__).resolve(), Path(local.__file__).resolve(), PROTOCOL,
        c.WORK / 'experiments/sit_measure_guidance_20260912/data.py']
    request = dict(model='sit_small', steps=3000, batch=32, seed=local.TRAIN_SEED,
        modes=list(MODES), normalization_batches=32, lr=.0003, weight_decay=.0001, ema=.995,
        real_data_only=True, sources=c.source_manifest(files),
        assets={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
        data={str(p): c.sha(p) for p in local.data_paths('sit_small')},
        heads={str(old / f): c.sha(old / f) for f in ('head.pt', 'request.json', 'summary.json')})
    path = TRAIN / 'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path, request)
    return path


def make_heads(rt):
    base = local.make_head(rt)
    return {mode: Head(base if mode == 'raw' else copy.deepcopy(base), mode) for mode in MODES}


def train():
    path = prepare()
    local.verify_request(path)
    if (TRAIN / 'summary.json').exists():
        return
    torch.manual_seed(local.TRAIN_SEED)
    rt = c.runtime('sit_small')
    assert rt.head.module.norm_final.eps == EPS
    data = local.RealData('sit_small')
    heads = make_heads(rt)
    g = torch.Generator(device='cuda').manual_seed(local.TRAIN_SEED)
    sums = {k: torch.zeros(2 if k == 'stats' else 384, device='cuda', dtype=torch.float64)
            for k in (*MODES, 'condition', 'stats')}
    squares = {k: v.clone() for k, v in sums.items()}
    counts = {k: 0 for k in sums}
    with torch.no_grad():
        for _ in range(32):
            feats, _, _, _, _ = local.training_batch(rt, data, 'train', g, 32)
            values = {mode: representation(feats['context'], mode)[0] for mode in MODES}
            values.update(condition=feats['condition'], stats=representation(feats['context'], 'ln_stats')[1])
            for k, x in values.items():
                x = x.reshape(-1, x.shape[-1]).double()
                sums[k] += x.sum(0)
                squares[k] += x.square().sum(0)
                counts[k] += len(x)
        for mode, head in heads.items():
            for key, prefix in ((mode, 'token'), ('condition', 'condition')):
                mean = sums[key] / counts[key]
                std = (squares[key] / counts[key] - mean.square()).clamp_min(1e-8).sqrt()
                getattr(head.base, prefix + '_mean').copy_(mean.float())
                getattr(head.base, prefix + '_std').copy_(std.float())
            if mode == 'ln_stats':
                mean = sums['stats'] / counts['stats']
                std = (squares['stats'] / counts['stats'] - mean.square()).clamp_min(1e-8).sqrt()
                head.stats_mean.copy_(mean.float())
                head.stats_std.copy_(std.float())
        assert torch.equal(heads['ln'](feats['context'], feats['condition']),
                           heads['ln_stats'](feats['context'], feats['condition']))
        norm, stats = representation(feats['context'], 'ln_stats')
        reconstruction = stats[..., :1] + stats[..., 1:] * norm
        reconstruction_error = float((reconstruction - feats['context']).abs().max())
        assert torch.allclose(reconstruction, feats['context'], rtol=1e-5, atol=1e-5)
    ema = {k: copy.deepcopy(h).eval().requires_grad_(False) for k, h in heads.items()}
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=.0003, weight_decay=.0001) for k, h in heads.items()}
    history = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for step in range(1, 3001):
        feats, target, _, _, _ = local.training_batch(rt, data, 'train', g, 32)
        losses = {}
        for mode, head in heads.items():
            optimizers[mode].zero_grad(set_to_none=True)
            with rt.context():
                prediction = head(feats['context'], feats['condition'])
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
            row = dict(step=step, **losses)
            history.append(row)
            c.atomic(TRAIN / 'progress.json', dict(pid=os.getpid(), **row))
            print(row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    validation = {k: [] for k in MODES}
    vg = torch.Generator(device='cuda').manual_seed(local.TRAIN_SEED + 7)
    with torch.no_grad():
        for _ in range(32):
            feats, target, _, _, _ = local.training_batch(rt, data, 'validation', vg, 32)
            with rt.context():
                for mode, head in ema.items():
                    validation[mode].append(float((head(feats['context'], feats['condition']).float() - target).square().mean()))
    old = torch.load(local.ROOT / 'sit_small/input_local_training/head.pt', map_location='cpu', weights_only=True)['ema']['context']
    current = ema['raw'].base.state_dict()
    differences = {k: float((current[k].cpu() - value).abs().max()) for k, value in old.items()}
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    target = TRAIN / 'head.pt'
    temporary = target.with_suffix('.tmp')
    torch.save(dict(ema={k: h.state_dict() for k, h in ema.items()}, steps=3000,
        request_sha256=c.sha(path)), temporary)
    temporary.replace(target)
    summary = dict(complete=True, steps=3000, batch=32, training_seconds=seconds,
        history=history, validation_mse={k: float(np.mean(v)) for k, v in validation.items()},
        parameters={k: sum(p.numel() for p in h.parameters()) for k, h in ema.items()},
        raw_replay_max_difference=max(differences.values()), raw_replay_tensor_differences=differences,
        raw_replay_exact=all(v == 0 for v in differences.values()), normalization_reconstruction_max_error=reconstruction_error,
        ln_and_ln_stats_initial_outputs_exact=True, strong_parameters_frozen=True,
        request_sha256=c.sha(path), head_sha256=c.sha(target))
    c.atomic(TRAIN / 'summary.json', summary)
    print('Completed', summary['validation_mse'], 'raw replay difference', summary['raw_replay_max_difference'], flush=True)


def load_heads(rt):
    local.verify_request(TRAIN / 'request.json')
    summary = c.read(TRAIN / 'summary.json')
    assert summary['complete'] and summary['head_sha256'] == c.sha(TRAIN / 'head.pt')
    state = torch.load(TRAIN / 'head.pt', map_location='cpu', weights_only=True)
    assert state['request_sha256'] == c.sha(TRAIN / 'request.json') and state['steps'] == 3000
    heads = make_heads(rt)
    for mode, head in heads.items():
        head.load_state_dict(state['ema'][mode], strict=True)
        head.eval().requires_grad_(False)
    return heads


if __name__ == '__main__':
    train()
