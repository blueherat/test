"""Actual-model replay audit across changed inputs, labels and weak weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new audit output file')
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
    gpu = next(r for r in gpu_snapshot() if str(r['index']) == args.gpu)
    lease = Path('/tmp', f"eqvae_idle_{gpu['uuid']}.lock").open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert eligible(next(r for r in gpu_snapshot() if r['uuid'] == gpu['uuid']))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    import torch
    from .adapters import load, materialize_autocast_weights
    from .sampler import for_adapter
    from .reference import sample
    from .features import enable_backbone_checkpointing
    from experiments.guidance_loss_50k_20260914.config import sha

    torch.manual_seed(2026091902)
    adapter, head, provenance = load(args.model)
    if args.model != 'sit_small':
        materialize_autocast_weights(adapter)
    if args.checkpoint_backbone:
        enable_backbone_checkpointing(adapter)
    noise = torch.randn(args.batch, *adapter.cfg['shape'], device='cuda')
    labels = torch.arange(args.batch, device='cuda')
    coefficient = 1.05 if args.model == 'sit_small' else .3 if args.model == 'jit' else .35
    engine = for_adapter(adapter, head, noise, labels, coefficient)
    parameters = tuple(head.parameters())
    rows = []
    for iteration in range(2):
        # The second pass uses new labels/noise and in-place-updated weak weights.
        x = torch.randn_like(noise)
        y = (labels*17+iteration*101) % adapter.cfg['classes']
        expected = sample(adapter, head, x, y, coefficient)
        reference = torch.autograd.grad(expected.square().mean(), parameters)
        actual = engine(x, y)
        derivatives = torch.autograd.grad(actual.square().mean(), parameters)
        g, r = (torch.cat([v.flatten() for v in values]).float() for values in (derivatives, reference))
        row = dict(iteration=iteration, labels=y.tolist(),
                   endpoint_max_error=float((actual.detach()-expected.detach()).abs().max()),
                   gradient_relative_error=float((g-r).norm()/r.norm()),
                   gradient_cosine=float(torch.nn.functional.cosine_similarity(g, r, dim=0)))
        assert row['endpoint_max_error'] == 0, row
        assert row['gradient_relative_error'] < .005 and row['gradient_cosine'] > .9999, row
        print(row, flush=True)
        rows.append(row)
        with torch.no_grad():
            for parameter, derivative in zip(parameters, reference):
                parameter.add_(derivative, alpha=-1e-4/(float(r.norm())+1e-12))
        del expected, actual, reference, derivatives, g, r
        adapter.values.clear()
    assert all(p.grad is None for p in adapter.model.parameters())
    sources = {str(p.resolve()): sha(p) for p in Path(__file__).parent.glob('*.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(complete=True, model=args.model, batch=args.batch,
        checkpoint_backbone=args.checkpoint_backbone, gpu=gpu, torch=torch.__version__,
        head_provenance=provenance, sources=sources, checks=rows), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True)
    p.add_argument('--model', choices=('sit_small', 'jit', 'raev2'), required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--checkpoint-backbone', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
