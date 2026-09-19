"""Disposable reproduction of JiT's failed checkpoint preflight, outside frozen sources."""
import copy
import fcntl
import json
import os
from pathlib import Path
import time


def main():
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
    all_visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    visible = all_visible[local_rank]
    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    os.environ['LOCAL_RANK'] = '0'
    lease = Path('/tmp', f'eqvae_idle_{visible}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert eligible(next(row for row in gpu_snapshot() if row['uuid'] == visible))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from experiments.guidance_dynamic_50k_20260915 import config as k, training as train
    from experiments.guidance_dynamic_50k_20260915.data import Stream
    from experiments.guidance_dynamic_50k_20260915.models import Adapter
    from experiments import train_imagenet100_sit_flow as base
    root = k.ROOT / 'repair_20260915' / os.environ.get('DIAG_NAME', f'diagnostic_{time.time_ns()}')
    root.mkdir(parents=True, exist_ok=True)
    print('ROOT', root, flush=True)
    context = base.initialize_distributed('cuda')
    if not dist.is_initialized():
        dist.init_process_group('nccl', init_method='file://' + str(root / 'rendezvous'), rank=0, world_size=1)
    k.GLOBAL_BATCH = 64 * context.world_size  # Production per-rank batch, diagnostic world size.
    adapter = Adapter('jit')
    stream = Stream('jit', 'real', context)
    g = torch.Generator(device='cuda').manual_seed(adapter.cfg['seed'] + 121 + context.rank)
    first = stream.draw(g)
    head = train.initialized_head(adapter)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0., betas=(.9, .999), fused=True)
    ddp = DDP(head, device_ids=[0], broadcast_buffers=False, static_graph=True)
    state = train.State(root / 'checkpoint', head, ema, opt, g, context)
    snapshots = {}
    phase = ['first']
    original = adapter.features
    def features(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots[phase[0] + '_features'] = {name: value.detach().clone() for name, value in result.items()}
        return result
    adapter.features = features
    def update(batch):
        opt.zero_grad(set_to_none=True)
        loss = train.compute_loss(adapter, ddp, batch, k.ARMS['real'])
        loss.backward()
        snapshots[phase[0] + '_grads'] = {name: p.grad.detach().clone() for name, p in head.named_parameters()}
        snapshots[phase[0] + '_loss'] = float(loss.detach())
        opt.step()
        snapshots[phase[0] + '_weights'] = {name: p.detach().clone() for name, p in head.state_dict().items()}
    warmup = int(os.environ.get('DIAG_WARMUP', '1'))
    update(first)
    for _ in range(1, warmup):
        update(stream.draw(g))
    state.step = warmup
    state.save()
    phase[0] = 'expected'
    expected_batch = stream.draw(g)
    update(expected_batch)
    train.State(root / 'checkpoint', head, ema, opt, g, context)
    repeated = Stream('jit', 'real', context, start_step=warmup).draw(g)
    assert all(torch.equal(expected_batch[key], repeated[key]) for key in expected_batch)
    phase[0] = 'restored'
    update(repeated)
    def compare(left, right):
        return {name: dict(equal=torch.equal(value, right[name]),
                    max_abs=float((value.float() - right[name].float()).abs().max()),
                    unequal=int((value != right[name]).sum())) for name, value in left.items()}
    result = dict(batch_exact=True, losses=[snapshots[p + '_loss'] for p in ('expected', 'restored')],
        **{kind: compare(snapshots['expected_' + kind], snapshots['restored_' + kind])
           for kind in ('features', 'grads', 'weights')})
    result.update(rank=context.rank, world_size=context.world_size, warmup=warmup,
                  ddp=ddp._get_ddp_logging_data())
    k.atomic(root / f'result_rank{context.rank}.json', result)
    print(json.dumps(result, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
