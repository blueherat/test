"""Check serialization exactly without assuming that distinct NCCL reductions are bitwise equal."""
import copy
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from experiments.guidance_dynamic_50k_20260915 import config as k, training as train
from experiments.guidance_dynamic_50k_20260915.data import Stream
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.sampling import integrate
from experiments import train_imagenet100_sit_flow as base
from . import AMENDMENT


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def gpu(model, context, *, diagnostic_root=None):
    diagnostic = diagnostic_root is not None
    if not diagnostic:
        assert context.world_size == k.WORLD
    root = diagnostic_root or k.model_root(model) / f'checks_runtime{context.world_size}'
    root.mkdir(parents=True, exist_ok=True)
    details = dict(rank=context.rank, world_size=context.world_size, passed=False,
                   gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], diagnostic_only=diagnostic)
    def record():
        k.atomic(root / f'diagnostics_rank{context.rank}.json', details)
    def require(name, ok):
        details[name] = bool(ok)
        record()
        assert ok, name

    adapter = Adapter(model)
    initial_hash = fingerprint(adapter.model)
    base.configure_runtime(adapter.cfg['seed'], context.rank, True)
    stream = Stream(model, 'real', context)
    generator = torch.Generator(device='cuda').manual_seed(adapter.cfg['seed'] + 121 + context.rank)
    first = stream.draw(generator)
    if model == 'sit_small':
        native_loader, _ = base.create_loader(cache_dir=k.SIT_DATA, split='train',
            local_batch_size=k.GLOBAL_BATCH // context.world_size, context=context,
            seed=adapter.cfg['seed'], shuffle=True, num_workers=0, prefetch_factor=4, drop_last=True)
        moments, labels = next(iter(native_loader))
        require('original_full_data_loader', np.array_equal(labels.numpy(), first['true_labels'].cpu().numpy()))
        moments = moments.cuda()
        a = base.sample_sdvae_posterior(moments, torch.randn_like(first['positive']))
        b = base.sample_sdvae_posterior(moments, torch.randn_like(first['positive']))
        require('fresh_posterior_draw', not torch.equal(a, b))

    head = train.initialized_head(adapter)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=0., fused=True)
    local_gradients = {}
    handles = []
    for name, parameter in head.named_parameters():
        def capture(gradient, name=name):
            local_gradients[name] = gradient.detach().clone()
            return gradient
        handles.append(parameter.register_hook(capture))
    ddp = DDP(head, device_ids=[0], broadcast_buffers=False, find_unused_parameters=False,
              gradient_as_bucket_view=True, static_graph=True)
    state = train.State(root / 'resume_test', head, ema, opt, generator, context)
    require('disposable_checkpoint_fresh', state.step == 0)

    def backward(batch):
        local_gradients.clear()
        opt.zero_grad(set_to_none=True)
        loss = train.compute_loss(adapter, ddp, batch, k.ARMS['real'])
        loss.backward()
        require('finite_loss_and_gradients', bool(torch.isfinite(loss)) and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
        return loss.detach().clone()

    def update():
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), .9999)
            torch._foreach_add_(list(ema.parameters()), list(head.parameters()), alpha=.0001)

    backward(first)
    update()
    state.step = 1
    state.save()
    saved = train.load_torch(root / 'resume_test' / k.read(root / 'resume_test/latest_pointer.json')['file'])
    expected_batch = stream.draw(generator)
    expected_loss = backward(expected_batch)
    expected_local = {name: value.clone() for name, value in local_gradients.items()}
    expected_reduced = {name: p.grad.detach().clone() for name, p in head.named_parameters()}
    update()
    expected = dict(online=copy.deepcopy(head.state_dict()), ema=copy.deepcopy(ema.state_dict()),
                    optimizer=copy.deepcopy(opt.state_dict()))

    restored = train.State(root / 'resume_test', head, ema, opt, generator, context)
    require('checkpoint_state_exact', restored.step == 1 and exact(saved['online'], head.state_dict())
            and exact(saved['ema'], ema.state_dict()) and exact(saved['optimizer'], opt.state_dict()))
    require('rng_state_exact', exact(saved['rng'][context.rank]['data'], generator.get_state()) and
            exact(saved['rng'][context.rank]['runtime'], base.capture_rng_state(context.device)))
    repeated = Stream(model, 'real', context, start_step=1).draw(generator)
    require('training_batch_exact', exact(expected_batch, repeated))
    repeated_loss = backward(repeated)
    require('loss_exact', torch.equal(expected_loss, repeated_loss))
    require('local_gradients_exact', exact(expected_local, local_gradients))

    # Each summation can accumulate FP32 rounding. Bound the difference by the
    # absolute local contributions, not by the potentially cancelling mean.
    # This does not tolerate changed local gradients or changed checkpoint state.
    reduction = {}
    eps = torch.finfo(torch.float32).eps
    tiny = torch.finfo(torch.float32).tiny
    for name, p in head.named_parameters():
        mean_absolute = expected_local[name].abs().float()
        dist.all_reduce(mean_absolute)
        mean_absolute /= context.world_size
        bound = 8 * context.world_size * eps * mean_absolute + context.world_size * tiny
        difference = (p.grad - expected_reduced[name]).abs()
        reduction[name] = dict(bitwise_equal=torch.equal(p.grad, expected_reduced[name]),
            max_abs=float(difference.max()), max_bound=float(bound.max()),
            max_fraction_of_bound=float((difference / bound).max()),
            bounded=bool((difference <= bound).all()))
    details['reduction'] = reduction
    require('collective_rounding_bounded', all(row['bounded'] for row in reduction.values()))

    # Replay with the exact same reduced gradient to isolate optimizer recovery.
    # Only this disposable check copies a gradient; the training loop is unchanged.
    for name, p in head.named_parameters():
        p.grad.copy_(expected_reduced[name])
    update()
    require('optimizer_update_given_same_gradient_exact', exact(expected['online'], head.state_dict())
            and exact(expected['ema'], ema.state_dict()) and exact(expected['optimizer'], opt.state_dict()))
    flat = torch.cat([p.detach().flatten() for p in head.parameters()])
    reference = flat.clone()
    dist.broadcast(reference, 0)
    require('ddp_parameters_identical', torch.equal(flat, reference))
    for handle in handles:
        handle.remove()

    bank = k.model_root(model) / 'quality_inputs'
    n = adapter.cfg['sample_batch']
    noise = torch.from_numpy(np.array(np.load(bank / 'noise.npy', mmap_mode='r')[:n])).cuda()
    labels = torch.from_numpy(np.load(bank / 'labels.npy')[:n]).cuda()
    first_zero, counts = integrate(adapter, None, 'strong', 0., noise, labels)
    second_zero, other = integrate(adapter, head, 'real', 0., noise, labels)
    require('zero_guidance_exact', torch.equal(first_zero, second_zero) and counts == other and counts['head'] == 0)
    require('source_unchanged', fingerprint(adapter.model) == initial_hash and
            all(p.grad is None and not p.requires_grad for p in adapter.model.parameters()))
    details.update(passed=True, zero_counts=counts, request_sha256=k.sha(k.ROOT / 'request.json'),
        runtime_amendment_sha256=k.sha(AMENDMENT),
        note='Exact checkpoint/data/local-gradient/optimizer recovery; live collective replay uses a summation error bound.')
    record()
    k.atomic(root / f'rank{context.rank}.json', details)
    dist.barrier()
    if context.is_main:
        rows = [k.read(root / f'rank{rank}.json') for rank in range(context.world_size)]
        assert len({row['gpu_uuid'] for row in rows}) == context.world_size
        k.atomic(root / 'complete.json', dict(complete=True, passed=True, world_size=context.world_size,
            ranks=rows, diagnostic_only=diagnostic, request_sha256=k.sha(k.ROOT / 'request.json'),
            runtime_amendment_sha256=k.sha(AMENDMENT)))
    dist.barrier()
