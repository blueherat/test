"""Paired actual-model timing, endpoint and gradient checks on one idle GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def main(args):
    if (args.output/'result.json').exists():
        raise FileExistsError('Use a new benchmark output directory')
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
    gpu = next(r for r in gpu_snapshot() if str(r['index']) == args.gpu or r['uuid'] == args.gpu)
    lease = Path('/tmp', f"eqvae_idle_{gpu['uuid']}.lock").open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert eligible(next(r for r in gpu_snapshot() if r['uuid'] == gpu['uuid'])), gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    import gc
    import statistics
    import torch
    from .adapters import load, materialize_autocast_weights
    from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
    from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import recomputed_rollout
    from .sampler import for_adapter

    torch.manual_seed(2026091901)
    adapter, head, provenance = load(args.model)
    if args.model == 'sit_small':
        state = torch.load('/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/checkpoint_001456.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(state['head'])
        del state
    shape = adapter.cfg['shape']
    noise = torch.randn(args.batch, *shape, device='cuda')
    labels = torch.arange(args.batch, device='cuda')
    coefficient = 1.05 if args.model == 'sit_small' else .35 if args.model == 'raev2' else .3
    if args.model == 'raev2':
        from experiments.raev2_shallow_ig_20260914.core import predict
        class LegacySteps:
            def __len__(self): return 100
            def __call__(self, i, x):
                rt=adapter.rt; rt.labels=labels
                t,u=rt.grid[i],rt.grid[i+1]
                with rt.context():
                    full,_=rt.pair(x,t)
                    if .1<=float(t)<=1.:
                        clean=adapter.unpatch(predict('mlp',head,adapter.values)).float()
                        weak=rt.native.clean_to_velocity(clean,x,rt.times(x,t),denominator_floor=float(rt.cfg.transport.t_eps))
                        full=full+coefficient*(full-weak)
                    return x+(u-t)*full
        old_steps=LegacySteps()
    else:
        old_steps = GuidanceSteps(adapter, head, labels, coefficient, method='real')
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(model=args.model, batch=args.batch, gpu=gpu, torch=torch.__version__,
                  precision='original', coefficient=coefficient, steps=len(old_steps), repeats=args.repeats,
                  head_provenance=provenance)

    def publish(stage, **fields):
        result.update(fields)
        (args.output/'progress.json').write_text(json.dumps(dict(result,stage=stage),indent=2)+'\n')
        print(stage, fields, flush=True)

    def measured(run):
        rows=[]
        for repeat in range(args.repeats+1):
            adapter.values.clear(); gc.collect(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start=time.perf_counter()
            endpoint=run()
            # Fixed, smooth endpoint objective isolates the sampler derivative.
            gradient=torch.autograd.grad(endpoint.square().mean(), tuple(head.parameters()))
            torch.cuda.synchronize()
            rows.append(dict(seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                             reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
        return endpoint.detach().clone(),torch.cat([x.flatten() for x in gradient]).detach(),dict(
            iterations=rows,median_seconds=statistics.median(x['seconds'] for x in rows[1:]),
            peak_gib=max(x['peak_gib'] for x in rows[1:]),reserved_gib=max(x['reserved_gib'] for x in rows[1:]))

    publish('baseline_start')
    expected,reference,baseline=measured(lambda:recomputed_rollout(old_steps,len(old_steps),noise,head.parameters()))
    publish('baseline_complete',baseline=baseline)
    start=time.perf_counter()
    if args.precast:
        publish('frozen_weights_materialized',saved_weight_bytes=materialize_autocast_weights(adapter))
    engine=for_adapter(adapter,head,noise,labels,coefficient,graphs=not args.eager)
    torch.cuda.synchronize()
    publish('engine_ready',setup_seconds=time.perf_counter()-start)
    actual,grad,fast=measured(lambda:engine(noise,labels))
    relative=float((grad-reference).norm()/reference.norm())
    cosine=float(torch.nn.functional.cosine_similarity(grad,reference,dim=0))
    error=float((expected-actual).abs().max())
    publish('sampler_complete',optimized=fast,endpoint_max_error=error,gradient_relative_error=relative,
            gradient_cosine=cosine,speedup=baseline['median_seconds']/fast['median_seconds'])
    assert error < (1e-5 if args.model=='sit_small' else 1e-3),error
    assert relative < .005 and cosine > .9999,(relative,cosine)
    assert not any(p.grad is not None for p in adapter.model.parameters())
    (args.output/'result.json').write_text(json.dumps(dict(result,complete=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--model',choices=('sit_small','jit','raev2'),default='sit_small')
    p.add_argument('--batch',type=int,default=6)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--eager',action='store_true')
    p.add_argument('--precast',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
