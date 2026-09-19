"""Full binary-GAN update benchmark; run modes in separate fresh processes."""
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
    gpu=next(r for r in gpu_snapshot() if str(r['index'])==args.gpu or r['uuid']==args.gpu)
    lease=Path('/tmp',f"eqvae_idle_{gpu['uuid']}.lock").open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert eligible(next(r for r in gpu_snapshot() if r['uuid']==gpu['uuid']))
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    os.environ['TORCH_COMPILE_DISABLE']='1'
    import gc
    import statistics
    import numpy as np
    import torch
    from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    from .adapters import load,image_decoder,materialize_autocast_weights
    from .data import ImageNetData
    from .reference import sample as reference_sample
    from .sampler import for_adapter
    from .training import step
    from .features import enable_feedback_checkpointing,enable_backbone_checkpointing

    torch.manual_seed(2026091901)
    adapter,head,provenance=load(args.model)
    checkpoint=Path('/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/checkpoint_001456.pt')
    critic=BinaryCritic(classes=adapter.cfg['classes']).cuda()
    if args.model=='sit_small':
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        head.load_state_dict(state['head']);critic.load_state_dict(state['critic'])
    if args.precast:
        materialize_autocast_weights(adapter)
    feature=DifferentiableInception2048().cuda().eval().requires_grad_(False)
    decode=image_decoder(adapter)
    if args.checkpoint_feedback:
        enable_feedback_checkpointing(adapter,feature)
    if args.checkpoint_backbone:
        enable_backbone_checkpointing(adapter)
    if args.channels_last:
        adapter.rt.vae.to(memory_format=torch.channels_last)
        feature.to(memory_format=torch.channels_last)
    real,noise,labels=ImageNetData(args.model,2026091901).draw(args.batch)
    ow=torch.optim.Adam(head.parameters(),lr=1e-6,betas=(.9,.99))
    od=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.99))
    if args.model!='sit_small':
        cpu=lambda m:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        state=dict(head=cpu(head),critic=cpu(critic),optimizer_w=ow.state_dict(),optimizer_d=od.state_dict())
        checkpoint=None
    coefficient=1.05 if args.model=='sit_small' else .3 if args.model=='jit' else .35
    start=time.perf_counter()
    if args.mode=='baseline':
        sample=lambda z,y:reference_sample(adapter,head,z,y,coefficient)
    else:
        sample=for_adapter(adapter,head,noise,labels,coefficient,graphs=True)
    torch.cuda.synchronize()
    setup=time.perf_counter()-start
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for repeat in range(args.repeats+1):
        # Exact same saved D/W/Adam states for every paired update.
        head.load_state_dict(state['head']);critic.load_state_dict(state['critic'])
        import copy
        ow.load_state_dict(copy.deepcopy(state['optimizer_w']))
        od.load_state_dict(copy.deepcopy(state['optimizer_d']))
        ow.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True)
        adapter.values.clear();gc.collect();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        metrics=step(head=head,critic=critic,optimizer_w=ow,optimizer_d=od,sample=sample,
                     decode=decode,feature=feature,real=real,noise=noise,labels=labels,
                     feature_chunk=0 if args.mode=='baseline' else args.chunk)
        torch.cuda.synchronize()
        row=dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,metrics=metrics.tolist())
        rows.append(row)
        print(repeat,row,flush=True)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.save(args.output/'head_gradient.npy',torch.cat([p.grad.flatten() for p in head.parameters()]).cpu().numpy())
    np.save(args.output/'head_after.npy',torch.cat([p.detach().flatten() for p in head.parameters()]).cpu().numpy())
    np.save(args.output/'critic_after.npy',torch.cat([p.detach().flatten() for p in critic.parameters()]).cpu().numpy())
    assert not any(p.grad is not None for p in adapter.model.parameters())
    result=dict(complete=True,kind='paired_full_binary_gan_update',model=args.model,batch=args.batch,
                mode=args.mode,chunk=args.chunk,channels_last=args.channels_last,
                checkpoint_feedback=args.checkpoint_feedback,checkpoint_backbone=args.checkpoint_backbone,
                precast=args.precast,gpu=gpu,torch=torch.__version__,
                checkpoint=str(checkpoint) if checkpoint else None,head_provenance=provenance,
                setup_seconds=setup,iterations=rows,
                median_seconds=statistics.median(r['seconds'] for r in rows[1:]),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows[1:]),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows[1:]),
                excludes='one-time model/data loading, graph capture, checkpoint writes',
                unchanged='operator compute precision, batch, deployed step count/solver, loss, R1, alternating update, full input Jacobians')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--mode',choices=('baseline','optimized'),required=True)
    p.add_argument('--model',choices=('sit_small','jit','raev2'),default='sit_small')
    p.add_argument('--batch',type=int,default=6)
    p.add_argument('--chunk',type=int,default=1)
    p.add_argument('--channels-last',action='store_true')
    p.add_argument('--checkpoint-feedback',action='store_true')
    p.add_argument('--precast',action='store_true')
    p.add_argument('--checkpoint-backbone',action='store_true')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
