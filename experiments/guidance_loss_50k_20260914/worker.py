"""One bounded GPU/CPU task from the durable queue."""
import argparse
import json
import torch
from . import config as k
from . import components as x


@torch.no_grad()
def gpu_preflight():
    # Re-run shared features, original Context/native trajectories and zero guidance.
    x.legacy.gpu_preflight()
    from .sampling import Counter
    rt=x.legacy.runtime()
    capture=x.local.Capture(rt)
    head=x.legacy.make_head(rt).eval().requires_grad_(False)
    labels=torch.tensor([0,1],device='cuda')
    g=torch.Generator(device='cuda').manual_seed(k.SEED+88)
    noise=torch.randn((2,4,32,32),device='cuda',generator=g)
    original=x.legacy.make_head(rt,old=True).eval().requires_grad_(False)
    rows={}
    for arm in ('native','context','contrast_s','contrast_weak','covariance','ig_contrast','cfg_native','cfg_contrast'):
        selected=None if arm in ('native','cfg_native') else original if arm in ('context','contrast_weak') else head
        counter=Counter(rt,selected)
        value,counts=x.c.integrate(rt,noise,labels,lambda z,t,left,i,j:x.guided_field(rt,selected,capture,arm,z,t,left))
        expected=k.inference_counts(arm)
        assert counts==dict(full=expected['full'],prefix=0)
        assert counter.blocks==[expected['full']]*12
        assert counter.head==expected['head'] and counter.native_head==expected['native_head']
        counter.close()
        if arm in ('native','context','contrast_weak'):
            old_kind='native_base' if arm=='native' else 'context_base'
            reference,_=x.c.integrate(rt,noise,labels,lambda z,t,left,i,j:x.local.field(rt,{'context':original},capture,old_kind,z,t,left))
        elif arm=='ig_contrast':
            reference,_=x.c.integrate(rt,noise,labels,lambda z,t,left,i,j:x.guided_field(rt,None,capture,'native',z,t,left))
        elif arm in ('cfg_native','cfg_contrast'):
            def direct_cfg(z,t,left,i,j):
                strong=rt.field(z,t,'full')
                a=x.c.amount(rt,left,'cfg')
                if not a:return strong
                uncond=x.unconditional(rt,z,t)
                return strong+a*(strong-uncond)
            reference,_=x.c.integrate(rt,noise,labels,direct_cfg)
        else:
            reference,_=x.c.integrate(rt,noise,labels,lambda z,t,left,i,j:rt.field(z,t,'full'))
        assert torch.equal(value,reference),arm
        rows[arm]=dict(counts=expected,zero_or_legacy_trajectory_exact=True)
    # The CFG correction must read conditional features, before null overwrites the hook.
    t=noise.new_tensor(.3);rt.labels=labels
    strong=rt.field(noise,t,'full')
    retained=dict(capture.values)
    uncond=x.unconditional(rt,noise,t)
    expected=strong+1.25*(strong-uncond)+x.local.unpatchify(rt,original(retained['context'],retained['condition'])).float()
    actual=x.guided_field(rt,original,capture,'cfg_contrast',noise,t,.3)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    batch_t=noise.new_tensor([.18,.62])
    for kind in ('strong','ig','cfg'):
        base=x.base_velocity(rt,noise,batch_t,labels,kind)
        assert base.shape==noise.shape and torch.isfinite(base).all() and not base.requires_grad
        assert torch.equal(rt.labels,labels)
    # Real features and loss gradients for every objective; no nuisance/backbone gradients.
    with torch.enable_grad():
        train_head=x.legacy.make_head(rt)
        eta=x.SourceHead(train_head).cuda()
        features=x.local.features(rt,noise,t.expand(2),labels)
        pred=train_head(features['context'],features['condition'])
        target=x.local.patchify(rt,noise)
        probability=torch.sigmoid(eta(features['context'],features['condition'],t.expand(2)))
        for kind in ('fm','residual','guided_weak','contrast','contrast_weak','covariance','null'):
            train_head.zero_grad(set_to_none=True)
            loss=x.objective(pred,target,noise.new_tensor([0,1]),probability,torch.zeros_like(pred),kind)
            loss.backward(retain_graph=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in train_head.parameters())
            assert all(p.grad is None for p in eta.parameters())
        assert all(p.grad is None for p in rt.model.parameters())
    capture.close()
    receipt=k.read(k.ROOT/'gpu_preflight.json')
    receipt.update(new_objectives_and_cfg_features_checked=True,extended_sampling_checks=rows,
        additional_complete_check_paths=32,all_loss_gradients_finite=True,per_sample_training_times_checked=True)
    k.atomic(k.ROOT/'gpu_preflight.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['gpu_preflight','train','nuisance','endpoints','features','weights','sample','evaluate'])
    parser.add_argument('--arm',choices=k.SCREEN_ARMS)
    parser.add_argument('--source',choices=['weak',*k.NUISANCE_SOURCES])
    parser.add_argument('--fold',type=int,choices=[0,1])
    parser.add_argument('--stage',choices=['screen1000','confirm5000'],default='screen1000')
    args=parser.parse_args()
    if args.action=='gpu_preflight':gpu_preflight()
    elif args.action=='features':x.legacy.endpoint_features()
    elif args.action=='weights':
        from .weights import fit
        fit()
    elif args.action=='train':
        from .training import train
        train(args.arm)
    elif args.action=='nuisance':
        from .training import nuisance
        nuisance(args.source,args.fold)
    elif args.action=='endpoints':
        from .sampling import endpoints
        endpoints(args.source)
    elif args.action=='sample':
        from .sampling import sample
        sample(args.arm,args.stage)
    else:x.legacy.evaluate(args.arm,args.stage)


if __name__=='__main__':
    try:main()
    except k.RequestedStop as error:
        print(error,flush=True)
        raise SystemExit(75)
