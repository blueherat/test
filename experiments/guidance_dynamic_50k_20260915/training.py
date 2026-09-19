import copy
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from . import config as k
from .data import Stream,loader
from .models import Adapter,SourceHead,fingerprint
from experiments import train_imagenet100_sit_flow as base
from experiments.weak_reference_loss_20260914.objectives import smoothing_pair,weighted_mse
from experiments.endpoint_contrast_20260914.objectives import contrast_loss,covariance_target


def distributed():
    # The existing audited runtimes address cuda:0. Give each rank its own GPU.
    local=int(os.environ.get('LOCAL_RANK','0'))
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible)==k.WORLD and len(set(visible))==k.WORLD
    os.environ['CUDA_VISIBLE_DEVICES']=visible[local]
    os.environ['LOCAL_RANK']='0'
    context=base.initialize_distributed('cuda')
    assert context.world_size==k.WORLD
    return context


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    k.atomic(path.with_suffix('.json'),dict(sha256=k.sha(path),request_sha256=k.sha(k.ROOT/'request.json')))


def load_torch(path):
    receipt=k.read(path.with_suffix('.json'))
    assert receipt['request_sha256']==k.sha(k.ROOT/'request.json') and receipt['sha256']==k.sha(path)
    return torch.load(path,map_location='cpu',weights_only=False)


def pair(adapter,batch,spec):
    if spec['loss'] in ('fm','residual','guided_weak'):
        clean=batch['positive'] if spec['source']=='real' else batch['negative']
    else:clean=batch['clean']
    smooth=batch['coin']<spec.get('smooth',0.)
    tau=0.
    if spec.get('smooth'):
        tau=k.read(k.model_root(adapter.name)/'endpoints/strong/train/complete.json')['tau']
    z,target=smoothing_pair(clean,batch['t'],batch['noise'],tau,smooth)
    if adapter.name=='jit':
        target=torch.where(smooth[:,None,None,None],z+(1-batch['t'])[:,None,None,None]*target,clean)
    return z,adapter.patch(target)


def initialized_head(adapter):
    head=adapter.make_head()
    state=load_torch(k.model_root(adapter.name)/'calibration/head.pt')
    head.load_state_dict(state['initial'])
    return head


class Nuisance(nn.Module):
    def __init__(self,head):
        super().__init__();self.mean=head;self.eta=SourceHead(head).cuda()
    def forward(self,tokens,condition,t):
        return self.mean(tokens,condition),self.eta(tokens,condition,t)


def load_nuisances(adapter,source):
    result=[]
    for fold in (0,1):
        root=k.model_root(adapter.name)/'nuisance'/source/f'fold{fold}'
        receipt=k.read(root/'complete.json');assert receipt['steps']==k.STEPS
        assert receipt['head_sha256']==k.sha(root/'head.pt')
        value=load_torch(root/'head.pt')
        net=Nuisance(initialized_head(adapter));net.load_state_dict(value['ema'])
        result.append(net.eval().requires_grad_(False))
    return result


def cross_prediction(models,features,t,folds):
    mean=torch.zeros((len(t),features['context'].shape[1],models[0].mean.output.out_features),device='cuda')
    eta=torch.empty(len(t),device='cuda')
    with torch.no_grad():
        for fold,net in enumerate(models):
            mask=folds==fold
            if mask.any():
                m,e=net(features['context'][mask],features['condition'][mask],t[mask])
                mean[mask]=m.float();eta[mask]=e.float().sigmoid()
    return eta,mean


def objective(pred,target,source,eta,baseline,kind,alpha,weights=None):
    target=target.detach();baseline=baseline.detach()
    if kind=='fm':return weighted_mse(pred,target,torch.ones(len(target),device='cuda') if weights is None else weights)
    if kind=='residual':return (pred-(target-baseline)).square().mean()
    if kind=='guided_weak':return (baseline+alpha*(baseline-pred)-target).square().mean()
    if kind in ('contrast','null','contrast_weak'):
        direction=alpha*(baseline-pred) if kind=='contrast_weak' else pred
        return contrast_loss(direction,target,source,eta,baseline)
    if kind=='covariance':return (pred-covariance_target(target,source,eta,baseline)).square().mean()
    raise ValueError(kind)


def compute_loss(adapter,net,batch,spec,nuisances=None,weights=None,nuisance=False):
    z,target=pair(adapter,batch,spec)
    with torch.no_grad(),adapter.autocast(training=True):
        features=adapter.features(z,batch['t'],batch['labels'])
        baseline=torch.zeros_like(target);eta=torch.full_like(batch['source'],.5)
        if not nuisance and spec['loss']!='fm':
            if spec.get('baseline')=='mixture':
                eta,baseline=cross_prediction(nuisances,features,batch['t'],batch['fold'])
            else:
                baseline=adapter.patch(adapter.base(z,batch['t'],batch['labels'],spec['base']))
                if nuisances is not None:eta,_=cross_prediction(nuisances,features,batch['t'],batch['fold'])
    den=(1-batch['t']).clamp_min(.05)[:,None,None] if adapter.name=='jit' else 1.
    with adapter.autocast(training=True):
        if nuisance:
            pred,eta_logits=net(features['context'].detach(),features['condition'].detach(),batch['t'])
            mean_loss=((pred.float()-target)/den).square().mean()
            source_loss=torch.nn.functional.binary_cross_entropy_with_logits(eta_logits.float(),batch['source'])
            return mean_loss+source_loss
        pred=net(features['context'].detach(),features['condition'].detach())
    source=batch['null_source'] if spec['loss']=='null' else batch['source']
    w=None if weights is None else torch.from_numpy(np.array(weights[batch['neg_id'].cpu().numpy()])).cuda()
    return objective(pred.float()/den,target/den,source,eta,baseline.float()/den,
        spec['loss'],adapter.cfg['alpha'],w)


def normalize(model,context):
    root=k.model_root(model)/'calibration';root.mkdir(parents=True,exist_ok=True)
    adapter=Adapter(model);seed=k.settings(model)['seed']
    torch.manual_seed(seed);head=adapter.make_head()
    before=fingerprint(adapter.model)
    stream=Stream(model,'real',context)
    generator=torch.Generator(device='cuda').manual_seed(seed+9901+context.rank)
    sums={name:torch.zeros(adapter.cfg['width'],device='cuda',dtype=torch.float64) for name in ('token','condition')}
    squares={name:value.clone() for name,value in sums.items()}
    counts={name:torch.zeros((),device='cuda',dtype=torch.float64) for name in sums}
    with torch.no_grad():
        for _ in range(32):
            batch=stream.draw(generator);z,_=pair(adapter,batch,dict(loss='fm',source='real'))
            with adapter.autocast(training=True):features=adapter.features(z,batch['t'],batch['labels'])
            for name,key in (('token','context'),('condition','condition')):
                values=features[key].reshape(-1,adapter.cfg['width']).double()
                sums[name]+=values.sum(0);squares[name]+=values.square().sum(0);counts[name]+=len(values)
        for name in sums:
            for value in (sums[name],squares[name],counts[name]):dist.all_reduce(value)
            mean=sums[name]/counts[name]
            std=(squares[name]/counts[name]-mean.square()).clamp_min(1e-8).sqrt()
            getattr(head,name+'_mean').copy_(mean.float());getattr(head,name+'_std').copy_(std.float())
    assert fingerprint(adapter.model)==before
    if context.is_main:
        save_torch(root/'head.pt',dict(initial=head.state_dict(),request_sha256=k.sha(k.ROOT/'request.json')))
        k.atomic(root/'complete.json',dict(complete=True,request_sha256=k.sha(k.ROOT/'request.json'),
            head_sha256=k.sha(root/'head.pt'),statistics_images=32*k.GLOBAL_BATCH,
            train_images=k.read(k.ROOT/'request.json')['real_data'][model]['train_images'],
            old_head_parameters_reused=False,strong_unchanged=True))
    dist.barrier()


class State:
    def __init__(self,root,net,ema,opt,generator,context):
        self.root,self.net,self.ema,self.opt,self.generator,self.context=root,net,ema,opt,generator,context
        root.mkdir(parents=True,exist_ok=True);self.step=0;self.history=[];self.seconds=0.;self.slot=0
        if (root/'latest_pointer.json').exists():
            pointer=k.read(root/'latest_pointer.json')
            state=load_torch(root/pointer['file'])
            assert state['request_sha256']==k.sha(k.ROOT/'request.json') and state['world']==context.world_size
            net.load_state_dict(state['online']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer'])
            self.step=state['step'];self.history=state['history'];self.seconds=state['seconds'];self.slot=pointer['slot']
            saved=state['rng'][context.rank]
            generator.set_state(saved['data']);base.restore_rng_state(saved['runtime'],context.device)
    def save(self):
        local=dict(data=self.generator.get_state(),runtime=base.capture_rng_state(self.context.device))
        rng=[None]*self.context.world_size;dist.all_gather_object(rng,local)
        if self.context.is_main:
            slot=1-self.slot;name=f'recovery{slot}.pt'
            value=dict(step=self.step,world=self.context.world_size,online=self.net.state_dict(),ema=self.ema.state_dict(),
                optimizer=self.opt.state_dict(),rng=rng,history=self.history,seconds=self.seconds,
                request_sha256=k.sha(k.ROOT/'request.json'))
            save_torch(self.root/name,value)
            k.atomic(self.root/'latest_pointer.json',dict(file=name,slot=slot,step=self.step))
            if self.step in (3000,10000,20000,30000,40000,50000):save_torch(self.root/f'step{self.step:06d}.pt',value)
        self.slot=1-self.slot
        dist.barrier()


def train(model,arm,context,source=None,fold=None):
    nuisance=source is not None
    spec=dict(loss='nuisance',source=source,base='strong') if nuisance else k.ARMS[arm]
    root=k.model_root(model)/('nuisance' if nuisance else 'training')/(source if nuisance else arm)
    if nuisance:root=root/f'fold{fold}'
    adapter=Adapter(model);seed=adapter.cfg['seed'];base.configure_runtime(seed,context.rank,True)
    torch.manual_seed(seed);head=initialized_head(adapter)
    net=Nuisance(head) if nuisance else head
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=0.,fused=True)
    wrapped=DDP(net,device_ids=[0],broadcast_buffers=False,find_unused_parameters=False,
        gradient_as_bucket_view=True,static_graph=True)
    generator=torch.Generator(device='cuda').manual_seed(seed+context.rank)
    state=State(root,net,ema,opt,generator,context)
    stream=Stream(model,spec['source'],context,state.step,fold if nuisance else None)
    nuisances=load_nuisances(adapter,spec['source']) if spec['loss'] in ('contrast','contrast_weak','covariance') else None
    weights=None
    if spec.get('weight'):weights=np.load(k.model_root(model)/'weights'/f'{spec["weight"]}_train.npy')
    before=fingerprint(adapter.model)
    total=torch.zeros((),device='cuda',dtype=torch.float64);acc=0;start=time.perf_counter()
    stop=torch.zeros((),device='cuda',dtype=torch.int32)
    for step in range(state.step+1,k.STEPS+1):
        stop.fill_(int((k.ROOT/'STOP_AFTER_CURRENT').exists()) if context.is_main else 0)
        dist.broadcast(stop,0)
        if stop.item():
            state.seconds+=time.perf_counter()-start;state.save();return
        batch=stream.draw(generator);opt.zero_grad(set_to_none=True)
        loss=compute_loss(adapter,wrapped,batch,spec,nuisances,weights,nuisance)
        finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError(f'Nonfinite loss {model} {arm or source} step {step}')
        loss.backward();opt.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()),.9999)
            torch._foreach_add_(list(ema.parameters()),list(net.parameters()),alpha=.0001)
        state.step=step;total+=loss.detach().double();acc+=1
        if step==1 or step%100==0:
            dist.all_reduce(total);elapsed=time.perf_counter()-start
            row=dict(step=step,loss=float(total/(context.world_size*acc)),global_batch=k.GLOBAL_BATCH,
                world_size=context.world_size,seconds_per_step=elapsed/acc)
            state.seconds+=elapsed;state.history.append(row)
            if context.is_main:
                k.atomic(root/'progress.json',dict(pid=os.getpid(),phase='training',model=model,arm=arm,
                    source=source,fold=fold,target_steps=k.STEPS,**row))
                print(model,arm or source,row,flush=True)
            total.zero_();acc=0;start=time.perf_counter()
        if step%5000==0:
            validation=Stream(model,spec['source'],context,validation=True)
            vg=torch.Generator(device='cuda').manual_seed(seed+7001+context.rank)
            vals=torch.zeros(2,device='cuda',dtype=torch.float64)
            vw=None if weights is None else np.load(k.model_root(model)/'weights'/f'{spec["weight"]}_validation.npy')
            with torch.no_grad():
                for _ in range(min(8,len(validation.loader))):
                    b=validation.draw(vg);value=compute_loss(adapter,ema,b,spec,nuisances,vw,nuisance)
                    vals+=torch.tensor([float(value)*len(b['t']),len(b['t'])],device='cuda')
            dist.all_reduce(vals)
            row=dict(step=step,validation_objective=float(vals[0]/vals[1]),samples=int(vals[1]))
            state.history.append(row)
            if context.is_main:k.atomic(root/f'validation{step:06d}.json',row)
            del validation
            start=time.perf_counter()
        if step%500==0 or step in (3000,k.STEPS):state.save();start=time.perf_counter()
    assert all(p.grad is None and not p.requires_grad for p in adapter.model.parameters())
    assert fingerprint(adapter.model)==before
    if context.is_main:
        save_torch(root/'head.pt',dict(step=k.STEPS,ema=ema.state_dict(),request_sha256=k.sha(k.ROOT/'request.json')))
        k.atomic(root/'complete.json',dict(complete=True,steps=k.STEPS,model=model,arm=arm,source=source,fold=fold,
            head_sha256=k.sha(root/'head.pt'),request_sha256=k.sha(k.ROOT/'request.json'),
            global_batch=k.GLOBAL_BATCH,world_size=context.world_size,history=state.history,
            parameters=sum(p.numel() for p in net.parameters()),strong_unchanged=True,
            full_dynamic_real_data=True,training_seconds=state.seconds))
    dist.barrier()
