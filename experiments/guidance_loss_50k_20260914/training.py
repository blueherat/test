"""50K training with exact RNG/optimizer recovery and clean-endpoint validation."""
import copy
import os
import time
import numpy as np
import torch
from torch.nn import functional as F

from . import config as k
from . import components as x


class State:
    def __init__(self, root, modules, optimizers, generator, dependencies):
        self.root, self.modules, self.optimizers = root, modules, optimizers
        self.generator, self.dependencies = generator, dependencies
        self.step, self.history, self.elapsed = 0, [], 0.
        root.mkdir(parents=True,exist_ok=True)
        pointer=root/'latest_pointer.json'
        if pointer.exists():
            reference=k.read(pointer)
            old = x.legacy.load_torch(root/reference['file'])
            assert old['step']==reference['step']
            assert old['dependencies']==dependencies
            for name,module in modules.items():module.load_state_dict(old['modules'][name])
            for name,opt in optimizers.items():opt.load_state_dict(old['optimizers'][name])
            generator.set_state(old['data_rng'])
            torch.set_rng_state(old['cpu_rng'])
            torch.cuda.set_rng_state(old['cuda_rng'])
            self.step,self.history,self.elapsed=old['step'],old['history'],old['seconds']
        self.started=time.perf_counter()

    def checkpoint(self):
        torch.cuda.synchronize()
        state=dict(step=self.step,modules={n:m.state_dict() for n,m in self.modules.items()},
            optimizers={n:o.state_dict() for n,o in self.optimizers.items()},
            data_rng=self.generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            history=self.history,seconds=self.elapsed+time.perf_counter()-self.started,dependencies=self.dependencies)
        pointer=self.root/'latest_pointer.json'
        slot=1-k.read(pointer)['slot'] if pointer.exists() else 0
        filename=f'recovery_{slot}.pt'
        # The old slot remains valid if the process dies before pointer publication.
        x.legacy.save_torch(self.root/filename,state)
        k.atomic(pointer,dict(file=filename,slot=slot,step=self.step))
        if self.step in (3000,10000,20000,30000,40000,50000):
            x.legacy.save_torch(self.root/'checkpoints'/f'step_{self.step:06d}.pt',state)

    def progress(self, metrics):
        value=dict(step=self.step,**metrics)
        self.history.append(value)
        k.atomic(self.root/'progress.json',dict(pid=os.getpid(),target_steps=k.STEPS,**value))
        print(value,flush=True)


def receipt_valid(root):
    if not (root/'complete.json').exists():
        return False
    row=k.read(root/'complete.json')
    assert row['complete'] and row['steps']==k.STEPS
    assert row['request_sha256']==k.sha(k.ROOT/'request.json')
    assert row['head_sha256']==k.sha(root/'head.pt')
    for path,digest in row['dependencies'].items():
        assert k.sha(path)==digest
    return True


def finish(state, final, before, rt, extra):
    assert state.step==k.STEPS
    state.checkpoint()
    assert x.legacy.tensor_fingerprint(rt.model)==before
    assert all(p.grad is None and not p.requires_grad for p in rt.model.parameters())
    x.legacy.save_torch(state.root/'head.pt',dict(step=k.STEPS,**final))
    k.atomic(state.root/'complete.json',dict(complete=True,steps=k.STEPS,head_sha256=k.sha(state.root/'head.pt'),
        request_sha256=k.sha(k.ROOT/'request.json'),history=state.history,dependencies=state.dependencies,
        training_seconds=state.elapsed+time.perf_counter()-state.started,strong_unchanged=True,
        strong_fingerprint=before,validation='last 2 of every 20 clean endpoints, fresh fixed noise; never used for head training',
        **extra))


def nuisance(source, fold):
    k.verify()
    assert k.read(k.ROOT/'gpu_preflight.json')['passed']
    root=k.ROOT/'nuisance'/source/f'fold{fold}'
    if receipt_valid(root):return
    rt=x.legacy.runtime()
    before=x.legacy.tensor_fingerprint(rt.model)
    mean=x.legacy.make_head(rt)
    eta=x.SourceHead(mean).cuda()
    mean_ema=copy.deepcopy(mean).eval().requires_grad_(False)
    eta_ema=copy.deepcopy(eta).eval().requires_grad_(False)
    opts=dict(mean=x.optimizer(mean),eta=x.optimizer(eta))
    bank=x.SourceBank(source)
    g=torch.Generator(device='cuda').manual_seed(k.SEED+100+fold)
    state=State(root,dict(mean=mean,eta=eta,mean_ema=mean_ema,eta_ema=eta_ema),opts,g,bank.provenance)

    def losses(batch, mean_model, eta_model):
        t=batch['t']; a=t[:,None,None,None]
        z=a*batch['clean']+(1-a)*batch['noise']
        features=x.local.features(rt,z,t,batch['labels'])
        target=x.local.patchify(rt,batch['clean']-batch['noise'])
        prediction=mean_model(features['context'],features['condition'])
        logit=eta_model(features['context'],features['condition'],t)
        return (prediction-target).square().mean(),F.binary_cross_entropy_with_logits(logit,batch['source'])

    try:
        for step in range(state.step+1,k.STEPS+1):
            k.check_stop()
            batch=bank.draw(g,train_excluding_fold=fold)
            assert torch.all(batch['fold'] != fold)
            mse,bce=losses(batch,mean,eta)
            assert torch.isfinite(mse+bce)
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            (mse+bce).backward()
            for module in (mean,eta):
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            for opt in opts.values():opt.step()
            x.ema_update(mean_ema,mean);x.ema_update(eta_ema,eta)
            state.step=step
            metrics=dict(mean_mse=float(mse.detach()),source_bce=float(bce.detach()))
            if step%5000==0:
                vg=torch.Generator(device='cuda').manual_seed(k.SEED+771)
                total=np.zeros(2)
                with torch.no_grad():
                    for _ in range(8):
                        mv,ev=losses(bank.draw(vg,validation=True),mean_ema,eta_ema)
                        total+=np.array([float(mv),float(ev)])/8
                metrics.update(validation_mean_mse=total[0],validation_source_bce=total[1])
            if step==1 or step%100==0:state.progress(metrics)
            if step%500==0:state.checkpoint()
    except k.RequestedStop:
        state.checkpoint()
        raise
    finish(state,dict(mean_ema=mean_ema.state_dict(),eta_ema=eta_ema.state_dict(),source=source,fold=fold),
           before,rt,dict(source=source,fold=fold,training_fold=1-fold,source_label_one='real',
               parameters=sum(p.numel() for p in mean.parameters())+sum(p.numel() for p in eta.parameters())))


def train(arm):
    request=k.verify()
    assert k.read(k.ROOT/'gpu_preflight.json')['passed']
    root=k.ROOT/'training'/arm
    if receipt_valid(root):return
    spec=k.ARMS[arm]
    rt=x.legacy.runtime()
    before=x.legacy.tensor_fingerprint(rt.model)
    head=x.legacy.make_head(rt)
    ema=copy.deepcopy(head).eval().requires_grad_(False)
    opt=x.optimizer(head)
    bank=x.SourceBank(spec['source'])
    dependencies=dict(bank.provenance)
    nuisance_models=None
    if spec['loss'] in ('contrast','contrast_weak','covariance'):
        nuisance_models,extra=x.load_nuisances(rt,spec['source'])
        dependencies.update(extra)
    weights=None
    if spec.get('weight'):
        path=k.ROOT/'weights'/f"{spec['weight']}.npy"
        receipt=k.read(k.ROOT/'weights/complete.json')
        assert receipt['files'][str(path)]==k.sha(path)
        weights=torch.from_numpy(np.load(path)).cuda()
        dependencies[str(path)]=k.sha(path)
    g=torch.Generator(device='cuda').manual_seed(k.SEED)
    state=State(root,dict(head=head,ema=ema),dict(head=opt),g,dependencies)

    def loss_for(batch, model):
        if spec['loss'] in ('fm','residual','guided_weak'):
            clean=batch['positive'] if spec['source']=='real' else batch['negative']
        else:
            clean=batch['clean']
        z,target=x.smoothing_pair(clean,batch['t'],batch['noise'],request['tau'],batch['coin']<spec.get('smooth',0.))
        # One frozen prefix for readout; baseline full calls are offline training work.
        features=x.local.features(rt,z,batch['t'],batch['labels'])
        target=x.local.patchify(rt,target)
        baseline=torch.zeros_like(target)
        eta=torch.full_like(batch['source'],.5)
        source=batch['source']
        if spec['loss']!='fm':
            if spec.get('baseline')=='mixture':
                eta,baseline=x.nuisance_prediction(nuisance_models,features,batch['t'],batch['fold'])
            else:
                baseline=x.local.patchify(rt,x.base_velocity(rt,z,batch['t'],batch['labels'],spec['base']))
                if nuisance_models is not None:
                    eta,_=x.nuisance_prediction(nuisance_models,features,batch['t'],batch['fold'])
            if spec['loss']=='null':source=batch['null_source']
        pred=model(features['context'],features['condition'])
        w=None if weights is None else weights[batch['labels'],batch['neg_id']]
        return x.objective(pred,target,source,eta,baseline,spec['loss'],w)

    try:
        for step in range(state.step+1,k.STEPS+1):
            k.check_stop()
            loss=loss_for(bank.draw(g),head)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            opt.step();x.ema_update(ema,head)
            state.step=step
            metrics=dict(loss=float(loss.detach()))
            if step%5000==0:
                vg=torch.Generator(device='cuda').manual_seed(k.SEED+772)
                with torch.no_grad():
                    metrics['validation_objective']=sum(float(loss_for(bank.draw(vg,validation=True),ema)) for _ in range(8))/8
            if step==1 or step%100==0:state.progress(metrics)
            if step%500==0:state.checkpoint()
    except k.RequestedStop:
        state.checkpoint()
        raise
    finish(state,dict(arm=arm,ema=ema.state_dict()),before,rt,
           dict(arm=arm,spec=spec,parameters=sum(p.numel() for p in head.parameters())))
