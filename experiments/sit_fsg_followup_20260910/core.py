"""Counted inference-only implementations of the twenty registered hypotheses."""
from __future__ import annotations
import contextlib
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from experiments import run_sit_fsg_20ideas_20260910 as catalog
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.lifting_scale_sweep_20260909 import array_sha

IDEAS={r['key']:r for r in catalog.IDEAS}
FAMILIES={f'f{r["id"]:02d}_{r["key"]}':r for r in catalog.IDEAS}
configurations=catalog.configurations
FIELD_KEYS={'layer_interaction','transported_memory','commitment_latch','spatial_condition_interaction'}
SPECIAL=set(IDEAS)|{'fsg_operator_control','plain_root_control'}
EPS=1e-10
norm,inner,projection,cap=old.norm,old.inner,old.projection,old.cap


def install():
    # This extension does not replace the original module's dispatch function.
    pass


def unit(x):return x/norm(x).clamp_min(EPS)
def energy(x):return x.flatten(1).square().mean(1)
def flat(x):return x.flatten(1)
def choose(mask,x,y):return torch.where(mask.reshape(-1,1,1,1),x,y)


def basis(first,second):
    q1=unit(first);q2=second-projection(second,q1);q2=unit(q2)
    return torch.stack((flat(q1),flat(q2)),-1)


def combine(q,coeff,like):return torch.bmm(q,coeff.unsqueeze(-1)).squeeze(-1).reshape_as(like)


def ridge_step(residual,jacobian):
    """Solve the actual local least-squares system in FP64, with a relative ridge."""
    with old.exact_matmul():
        j=jacobian.double();r=residual.double()
        gram=j.transpose(1,2)@j
        ridge=.001*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
        matrix=gram+ridge[:,None,None]*torch.eye(2,device=j.device,dtype=j.dtype)
        rhs=-(j.transpose(1,2)@r.unsqueeze(-1))
        answer=torch.linalg.solve(matrix,rhs).squeeze(-1)
    return answer.to(residual.dtype)


def patch_permute(x,order):
    b,c,h,w=x.shape;assert h==w==32 and order.shape==(b,64)
    values=x.reshape(b,c,8,4,8,4).permute(0,2,4,1,3,5).reshape(b,64,c*16)
    values=torch.gather(values,1,order[:,:,None].expand_as(values))
    return values.reshape(b,8,8,c,4,4).permute(0,3,1,4,2,5).reshape_as(x)


class Probe:
    def __init__(self,rt,z,t,ctx):
        self.rt,self.z,self.t,self.ctx=rt,z,float(t),ctx
        self.labels=rt.labels
        self.c,self.w=rt.pair(z,z.new_tensor(t))
        self.u=self.field(z,t,'u');self.d=self.c-self.u
        self.m=z+(1-t)*self.c;self.noise=z-t*self.c
        table=rt.model.y_embedder.embedding_table
        self.ec=table(self.labels);self.eu=table(torch.full_like(self.labels,100))
        self.eps=.001*norm(z).clamp_min(math.sqrt(z[0].numel()))

    def field(self,z,t,condition='c'):
        before=self.rt.labels
        embedding=None
        if isinstance(condition,torch.Tensor):
            if condition.dtype==torch.long:self.rt.labels=condition
            else:self.rt.labels=self.labels;embedding=condition
        else:self.rt.labels=self.labels if condition=='c' else torch.full_like(self.labels,100)
        try:
            with old.embedding_override(self.rt,embedding):
                return self.rt.field(z,z.new_tensor(t),'full')
        finally:self.rt.labels=before

    def mean(self,z,t,condition='c'):return z+(1-t)*self.field(z,t,condition)

    def flow(self,z,condition='c',fraction=.25):
        h=fraction*(1-self.t)/2
        first=self.field(z,self.t,condition)
        middle=z+h*first
        return middle+h*self.field(middle,self.t+h,condition)

    def endpoints(self,z,fraction=.25):
        return self.flow(z,'c',fraction),self.flow(z,'u',fraction)

    def gap(self,z,fraction=.25):
        c,u=self.endpoints(z,fraction);return c-u

    def divergence(self,z):
        # Gaussian Hutchinson probe, frozen across every candidate at this event.
        xi=self.ctx.xi;eps=.001*max(1-self.t,.1)
        diff=(self.field(z+eps*xi,self.t)-self.field(z-eps*xi,self.t))/(2*eps)
        return (diff*xi).flatten(1).mean(1)

    def layer_field(self,keep):
        handles=[];shared_time={}
        try:
            def capture_time(module,args,out):shared_time['value']=out
            handles.append(self.rt.model.t_embedder.register_forward_hook(capture_time))
            modules=list(self.rt.model.blocks)+[self.rt.model.final_layer]
            for index,block in enumerate(modules):
                assert keep in ('all','none','early','late')
                if keep=='all' or (keep!='none' and (index<4)==(keep=='early')):continue
                def replace(module,args):
                    assert len(args)==2
                    return args[0],shared_time['value']+self.eu
                handles.append(block.register_forward_pre_hook(replace))
            return self.field(self.z,self.t)
        finally:
            for handle in handles:handle.remove()


def objective(probe,key,theta):
    z,t=probe.z,probe.t
    if key=='neighborhood_root':
        shift=theta*(1-t)*probe.ctx.xi
        return lambda x:torch.cat([flat(probe.gap(x+s*shift))/math.sqrt(2) for s in (-1,1)],1)
    if key=='covariance_agreement':
        xi=probe.ctx.xi;eps=.002*max(1-t,.1)
        def quantities(x):
            def gap(y):return probe.mean(y,t)-probe.mean(y,t,'u')
            a=gap(x);j=(gap(x+eps*xi)-gap(x-eps*xi))/(2*eps)
            return a,j
        a,j=quantities(z)
        sa=flat(a).norm(dim=1,keepdim=True).clamp_min(1e-5)
        sj=flat(j).norm(dim=1,keepdim=True).clamp_min(1e-5)
        def value(x):
            a,j=quantities(x)
            return torch.cat((flat(a)/sa,math.sqrt(theta)*flat(j)/sj),1)
        return value
    if key in ('renoise_readability','conditional_idempotence'):
        s=t+theta*(1-t);target=probe.m
        def value(x):
            c=probe.field(x,t);m=x+(1-t)*c;n=x-t*c
            xi=probe.ctx.xi if key=='renoise_readability' else n
            view=s*m+(1-s)*xi
            recovered=probe.mean(view,s,'u' if key=='renoise_readability' else 'c')
            return flat(recovered-(target if key=='renoise_readability' else m))
        return value
    c0,u0=probe.endpoints(z);r0=c0-u0
    if key=='patch_bottleneck':
        e=F.avg_pool2d(r0.square().mean(1,keepdim=True),4)
        logits=theta*e/e.mean((2,3),keepdim=True).clamp_min(EPS)
        weight=torch.softmax(logits.flatten(1),dim=1).reshape_as(e)*e[0].numel()
        weight=F.interpolate(weight,scale_factor=4,mode='nearest').sqrt()
    if key=='condition_continuum':embedding=probe.eu+theta*(probe.ec-probe.eu)
    def value(x):
        c,u=probe.endpoints(x);r=c-u
        if key=='anchor_root':return torch.cat((flat(r),math.sqrt(theta)*flat(c-c0)),1)
        if key=='horizon_consensus':
            return torch.cat((math.sqrt(1-theta)*flat(probe.gap(x,.125)),math.sqrt(theta)*flat(r)),1)
        if key=='patch_bottleneck':return flat(r*weight)
        if key=='condition_continuum':return torch.cat((flat(r),flat(probe.flow(x,embedding)-u)),1)
        if key=='content_quotient':
            def content(v):
                v=v-v.mean((2,3),keepdim=True)
                v=v/v.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-6)
                return flat(F.adaptive_avg_pool2d(v,int(theta)))
            return content(c)-content(u)
        if key=='orbit_agreement':
            ue=probe.field(u,t+.25*(1-t),'u');par=projection(r,ue)
            return flat(r-par+math.sqrt(theta/(1+theta))*par)
        assert key in ('noise_shell_root','plain_root_control'),key
        return flat(r)
    return value


def covariance_matrices(probe,z,q):
    matrices=[];negative=z.new_zeros(())
    for cond in ('c','u'):
        mean=probe.mean(z,probe.t,cond);cols=[]
        for index in range(2):
            direction=q[:,:,index].reshape_as(z)
            changed=probe.mean(z+probe.eps*direction,probe.t,cond)
            cols.append(flat((changed-mean)/probe.eps))
        with old.exact_matmul():
            j=torch.stack(cols,-1).double();m=q.double().transpose(1,2)@j
            m=(m+m.transpose(1,2))/2
            vals,vec=torch.linalg.eigh(m)
            negative+=(vals<0).sum()
            matrices.append((vec*vals.clamp_min(0)[:,None,:])@vec.transpose(1,2))
    return matrices,negative


def calibrate(rt,z,t,config,ctx,amount):
    key,theta=config['key'],config['theta'];empty=dict(accepted=0.,before=0.,after=0.,shift=0.,negative=0.)
    if not amount or config.get('parameters',{}).get('calibration_multiplier',1.)==0:return z,empty
    if key in ('covariance_agreement','shared_ambiguity','information_volume') and t==0:return z,empty
    p=Probe(rt,z,t,ctx);radius=(4/64)*amount*norm(p.d)
    if key in ('noise_shell_root','plain_root_control'):radius=radius*theta
    q=basis(p.d,ctx.xi)
    if key=='noise_shell_root':
        q=basis(p.d-projection(p.d,p.noise),ctx.xi-projection(ctx.xi,p.noise))
    if key in ('fsg_operator_control','cycle_defect'):
        h=theta if key=='fsg_operator_control' else .125
        h=min(h,1-t-1e-5)
        moved=z+h*(p.c+amount*p.d)
        delta=moved-h*p.field(moved,t+h,'u')-z
        if key=='cycle_defect':
            same=z+h*p.u
            defect=same-h*p.field(same,t+h,'u')-z
            delta=delta-theta*defect
        candidate=z+cap(delta,radius)
        if key=='fsg_operator_control':
            return candidate,dict(accepted=float(len(z)),before=0.,after=0.,shift=float(norm(candidate-z).sum()),negative=0.)
        initial=energy(p.gap(z));ending=energy(p.gap(candidate));ok=ending<initial
    elif key=='endpoint_cotangent':
        end=p.flow(z,'u',theta);s=t+theta*(1-t)
        evidence=p.field(end,s)-p.field(end,s,'u')
        coefficients=[]
        for index in range(2):
            changed=p.flow(z+p.eps*q[:,:,index].reshape_as(z),'u',theta)
            coefficients.append(inner((changed-end)/p.eps,evidence).flatten())
        candidate=z+cap(combine(q,torch.stack(coefficients,1),z),radius)
        delta_end=p.flow(candidate,'u',theta)-end
        progress=inner(delta_end,evidence).flatten();ok=progress>0
        initial=torch.zeros_like(progress);ending=-progress
    elif key in ('shared_ambiguity','rival_margin','information_volume'):
        candidates=[z,z+radius*q[:,:,0].reshape_as(z),z-radius*q[:,:,0].reshape_as(z)]
        scores=[];negative=0.
        if key=='information_volume':
            mats,neg=covariance_matrices(p,z,q);negative+=float(neg)
            scale=sum(m.diagonal(dim1=-2,dim2=-1).mean(-1) for m in mats).mul(.5).clamp_min(1e-4)
            def score(matrices):
                ridge=theta*scale[:,None,None]*torch.eye(2,device=z.device,dtype=torch.float64)
                return (torch.linalg.slogdet(matrices[0]+ridge).logabsdet-
                        torch.linalg.slogdet(matrices[1]+ridge).logabsdet).float()
            scores.append(score(mats))
            for x in candidates[1:]:
                mats,neg=covariance_matrices(p,x,q);negative+=float(neg);scores.append(score(mats))
        else:
            r0=p.gap(z);scale=energy(r0).clamp_min(1e-6)
            if key=='shared_ambiguity':base_div=p.divergence(z)
            else:
                table=F.normalize(rt.model.y_embedder.embedding_table.weight[:100],dim=-1)
                sims=table[p.labels]@table.T;sims.scatter_(1,p.labels[:,None],-torch.inf)
                rival=sims.argmax(1)
            for x in candidates:
                c,u=p.endpoints(x);value=energy(c-u)/scale
                if key=='shared_ambiguity':value=value+theta*(p.divergence(x)-base_div)/base_div.abs().clamp_min(1.)
                else:value=value-theta*energy(u-p.flow(x,rival))/scale
                scores.append(value)
        scores=torch.stack(scores,1);index=scores.argmin(1)
        candidate=z.clone()
        for k in (1,2):candidate=choose(index==k,candidates[k],candidate)
        initial=scores[:,0];ending=scores.gather(1,index[:,None]).squeeze(1);ok=ending<initial
        empty['negative']=negative
    else:
        fn=objective(p,key,theta);r=fn(z);cols=[]
        for index in range(2):
            changed=fn(z+p.eps*q[:,:,index].reshape_as(z))
            cols.append((changed-r)/p.eps.flatten(1))
        coeff=ridge_step(r,torch.stack(cols,-1))
        delta=cap(combine(q,coeff,z),radius);candidate=z+delta
        if key=='noise_shell_root':
            n=p.noise+delta/max(1-t,1e-6)
            candidate=t*p.m+(1-t)*unit(n)*norm(p.noise)
        initial=energy(r);ending=energy(fn(candidate));ok=ending<initial
    assert torch.isfinite(initial).all() and torch.isfinite(ending).all(),key
    result=choose(ok,candidate,z)
    actual=torch.where(ok,ending,initial)
    assert (actual<=initial+1e-6*initial.abs().clamp_min(1)).all(),key
    return result,dict(accepted=float(ok.sum()),before=float(initial.sum()),after=float(actual.sum()),
        shift=float(norm(result-z).sum()),negative=empty['negative'])


def evaluate(rt,z,t,config,amount,ctx):
    if amount==0:return old.evaluate(rt,z,t,config,amount,ctx)
    key,theta=config['key'],config['theta']
    q=old.Queries(rt,z,t,key,ctx);d=q.s-q.null()[0];u=d
    info=dict(clean=q.m,gap=d)
    if key=='layer_interaction' and theta:
        p=Probe(rt,z,t,ctx)
        interaction=q.s-p.layer_field('early')-p.layer_field('late')+q.null()[0]
        u=d-theta*interaction
    elif key=='transported_memory' and theta:
        memory=ctx.history.get('class_memory',torch.zeros_like(z))
        u=d-theta*projection(d,memory)
    elif key=='commitment_latch':
        u=d*ctx.history.get('latch_active',torch.ones((len(z),1,1,1),device=z.device))
    elif key=='spatial_condition_interaction' and theta:
        permutation=ctx.block_permutation;inverse=torch.argsort(permutation,dim=1)
        moved=patch_permute(z,permutation)
        other=q.call(moved,kind='full')-q.call(moved,kind='full',labels=torch.full_like(rt.labels,100))
        u=d-theta*patch_permute(other,inverse)
    value=q.s+amount*u;off=(value-q.s)-projection(value-q.s,d)
    info['correction']=value-q.s
    info['diagnostics']=torch.stack(((norm(value-q.s)/(amount*norm(d)).clamp_min(EPS)).flatten(),
        (old.squared(off)/old.squared(value-q.s).clamp_min(EPS)).flatten(),
        (norm(z+(1-t)*value)/norm(q.m).clamp_min(EPS)).flatten(),
        z.new_zeros(len(z)),(norm(d).flatten()<=EPS).to(z.dtype)),-1)
    return value,info


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    seed=int(array_sha(noise.detach().cpu().numpy())[:15],16)
    if config['key'] not in SPECIAL or zero or not config['strength']:
        return old.sample(rt,noise,labels,config,batch_seed=seed,zero=zero)
    assert config['solver']=='heun64' and config['source']=='cfg'
    rt.labels=labels;z=noise.clone();begin=rt.counts.copy()
    ctx=old.StepContext(shadow=noise.clone());generator=torch.Generator(device=z.device).manual_seed(seed)
    diagnostics=torch.zeros((len(z),5),device=z.device,dtype=torch.float64);queries=0
    extra=dict(calibration_events=0,accepted_calibrations=0.,calibration_loss_before_sum=0.,
        calibration_loss_after_sum=0.,calibration_shift_norm_sum=0.,negative_covariance_eigenvalues=0.,
        calibration_full_calls=0)
    # Match the old RNG consumption so controls and mechanisms have the same
    # per-step Gaussian probes. Extra block permutations use that step's order.
    offsets=torch.randint(1,100,(len(z),4),device=z.device,generator=generator)
    ctx.alternate_labels=(labels[:,None]+offsets)%100
    for k in range(64):
        t=k/64;h=1/64;amount=old.amount_at(config,t)
        ctx.xi=torch.randn(z.shape,device=z.device,generator=generator,dtype=z.dtype)
        ctx.permutations=torch.argsort(torch.rand((len(z),256),device=z.device,generator=generator),dim=-1)
        heads=rt.model.blocks[0].attn.num_heads
        ctx.head_orders=torch.argsort(torch.rand((len(z),4,heads),device=z.device,generator=generator),dim=-1)
        ctx.block_permutation=torch.argsort(ctx.permutations[:,:64],dim=1)
        if amount and k%4==0:
            before=rt.counts['full']
            if config['key']=='commitment_latch':
                p=Probe(rt,z,t,ctx);residual=norm(p.gap(z))/(.25*(1-t)*norm(p.c)).clamp_min(EPS)
                low=ctx.history.get('latch_low',torch.zeros_like(residual))
                low=torch.where(residual<config['theta'],low+1,torch.zeros_like(low))
                active=ctx.history.get('latch_active',torch.ones_like(residual))
                active=torch.where(low>=2,torch.zeros_like(active),active)
                active=torch.where(residual>2*config['theta'],torch.ones_like(active),active)
                ctx.history.update(latch_low=low,latch_active=active)
            elif config['key'] not in FIELD_KEYS:
                z,rec=calibrate(rt,z,t,config,ctx,amount)
                extra['calibration_events']+=1
                for dest,src in [('accepted_calibrations','accepted'),('calibration_loss_before_sum','before'),
                    ('calibration_loss_after_sum','after'),('calibration_shift_norm_sum','shift'),
                    ('negative_covariance_eigenvalues','negative')]:extra[dest]+=rec[src]
            extra['calibration_full_calls']+=rt.counts['full']-before
        left=z
        first,info=evaluate(rt,z,t,config,amount,ctx)
        second,end=evaluate(rt,z+h*first,t+h,config,amount,ctx)
        z=z+(h/2)*(first+second)
        if amount:
            diagnostics.add_(info['diagnostics'].double()+end['diagnostics'].double());queries+=2
            if config['key']=='transported_memory':
                memory=ctx.history.get('class_memory',torch.zeros_like(z))
                if k>0:
                    before=rt.labels;rt.labels=torch.full_like(labels,100)
                    try:
                        eps=.001*norm(left).clamp_min(math.sqrt(left[0].numel()))
                        u=rt.field(left,left.new_tensor(t),'full')
                        up=rt.field(left+eps*unit(memory),left.new_tensor(t),'full')
                    finally:rt.labels=before
                    memory=memory+h*(up-u)/eps*norm(memory)
                memory=memory+(h/2)*(info['correction']+end['correction'])
                ctx.history['class_memory']=cap(memory,4*norm(z).clamp_min(1)).detach()
        old.commit_history(ctx,info)
        if not torch.isfinite(z).all() or z.abs().max()>1e6:
            raise FloatingPointError(f'{config["arm"]}: nonfinite or |state|>1e6 at step {k}')
    full=rt.counts['full']-begin['full'];prefix=rt.counts['prefix']-begin['prefix']
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=max(0,full-224),
        diagnostic_queries=queries,strang_active_steps=0,
        diagnostics=(diagnostics/max(queries,1)).cpu().numpy(),**extra)


@torch.inference_mode()
def limiting_checks(rt,noise,labels):
    rt.labels=labels;ctx=old.StepContext(xi=torch.ones_like(noise))
    original=rt.field(noise,noise.new_tensor(.25),'full')
    for row in catalog.IDEAS:
        cfg=next(c for c in configurations() if c['key']==row['key'])
        got,_=evaluate(rt,noise,.25,cfg,0.,ctx)
        assert torch.equal(got,original),row['key']
        changed=dict(cfg,parameters={'calibration_multiplier':0.})
        got,_=calibrate(rt,noise,.25,changed,ctx,1.)
        assert torch.equal(got,noise),row['key']
    for key in ('layer_interaction','transported_memory','spatial_condition_interaction'):
        cfg=next(c for c in configurations() if c['key']==key);cfg=dict(cfg,theta=0.)
        value,_=evaluate(rt,noise,.25,cfg,1.25,ctx)
        expected,_=old.evaluate(rt,noise,.25,dict(cfg,key='native_cfg'),1.25,ctx)
        assert torch.equal(value,expected),key
    assert rt.labels is labels
    return dict(zero_field_exact=20,zero_state_write_exact=20,zero_interaction_native_cfg_exact=3)


def cpu_checks():
    configs=configurations();assert len(configs)==256
    for row in catalog.IDEAS:
        assert (old.assets.WORK/'docs'/row['prior']).exists(),row['prior']
        assert len(row['values'])==2 and all(r in catalog.SOURCES for r in row['refs'])
    torch.manual_seed(202610079)
    x=torch.randn(3,4,32,32,dtype=torch.float64)
    order=torch.argsort(torch.rand(3,64),dim=1)
    assert torch.equal(patch_permute(patch_permute(x,order),torch.argsort(order,dim=1)),x)
    q=basis(x,torch.randn_like(x));gram=q.transpose(1,2)@q
    assert torch.allclose(gram,torch.eye(2,dtype=x.dtype)[None].expand_as(gram),atol=1e-12)
    j=torch.randn(3,20,2,dtype=torch.float64);r=torch.randn(3,20,dtype=torch.float64)
    delta=ridge_step(r,j);changed=r+(j@delta[:,:,None]).squeeze(-1)
    assert (energy(changed)<energy(r)).all()
    # Endpoint derivative transpose is not an inverse response operator.
    a=torch.tensor([[2.,1.],[0.,3.]],dtype=torch.float64);g=torch.tensor([1.,-2.],dtype=torch.float64)
    assert torch.equal(a.T@g,torch.tensor([2.,-5.],dtype=torch.float64))
    assert not torch.allclose(a.T@g,torch.linalg.solve(a,g))
    # Mean agreement alone leaves conditional covariance unconstrained.
    c=torch.diag(torch.tensor([1.,4.],dtype=torch.float64));u=torch.eye(2,dtype=torch.float64)
    assert not torch.allclose(c,u) and torch.linalg.slogdet(c).logabsdet>torch.linalg.slogdet(u).logabsdet
    # The two-factor interaction vanishes for additive early/late effects.
    a,b,d=torch.randn(3,2,3)
    assert torch.allclose((a+b+d)-(a+b)-(a+d)+a,torch.zeros_like(a),atol=1e-6)
    return dict(passed=True,ideas=20,candidates=200,controls=56,total=256,
        permutation_inverse=True,least_squares_descent=True,cotangent_is_not_inverse=True,
        equal_means_need_not_equal_covariances=True,additive_condition_interaction_zero=True)
