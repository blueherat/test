"""Concrete inference operators for the frozen research catalog.

All histories and stochastic queries are fixed during a Heun step. No model
weights are changed. Additional full/prefix calls use the counted runtime API.
"""
from __future__ import annotations
import contextlib
import math
from dataclasses import dataclass,field
import torch
from torch.nn import functional as F
from experiments import small_sit_predictable_gap_20260909 as residual
from . import assets

EPS=1e-12
PATCH=residual.cache_source.patchify
UNPATCH=residual.cache_source.unpatchify


def squared(x):return x.square().flatten(1).sum(1).reshape(-1,1,1,1)
def norm(x):return squared(x).sqrt()
def inner(x,y):return (x*y).flatten(1).sum(1).reshape(-1,1,1,1)
def projection(x,onto):return onto*(inner(x,onto)/squared(onto).clamp_min(EPS))
def cap(x,radius):return x*(radius/norm(x).clamp_min(EPS)).clamp_max(1.)
def match(x,reference):
    return torch.where(norm(x)>EPS,x*(norm(reference)/norm(x).clamp_min(EPS)),reference)


@contextlib.contextmanager
def exact_matmul():
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def patch_matrix(x,matrix):
    with exact_matmul():return UNPATCH(PATCH(x)@matrix.T)


def haar_shrink(x,amount):
    a,b,c,d=x[:,:,0::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,0::2],x[:,:,1::2,1::2]
    ll,lh,hl,hh=(a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2
    details=torch.stack((lh,hl,hh),1)
    threshold=amount*details.abs().flatten(1).median(1).values.reshape(-1,1,1,1,1)
    lh,hl,hh=(details.sign()*(details.abs()-threshold).clamp_min(0)).unbind(1)
    out=torch.empty_like(x)
    out[:,:,0::2,0::2]=(ll+lh+hl+hh)/2
    out[:,:,0::2,1::2]=(ll-lh+hl-hh)/2
    out[:,:,1::2,0::2]=(ll+lh-hl-hh)/2
    out[:,:,1::2,1::2]=(ll-lh-hl+hh)/2
    return out


def gradient(x):return torch.roll(x,-1,-1)-x,torch.roll(x,-1,-2)-x
def divergence(px,py):return px-torch.roll(px,1,-1)+py-torch.roll(py,1,-2)


def tv_prox(x,amount,iterations=12):
    if amount==0:return x
    gx,gy=gradient(x)
    scale=(gx.square()+gy.square()).flatten(1).mean(1).sqrt().reshape(-1,1,1,1)
    penalty=(amount*scale).clamp_min(EPS)
    px,py=torch.zeros_like(x),torch.zeros_like(x)
    for _ in range(iterations):
        u=x+penalty*divergence(px,py)
        gx,gy=gradient(u)
        px,py=px+.125*gx/penalty,py+.125*gy/penalty
        radius=(px.square()+py.square()).sqrt().clamp_min(1)
        px,py=px/radius,py/radius
    return x+penalty*divergence(px,py)


def edge_resolvent(d,m,amount,iterations=8):
    dx,dy=gradient(m)
    scale=(dx.square()+dy.square()).sum(1,keepdim=True).mean((2,3),keepdim=True).clamp_min(EPS)
    right=torch.exp(-dx.square().sum(1,keepdim=True)/scale)
    down=torch.exp(-dy.square().sum(1,keepdim=True)/scale)
    left,up=torch.roll(right,1,-1),torch.roll(down,1,-2)
    total=right+left+down+up
    u=d
    for _ in range(iterations):
        neighbors=right*torch.roll(u,-1,-1)+left*torch.roll(u,1,-1)+down*torch.roll(u,-1,-2)+up*torch.roll(u,1,-2)
        u=(d+amount*neighbors)/(1+amount*total)
    return u


def channel_metric(d,m,ridge):
    flat=m.flatten(2);flat=flat-flat.mean(2,keepdim=True)
    with exact_matmul():
        cov=flat@flat.transpose(1,2)/flat.shape[-1]
        mean=cov.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(EPS)
        vals,vec=torch.linalg.eigh(cov+ridge*mean[:,None,None]*torch.eye(4,device=m.device,dtype=m.dtype))
        vals=vals.clamp_min(EPS)
        root=(vec*vals.sqrt()[:,None,:])@vec.transpose(1,2)
        invroot=(vec*vals.rsqrt()[:,None,:])@vec.transpose(1,2)
        whitened=invroot@d.flatten(2)
        lengths=whitened.square().sum(1,keepdim=True).sqrt()
        budget=2*lengths.square().mean(2,keepdim=True).sqrt()
        whitened=whitened*(budget/lengths.clamp_min(EPS)).clamp_max(1.)
        return (root@whitened).reshape_as(d)


def lowpass(x):
    axis=torch.fft.fftfreq(x.shape[-1],device=x.device)*x.shape[-1]
    mask=(axis[:,None].square()+axis[None,:].square())<=16
    return torch.fft.ifft2(torch.fft.fft2(x,norm='ortho')*mask,norm='ortho').real


def huber_consensus(values,radius,iterations=8):
    points=torch.stack(values,1)
    center=points.mean(1)
    scale=norm(values[0]).reshape(-1,1,1,1,1)*radius
    for _ in range(iterations):
        distance=(points-center[:,None]).flatten(2).norm(dim=2).reshape(len(center),len(values),1,1,1)
        weights=distance.clamp_min(scale).clamp_min(EPS).reciprocal()
        center=(points*weights).sum(1)/weights.sum(1)
    return center


@dataclass
class StepContext:
    history:dict=field(default_factory=dict)
    shadow:torch.Tensor|None=None
    xi:torch.Tensor|None=None
    permutations:torch.Tensor|None=None
    head_orders:torch.Tensor|None=None
    alternate_labels:torch.Tensor|None=None
    h:float=1/64


@contextlib.contextmanager
def embedding_override(rt,value):
    if value is None:yield;return
    handle=rt.model.y_embedder.register_forward_hook(lambda module,args,output:value)
    try:yield
    finally:handle.remove()


@contextlib.contextmanager
def perturb_prefix(rt,key,amount,ctx):
    handles=[];originals=[]
    if amount==0 and key!='ig_entropy_reference':yield;return
    try:
        for layer,block in enumerate(rt.model.blocks[:4]):
            attn=block.attn
            if key=='ig_mlp_contraction':
                handles.append(block.mlp.register_forward_hook(lambda module,args,out:(1-amount)*out))
            elif key=='ig_soft_pag':
                captured={}
                def capture(module,args,out,saved=captured,heads=attn.num_heads):
                    b,n,c3=out.shape
                    saved['v']=out.reshape(b,n,3,heads,c3//(3*heads))[:,:,2].reshape(b,n,c3//3)
                handles.append(attn.qkv.register_forward_hook(capture))
                def mix(module,args,out,saved=captured):
                    identity=module.proj_drop(module.proj(saved['v']))
                    return (1-amount)*out+amount*identity
                handles.append(attn.register_forward_hook(mix))
            elif key=='ig_local_attention':
                originals.append((attn,attn.forward))
                def localized(x,module=attn):
                    b,n,c=x.shape
                    q,k,v=module.qkv(x).reshape(b,n,3,module.num_heads,c//module.num_heads).permute(2,0,3,1,4).unbind(0)
                    q,k=module.q_norm(q),module.k_norm(k)
                    side=round(n**.5);assert side*side==n
                    axis=torch.arange(side,device=x.device,dtype=x.dtype)/(side-1)
                    yy,xx=torch.meshgrid(axis,axis,indexing='ij');positions=torch.stack((yy.flatten(),xx.flatten()),-1)
                    distance=(positions[:,None]-positions[None,:]).square().sum(-1)
                    value=F.scaled_dot_product_attention(q,k,v,attn_mask=-amount*distance,dropout_p=0.,scale=module.scale)
                    value=value.transpose(1,2).reshape(b,n,c)
                    return module.proj_drop(module.proj(value))
                attn.forward=localized
            else:
                def modify(module,args,out,index=layer,heads=attn.num_heads):
                    b,n,c3=out.shape
                    values=out.reshape(b,n,3,heads,c3//(3*heads)).clone()
                    if key=='ig_entropy_reference':values[:,:,0]*=amount
                    elif key=='ig_query_contrast':
                        q=values[:,:,0];values[:,:,0]=q.mean(1,keepdim=True)+(1-amount)*(q-q.mean(1,keepdim=True))
                    elif key=='ig_value_contrast':
                        v=values[:,:,2];values[:,:,2]=v.mean(1,keepdim=True)+(1-amount)*(v-v.mean(1,keepdim=True))
                    elif key=='ig_attention_head_dropout':
                        mask=torch.ones((b,heads),device=out.device,dtype=out.dtype)
                        mask.scatter_(1,ctx.head_orders[:,index,:int(amount)],0.)
                        values[:,:,2]*=mask[:,None,:,None]
                    elif key=='ig_value_permutation':
                        v=values[:,:,2]
                        index_tensor=ctx.permutations[:,:,None,None].expand_as(v)
                        values[:,:,2]=(1-amount)*v+amount*torch.gather(v,1,index_tensor)
                    else:raise KeyError(key)
                    return values.reshape_as(out)
                handles.append(attn.qkv.register_forward_hook(modify))
        yield
    finally:
        for handle in handles:handle.remove()
        for module,original in originals:module.forward=original


class Queries:
    def __init__(self,rt,z,time_value,key,ctx):
        self.rt,self.z,self.tv,self.key,self.ctx=rt,z,float(time_value),key,ctx
        self.t=z.new_tensor(time_value);self.b=1-self.t
        self.labels=rt.labels
        if key=='ig_robust_head_consensus':
            rt.counts['full']+=1
            self.s,self.multi,_=rt.small.evaluate_source_with_heads(rt.model,z,rt.times(z,self.t),rt.labels,
                heads=rt.portfolio_heads,source_semantics=rt.semantics)
            self.w=self.multi['depth4_v']
        else:self.s,self.w=rt.pair(z,self.t)
        self.h=rt.capture_weak.value.detach()
        self.d=self.s-self.w;self.m=z+self.b*self.s
        self.unconditional=None;self._c=None;self._x=None

    def call(self,z=None,time_value=None,kind='pair',labels=None,embedding=None,perturb=None):
        z=self.z if z is None else z
        t=self.t if time_value is None else z.new_tensor(time_value)
        before=self.rt.labels
        self.rt.labels=self.labels if labels is None else labels
        perturbation=perturb_prefix(self.rt,*perturb,self.ctx) if perturb is not None else contextlib.nullcontext()
        try:
            with embedding_override(self.rt,embedding),perturbation:
                if kind=='pair':return self.rt.pair(z,t)
                return self.rt.field(z,t,'base' if kind=='weak' else 'full')
        finally:self.rt.labels=before

    def null(self):
        if self.unconditional is None:self.unconditional=self.call(labels=torch.full_like(self.labels,100))
        return self.unconditional

    def gap(self,z):
        s,w=self.call(z);return s-w

    def design(self):
        if self._x is None:
            state=self.rt.portfolio_assets
            self._x=(self.h-state['mean'])/state['std']
        return self._x

    def predict(self,coefficient):
        with exact_matmul():return UNPATCH(self.design()@coefficient[:-1]+coefficient[-1])

    def predictable(self):
        if self._c is None:self._c=self.predict(self.rt.portfolio_assets['original_coefficient'])
        return self._c

    def class_embeddings(self):
        table=self.rt.model.y_embedder.embedding_table
        return table(self.labels),table(torch.full_like(self.labels,100))

    def response_basis(self):
        first=self.d/norm(self.d).clamp_min(EPS)
        second=self.predictable()-projection(self.predictable(),first)
        second=second/norm(second).clamp_min(EPS)
        basis=torch.stack((first.flatten(1),second.flatten(1)),-1)
        epsilon=.001*norm(self.z).clamp_min(1.)
        responses=[]
        for direction in (first,second):
            z=self.z+epsilon*direction
            s=self.call(z,kind='full')
            responses.append(((z+self.b*s-self.m)/epsilon).flatten(1))
        return basis,torch.stack(responses,-1)


def evaluate(rt,z,time_value,config,amount,ctx):
    if amount==0:
        return rt.field(z,z.new_tensor(time_value),'full'),dict(diagnostics=z.new_zeros((len(z),5)))
    key=config['key'];r=config['theta'];q=Queries(rt,z,time_value,key,ctx)
    state=rt.portfolio_assets
    d=q.s-q.null()[0] if config['source']=='cfg' and key!='cfg_alt_class_reference' else q.d
    info=dict(gap=d,clean=q.m)
    constraint=z.new_zeros((len(z),))
    if key in ('native_ig','native_cfg'):u=d
    elif key=='cfg_apg_momentum':
        momentum=d+r*ctx.history.get('momentum',torch.zeros_like(d))
        info['momentum']=momentum
        bounded=cap(momentum,2*norm(d));u=bounded-projection(bounded,q.m)
    elif key=='cfg_zero_ridge':
        weak=q.null()[0]
        ratio=(inner(q.s,weak)+r*squared(weak))/((1+r)*squared(weak)).clamp_min(EPS)
        u=q.s-ratio*weak
    elif key=='cfg_channel_rescale':
        guided=q.m+q.b*amount*d
        mean=guided.mean((2,3),keepdim=True)
        sd=(q.m-q.m.mean((2,3),keepdim=True)).square().mean((2,3),keepdim=True).sqrt()
        gd=(guided-mean).square().mean((2,3),keepdim=True).sqrt()
        calibrated=mean+(guided-mean)*(sd/gd.clamp_min(EPS))
        value=(1-r)*guided+r*calibrated
        u=(value-q.m)/(q.b*amount)
    elif key=='cfg_patch_budget':
        p=PATCH(d)
        local=p.square().mean(-1,keepdim=True).sqrt()
        budget=r*d.square().flatten(1).mean(1).sqrt()[:,None,None]
        u=UNPATCH(p*(budget/local.clamp_min(EPS)).clamp_max(1))
    elif key=='ig_channel_metric':u=channel_metric(d,q.m,r)
    elif key=='ig_wavelet_shrink':u=haar_shrink(d,r)
    elif key=='ig_huber_innovation':
        innovation=d-q.predictable()
        scale=r*innovation.square().flatten(1).mean(1).sqrt().reshape(-1,1,1,1)
        u=match(innovation/torch.sqrt(1+(innovation/scale.clamp_min(EPS)).square()),d)
    elif key=='ig_edge_resolvent':u=edge_resolvent(d,q.m,r)
    elif key=='ig_partial_residual':u=match(d-r*q.predictable(),d)
    elif key=='ig_predictable_subspace':u=match(d-r*projection(d,q.predictable()),d)
    elif key=='ig_residual_whitening':u=match(patch_matrix(d-q.predictable(),state['whitening'][str(r)]),d)
    elif key=='ig_explained_eigenfilter':u=match(patch_matrix(d,state['eigenfilter'][str(r)]),d)
    elif key=='ig_leverage_residual':
        x=q.design();inv=state['inverse_gram']
        with exact_matmul():ell=((x@inv[:-1,:-1])*x).sum(-1)+2*x@inv[:-1,-1]+inv[-1,-1]
        weight=1/(1+ell.clamp_min(0)/(r*state['leverage_median']))
        u=match(d-UNPATCH(PATCH(q.predictable())*weight[:,:,None]),d)
    elif key=='ig_target_error_readout':u=d+r*cap(q.predict(state['target_coefficient']),norm(d))
    elif key=='ig_time_residual':
        bins=int(r);index=min(bins-1,int(q.tv*2*bins))
        u=match(d-q.predict(state['time_models'][str(bins)][index]),d)
    elif key=='ig_gap_ema':
        u=(1-r)*d+r*ctx.history.get('ema',d);info['ema']=u
    elif key=='ig_gap_linear_trend':
        previous=ctx.history.get('level',d);trend=ctx.history.get('trend',torch.zeros_like(d))
        level=(1-r)*d+r*(previous+trend)
        trend=(1-r)*(level-previous)+r*trend
        info.update(level=level,trend=trend);u=cap(level+trend,2*norm(d))
    elif key=='cfg_higs':
        guided=q.m+q.b*amount*d
        previous=ctx.history.get('guided_ema',guided)
        u=d+r*cap((guided-previous)/q.b,norm(d))
        info['guided_ema']=.8*previous+.2*guided
    elif key=='ig_clean_curvature':
        previous=ctx.history.get('clean',q.m);older=ctx.history.get('clean_previous',2*previous-q.m)
        u=d+r*cap((q.m-2*previous+older)/q.b,norm(d))
    elif key in ('ig_leaky_shadow','ig_shadow_transport'):
        y=z if ctx.shadow is None else ctx.shadow
        sy,wy=q.call(y);dy=sy-wy
        tether=r if key=='ig_leaky_shadow' else 4.
        info['shadow_velocity']=sy+tether*(z-y)
        if key=='ig_shadow_transport':dy=dy+r*(q.gap(y+.1*(z-y))-dy)/.1
        u=match(dy,d)
    elif key=='ig_implicit_gap':
        u=d
        for _ in range(2):u=q.gap(z+r*ctx.h*u)
        u=cap(u,2*norm(d))
    elif key=='ig_strang_split':raise RuntimeError('Strang has a dedicated finite-step implementation')
    elif key=='ig_flip_average':
        response=torch.flip(q.gap(torch.flip(z,(-1,))),(-1,))
        u=(1-r)*d+r*(d+response)/2
    elif key=='ig_patch_phase_average':
        shift=int(r)
        plus=torch.roll(q.gap(torch.roll(z,shift,-1)),-shift,-1)
        minus=torch.roll(q.gap(torch.roll(z,-shift,-1)),shift,-1)
        u=(d+plus+minus)/3
    elif key=='ig_antithetic_query':u=(q.gap(z+r*q.b*ctx.xi)+q.gap(z-r*q.b*ctx.xi))/2
    elif key=='ig_noise_rotation':
        predicted_noise=z-q.t*q.s
        center=q.t*q.m+q.b*math.sqrt(1-r*r)*predicted_noise
        u=(q.gap(center+q.b*r*ctx.xi)+q.gap(center-q.b*r*ctx.xi))/2
    elif key=='ig_cross_band_response':
        changed=q.gap(z-r*q.t*lowpass(q.m))
        response=(d-changed)/r;response=response-lowpass(response)
        u=d+.1*cap(response,norm(d))
    elif key=='ig_tv_resolvent':u=tv_prox(d,r)
    elif key=='ig_commutator':
        epsilon=.001
        first=(q.gap(z+epsilon*q.s)-d)/epsilon
        second=(q.call(z+epsilon*d,kind='full')-q.s)/epsilon
        bracket=cap(first-second,norm(d)/ctx.h)
        u=d+r*ctx.h*bracket/2
    elif key=='ig_condition_interaction':
        su,wu=q.null();u=match(d-r*(su-wu),d)
    elif key=='ig_remove_condition_overlap':
        dc=q.s-q.null()[0];u=match(d-r*projection(d,dc),d)
    elif key=='cfg_remove_capacity_overlap':u=match(d-r*projection(d,q.d),d)
    elif key=='ig_condition_guard':
        dc=q.s-q.null()[0]
        deficit=(r*norm(d)*norm(dc)-inner(d,dc)).clamp_min(0)
        constraint=(deficit.flatten()>0).to(z.dtype)
        u=d+deficit/squared(dc).clamp_min(EPS)*dc
    elif key=='ig_cfg_pcgrad':
        dc=q.s-q.null()[0];dot=inner(d,dc).clamp_max(0)
        first=d-dot/squared(dc).clamp_min(EPS)*dc
        second=dc-dot/squared(d).clamp_min(EPS)*d
        u=first+r*second;constraint=(dot.flatten()<0).to(z.dtype)
    elif key=='cfg_alt_class_reference':
        references=[q.call(labels=ctx.alternate_labels[:,i],kind='full') for i in range(int(r))]
        u=q.s-torch.stack(references).mean(0)
        d=u;info['gap']=d
    elif key=='cfg_symmetric_condition_response':
        conditional,unconditional=q.class_embeddings();delta=conditional-unconditional
        plus=q.call(kind='full',embedding=unconditional+r*delta)
        minus=q.call(kind='full',embedding=unconditional-r*delta)
        u=match((plus-minus)/(2*r),d)
    elif key=='cfg_condition_secant':
        conditional,unconditional=q.class_embeddings()
        changed=q.call(kind='full',embedding=unconditional+r*(conditional-unconditional))
        u=match(changed-q.null()[0],d)
    elif key in ('ig_soft_pag','ig_local_attention','ig_entropy_reference','ig_query_contrast',
                 'ig_value_contrast','ig_attention_head_dropout','ig_mlp_contraction','ig_value_permutation'):
        weak=q.call(kind='weak',perturb=(key,r))
        u=match(q.s-weak,d)
    elif key in ('ig_response_metric','ig_inverse_response'):
        basis,response=q.response_basis()
        with exact_matmul():
            gram=response.transpose(1,2)@response
            eye=torch.eye(2,device=z.device,dtype=z.dtype)
            if key=='ig_response_metric':
                scale=gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(EPS)
                system=eye+r*gram/scale[:,None,None]
                rhs=basis.transpose(1,2)@d.flatten(1)[:,:,None]
            else:
                system=gram+r*eye
                rhs=response.transpose(1,2)@(q.b*d).flatten(1)[:,:,None]
            coefficients=torch.linalg.solve(system,rhs)
            u=(basis@coefficients).squeeze(-1).reshape_as(d)
        if key=='ig_inverse_response':u=cap(u,2*norm(d))
    elif key=='ig_conservative_error':
        index=min(3,int(q.tv*8))
        with exact_matmul():e=UNPATCH(PATCH(z)@state['conservative_a'][index]+state['conservative_b'][index])
        u=d+r*cap(e,norm(d))
    elif key=='ig_robust_head_consensus':
        directions=[d]+[match(q.s-q.multi[f'depth{depth}_v'],d) for depth in (6,10)]
        u=huber_consensus(directions,r)
    else:raise KeyError(key)
    value=q.s+amount*u
    correction=value-q.s
    off=correction-projection(correction,d)
    diag=torch.stack(((norm(correction)/(amount*norm(d)).clamp_min(EPS)).flatten(),
        (squared(off)/squared(correction).clamp_min(EPS)).flatten(),
        (norm(z+q.b*value)/norm(q.m).clamp_min(EPS)).flatten(),constraint,
        (norm(d).flatten()<=EPS).to(z.dtype)),-1)
    info['diagnostics']=diag
    return value,info


def amount_at(config,left_time):
    if left_time>=config['cutoff']:return 0.
    peak=config['strength']
    if config['source']=='cfg':return peak
    if peak==.7:return .6 if left_time<.25 else .7
    return peak*(6/7) if left_time<.25 else peak


def commit_history(ctx,info):
    if 'clean' in info:
        if 'clean' in ctx.history:ctx.history['clean_previous']=ctx.history['clean']
        ctx.history['clean']=info['clean'].detach()
    for key in ('gap','momentum','ema','level','trend','guided_ema'):
        if key in info:ctx.history[key]=info[key].detach()


def strong_heun(rt,z,t,h):
    s=rt.field(z,z.new_tensor(t),'full')
    sp=rt.field(z+h*s,z.new_tensor(t+h),'full')
    return z+(h/2)*(s+sp)


def strang_step(rt,z,t,h,amount,substeps):
    z=strong_heun(rt,z,t,h/2)
    midpoint=z.new_tensor(t+h/2);step=h/substeps
    for _ in range(substeps):
        s,w=rt.pair(z,midpoint);first=amount*(s-w)
        s,w=rt.pair(z+step*first,midpoint);second=amount*(s-w)
        z=z+(step/2)*(first+second)
    return strong_heun(rt,z,t+h/2,h/2)


@torch.inference_mode()
def sample(rt,noise,labels,config,*,batch_seed,zero=False):
    rt.labels=labels;z=noise.clone();begin=rt.counts.copy()
    diagnostics=torch.zeros((len(z),5),device=z.device,dtype=torch.float64)
    queried=0;strang_steps=0
    if config['solver']=='dopri5':
        from torchdiffeq import odeint
        assert config['key']=='native_ig'
        grid=z.new_tensor((0.,.125,.25,.375,.5,1.))
        for k in range(len(grid)-1):
            amount=0. if zero else amount_at(config,float(grid[k]))
            z=odeint(lambda t,x:rt.guided(x,t,amount),z,grid[k:k+2],method='dopri5',rtol=.001,atol=1e-6)[-1]
    else:
        assert config['solver']=='heun64'
        generator=torch.Generator(device=z.device).manual_seed(batch_seed)
        ctx=StepContext(shadow=noise.clone())
        offsets=torch.randint(1,100,(len(z),4),device=z.device,generator=generator)
        ctx.alternate_labels=(labels[:,None]+offsets)%100
        for k in range(64):
            t=k/64;h=1/64
            ctx.xi=torch.randn(z.shape,device=z.device,generator=generator,dtype=z.dtype)
            ctx.permutations=torch.argsort(torch.rand((len(z),256),device=z.device,generator=generator),dim=-1)
            heads=rt.model.blocks[0].attn.num_heads
            ctx.head_orders=torch.argsort(torch.rand((len(z),4,heads),device=z.device,generator=generator),dim=-1)
            amount=0. if zero else amount_at(config,t)
            if config['key']=='ig_strang_split' and amount:
                z=strang_step(rt,z,t,h,amount,int(config['theta']));strang_steps+=1
            else:
                first,info=evaluate(rt,z,t,config,amount,ctx)
                shadow=ctx.shadow
                if 'shadow_velocity' in info:ctx.shadow=shadow+h*info['shadow_velocity']
                second,end_info=evaluate(rt,z+h*first,t+h,config,amount,ctx)
                if 'shadow_velocity' in end_info:
                    ctx.shadow=shadow+(h/2)*(info['shadow_velocity']+end_info['shadow_velocity'])
                z=z+(h/2)*(first+second)
                if amount:
                    diagnostics.add_(info['diagnostics'].double()+end_info['diagnostics'].double());queried+=2
                commit_history(ctx,info)
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError(f"{config['arm']}: nonfinite or |state|>1e6 at step {k}")
    if not torch.isfinite(z).all():raise FloatingPointError(f"{config['arm']}: nonfinite endpoint")
    full=rt.counts['full']-begin['full'];prefix=rt.counts['prefix']-begin['prefix']
    native_budget=128+(2*round(config['cutoff']*64)
        if config['source']=='cfg' and config['strength'] and not zero else 0)
    auxiliary=0 if config['solver']=='dopri5' else max(0,full-native_budget)
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=auxiliary,
        diagnostic_queries=queried,strang_active_steps=strang_steps,
        diagnostics=(diagnostics/max(queried,1)).cpu().numpy())


def make_runtime():
    rt=residual.make_runtime('sit_small')
    def move(value):
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,dict):return {key:move(item) for key,item in value.items()}
        return value
    rt.portfolio_assets=move(torch.load(assets.ROOT/'assets/fitted.pt',map_location='cpu',weights_only=True))
    rt.portfolio_heads={'depth4_v':rt.head}
    module,source=rt.small.load_official_sit_module(rt.small.DEFAULT_OFFICIAL_SIT_REPO,verify_source=True)
    for name,path in assets.EXTRA_HEADS.items():
        rt.portfolio_heads[name]=rt.small.load_internal_head_for_source(checkpoint_path=path,name=name,
            head_weights='ema',model=rt.model,sit_module=module,
            source_checkpoint_path=residual.infrastructure.WORK.parent/'data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0/checkpoints/step_00800000.pt',
            source_metadata=source,device=torch.device('cuda'))
    return rt
