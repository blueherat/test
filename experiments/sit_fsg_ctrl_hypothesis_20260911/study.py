"""Image-visible suffixes, fixed-target interventions, and directional flow checks."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core, pipeline as p
from experiments.sit_fsg_pasted_20260910 import core as semantic, readout
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

TRAJECTORIES = ('cfg_tuned','cfg_high','fsg_high','fsg_length_high','smc_high',
                'instant_high','soft_high','norm_high','refined_high')
TIMES=(8,16,24,32,40,48)
INTERVENTION_TIMES=(8,24,40)


def scalar(value):
    return value.detach().flatten().cpu().numpy()


def cosine(a,b):
    return (core.inner(a,b)/(core.norm(a)*core.norm(b)).clamp_min(1e-10)).flatten()


def relative(a,b):
    return (core.norm(a)/core.norm(b).clamp_min(1e-10)).flatten()


def encode_records(size,context,values):
    arrays={key:scalar(value) if isinstance(value,torch.Tensor) else np.asarray(value)
            for key,value in values.items()}
    assert all(np.asarray(value).shape==(size,) for value in arrays.values()), {
        key:np.asarray(value).shape for key,value in arrays.items()}
    assert all(np.isfinite(value).all() for value in arrays.values())
    return [dict(context,index=context['start']+j,**{key:float(value[j]) for key,value in arrays.items()})
            for j in range(size)]


class Readouts:
    def __init__(self,rt):
        self.rt=rt;self.resnet=readout.IndependentReadout()
        self.conv=semantic.SemanticReadout(rt)

    def images(self,pixels,labels):
        values=self.resnet.images(pixels,scalar(labels).astype(np.int64))
        image=torch.from_numpy(np.array(pixels)).cuda().permute(0,3,1,2).float()/255
        with core.old.exact_matmul():
            logits=self.conv.model(self.conv.transform(image))
        target=self.conv.original_labels[labels]
        prob=logits.softmax(-1).gather(1,target[:,None]).squeeze(1)
        correct=logits.argmax(-1)==target
        return dict(resnet_p=values[:,0],resnet_top1=values[:,1],resnet_prediction=values[:,2],
            convnext_p=scalar(prob),convnext_top1=scalar(correct),convnext_prediction=scalar(logits.argmax(-1)))

    def latents(self,z,labels,folder=None,start=0):
        pixels=self.rt.decode(z)
        if folder is not None:
            folder.mkdir(parents=True,exist_ok=True)
            for j,image in enumerate(pixels):
                Image.fromarray(image).save(folder/f'{start+j:03d}.png')
        return self.images(pixels,labels)


def bundle_readouts(reader,values,labels,folder,start):
    result={}
    for key,value in values.items():
        got=reader.latents(value,labels,folder/key,start)
        result.update({f'{key}_{name}':item for name,item in got.items()})
    return result


def state_path(method,start):
    return p.ROOT/'trajectories'/f'{method}_{start:03d}.npz'


def load_state(method,start,step):
    with np.load(state_path(method,start)) as data:
        return torch.from_numpy(data[f'x{step:02d}'].copy()).cuda()


def trajectories(rt,reader,noise,labels,rank,request_hash):
    folder=p.ROOT/'trajectories'
    for start in range(rank*8,len(noise),32):
        n,y=noise[start:start+8],labels[start:start+8]
        for method in TRAJECTORIES:
            destination=folder/f'{method}_{start:03d}.json'
            if destination.exists():
                assert read(destination)['request_sha256']==request_hash
                continue
            begin=time.perf_counter();initial=rt.counts.copy()
            endpoint,stats,snapshots,traces=core.trajectory(rt,n,y,method,capture=TIMES,trace=True)
            images=p.ROOT/'images'/'trajectories'/method
            final_quality=reader.latents(endpoint,y,images/'guided',start)
            rows=encode_records(len(n),dict(start=start,method=method,k=64,suffix='guided'),final_quality)
            arrays=dict(noise=scalar(n).reshape(n.shape),labels=scalar(y),endpoint=endpoint.cpu().numpy())
            for step,snapshot in snapshots.items():
                x=snapshot['state'];arrays[f'x{step:02d}']=x.cpu().numpy()
                if snapshot['memory'] is not None:
                    arrays[f'memory{step:02d}']=snapshot['memory'].cpu().numpy()
                c,u=core.pair(rt,x,step/64,y)
                gap=c-u
                for suffix in ('null','conditional'):
                    before=rt.counts.copy()
                    end=core.future(rt,x,step,y,suffix)
                    quality=reader.latents(end,y,images/f'{suffix}_{step:02d}',start)
                    values=dict(quality,gap_rms=core.rms(gap),clean_gap_rms=(1-step/64)*core.rms(gap),
                        noise_gap_rms=(step/64)*core.rms(gap),
                        guided_endpoint_distance=core.rms(end-endpoint),
                        prefix_full_calls=np.full(len(n),snapshot['full_calls']),
                        suffix_full_calls=np.full(len(n),rt.counts['full']-before['full']))
                    rows+=encode_records(len(n),dict(start=start,method=method,k=step,suffix=suffix),values)
                    arrays[f'{suffix}{step:02d}']=end.cpu().numpy()
            trace_rows=[]
            for trace in traces:
                trace_rows+=encode_records(len(n),dict(start=start,method=method,k=trace['k']),
                    {key:value for key,value in trace.items() if key!='k'})
            with state_path(method,start).open('wb') as stream:
                np.savez(stream,**arrays)
            atomic(destination,dict(request_sha256=request_hash,method=method,start=start,
                rows=rows,traces=trace_rows,sampling_full_calls=stats['full_calls'],
                diagnostic_full_calls=rt.counts['full']-initial['full']-stats['full_calls'],
                elapsed_seconds=time.perf_counter()-begin,state_sha256=sha(state_path(method,start))))
            print(dict(part='trajectories',rank=rank,method=method,start=start,
                seconds=round(time.perf_counter()-begin,2)),flush=True)


@core.old.exact_matmul()
def interventions(rt,reader,noise,labels,rank,request_hash):
    folder=p.ROOT/'interventions'
    for start in range(rank*8,64,32):
        n,y=noise[start:start+8],labels[start:start+8]
        for base,amount in (('cfg_tuned',1.25),('cfg_high',2.75)):
            for step in INTERVENTION_TIMES:
                path=folder/f'{base}_{step:02d}_{start:03d}.json'
                if path.exists():
                    assert read(path)['request_sha256']==request_hash
                    continue
                begin=time.perf_counter();before=rt.counts.copy()
                state=load_state(base,start,step)
                c0=core.future(rt,state,step,y,'conditional')
                u0=core.future(rt,state,step,y,'null')
                guide0=core.future(rt,state,step,y,'guided',amount=amount)
                candidates,setup,records=core.proposals(rt,state,step,y,amount)
                frozen_images=p.ROOT/'images'/'interventions'/base/f'{step:02d}'
                frozen=bundle_readouts(reader,dict(reference_c=c0,reference_guided=guide0),y,
                    frozen_images,start)
                if start<32:
                    inv=core.inverse_null(rt,c0,step,y)
                    candidates['full_inverse_oracle']=inv
                    candidates['full_inverse_capped']=state+core.cap(inv-state,setup['radius'])
                rows=[];array_values={}
                for variant,moved in candidates.items():
                    c=core.future(rt,moved,step,y,'conditional')
                    u=core.future(rt,moved,step,y,'null')
                    vc,vu=core.pair(rt,moved,step/64,y)
                    metrics=dict(shift_rms=core.rms(moved-state),radius_rms=setup['radius'].flatten()/64,
                        local_gap_rms=core.rms(vc-vu),write_rms=core.rms(u-c0),agreement_rms=core.rms(u-c),
                        guided_write_rms=core.rms(u-guide0),conditional_drift_rms=core.rms(c-c0),
                        null_change_rms=core.rms(u-u0),direction_cos_cfg=cosine(moved-state,setup['gap']),
                        coarse_reference_c_error=core.rms(setup['baseline']['c']-c0),
                        coarse_reference_u_error=core.rms(setup['baseline']['u']-u0))
                    metrics.update(bundle_readouts(reader,dict(c=c,u=u),y,frozen_images/variant,start))
                    metrics.update(frozen)
                    rows+=encode_records(len(n),dict(start=start,base=base,amount=amount,k=step,variant=variant),metrics)
                    array_values[f'{variant}_c']=c.cpu().numpy();array_values[f'{variant}_u']=u.cpu().numpy()
                    array_values[f'{variant}_state']=moved.cpu().numpy()
                fits={key:{name:scalar(value).tolist() for name,value in record.items()
                           if name in ('before','after','shift_rms','index')} for key,record in records.items()}
                arrays=path.with_suffix('.npz')
                np.savez(arrays,reference_c=c0.cpu().numpy(),reference_u=u0.cpu().numpy(),
                         reference_guided=guide0.cpu().numpy(),**array_values)
                gram=setup['q'].transpose(1,2)@setup['q']
                identity=torch.eye(4,device=state.device)[None]
                atomic(path,dict(request_sha256=request_hash,rows=rows,fits=fits,
                    basis_orthogonality_max_error=float((gram-identity).abs().max()),
                    total_full_calls=rt.counts['full']-before['full'],arrays_sha256=sha(arrays),
                    elapsed_seconds=time.perf_counter()-begin))
                print(dict(part='interventions',rank=rank,base=base,k=step,start=start,
                    seconds=round(time.perf_counter()-begin,2)),flush=True)


@core.old.exact_matmul()
def jacobian(rt,reader,noise,labels,rank,request_hash):
    folder=p.ROOT/'jacobian'
    for start in range(rank*8,32,32):
        y=labels[start:start+8]
        for base,amount in (('cfg_tuned',1.25),('cfg_high',2.75)):
            for step in INTERVENTION_TIMES:
                path=folder/f'{base}_{step:02d}_{start:03d}.json'
                if path.exists():
                    assert read(path)['request_sha256']==request_hash
                    continue
                state=load_state(base,start,step);t=step/64
                _,_,gap=core.fsg_delta(rt,state,t,y,amount)
                eps=.001*core.norm(state).clamp_min(64.)
                rows=[];begin=time.perf_counter();before=rt.counts.copy()
                for horizon in (2/64,4/64,8/64,16/64):
                    end=t+horizon
                    def u(x):return core.interval(rt,x,t,end,y,null=True,steps=8)
                    # This local conditional future includes the full guidance weight w=1+a.
                    condition=core.interval(rt,state,t,end,y,amount=amount,steps=8)
                    null=u(state);desired=condition-null
                    pulled=core.interval(rt,condition,end,t,y,null=True,steps=8)
                    exact_delta=pulled-state
                    same=core.interval(rt,null,end,t,y,null=True,steps=8)-state
                    euler,radius,_=core.fsg_delta(rt,state,t,y,amount,horizon)
                    euler_same,_,_=core.fsg_delta(rt,state,t,y,amount,horizon,True)
                    torch.backends.cuda.matmul.allow_tf32=True
                    try:
                        native_euler,_,_=core.fsg_delta(rt,state,t,y,amount,horizon)
                    finally:
                        torch.backends.cuda.matmul.allow_tf32=False
                    euler_capped=core.cap(euler,radius)
                    directions=dict(flow_inverse=exact_delta,flow_cfg_equal=core.norm(exact_delta)*core.unit(gap),
                        euler_fsg=euler,euler_cfg_equal=core.norm(euler)*core.unit(gap),
                        euler_capped=euler_capped,capped_cfg_equal=core.norm(euler_capped)*core.unit(gap),
                        flow_capped=core.cap(exact_delta,radius),
                        euler_debiased=euler-euler_same,same_field_flow=same,same_field_euler=euler_same)
                    for variant,delta in directions.items():
                        d=core.unit(delta)
                        j=(u(state+eps*d)-u(state-eps*d))/(2*eps)
                        j2=(u(state+2*eps*d)-u(state-2*eps*d))/(4*eps)
                        jd=j*core.norm(delta)
                        actual=u(state+delta)-null
                        gain=(core.inner(jd,desired)/core.inner(jd,jd).clamp_min(1e-10)).clamp_min(0.)
                        metrics=dict(displacement_rms=core.rms(delta),desired_rms=core.rms(desired),
                            direction_cos_cfg=cosine(delta,gap),
                            local_cfg_limit_error=relative(delta-horizon*(1+amount)*gap,horizon*(1+amount)*gap),
                            jacobian_target_relative_error=relative(jd-desired,desired),
                            finite_target_relative_error=relative(actual-desired,desired),
                            linearization_relative_error=relative(actual-jd,actual),
                            finite_difference_relative_disagreement=relative(j2-j,j),
                            response_target_cosine=cosine(jd,desired),
                            best_linear_ray_relative_error=relative(gain*jd-desired,desired),
                            achieved_progress_fraction=(core.inner(actual,desired)/core.inner(desired,desired).clamp_min(1e-10)).flatten(),
                            native_euler_relative_difference=relative(native_euler-euler,euler),
                            flow_same_field_relative_error=relative(same,exact_delta))
                        rows+=encode_records(len(state),dict(start=start,base=base,amount=amount,k=step,
                            horizon=horizon,variant=variant),metrics)
                    # Numerical resolution control for the ideal forward/inverse map.
                    fine_c=core.interval(rt,state,t,end,y,amount=amount,steps=16)
                    fine_pull=core.interval(rt,fine_c,end,t,y,null=True,steps=16)
                    rows+=encode_records(len(state),dict(start=start,base=base,amount=amount,k=step,
                        horizon=horizon,variant='resolution'),dict(
                            inverse_resolution_relative_error=relative(fine_pull-pulled,exact_delta),
                            conditional_resolution_relative_error=relative(fine_c-condition,desired)))
                for horizon in (1/1024,1/256):
                    end=t+horizon
                    condition=core.interval(rt,state,t,end,y,amount=amount,steps=4)
                    pulled=core.interval(rt,condition,end,t,y,null=True,steps=4)
                    delta=pulled-state
                    rows+=encode_records(len(state),dict(start=start,base=base,amount=amount,k=step,
                        horizon=horizon,variant='short_limit'),dict(direction_cos_cfg=cosine(delta,gap),
                            local_cfg_limit_error=relative(delta-horizon*(1+amount)*gap,horizon*(1+amount)*gap)))
                atomic(path,dict(request_sha256=request_hash,rows=rows,
                    total_full_calls=rt.counts['full']-before['full'],elapsed_seconds=time.perf_counter()-begin))
                print(dict(part='jacobian',rank=rank,base=base,k=step,start=start,
                    seconds=round(time.perf_counter()-begin,2)),flush=True)


@torch.inference_mode()
def run_rank(part,rank):
    request=p.verify();request_hash=sha(p.ROOT/'study_request.json')
    folder=p.ROOT/part;folder.mkdir(exist_ok=True)
    rt=core.old.make_runtime()
    for source,digest in rt.sources.items():
        assert request['sources'][source]==digest,source
    noise=torch.from_numpy(np.load(p.ROOT/'inputs/noise.npy').copy()).cuda()
    labels=torch.from_numpy(np.load(p.ROOT/'inputs/labels.npy').copy()).cuda()
    reader=Readouts(rt) if part!='jacobian' else None
    begin=time.perf_counter()
    globals()[part](rt,reader,noise,labels,rank,request_hash)
    atomic(folder/f'rank{rank}.json',dict(complete=True,rank=rank,request_sha256=request_hash,
        elapsed_seconds=time.perf_counter()-begin))
