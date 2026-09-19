"""Independent audit of guidance conventions, JVPs, and two distinct future targets."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import signal
import time
import numpy as np
import torch
from scipy.linalg import expm
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core,pipeline as p,study
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha,WORK

ROOT=p.ROOT/'independent_jacobian_audit'
PROTOCOL=WORK/'docs/SIT_FSG_JACOBIAN_AUDIT_PROTOCOL_20260911_ZH.md'


class Affine:
    def __init__(self):
        self.a=torch.tensor([[.3,1.],[-.2,-.7]],dtype=torch.float64)
        self.b=torch.tensor([.5,-.2],dtype=torch.float64)
        self.q=torch.tensor([.2,.1],dtype=torch.float64)
        self.change=torch.tensor([[.4,-.1],[.3,.2]],dtype=torch.float64)
        self.offset=torch.tensor([.1,.3],dtype=torch.float64)
        self.labels=torch.zeros(1,dtype=torch.long);self.counts=dict(full=0,prefix=0)

    def field(self,x,t,kind):
        self.counts['full']+=1
        a,b=self.a,self.b
        if int(self.labels[0])!=100:a,b=a+self.change,b+self.offset
        return (x.flatten(1)@a.T+b+float(t)*self.q).reshape_as(x)


def cpu_audit():
    rt=Affine();x=torch.tensor([.7,-.4],dtype=torch.float64).reshape(1,1,1,2)
    labels=torch.zeros(1,dtype=torch.long);rows=[]
    for horizon in (1/1024,1/32,.125,.25):
        direction=torch.tensor([.2,-.3],dtype=torch.float64).reshape_as(x)
        function=lambda z:core.interval(rt,z,.375,.375+horizon,labels,null=True,steps=8)
        epsilon=.001;d=direction/core.norm(direction)
        finite=(function(x+epsilon*d)-function(x-epsilon*d))/(2*epsilon)*core.norm(direction)
        step=horizon/8
        one=torch.eye(2,dtype=torch.float64)+step*rt.a+(step*step/2)*(rt.a@rt.a)
        discrete=torch.linalg.matrix_power(one,8)@direction.flatten()
        exact=torch.from_numpy(expm(horizon*rt.a.numpy()))@direction.flatten()
        error=float((finite.flatten()-discrete).norm()/discrete.norm())
        assert error<1e-10,error
        rows.append(dict(horizon=horizon,finite_difference_vs_exact_discrete_jvp=error,
            discrete_vs_continuous_jvp=float((discrete-exact).norm()/exact.norm())))
    # The limiting displacement is H(1+a)g for Cw followed by U inverse.
    g=torch.tensor([.15,-.2],dtype=torch.float64)
    u=torch.tensor([.8,.3],dtype=torch.float64)
    for amount in (0.,1.25,2.75):
        for horizon in (1/1024,1/32,.125):
            c=u+g
            delta=horizon*(c+amount*g)-horizon*u
            torch.testing.assert_close(delta,horizon*(1+amount)*g,rtol=1e-12,atol=1e-14)
    return dict(passed=True,affine_jvp=rows,constant_field_weight_w_equals_1_plus_a=True)


def pause_sampling():
    status=read(p.ROOT/'status.json')
    if status['phase']=='complete':return None
    assert status['phase'] in ('sampling','evaluation','preflight'),status
    pid=status['controller_pid']
    assert 'sit_fsg_ctrl_hypothesis_20260911.pipeline' in Path(f'/proc/{pid}/cmdline').read_text()
    os.kill(pid,signal.SIGSTOP)
    atomic(ROOT/'pause.json',dict(status=status,paused_unix=time.time()))
    try:
        if status['phase']=='sampling':
            paths=[p.ROOT/p.STAGE/status['arm']/f'rank{rank}/summary.json' for rank in range(4)]
        elif status['phase']=='preflight':
            paths=[p.ROOT/p.STAGE/'runs'/status['run_id']/f'ready{rank}.json' for rank in range(4)]
        else:paths=[]
        while not all(path.exists() and read(path).get('run_id')==status['run_id'] for path in paths):time.sleep(1)
        return pid
    except BaseException:
        os.kill(pid,signal.SIGCONT);raise


def neural_audit():
    rt=core.old.make_runtime()
    rows=[];precision=[]
    labels=np.load(p.ROOT/'inputs/labels.npy')
    for base,amount in (('cfg_tuned',1.25),('cfg_high',2.75)):
        with np.load(study.state_path(base,0)) as data:states=data['x24'].copy()
        for index in (0,3):
            x=torch.from_numpy(states[index:index+1]).cuda()
            y=torch.from_numpy(labels[index:index+1].copy()).cuda()
            for horizon in (1/32,.125):
                t=.375;end=t+horizon
                # Use the differentiable math attention backend for both AD and FD.
                with core.old.exact_matmul(),torch.backends.cuda.sdp_kernel(
                        enable_flash=False,enable_math=True,enable_mem_efficient=False,enable_cudnn=False):
                    function=lambda z:core.interval(rt,z,t,end,y,null=True,steps=8)
                    with torch.no_grad():
                        c,u=core.pair(rt,x,t,y);gap=c-u
                        null=function(x)
                        flow_target=core.interval(rt,x,t,end,y,amount=amount,steps=8)
                        euler_target=x+horizon*(c+amount*gap)
                        actual=euler_target-horizon*core.field(rt,euler_target,end,y,True)-x
                        epsilon=.001*core.norm(x).clamp_min(64.)
                        radius=(4/64)*amount*core.norm(gap)
                        flow_pull=core.interval(rt,flow_target,end,t,y,null=True,steps=8)-x
                        euler_pull=core.interval(rt,euler_target,end,t,y,null=True,steps=8)-x
                        variants=dict(euler_fsg=actual,cfg_equal=core.norm(actual)*core.unit(gap),
                            cfg_first_order=horizon*(1+amount)*gap,flow_inverse=flow_pull,
                            inverse_euler_target=euler_pull,capped_euler=core.cap(actual,radius),
                            capped_cfg=core.norm(core.cap(actual,radius))*core.unit(gap))
                    for variant,delta in variants.items():
                        with torch.no_grad():
                            d=core.unit(delta)
                            fd=(function(x+epsilon*d)-function(x-epsilon*d))/(2*epsilon)*core.norm(delta)
                            fd2=(function(x+2*epsilon*d)-function(x-2*epsilon*d))/(4*epsilon)*core.norm(delta)
                            finite=function(x+delta)-null
                            values=dict(displacement_rms=core.rms(delta),
                                effective_coefficient_per_H=float((core.norm(delta)/(horizon*core.norm(gap))).item()),
                                first_order_length_ratio=float((core.norm(delta)/(horizon*(1+amount)*core.norm(gap))).item()),
                                finite_error_flow_target=float(study.relative(finite-(flow_target-null),flow_target-null).item()),
                                finite_error_own_euler_target=float(study.relative(finite-(euler_target-null),euler_target-null).item()),
                                linear_error_flow_target=float(study.relative(fd-(flow_target-null),flow_target-null).item()),
                                linear_error_own_euler_target=float(study.relative(fd-(euler_target-null),euler_target-null).item()),
                                fd_epsilon_vs_double=float(study.relative(fd-fd2,fd).item()),
                                flow_vs_euler_target_difference=float(study.relative(flow_target-euler_target,flow_target-null).item()))
                            values['displacement_rms']=float(values['displacement_rms'].item())
                        if variant in ('euler_fsg','cfg_equal'):
                            _,automatic=torch.autograd.functional.jvp(function,x,delta,create_graph=False,strict=True)
                            ad_error=float(study.relative(automatic-fd,automatic).item())
                            values['autograd_vs_finite_relative_error']=ad_error
                            values['autograd_vs_finite_cosine']=float(study.cosine(automatic,fd).item())
                            assert ad_error<.02,(base,index,horizon,variant,ad_error)
                        rows.append(dict(base=base,index=index,amount=amount,w=1+amount,k=24,
                            horizon=horizon,variant=variant,**values))
                    print(dict(base=base,index=index,horizon=horizon,completed=True),flush=True)
    return rows


def main(pause=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    request=dict(sources={str(path):sha(path) for path in [Path(__file__),PROTOCOL,Path(core.__file__),Path(study.__file__)]},
        original_request_sha256=sha(p.ROOT/'study_request.json'),states=[0,3],time=.375,
        bases=['cfg_tuned','cfg_high'],horizons=[1/32,.125],
        epsilon='.001 max(norm(x),64)',jacobian='Directional JVP of the discrete eight-step NULL Heun flow',
        independent_targets=['high-accuracy guided flow','the adapter own Euler forward endpoint'],
        created_unix=time.time())
    if (ROOT/'request.json').exists():
        old=read(ROOT/'request.json');request['created_unix']=old['created_unix'];assert request==old
    else:atomic(ROOT/'request.json',request)
    cpu=cpu_audit();atomic(ROOT/'cpu.json',cpu)
    pid=pause_sampling() if pause else None
    try:
        rows=neural_audit()
        for path,digest in request['sources'].items():assert sha(path)==digest,path
        atomic(ROOT/'results.json',dict(passed=True,cpu=cpu,rows=rows,request_sha256=sha(ROOT/'request.json')))
    finally:
        if pid is not None:
            os.kill(pid,signal.SIGCONT);atomic(ROOT/'resumed.json',dict(controller_pid=pid,resumed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pause-small',action='store_true')
    main(parser.parse_args().pause_small)
